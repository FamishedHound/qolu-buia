"""Tests for the categorical (C51-style) distributional RL primitives.

These follow Bellemare et al. (2017) "A Distributional Perspective on
Reinforcement Learning", which the paper explicitly cites for its atom support.
"""

from __future__ import annotations

import math

import pytest
import torch

from chapter4_qolu.agent.distributional import (
    AtomSupport,
    categorical_kl_divergence,
    combine_dueling_streams,
    project_categorical_distribution,
)


class TestAtomSupport:
    def test_paper_defaults_match_table_1(self) -> None:
        support = AtomSupport()
        assert support.n_atoms == 51
        assert support.v_min == pytest.approx(-200.0)
        assert support.v_max == pytest.approx(200.0)

    def test_delta_z_matches_paper_formula(self) -> None:
        support = AtomSupport(n_atoms=51, v_min=-200.0, v_max=200.0)
        assert support.delta_z == pytest.approx((200.0 - -200.0) / (51 - 1))
        assert support.delta_z == pytest.approx(8.0)

    def test_atoms_are_equally_spaced_and_span_the_interval(self) -> None:
        support = AtomSupport(n_atoms=5, v_min=-2.0, v_max=2.0)
        atoms = support.atoms()
        assert atoms.shape == (5,)
        torch.testing.assert_close(atoms, torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))

    def test_atom_i_equals_vmin_plus_i_delta_z(self) -> None:
        support = AtomSupport(n_atoms=11, v_min=-5.0, v_max=5.0)
        atoms = support.atoms()
        for i in range(support.n_atoms):
            assert atoms[i].item() == pytest.approx(support.v_min + i * support.delta_z)

    def test_rejects_degenerate_configuration(self) -> None:
        with pytest.raises(ValueError):
            AtomSupport(n_atoms=1, v_min=-1.0, v_max=1.0)
        with pytest.raises(ValueError):
            AtomSupport(n_atoms=51, v_min=1.0, v_max=-1.0)


class TestCategoricalProjection:
    def test_terminal_transition_places_mass_on_reward(self) -> None:
        # delta_z = 8.0; a reward of 4.0 sits exactly half-way between the atoms
        # at 0.0 (index 25) and 8.0 (index 26).
        support = AtomSupport()
        next_probs = torch.zeros(1, 51)
        next_probs[0, 0] = 1.0  # ignored: transition is terminal
        projected = project_categorical_distribution(
            next_probs=next_probs,
            rewards=torch.tensor([4.0]),
            dones=torch.tensor([1.0]),
            discount=0.99,
            support=support,
        )
        assert projected[0, 25].item() == pytest.approx(0.5)
        assert projected[0, 26].item() == pytest.approx(0.5)
        assert projected.sum().item() == pytest.approx(1.0)

    def test_exact_atom_hit_puts_all_mass_on_one_atom(self) -> None:
        support = AtomSupport()
        next_probs = torch.zeros(1, 51)
        next_probs[0, 10] = 1.0
        projected = project_categorical_distribution(
            next_probs=next_probs,
            rewards=torch.tensor([8.0]),  # lands exactly on atom index 26 (= 8.0)
            dones=torch.tensor([1.0]),
            discount=0.99,
            support=support,
        )
        assert projected[0, 26].item() == pytest.approx(1.0)
        assert projected.sum().item() == pytest.approx(1.0)

    def test_bootstrapped_transition_uses_reward_plus_discounted_atom(self) -> None:
        support = AtomSupport(n_atoms=5, v_min=-2.0, v_max=2.0)  # delta_z = 1.0
        next_probs = torch.zeros(1, 5)
        next_probs[0, 4] = 1.0  # next-state return concentrated at z = 2.0
        projected = project_categorical_distribution(
            next_probs=next_probs,
            rewards=torch.tensor([0.0]),
            dones=torch.tensor([0.0]),
            discount=0.5,
            support=support,
        )
        # target value = 0 + 0.5 * 2.0 = 1.0 -> atom index 3 exactly.
        assert projected[0, 3].item() == pytest.approx(1.0)

    def test_linear_interpolation_between_neighbouring_atoms(self) -> None:
        support = AtomSupport(n_atoms=5, v_min=-2.0, v_max=2.0)
        next_probs = torch.zeros(1, 5)
        next_probs[0, 4] = 1.0  # z = 2.0
        projected = project_categorical_distribution(
            next_probs=next_probs,
            rewards=torch.tensor([0.0]),
            dones=torch.tensor([0.0]),
            discount=0.75,
            support=support,
        )
        # target value = 1.5 -> halfway between atoms 3 (1.0) and 4 (2.0).
        assert projected[0, 3].item() == pytest.approx(0.5)
        assert projected[0, 4].item() == pytest.approx(0.5)

    def test_values_above_vmax_are_clipped_to_the_last_atom(self) -> None:
        support = AtomSupport(n_atoms=5, v_min=-2.0, v_max=2.0)
        next_probs = torch.zeros(1, 5)
        next_probs[0, 2] = 1.0
        projected = project_categorical_distribution(
            next_probs=next_probs,
            rewards=torch.tensor([99.0]),
            dones=torch.tensor([1.0]),
            discount=0.99,
            support=support,
        )
        assert projected[0, 4].item() == pytest.approx(1.0)

    def test_values_below_vmin_are_clipped_to_the_first_atom(self) -> None:
        support = AtomSupport(n_atoms=5, v_min=-2.0, v_max=2.0)
        next_probs = torch.zeros(1, 5)
        next_probs[0, 2] = 1.0
        projected = project_categorical_distribution(
            next_probs=next_probs,
            rewards=torch.tensor([-99.0]),
            dones=torch.tensor([1.0]),
            discount=0.99,
            support=support,
        )
        assert projected[0, 0].item() == pytest.approx(1.0)

    def test_projection_preserves_total_probability_mass_for_a_batch(self) -> None:
        support = AtomSupport()
        generator = torch.Generator().manual_seed(0)
        logits = torch.randn(16, 51, generator=generator)
        next_probs = torch.softmax(logits, dim=-1)
        rewards = torch.randn(16, generator=generator) * 10.0
        dones = (torch.rand(16, generator=generator) < 0.3).float()
        projected = project_categorical_distribution(
            next_probs=next_probs,
            rewards=rewards,
            dones=dones,
            discount=0.99,
            support=support,
        )
        assert projected.shape == (16, 51)
        torch.testing.assert_close(projected.sum(dim=-1), torch.ones(16))
        assert torch.all(projected >= 0.0)

    def test_accepts_per_sample_discounts_for_n_step_returns(self) -> None:
        support = AtomSupport(n_atoms=5, v_min=-2.0, v_max=2.0)
        next_probs = torch.zeros(2, 5)
        next_probs[:, 4] = 1.0
        projected = project_categorical_distribution(
            next_probs=next_probs,
            rewards=torch.zeros(2),
            dones=torch.zeros(2),
            discount=torch.tensor([0.5, 0.0]),
            support=support,
        )
        assert projected[0, 3].item() == pytest.approx(1.0)  # 0.5 * 2.0 = 1.0
        assert projected[1, 2].item() == pytest.approx(1.0)  # 0.0 * 2.0 = 0.0

    def test_preserves_the_bellman_mean_when_nothing_is_clipped(self) -> None:
        # The categorical projection is mean-preserving as long as the target
        # values stay inside [V_min, V_max]; this pins the interpolation weights
        # down independently of the point-mass tests above.
        support = AtomSupport()
        generator = torch.Generator().manual_seed(11)
        # Keep the next-state mass away from the ends of the support so that
        # r + gamma * z can never leave [V_min, V_max] and get truncated.
        logits = torch.full((32, 51), -1e9)
        logits[:, 10:41] = torch.randn(32, 31, generator=generator)
        next_probs = torch.softmax(logits, dim=-1)
        rewards = torch.randn(32, generator=generator) * 5.0
        dones = torch.zeros(32)
        discount = 0.99
        atoms = support.atoms()

        projected = project_categorical_distribution(
            next_probs=next_probs,
            rewards=rewards,
            dones=dones,
            discount=discount,
            support=support,
        )
        expected = rewards + discount * (next_probs * atoms).sum(dim=-1)
        torch.testing.assert_close(
            (projected * atoms).sum(dim=-1), expected, atol=1e-4, rtol=1e-4
        )

    def test_rejects_mismatched_batch_shapes(self) -> None:
        support = AtomSupport(n_atoms=5, v_min=-2.0, v_max=2.0)
        with pytest.raises(ValueError):
            project_categorical_distribution(
                next_probs=torch.zeros(3, 5),
                rewards=torch.zeros(2),
                dones=torch.zeros(3),
                discount=0.9,
                support=support,
            )

    def test_rejects_support_size_mismatch(self) -> None:
        support = AtomSupport(n_atoms=5, v_min=-2.0, v_max=2.0)
        with pytest.raises(ValueError):
            project_categorical_distribution(
                next_probs=torch.zeros(2, 7),
                rewards=torch.zeros(2),
                dones=torch.zeros(2),
                discount=0.9,
                support=support,
            )


class TestCategoricalKLDivergence:
    def test_zero_when_distributions_are_identical(self) -> None:
        probs = torch.tensor([[0.2, 0.3, 0.5]])
        loss = categorical_kl_divergence(probs, probs.log())
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_matches_analytic_kl_value(self) -> None:
        target = torch.tensor([[0.25, 0.75]])
        predicted = torch.tensor([[0.5, 0.5]])
        expected = 0.25 * math.log(0.25 / 0.5) + 0.75 * math.log(0.75 / 0.5)
        loss = categorical_kl_divergence(target, predicted.log())
        assert loss.item() == pytest.approx(expected, rel=1e-6)

    def test_is_non_negative(self) -> None:
        generator = torch.Generator().manual_seed(1)
        target = torch.softmax(torch.randn(8, 51, generator=generator), dim=-1)
        predicted = torch.softmax(torch.randn(8, 51, generator=generator), dim=-1)
        loss = categorical_kl_divergence(target, predicted.log(), reduction="none")
        assert torch.all(loss >= -1e-6)

    def test_reduction_modes(self) -> None:
        target = torch.tensor([[0.25, 0.75], [0.6, 0.4]])
        predicted = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        per_sample = categorical_kl_divergence(
            target, predicted.log(), reduction="none"
        )
        assert per_sample.shape == (2,)
        mean = categorical_kl_divergence(target, predicted.log(), reduction="mean")
        total = categorical_kl_divergence(target, predicted.log(), reduction="sum")
        assert mean.item() == pytest.approx(per_sample.mean().item())
        assert total.item() == pytest.approx(per_sample.sum().item())

    def test_supports_importance_weights(self) -> None:
        target = torch.tensor([[0.25, 0.75], [0.6, 0.4]])
        predicted = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        per_sample = categorical_kl_divergence(
            target, predicted.log(), reduction="none"
        )
        weights = torch.tensor([2.0, 0.0])
        weighted = categorical_kl_divergence(
            target, predicted.log(), reduction="mean", weights=weights
        )
        assert weighted.item() == pytest.approx(per_sample[0].item() * 2.0 / 2.0)

    def test_gradients_flow_to_the_prediction(self) -> None:
        logits = torch.zeros(1, 3, requires_grad=True)
        target = torch.tensor([[0.1, 0.2, 0.7]])
        loss = categorical_kl_divergence(target, torch.log_softmax(logits, dim=-1))
        loss.backward()
        assert logits.grad is not None
        assert torch.any(logits.grad != 0.0)

    def test_rejects_invalid_reduction(self) -> None:
        probs = torch.tensor([[0.5, 0.5]])
        with pytest.raises(ValueError):
            categorical_kl_divergence(probs, probs.log(), reduction="median")

    def test_descending_the_gradient_moves_the_prediction_towards_the_target(
        self,
    ) -> None:
        target = torch.tensor([[0.1, 0.2, 0.7]])
        logits = torch.zeros(1, 3, requires_grad=True)
        optimiser = torch.optim.SGD([logits], lr=1.0)
        first = categorical_kl_divergence(target, torch.log_softmax(logits, dim=-1))
        for _ in range(50):
            optimiser.zero_grad()
            loss = categorical_kl_divergence(target, torch.log_softmax(logits, dim=-1))
            loss.backward()
            optimiser.step()
        final = categorical_kl_divergence(target, torch.log_softmax(logits, dim=-1))
        assert final.item() < first.item()
        torch.testing.assert_close(
            torch.softmax(logits.detach(), dim=-1), target, atol=1e-3, rtol=1e-3
        )

    def test_ignores_zero_probability_target_atoms(self) -> None:
        # Projected targets are sparse: most atoms carry exactly zero mass and
        # must not turn into NaNs when the prediction assigns them ~0 as well.
        target = torch.tensor([[0.0, 1.0, 0.0]])
        predicted_log_probs = torch.log(torch.tensor([[1e-45, 1.0, 1e-45]]))
        loss = categorical_kl_divergence(target, predicted_log_probs)
        assert torch.isfinite(loss)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


class TestDuelingCombination:
    def test_matches_the_papers_formula(self) -> None:
        value = torch.tensor([[[1.0, 2.0]]])  # (batch=1, 1, atoms=2)
        advantage = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])  # (1, actions=2, atoms=2)
        combined = combine_dueling_streams(value, advantage)
        mean_advantage = advantage.mean(dim=1, keepdim=True)
        expected = value + advantage - mean_advantage
        torch.testing.assert_close(combined, expected)
        assert combined.shape == (1, 2, 2)

    def test_advantage_is_mean_centred_across_actions(self) -> None:
        generator = torch.Generator().manual_seed(2)
        value = torch.randn(4, 1, 51, generator=generator)
        advantage = torch.randn(4, 3, 51, generator=generator)
        combined = combine_dueling_streams(value, advantage)
        centred = combined - value
        torch.testing.assert_close(
            centred.mean(dim=1), torch.zeros(4, 51), atol=1e-6, rtol=0
        )

    def test_is_invariant_to_constant_advantage_shifts(self) -> None:
        generator = torch.Generator().manual_seed(3)
        value = torch.randn(2, 1, 5, generator=generator)
        advantage = torch.randn(2, 4, 5, generator=generator)
        base = combine_dueling_streams(value, advantage)
        shifted = combine_dueling_streams(value, advantage + 7.5)
        torch.testing.assert_close(base, shifted, atol=1e-5, rtol=0)

    def test_value_shift_moves_every_action_equally(self) -> None:
        value = torch.zeros(1, 1, 3)
        advantage = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        base = combine_dueling_streams(value, advantage)
        shifted = combine_dueling_streams(value + 2.0, advantage)
        torch.testing.assert_close(shifted - base, torch.full_like(base, 2.0))

    def test_rejects_wrong_value_stream_shape(self) -> None:
        with pytest.raises(ValueError):
            combine_dueling_streams(torch.zeros(1, 2, 3), torch.zeros(1, 2, 3))
