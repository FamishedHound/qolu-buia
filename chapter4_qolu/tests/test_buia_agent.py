"""Tests for the bottom-up investor agent (BUIA, Algorithm 2 + Table 2)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from chapter4_qolu.agent.buia_agent import BUIAAgent, BUIAConfig, Proposal
from chapter4_qolu.agent.buia_network import BUIANetwork, BUIANetworkConfig
from chapter4_qolu.agent.proposal_sampling import sample_without_replacement
from chapter4_qolu.env.parcel_grid import ParcelIndex


def make_network(channels: int = 12, window: int = 5) -> BUIANetwork:
    torch.manual_seed(0)
    return BUIANetwork(BUIANetworkConfig(in_channels=channels, window=window))


def make_agent(**overrides: object) -> BUIAAgent:
    settings: dict[str, object] = {
        "observation_channels": 12,
        "observation_window": 5,
        "max_proposals": 3,
    }
    settings.update(overrides)
    return BUIAAgent(BUIAConfig(**settings), seed=0)  # type: ignore[arg-type]


def observations(count: int, channels: int = 12, window: int = 5) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.random((count, channels, window, window)).astype(np.float32)


class TestBUIANetworkMatchesTable2:
    def test_single_convolution_layer(self) -> None:
        network = make_network()
        convs = [m for m in network.modules() if isinstance(m, nn.Conv2d)]
        assert len(convs) == 1

    def test_kernel_and_filter_counts_match_the_input(self) -> None:
        network = make_network(channels=12, window=5)
        conv = next(m for m in network.modules() if isinstance(m, nn.Conv2d))
        assert conv.in_channels == 12
        assert conv.out_channels == 12  # "filter counts matching the input"
        assert conv.kernel_size == (5, 5)  # "kernel ... matching the input"

    def test_mlp_is_128_128_64(self) -> None:
        network = make_network()
        widths = [
            (m.in_features, m.out_features)
            for m in network.mlp.modules()
            if isinstance(m, nn.Linear)
        ]
        assert [out for _, out in widths] == [128, 128, 64]
        assert widths[0][0] == 12  # flattened conv output

    def test_outputs_one_logit_per_candidate_parcel(self) -> None:
        network = make_network()
        logits = network(torch.zeros(7, 12, 5, 5))
        assert logits.shape == (7,)

    def test_distribution_is_a_softmax_over_candidates(self) -> None:
        network = make_network()
        probabilities = network.distribution(torch.randn(6, 12, 5, 5))
        assert probabilities.shape == (6,)
        assert probabilities.sum().item() == pytest.approx(1.0)
        assert torch.all(probabilities >= 0)

    def test_gradients_reach_every_parameter(self) -> None:
        network = make_network()
        network(torch.randn(4, 12, 5, 5)).sum().backward()
        assert [name for name, p in network.named_parameters() if p.grad is None] == []

    def test_rejects_wrong_channel_count(self) -> None:
        network = make_network(channels=12)
        with pytest.raises((RuntimeError, ValueError)):
            network(torch.zeros(3, 5, 5, 5))


class TestSequentialSamplingWithoutReplacement:
    def test_draws_k_distinct_items(self) -> None:
        logits = torch.zeros(6)
        indices, _ = sample_without_replacement(logits, 3, np.random.default_rng(0))
        assert len(indices) == 3
        assert len(set(indices)) == 3

    def test_k_larger_than_n_returns_a_permutation(self) -> None:
        logits = torch.zeros(4)
        indices, _ = sample_without_replacement(logits, 10, np.random.default_rng(0))
        assert sorted(indices) == [0, 1, 2, 3]

    def test_log_probability_of_a_single_draw(self) -> None:
        logits = torch.tensor([0.0, math.log(3.0)])
        indices, log_probability = sample_without_replacement(
            logits, 1, np.random.default_rng(3)
        )
        expected = math.log(0.25) if indices[0] == 0 else math.log(0.75)
        assert log_probability.item() == pytest.approx(expected, rel=1e-6)

    def test_log_probability_follows_the_plackett_luce_chain(self) -> None:
        # p = [0.1, 0.2, 0.7]; drawing (2, 0) has probability 0.7 * 0.1/0.3.
        probabilities = torch.tensor([0.1, 0.2, 0.7])
        logits = probabilities.log()
        _, log_probability = sample_without_replacement(
            logits, 2, np.random.default_rng(0), forced_indices=(2, 0)
        )
        expected = math.log(0.7) + math.log(0.1 / (0.1 + 0.2))
        assert log_probability.item() == pytest.approx(expected, rel=1e-6)

    def test_never_draws_masked_out_candidates(self) -> None:
        logits = torch.tensor([0.0, -float("inf"), 0.0])
        for seed in range(20):
            indices, _ = sample_without_replacement(
                logits, 2, np.random.default_rng(seed)
            )
            assert 1 not in indices

    def test_first_draw_frequencies_match_the_softmax(self) -> None:
        logits = torch.tensor([0.0, math.log(3.0)])
        rng = np.random.default_rng(5)
        draws = [sample_without_replacement(logits, 1, rng)[0][0] for _ in range(4000)]
        share = float(np.mean([index == 1 for index in draws]))
        assert share == pytest.approx(0.75, abs=0.02)

    def test_log_probability_carries_gradient(self) -> None:
        logits = torch.zeros(4, requires_grad=True)
        _, log_probability = sample_without_replacement(
            logits, 2, np.random.default_rng(0)
        )
        log_probability.backward()
        assert logits.grad is not None
        assert torch.any(logits.grad != 0)

    def test_rejects_invalid_k(self) -> None:
        with pytest.raises(ValueError):
            sample_without_replacement(torch.zeros(3), 0, np.random.default_rng(0))


class TestBUIAProposals:
    def test_proposes_at_most_max_proposals_unique_parcels(self) -> None:
        agent = make_agent(max_proposals=3)
        candidates = tuple(ParcelIndex(0, index) for index in range(8))
        proposal = agent.propose(candidates, observations(8))
        assert isinstance(proposal, Proposal)
        assert len(proposal.parcels) == 3
        assert len(set(proposal.parcels)) == 3
        assert set(proposal.parcels).issubset(set(candidates))

    def test_handles_fewer_candidates_than_the_cap(self) -> None:
        agent = make_agent(max_proposals=5)
        candidates = (ParcelIndex(1, 1), ParcelIndex(2, 2))
        proposal = agent.propose(candidates, observations(2))
        assert len(proposal.parcels) == 2

    def test_indices_point_back_into_the_candidate_list(self) -> None:
        agent = make_agent(max_proposals=2)
        candidates = tuple(ParcelIndex(3, index) for index in range(5))
        proposal = agent.propose(candidates, observations(5))
        for index, parcel in zip(proposal.indices, proposal.parcels):
            assert candidates[index] == parcel

    def test_is_reproducible_for_a_fixed_seed(self) -> None:
        candidates = tuple(ParcelIndex(0, index) for index in range(6))
        first = make_agent().propose(candidates, observations(6))
        second = make_agent().propose(candidates, observations(6))
        assert first.parcels == second.parcels

    def test_log_probability_is_negative(self) -> None:
        agent = make_agent()
        candidates = tuple(ParcelIndex(0, index) for index in range(6))
        proposal = agent.propose(candidates, observations(6))
        assert proposal.log_probability < 0.0

    def test_rejects_mismatched_observations(self) -> None:
        agent = make_agent()
        with pytest.raises(ValueError):
            agent.propose((ParcelIndex(0, 0), ParcelIndex(0, 1)), observations(3))

    def test_rejects_an_empty_candidate_set(self) -> None:
        agent = make_agent()
        with pytest.raises(ValueError):
            agent.propose((), observations(0))

    def test_rejects_observations_with_the_wrong_window(self) -> None:
        agent = make_agent(observation_window=5)
        with pytest.raises(ValueError):
            agent.propose((ParcelIndex(0, 0),), observations(1, window=7))


class TestBUIAReinforceUpdate:
    def test_update_requires_a_proposal_first(self) -> None:
        agent = make_agent()
        with pytest.raises(RuntimeError):
            agent.update(1.0)

    def test_positive_advantage_increases_the_chosen_log_probability(self) -> None:
        # Uses the shipped learning rate: this checks the *direction* of the
        # score-function gradient, which only holds while the step is small
        # enough for the local linearisation (Adam at lr 0.1 overshoots far
        # enough to break the relation).
        agent = make_agent(max_proposals=2, learning_rate=1e-3, baseline_decay=1.0)
        candidates = tuple(ParcelIndex(0, index) for index in range(5))
        observed = observations(5)
        proposal = agent.propose(candidates, observed)
        before = agent.log_probability_of(proposal.indices, observed)
        agent.update(1.0)
        after = agent.log_probability_of(proposal.indices, observed)
        assert after > before

    def test_negative_advantage_decreases_the_chosen_log_probability(self) -> None:
        agent = make_agent(max_proposals=2, learning_rate=1e-3, baseline_decay=1.0)
        candidates = tuple(ParcelIndex(0, index) for index in range(5))
        observed = observations(5)
        proposal = agent.propose(candidates, observed)
        before = agent.log_probability_of(proposal.indices, observed)
        agent.update(-1.0)
        after = agent.log_probability_of(proposal.indices, observed)
        assert after < before

    def test_update_returns_a_finite_loss_and_changes_parameters(self) -> None:
        agent = make_agent()
        candidates = tuple(ParcelIndex(0, index) for index in range(6))
        before = [p.detach().clone() for p in agent.network.parameters()]
        agent.propose(candidates, observations(6))
        loss = agent.update(0.5)
        assert np.isfinite(loss)
        after = list(agent.network.parameters())
        assert any(not torch.allclose(a, b) for a, b in zip(before, after))

    def test_baseline_tracks_the_reward_history(self) -> None:
        agent = make_agent(baseline_decay=0.5)
        candidates = tuple(ParcelIndex(0, index) for index in range(4))
        assert agent.baseline == pytest.approx(0.0)
        agent.propose(candidates, observations(4))
        agent.update(1.0)
        assert agent.baseline == pytest.approx(0.5)
        agent.propose(candidates, observations(4))
        agent.update(1.0)
        assert agent.baseline == pytest.approx(0.75)

    def test_a_second_update_without_a_new_proposal_raises(self) -> None:
        agent = make_agent()
        candidates = tuple(ParcelIndex(0, index) for index in range(4))
        agent.propose(candidates, observations(4))
        agent.update(1.0)
        with pytest.raises(RuntimeError):
            agent.update(1.0)

    def test_state_dict_round_trip(self) -> None:
        agent = make_agent()
        restored = make_agent()
        observed = observations(4)
        restored.load_state_dict(agent.state_dict())
        np.testing.assert_allclose(
            restored.log_probability_of((0, 1), observed),
            agent.log_probability_of((0, 1), observed),
            rtol=1e-5,
        )
