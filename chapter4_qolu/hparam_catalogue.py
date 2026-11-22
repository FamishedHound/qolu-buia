"""Which hyper-parameters count as *key*, and what their defaults are.

Split out of :mod:`chapter4_qolu.hparam_sensitivity` so that *what is
perturbed* stays separate from *how it is perturbed* - the same split
:mod:`chapter4_qolu.sensitivity_plan` and
:mod:`chapter4_qolu.sensitivity_stages` make for the configuration sweep.
Nothing here computes a perturbation; it only says which scalar settings are in
scope, where each one lives and what has to hold of it.

Design note (the inclusion rule).  A hyper-parameter is *key*, and therefore
perturbed, when it is a **scalar** setting of the learner or of the environment
and it is either

(a) given a numeric value by the published Table 4.1 - the discount ``gamma``,
    the mini-batch size and the sampled-parcel count - or
(b) required by the implementation and left open by the published text - the
    learning rate, the replay capacity, the n-step length, the target-update
    interval, the epsilon schedule (both endpoints and its length) and the
    observation window.

Three families are deliberately **excluded**.  The architecture rows of
Table 4.1 (the 32-64-128 convolution stack, the 256-unit residual MLP, the 51
atoms and the [-200, 200] support) are published exactly, and +/-20 % of a
filter count changes the *model* rather than a setting.  The switches
(``noisy_nets``, ``prioritised_replay``, the reward reading, the consensus
reading) have no +/-20 %; they are readings, and
:mod:`chapter4_qolu.run_sensitivity` already sweeps them.  The scale settings of
the campaign itself (extent, conversion fraction, per-iteration cap) are the
axes of that same sweep and are held fixed here, so that this driver answers
one question only.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import MISSING, dataclass, fields
from typing import Any

from chapter4_qolu.config import EnvConfig, QOLUConfig

__all__ = [
    "KEY_HYPERPARAMETERS",
    "HyperParameter",
    "format_value",
    "hyperparameter_names",
    "resolve_hyperparameters",
]

#: Which configuration object a hyper-parameter lives on.
TARGETS: tuple[str, ...] = ("qolu", "environment")
#: Where a hyper-parameter came from; see the inclusion rule above.
SOURCES: tuple[str, ...] = ("published", "implementation")


def _default(cls: type, name: str) -> float:
    """The package default of one dataclass field, read rather than copied.

    Design note: baselines are taken off :mod:`chapter4_qolu.config` at import
    time instead of being written out here, so this analysis cannot drift away
    from the configuration it claims to be perturbing.  A field with no default
    is a programming error, not a configuration choice, so it raises.
    """
    for field in fields(cls):  # type: ignore[arg-type]
        if field.name == name:
            if field.default is MISSING:
                raise ValueError(f"{cls.__name__}.{name} has no default to perturb")
            return field.default  # type: ignore[return-value]
    raise ValueError(f"{cls.__name__} has no field {name!r}")


def format_value(value: float) -> str:
    """A hyper-parameter value as prose would write it.

    Design note (display only): 20 % below 0.05 is 0.04000000000000001 in
    binary floating point, and printing that in a sentence would suggest a
    precision the perturbation does not have.  The *stored* value is never
    touched - ``hparam_summary.json`` and ``hparam_runs.csv`` carry the exact
    double, as every other document in this package does - and six significant
    figures is more than any of these settings is meaningful to.
    """
    if isinstance(value, int):
        return str(value)
    text = f"{float(value):.6g}"
    # ... but a float stays visibly a float: an epsilon of "1" would read as a
    # count rather than as a probability.
    return text if any(mark in text for mark in ".e") else f"{text}.0"


@dataclass(frozen=True)
class HyperParameter:
    """One perturbable scalar setting and everything needed to perturb it."""

    name: str
    #: ``"qolu"`` (a :class:`~chapter4_qolu.config.QOLUConfig` field) or
    #: ``"environment"`` (a :class:`~chapter4_qolu.config.EnvConfig` field).
    target: str
    baseline: float
    #: Whether the configuration field is an integer count.
    integral: bool = False
    #: Whether it additionally has to stay odd (a centred square window).
    odd: bool = False
    minimum: float | None = None
    maximum: float | None = None
    #: ``"published"`` when the value appears in Table 4.1, ``"implementation"``
    #: when the published text leaves it open but the code needs a number.
    source: str = "implementation"
    note: str = ""

    def __post_init__(self) -> None:
        if self.target not in TARGETS:
            raise ValueError(
                f"unknown target {self.target!r}; expected one of {TARGETS}"
            )
        if self.source not in SOURCES:
            raise ValueError(
                f"unknown source {self.source!r}; expected one of {SOURCES}"
            )
        if self.odd and not self.integral:
            raise ValueError("an odd hyper-parameter must also be integral")

    def as_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target": self.target,
            "baseline": self.baseline,
            "baseline_text": format_value(self.baseline),
            "integral": self.integral,
            "odd": self.odd,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "source": self.source,
            "note": self.note,
        }


#: The eleven settings the inclusion rule above admits, learner first.
KEY_HYPERPARAMETERS: tuple[HyperParameter, ...] = (
    HyperParameter(
        name="learning_rate",
        target="qolu",
        baseline=_default(QOLUConfig, "learning_rate"),
        minimum=0.0,
        source="implementation",
        note=(
            "Table 4.1 names Adam but no step size; the package uses "
            "Bellemare et al.'s 2.5e-4."
        ),
    ),
    HyperParameter(
        name="gamma",
        target="qolu",
        baseline=_default(QOLUConfig, "gamma"),
        minimum=0.0,
        maximum=1.0,
        source="published",
        note="Table 4.1 states gamma = 0.99.",
    ),
    HyperParameter(
        name="batch_size",
        target="qolu",
        baseline=_default(QOLUConfig, "batch_size"),
        integral=True,
        minimum=1,
        source="published",
        note="Table 4.1 states a mini-batch of 256.",
    ),
    HyperParameter(
        name="replay_capacity",
        target="qolu",
        baseline=_default(QOLUConfig, "replay_capacity"),
        integral=True,
        minimum=1,
        source="implementation",
        note=(
            "Algorithm 1 samples from a replay buffer D whose size the text "
            "never fixes."
        ),
    ),
    HyperParameter(
        name="n_step",
        target="qolu",
        baseline=_default(QOLUConfig, "n_step"),
        integral=True,
        minimum=1,
        source="implementation",
        note="Algorithm 1 uses an n-step return but does not fix n.",
    ),
    HyperParameter(
        name="target_sync_interval",
        target="qolu",
        baseline=_default(QOLUConfig, "target_sync_interval"),
        integral=True,
        minimum=1,
        source="implementation",
        note='Algorithm 1 resets theta^- "every t_n steps"; t_n is never given.',
    ),
    HyperParameter(
        name="epsilon_start",
        target="qolu",
        baseline=_default(QOLUConfig, "epsilon_start"),
        minimum=0.0,
        maximum=1.0,
        source="implementation",
        note=(
            "Table 4.1 lists epsilon-greedy exploration but fixes no schedule; "
            "the package decays 1.0 -> 0.05 linearly."
        ),
    ),
    HyperParameter(
        name="epsilon_end",
        target="qolu",
        baseline=_default(QOLUConfig, "epsilon_end"),
        minimum=0.0,
        maximum=1.0,
        source="implementation",
        note="The floor of the same unpublished schedule.",
    ),
    HyperParameter(
        name="epsilon_decay_steps",
        target="qolu",
        baseline=_default(QOLUConfig, "epsilon_decay_steps"),
        integral=True,
        minimum=1,
        source="implementation",
        note=(
            "The length of the same unpublished schedule; included because a "
            "schedule's two endpoints say nothing without it."
        ),
    ),
    HyperParameter(
        name="samples_per_iteration",
        target="environment",
        baseline=_default(EnvConfig, "samples_per_iteration"),
        integral=True,
        minimum=1,
        source="published",
        note=(
            "Section 3.5 samples 50 parcels per iteration; genuinely free in "
            "the sense that nothing downstream depends on the exact count."
        ),
    ),
    HyperParameter(
        name="observation_window",
        target="environment",
        baseline=_default(EnvConfig, "observation_window"),
        integral=True,
        odd=True,
        minimum=1,
        source="implementation",
        note=(
            "The paper says only that the agents are partially observing; the "
            "9 x 9 patch is the package's reading."
        ),
    ),
)

_BY_NAME: dict[str, HyperParameter] = {item.name: item for item in KEY_HYPERPARAMETERS}


def hyperparameter_names() -> tuple[str, ...]:
    """Every name :func:`resolve_hyperparameters` accepts, in order."""
    return tuple(_BY_NAME)


def resolve_hyperparameters(names: Iterable[str]) -> tuple[HyperParameter, ...]:
    """Look up hyper-parameters by name, keeping the order they were asked in."""
    wanted = [str(name) for name in names]
    if not wanted:
        raise ValueError("at least one hyper-parameter is required")
    unknown = [name for name in wanted if name not in _BY_NAME]
    if unknown:
        raise ValueError(
            f"unknown hyper-parameter(s) {unknown}; expected any of {tuple(_BY_NAME)}"
        )
    return tuple(_BY_NAME[name] for name in wanted)
