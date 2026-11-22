"""Persisting a trained QOLU agent.

Without a checkpoint a run ends with its agents in memory and nothing on disk
but metrics, which is enough to reproduce a *table* and not enough to
interrogate a *model*.  The accuracy and the error of the learnt return
distributions can only be measured on a trained network, so the network has to
outlive the run that produced it.

What a checkpoint holds, and why each part is there:

======================  ====================================================
``config``              the :class:`~chapter4_qolu.config.QOLUConfig` as a
                        plain dict, so the network can be rebuilt without the
                        caller knowing the architecture
``support``            ``n_atoms`` / ``v_min`` / ``v_max`` written out
                        separately, because every distributional statistic is
                        meaningless without the atom locations and a reader
                        should not have to infer them from the config
``online`` / ``target`` both networks: the target network is what the n-step
                        bootstrap is evaluated with, so a diagnostic that
                        reconstructs the agent's own training target needs it
``optimizer``           Adam's moments, so a checkpoint can also be resumed
                        from rather than only read
``steps`` /             the epsilon and beta schedules are functions of these,
``gradient_steps``      so dropping them would silently change behaviour
``target_syncs``        how many times ``theta^- <- theta`` actually fired.
                        ``target_sync_interval`` can exceed a run's whole
                        gradient budget, in which case the reset never happens
                        and the stored ``target`` network *is* the random
                        initialisation - which the config alone cannot say
``numpy_rng_state``     the agent's own exploration stream
``torch_rng_state``     the process-wide torch stream, carried but *not*
                        restored automatically (see :func:`restore_into`)
======================  ====================================================

Design note (format): ``torch.save`` of a dict of plain Python values and
tensors, loaded with ``weights_only=True``.  Pickling the ``QOLUConfig``
dataclass itself would be shorter, but it would make every checkpoint
unreadable without trusting arbitrary code and would break the moment the
dataclass moved module; a dict of primitives survives both.

Design note (``target_syncs`` and :data:`CHECKPOINT_VERSION`): a version bump
makes :func:`restore_into` refuse every payload written under another version,
so an *additive* key does not bump it.  ``target_syncs`` is read with a default of
zero, so a payload without the key still loads.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from chapter4_qolu.config import QOLUConfig

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters for typing
    from chapter4_qolu.agent.qolu_agent import QOLUAgent

__all__ = [
    "CHECKPOINT_SUFFIX",
    "CHECKPOINT_VERSION",
    "STAKEHOLDER_STEM",
    "QOLUCheckpointMixin",
    "agent_checkpoint_paths",
    "checkpoint_payload",
    "config_from_payload",
    "in_memory_state",
    "load_agents",
    "read_checkpoint",
    "restore_in_memory",
    "restore_into",
    "save_agents",
    "write_checkpoint",
]

#: Bumped whenever the payload layout changes in a way a reader would notice.
CHECKPOINT_VERSION: int = 1
#: Suffix of a single agent checkpoint.
CHECKPOINT_SUFFIX: str = ".pt"
#: File-name stem of the ``i``-th top-down stakeholder of a joint run.
STAKEHOLDER_STEM: str = "stakeholder"

_STAKEHOLDER_FILE = re.compile(rf"\A{STAKEHOLDER_STEM}(\d+)\Z")


def in_memory_state(agent: QOLUAgent) -> dict[str, Any]:
    """The live tensors of ``agent`` - what ``state_dict`` hands back."""
    return {
        "online": agent.online_network.state_dict(),
        "target": agent.target_network.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "steps": int(agent.steps),
        "gradient_steps": int(agent.gradient_steps),
        "target_syncs": int(agent.target_syncs),
    }


def restore_in_memory(agent: QOLUAgent, state: Mapping[str, Any]) -> None:
    """Inverse of :func:`in_memory_state`.

    ``target_syncs`` is read with a default, so a payload without the key
    still loads; see the module docstring.
    """
    agent.online_network.load_state_dict(state["online"])
    agent.target_network.load_state_dict(state["target"])
    agent.optimizer.load_state_dict(state["optimizer"])
    agent.steps = int(state.get("steps", 0))
    agent.gradient_steps = int(state.get("gradient_steps", 0))
    agent.target_syncs = int(state.get("target_syncs", 0))


def checkpoint_payload(agent: QOLUAgent) -> dict[str, Any]:
    """Everything needed to rebuild ``agent`` exactly, as JSON-ish primitives."""
    payload = in_memory_state(agent)
    payload.update(
        {
            "version": CHECKPOINT_VERSION,
            "config": asdict(agent.config),
            "support": {
                "n_atoms": int(agent.support.n_atoms),
                "v_min": float(agent.support.v_min),
                "v_max": float(agent.support.v_max),
            },
            "numpy_rng_state": agent.rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
        }
    )
    return payload


def write_checkpoint(agent: QOLUAgent, path: Path | str) -> Path:
    """Write one agent to ``path`` (creating its parent directory)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(agent), target)
    return target


def read_checkpoint(path: Path | str) -> dict[str, Any]:
    """Read a checkpoint payload without building an agent from it."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} does not hold a QOLU checkpoint")
    return payload


def config_from_payload(payload: Mapping[str, Any]) -> QOLUConfig:
    """Rebuild the :class:`QOLUConfig` a checkpoint was written under."""
    stored = dict(payload["config"])
    known = {field.name for field in fields(QOLUConfig)}
    unknown = sorted(set(stored) - known)
    if unknown:
        raise ValueError(
            f"checkpoint carries settings this QOLUConfig does not have: {unknown}"
        )
    if "conv_channels" in stored:
        stored["conv_channels"] = tuple(stored["conv_channels"])
    return QOLUConfig(**stored)


def restore_into(agent: QOLUAgent, payload: Mapping[str, Any]) -> None:
    """Load a checkpoint payload into an already-constructed ``agent``.

    Design note (the torch stream): the process-wide torch generator is
    *carried* by the checkpoint but not restored here.  A diagnostics run loads
    three stakeholders one after another; restoring the global stream on each
    load would leave all three sharing whichever one happened to be loaded
    last, which is a silent, hard-to-find coupling.
    :meth:`~chapter4_qolu.agent.qolu_agent.QOLUAgent.restore_torch_rng` puts it
    back on request, which is what resuming training wants.
    """
    version = int(payload.get("version", 0))
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format version {version}; this package "
            f"writes and reads version {CHECKPOINT_VERSION}"
        )
    restore_in_memory(agent, payload)
    rng_state = payload.get("numpy_rng_state")
    if rng_state is not None:
        agent.rng.bit_generator.state = rng_state
    agent.torch_rng_state = payload.get("torch_rng_state")


def save_agents(agents: Sequence[QOLUAgent], directory: Path | str) -> tuple[Path, ...]:
    """Write one checkpoint per stakeholder into ``directory``.

    The file names are positional (``stakeholder0.pt``, ``stakeholder1.pt``,
    ...) because the stakeholder *order* is what assigns objectives in
    :func:`chapter4_qolu.runner.train_joint`; a name derived from the objective
    would stop being true the moment the stakeholder count changed.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    return tuple(
        write_checkpoint(
            agent, target / f"{STAKEHOLDER_STEM}{index}{CHECKPOINT_SUFFIX}"
        )
        for index, agent in enumerate(agents)
    )


def agent_checkpoint_paths(directory: Path | str) -> tuple[Path, ...]:
    """The ``stakeholder<i>.pt`` files of ``directory``, in stakeholder order."""
    folder = Path(directory)
    if not folder.is_dir():
        return ()
    found: list[tuple[int, Path]] = []
    for path in folder.glob(f"{STAKEHOLDER_STEM}*{CHECKPOINT_SUFFIX}"):
        match = _STAKEHOLDER_FILE.match(path.stem)
        if match:
            found.append((int(match.group(1)), path))
    return tuple(path for _, path in sorted(found))


def load_agents(directory: Path | str) -> tuple[QOLUAgent, ...]:
    """Rebuild every stakeholder a run left in ``directory``."""
    from chapter4_qolu.agent.qolu_agent import QOLUAgent

    paths = agent_checkpoint_paths(directory)
    if not paths:
        raise FileNotFoundError(
            f"no {STAKEHOLDER_STEM}<i>{CHECKPOINT_SUFFIX} checkpoints in {directory}"
        )
    return tuple(QOLUAgent.load(path) for path in paths)


class QOLUCheckpointMixin:
    """The save / load half of :class:`~chapter4_qolu.agent.qolu_agent.QOLUAgent`.

    Design note (why a mixin): persistence is a self-contained concern with a
    lot of prose attached to it - what is stored, what is deliberately not
    restored, what a version bump means - and keeping it beside the payload
    format rather than in the middle of Algorithm 1 keeps both files inside the
    package's ~400-line limit and keeps "how the agent learns" readable in one
    screenful.  It is a mixin rather than free functions only because
    ``QOLUAgent.load`` has to be a constructor.
    """

    def state_dict(self) -> dict[str, Any]:
        """The live tensors and counters, for an in-memory copy."""
        return in_memory_state(self)  # type: ignore[arg-type]

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Inverse of :meth:`state_dict`."""
        restore_in_memory(self, state)  # type: ignore[arg-type]

    def save(self, path: Path | str) -> Path:
        """Write this agent to ``path`` so a later run can interrogate it."""
        return write_checkpoint(self, path)  # type: ignore[arg-type]

    @classmethod
    def load(cls, path: Path | str, device: str | None = None) -> Any:
        """Rebuild the agent ``path`` was written from, exactly.

        The architecture comes from the checkpoint, not from the caller, so a
        diagnostic can be pointed at a run without being told how the agent was
        configured.  ``device`` overrides the device the agent was saved on.
        """
        payload = read_checkpoint(path)
        config = config_from_payload(payload)
        if device is not None:
            config = replace(config, device=device)
        agent = cls(config)  # type: ignore[call-arg]
        restore_into(agent, payload)
        return agent

    def restore_torch_rng(self) -> None:
        """Put the process-wide torch stream back to the checkpoint's state."""
        state = getattr(self, "torch_rng_state", None)
        if state is None:
            raise RuntimeError("this agent was not loaded from a checkpoint")
        torch.set_rng_state(state)
