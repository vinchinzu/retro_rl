"""Concrete SkillPolicy helpers for one-shot and multi-frame solver skills.

``SolverSession`` already loops on :class:`~retro_harness.solver_domain.SkillSignal.RUNNING`.
Production SM route adapters remain honest one-shot macros (one terminal
``SkillStep`` whose action runs an entire edge). Frame-level skills, tape
scripts, and harness tests use :class:`ScriptedSkillPolicy` (or any policy
that emits multiple RUNNING steps) so the multi-step lifecycle is exercised
rather than only implied by the protocol.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from retro_harness.solver_domain import (
    SkillSignal,
    SkillStep,
    SolverObservation,
)


class ScriptedSkillPolicy:
    """Deterministic multi-frame skill over a fixed :class:`SkillStep` sequence.

    Each ``step()`` after ``reset()`` yields the next scripted step. Exhausting
    a script that ends on ``RUNNING`` is a hard error; a terminal last step is
    re-emitted if the session calls ``step`` again without reset (defensive).
    """

    def __init__(self, steps: Sequence[SkillStep]) -> None:
        if not steps:
            raise ValueError("steps must be non-empty")
        materialised = tuple(steps)
        for index, step in enumerate(materialised):
            if not isinstance(step, SkillStep):
                raise TypeError(f"steps[{index}] must be a SkillStep")
        self._steps = materialised
        self._index = 0

    @property
    def steps(self) -> tuple[SkillStep, ...]:
        return self._steps

    @property
    def index(self) -> int:
        return self._index

    def reset(
        self,
        observation: SolverObservation,
        config: Mapping[str, Any],
    ) -> None:
        del observation, config
        self._index = 0

    def step(self, observation: SolverObservation) -> SkillStep:
        del observation
        if self._index >= len(self._steps):
            last = self._steps[-1]
            if last.signal is SkillSignal.RUNNING:
                raise RuntimeError(
                    "scripted skill exhausted while still SkillSignal.RUNNING"
                )
            return last
        step = self._steps[self._index]
        self._index += 1
        return step


class OneShotSkillPolicy:
    """Honest one-shot skill: exactly one terminal ``SkillStep`` per reset.

    Matches the production SM ``RouteCommandPolicy`` shape: macro controllers
    that apply a full edge inside ``apply_action`` and never need RUNNING.
    """

    def __init__(self, step: SkillStep) -> None:
        if not isinstance(step, SkillStep):
            raise TypeError("step must be a SkillStep")
        if step.signal is SkillSignal.RUNNING:
            raise ValueError(
                "OneShotSkillPolicy cannot use SkillSignal.RUNNING; "
                "use ScriptedSkillPolicy for multi-frame skills"
            )
        self._step = step
        self._fired = False

    @property
    def step_template(self) -> SkillStep:
        return self._step

    def reset(
        self,
        observation: SolverObservation,
        config: Mapping[str, Any],
    ) -> None:
        del observation, config
        self._fired = False

    def step(self, observation: SolverObservation) -> SkillStep:
        del observation
        if self._fired:
            raise RuntimeError(
                "OneShotSkillPolicy stepped more than once without reset"
            )
        self._fired = True
        return self._step


__all__ = [
    "OneShotSkillPolicy",
    "ScriptedSkillPolicy",
]
