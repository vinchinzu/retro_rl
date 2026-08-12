"""Exit-prep phase for CowChoresTask (rr-y80y)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from harvest.tasks.cow_care import exit_prep_escape_action
from harvest.tasks.cow_fsm import MAX_EXIT_PREP_FRAMES, CowPhase
from harvest.tasks.cow_geometry import COW_EXIT_PREP_STAND
from retro_harness import ActionResult, TaskResult, TaskStatus, WorldState


class CowExitMixin:
    """Leave barn interior staged for EXIT_BARN handoff."""

    def _begin_exit_prep(self) -> None:
        self._exit_prep_started_step = self._step_count
        self._verify_count = 0
        self._clear_navigation()
        self._reset_pixel_nav_progress()
        self._phase = CowPhase.EXIT_PREP_NAV

    def _exit_prep_escape_action(self) -> Optional[np.ndarray]:
        """Pixel route out of left/upper dead-ends toward the lower aisle."""
        return exit_prep_escape_action(
            self._navigator.current_pos.x,
            self._navigator.current_pos.y,
        )

    def _step_exit_prep_nav(self, world: WorldState) -> TaskResult:
        if self._exit_prep_started_step <= 0:
            self._exit_prep_started_step = self._step_count
        if self._step_count - self._exit_prep_started_step > MAX_EXIT_PREP_FRAMES:
            print(
                f"[COW] Exit prep timeout at {self._navigator.current_tile}; "
                "handing off to EXIT_BARN"
            )
            self._phase = CowPhase.DONE
            return TaskResult(status=TaskStatus.RUNNING)
        if (
            self._navigator.current_tile == COW_EXIT_PREP_STAND
            or self._navigator.at_tile(COW_EXIT_PREP_STAND)
        ):
            self._phase = CowPhase.DONE
            return TaskResult(status=TaskStatus.RUNNING)
        action = self._exit_prep_escape_action()
        if action is not None:
            self._clear_navigation()
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
        action = self._navigate_to_tile(world.ram, COW_EXIT_PREP_STAND)
        if action is not None:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
        self._phase = CowPhase.DONE
        return TaskResult(status=TaskStatus.RUNNING)
