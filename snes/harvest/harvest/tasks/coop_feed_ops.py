"""Chicken-feed phase arms for CoopChoresTask."""

from __future__ import annotations

from typing import Optional

import numpy as np

from harvest.core.animal_status import (
    ITEM_CHICKEN_FEED,
    read_fed_chickens_flags,
    read_fed_chickens_n,
    read_hay_count,
    read_item_on_hand,
)
from harvest.tasks.animal_navigation import align_to_pixel
from harvest.tasks.coop_layout import (
    CHICKEN_FEED_SPOTS,
    FEED_BIN_FACE,
    FEED_BIN_STAND,
    MAX_FEED_PLACE_FRAMES,
    MAX_FEED_SLOT_DEFERRALS,
    ChickenFeedSpot,
)
from harvest.tasks.nav import make_action
from harvest.tasks.skills import coop_nav_to_feed_bin_skill
from retro_harness import ActionResult, TaskResult, TaskStatus, WorldState


class CoopFeedMixin:
    """Feed bin pickup and trough placement phases."""

    def _queue_place_feed(self, face: str) -> None:
        self._action_queue.extend(make_action(**{face: True}) for _ in range(4))
        self._action_queue.extend(make_action(**{face: True, "a": True}) for _ in range(8))
        self._action_queue.extend(make_action(a=True) for _ in range(4))
        self._action_queue.extend(make_action(down=True) for _ in range(12))
        self._action_queue.extend(make_action() for _ in range(8))

    def _fed_count_now(self, ram: np.ndarray) -> int:
        flags = read_fed_chickens_flags(ram)
        flag_count = sum(1 for spot in CHICKEN_FEED_SPOTS if flags & spot.flag)
        return max(read_fed_chickens_n(ram), flag_count)

    def _next_feed_spot(self, ram: np.ndarray) -> Optional[ChickenFeedSpot]:
        flags = read_fed_chickens_flags(ram)
        blocked = self._chicken_tiles(ram)
        blocked.discard(self._navigator.current_tile)

        for spot in CHICKEN_FEED_SPOTS:
            if flags & spot.flag:
                continue
            if spot.flag in self._blocked_feed_flags:
                continue
            if spot.stand in blocked:
                continue
            return spot

        for spot in CHICKEN_FEED_SPOTS:
            if not (flags & spot.flag) and spot.flag not in self._blocked_feed_flags:
                return spot
        return None

    def _advance_after_feed(self, ram: np.ndarray) -> TaskResult:
        self._feed_registered = False
        fed_now = min(self._fed_count_now(ram), self._adult_count)
        self.fed_count = max(self.fed_count, fed_now)
        self._feed_remaining = max(0, self._adult_count - fed_now)
        self._current_feed_spot = None
        self._feed_place_started_step = 0
        if self._feed_remaining > 0:
            self._clear_left_top_route()
            if read_item_on_hand(ram) == ITEM_CHICKEN_FEED:
                self._phase = "feed_place_nav"
            else:
                self._phase = "feed_nav"
        elif self._collectable_egg_present(ram):
            return self._begin_egg_nav()
        else:
            return self._begin_exit_prep()
        return TaskResult(status=TaskStatus.RUNNING)

    def _step_feed_nav(self, world: WorldState) -> TaskResult:
        if self._fed_count_now(world.ram) >= self._adult_count:
            self._active_skill = None
            return self._advance_after_feed(world.ram)
        if read_item_on_hand(world.ram) == ITEM_CHICKEN_FEED:
            self._active_skill = None
            self._phase = "feed_place_nav"
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
        # Production path: skills.py factory + host left-top aisle routing.
        skill_result = self._step_nav_skill(
            world,
            skill_name="coop_nav_feed_bin",
            make_skill=lambda: coop_nav_to_feed_bin_skill(
                navigate=lambda w: self._navigate_to_left_top_goal(
                    w.ram, FEED_BIN_STAND
                ),
            ),
        )
        if skill_result is not None:
            return skill_result
        self._hay_before = read_hay_count(world.ram)
        self._phase = "feed_act"
        return self._step_feed_act(world)

    def _step_feed_act(self, world: WorldState) -> TaskResult:
        held_item = read_item_on_hand(world.ram)
        if held_item == ITEM_CHICKEN_FEED:
            self._phase = "feed_place_nav"
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
        if held_item != 0:
            self._phase = "feed_verify"
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
        if self._fed_count_now(world.ram) >= self._adult_count:
            return self._advance_after_feed(world.ram)
        # Cap feeds at available hay
        if read_hay_count(world.ram) <= 0:
            print(f"[COOP] Out of hay after feeding {self.fed_count}")
            if self._egg_present(world.ram):
                self._phase = "egg_nav"
            else:
                self._phase = "done"
            return TaskResult(status=TaskStatus.RUNNING)
        self._queue_press_a(FEED_BIN_FACE, hold_frames=20, settle_frames=30)
        self._feed_registered = False
        self._verify_count = 0
        self._phase = "feed_verify"
        return TaskResult(status=TaskStatus.RUNNING)

    def _step_feed_verify(self, world: WorldState) -> TaskResult:
        if read_item_on_hand(world.ram) == ITEM_CHICKEN_FEED:
            self._verify_count = 0
            self._phase = "feed_place_nav"
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
        self._verify_count += 1
        if self._verify_count > 40:
            # Feed pickup did not register; retry the bin interaction.
            self._phase = "feed_act"
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

    def _step_feed_place_nav(self, world: WorldState) -> TaskResult:
        if self._fed_count_now(world.ram) >= self._adult_count:
            return self._advance_after_feed(world.ram)
        if read_item_on_hand(world.ram) != ITEM_CHICKEN_FEED:
            self._phase = "feed_nav"
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        if self._feed_place_started_step <= 0:
            self._feed_place_started_step = self._step_count

        spot = self._current_feed_spot
        flags_now = read_fed_chickens_flags(world.ram)
        chicken_tiles = self._chicken_tiles(world.ram)
        if spot is None or (flags_now & spot.flag) or (
            spot.stand in chicken_tiles and spot.stand != self._navigator.current_tile
        ):
            spot = self._next_feed_spot(world.ram)
            self._current_feed_spot = spot
            self._feed_place_started_step = self._step_count
        if spot is None:
            print("[COOP] No open feed slot; continuing to eggs/exit")
            self._feed_remaining = 0
            return self._advance_after_feed(world.ram)

        timed_out = (
            self._step_count - self._feed_place_started_step > MAX_FEED_PLACE_FRAMES
        )
        if self._navigator.stasis > 120 or timed_out:
            deferred = self._deferred_feed_counts.get(spot.flag, 0)
            if deferred < MAX_FEED_SLOT_DEFERRALS and not timed_out:
                self._deferred_feed_counts[spot.flag] = deferred + 1
                print(
                    f"[COOP] Feed deferred flag=0x{spot.flag:04X} "
                    f"reason=stasis count={deferred + 1}"
                )
            else:
                reason = "slot_timeout" if timed_out else "stasis"
                print(
                    f"[COOP] Feed skipped flag=0x{spot.flag:04X} reason={reason}"
                )
                self._blocked_feed_flags.add(spot.flag)
            self._current_feed_spot = None
            self._feed_place_started_step = self._step_count
            self._navigator.path = []
            self._navigator.stasis = 0
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        if (
            self._navigator.current_tile[0] == spot.stand[0]
            and self._navigator.current_tile[1] > spot.stand[1]
        ):
            dx = spot.interact_px[0] - self._navigator.current_pos.x
            if abs(dx) > 1:
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(make_action(right=dx > 0, left=dx < 0)),
                )

        if self._navigator.current_tile != spot.stand:
            action = self._navigate_to_tile(world.ram, spot.stand)
            if action is not None:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

        action = align_to_pixel(
            (self._navigator.current_pos.x, self._navigator.current_pos.y),
            spot.interact_px,
            tolerance=1,
        )
        if action is not None:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

        self._fed_before = self._fed_count_now(world.ram)
        self._fed_flags_before = read_fed_chickens_flags(world.ram)
        self._queue_place_feed(spot.face)
        self._verify_count = 0
        self._feed_place_started_step = 0
        self._phase = "feed_place_verify"
        return TaskResult(status=TaskStatus.RUNNING)

    def _step_feed_place_verify(self, world: WorldState) -> TaskResult:
        fed_now = self._fed_count_now(world.ram)
        flags_now = read_fed_chickens_flags(world.ram)
        if fed_now > self._fed_before or flags_now != self._fed_flags_before:
            self.fed_count = max(self.fed_count, min(fed_now, self._adult_count))
            self._feed_remaining = max(0, self._adult_count - self.fed_count)
            print(
                f"[COOP] Feed OK count={self.fed_count} "
                f"remaining={self._feed_remaining} flags=0x{flags_now:04X}"
            )
            return self._advance_after_feed(world.ram)

        self._verify_count += 1
        if self._verify_count > 30:
            if read_item_on_hand(world.ram) == ITEM_CHICKEN_FEED:
                self._phase = "feed_place_nav"
            else:
                self._current_feed_spot = None
                self._phase = "feed_nav"
            self._verify_count = 0
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

    def _step_feed_clear_nav(self, world: WorldState) -> TaskResult:
        self._phase = "feed_place_nav"
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

    def _step_feed_clear_verify(self, world: WorldState) -> TaskResult:
        self._phase = "feed_place_verify"
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
