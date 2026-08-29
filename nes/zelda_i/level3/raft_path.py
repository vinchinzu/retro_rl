"""Level 3 Raft path: Darknuts → key door → stairs → mode-9 passage → ADDR_RAFT.

Door micros / west-key / north-chain live in ``level3_path``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.engine import DungeonPhase, GenericDungeonRoomController
from zelda_i.door_graph.core import DoorDir
from zelda_i.level3.geometry import (
    KEY_DOOR_Y,
    KEY_DOOR_Y_TOL,
    NORTH_DOOR_X,
    NORTH_DOOR_X_TOL,
    RAFT_CHANNEL_X,
    RAFT_CHANNEL_X_TOL,
    RAFT_PASSAGE_MODE,
    RAFT_PICKUP_X,
    RAFT_PICKUP_Y,
    RAFT_SOUTH_Y,
    RAFT_SOUTH_Y_TOL,
    STAIRS_69_RIGHT_Y,
)
from zelda_i.level3.dungeon import (
    DARKNUT_OBJECT_TYPE,
    ROOM_59_SPEC,
    ROOM_69_SPEC,
    ROOM_ITEM_COMPASS,
    ROOM_L3_COMPASS,
    ROOM_L3_DARKNUTS,
    ROOM_L3_RAFT_PASSAGE,
    ROOM_L3_SOUTH_DARKNUTS,
    ROOM_L3_WEST_DARKNUTS,
)
from zelda_i.dungeon.hop_controller import dungeon_align_then_push
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

# Path timing knobs (not room-table data).
KEY_DOOR_PUSH_FRAMES = 160  # short push can spend key without room change
SPAWN_SETTLE_FRAMES = 100  # Darknuts lag ~75–100f before clear registers
LEFT_5B_MAX_FRAMES = 1500
KEY_5A_MAX_FRAMES = 2500
CLEAR_59_MAX_FRAMES = 18000
DOWN_69_MAX_FRAMES = 2000
DOWN_69_LOWER_AISLE_Y = 173
# Diamonds block the mid-band. East leftover (192,157); dest_6b-clear
# leftover sat in the west mouth (48,141) and RIGHT no-op'd for 2000f.
DOWN_69_EAST_DIAMOND_X = 176
DOWN_69_WEST_DIAMOND_X = 64
CLEAR_69_MAX_FRAMES = 28000
STAIRS_69_MAX_FRAMES = 2500
PASSAGE_RAFT_MAX_FRAMES = 6000
RAFT_PATH_MAX_FRAMES = 55000

RAFT_PATH_PHASES: tuple[str, ...] = (
    "settle_5b",
    "left_to_5a",
    "key_to_59",
    "spawn_59",
    "clear_59",
    "down_to_69",
    "spawn_69",
    "clear_69",
    "stairs_to_0f",
    "passage_raft",
    "done",
    "failed",
)


def raft_passage_step(snap: ZeldaSnapshot) -> FrameAction:
    """One frame of mode-9 0x0f passage geometry to Raft pickup.

    LIVE residual: south band UP is solid except channel at x≈176. Path:
    DOWN y≈189 → RIGHT x≈176 → UP to y≈141 → LEFT x≈136 touch Raft.

    Once on the channel column, prefer vertical align + LEFT (do not re-south).
    """
    if snap.mode == 17:
        return FrameAction(nes_idle_action(), "link_death")
    # Scroll / mode settle into underworld.
    if snap.transitioning or snap.mode not in (PLAY_MODE, RAFT_PASSAGE_MODE):
        if snap.mode in (6, 7, 10):
            return FrameAction(nes_action("RIGHT"), "passage_scroll")
        return FrameAction(nes_idle_action(), f"passage_wait_mode_{snap.mode}")
    if snap.screen != ROOM_L3_RAFT_PASSAGE and snap.mode != RAFT_PASSAGE_MODE:
        return FrameAction(
            nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}"
        )

    at_channel = abs(snap.link_x - RAFT_CHANNEL_X) <= RAFT_CHANNEL_X_TOL
    near_channel = abs(snap.link_x - RAFT_CHANNEL_X) <= 16
    on_south = snap.link_y >= RAFT_SOUTH_Y - RAFT_SOUTH_Y_TOL
    on_pickup_band = abs(snap.link_y - RAFT_PICKUP_Y) <= 8

    # Mid horizontal band (raft corridor): do not re-south — walk to pickup x.
    # Drift off exact channel while walking LEFT is expected (176 → 136).
    if on_pickup_band and (at_channel or near_channel or snap.link_x <= RAFT_CHANNEL_X):
        if snap.link_x > RAFT_PICKUP_X + 2:
            return FrameAction(nes_action("LEFT"), "passage_to_raft")
        if snap.link_x < RAFT_PICKUP_X - 6:
            return FrameAction(nes_action("RIGHT"), "passage_raft_overshoot")
        return FrameAction(nes_action("LEFT"), "passage_raft_touch")

    # Channel column: vertical align to pickup band.
    if at_channel or (near_channel and snap.link_y > RAFT_PICKUP_Y):
        if abs(snap.link_x - RAFT_CHANNEL_X) > RAFT_CHANNEL_X_TOL:
            direction = "RIGHT" if snap.link_x < RAFT_CHANNEL_X else "LEFT"
            return FrameAction(nes_action(direction), "passage_recenter_channel")
        if snap.link_y > RAFT_PICKUP_Y:
            return FrameAction(nes_action("UP"), "passage_channel_up")
        if snap.link_y < RAFT_PICKUP_Y - KEY_DOOR_Y_TOL:
            return FrameAction(nes_action("DOWN"), "passage_channel_down")
        return FrameAction(nes_action("LEFT"), "passage_to_raft")

    # Off channel north: reach south band first (UP solid except channel).
    if not on_south:
        # If already east, prefer re-acquire channel over futile south into wall.
        if snap.link_x >= 140:
            direction = "RIGHT" if snap.link_x < RAFT_CHANNEL_X else "LEFT"
            return FrameAction(nes_action(direction), "passage_seek_channel")
        return FrameAction(nes_action("DOWN"), "passage_to_south")

    # South band: walk to channel x≈176.
    if snap.link_x < RAFT_CHANNEL_X:
        return FrameAction(nes_action("RIGHT"), "passage_to_channel")
    return FrameAction(nes_action("LEFT"), "passage_to_channel")


def _count_live_darknuts(snap: ZeldaSnapshot) -> int:
    return sum(
        1
        for o in snap.objects
        if 1 <= o.slot <= 10
        and o.type_id == DARKNUT_OBJECT_TYPE
        and o.hp > 0
    )


def _is_room_scroll(snap: ZeldaSnapshot) -> bool:
    """True during horizontal/vertical room scroll (modes 4/6/7/16)."""
    return snap.transitioning or snap.mode in (4, 6, 7, 16)


@dataclass
class Level3RaftPathController:
    """Assisted Survival: Level3Darknuts → ADDR_RAFT via Compass west path.

    Phases (see ``RAFT_PATH_PHASES``)::

        settle_5b → left_to_5a → key_to_59 → spawn_59 → clear_59
        → down_to_69 → spawn_69 → clear_69 → stairs_to_0f → passage_raft

    Intervention: Survival (``--infinite-life``). Not Clean STATUS.
    """

    frames: int = 0
    phase_frames: int = 0
    push_frames: int = 0
    success: bool = False
    failed: bool = False
    phase: str = "settle_5b"
    keys_at_key_door: int | None = None
    max_live_59: int = 0
    max_live_69: int = 0
    clear_59: GenericDungeonRoomController = field(
        default_factory=lambda: GenericDungeonRoomController(ROOM_59_SPEC)
    )
    clear_69: GenericDungeonRoomController = field(
        default_factory=lambda: GenericDungeonRoomController(ROOM_69_SPEC)
    )
    notes: list[str] = field(default_factory=list)
    max_frames: int = RAFT_PATH_MAX_FRAMES

    def _set_phase(self, phase: str, note: str = "") -> None:
        if phase != self.phase:
            self.phase = phase
            self.phase_frames = 0
            self.push_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self.failed = True
        self._set_phase("failed", note)
        return FrameAction(nes_idle_action(), "failed")

    def step(self, snap: ZeldaSnapshot, *, has_raft: bool | None = None) -> FrameAction:
        """One control frame; Raft ownership defaults to the RAM snapshot."""
        self.frames += 1
        self.phase_frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed:
            return FrameAction(nes_idle_action(), "failed")
        if self.frames >= self.max_frames:
            return self._fail("timeout")
        if snap.mode == 17:
            return self._fail("link_death")

        # Global success: Raft inventory bit (may set mid-passage).
        if has_raft is None:
            has_raft = bool(snap.raft)
        if has_raft:
            self.success = True
            self._set_phase("done", "raft_acquired")
            return FrameAction(nes_idle_action(), "done")

        # --- settle_5b: ignore Darknuts; brief spawn settle then leave west ---
        if self.phase == "settle_5b":
            if (
                snap.screen == ROOM_L3_COMPASS
                and snap.mode == PLAY_MODE
                and not snap.transitioning
            ):
                self._set_phase("key_to_59", "already_0x5a")
                return FrameAction(nes_idle_action(), "phase_handoff")
            if (
                snap.screen == ROOM_L3_WEST_DARKNUTS
                and snap.mode == PLAY_MODE
                and not snap.transitioning
            ):
                self._set_phase("spawn_59", "already_0x59")
                return FrameAction(nes_idle_action(), "phase_handoff")
            if (
                snap.screen == ROOM_L3_SOUTH_DARKNUTS
                and snap.mode == PLAY_MODE
                and not snap.transitioning
            ):
                self._set_phase("spawn_69", "already_0x69")
                return FrameAction(nes_idle_action(), "phase_handoff")
            if snap.mode == RAFT_PASSAGE_MODE or snap.screen == ROOM_L3_RAFT_PASSAGE:
                self._set_phase("passage_raft", "already_passage")
                return FrameAction(nes_idle_action(), "phase_handoff")
            if self.phase_frames < 40:
                return FrameAction(nes_idle_action(), "settle_5b")
            self._set_phase("left_to_5a", "leave_darknuts_west")
            return FrameAction(nes_idle_action(), "phase_handoff")

        # --- left_to_5a: open west door (no clear) @ y≈141 ---
        if self.phase == "left_to_5a":
            if self.phase_frames > LEFT_5B_MAX_FRAMES:
                return self._fail("left_5a_timeout")
            if snap.screen == ROOM_L3_COMPASS:
                if _is_room_scroll(snap) or snap.mode != PLAY_MODE:
                    return FrameAction(nes_action("LEFT"), "left_5a_scroll")
                self._set_phase("key_to_59", "entered_0x5a")
                return FrameAction(nes_idle_action(), "left_5a_arrived")
            if _is_room_scroll(snap):
                return FrameAction(nes_action("LEFT"), "left_5a_scroll")
            if snap.mode != PLAY_MODE:
                return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
            if snap.screen != ROOM_L3_DARKNUTS:
                return FrameAction(
                    nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}"
                )
            return dungeon_align_then_push(
                snap,
                push_dir="LEFT",
                target_x=32,
                target_y=KEY_DOOR_Y,
                y_tol=KEY_DOOR_Y_TOL,
                x_tol=NORTH_DOOR_X_TOL,
                door_plane=48,
                reason="left_5a",
            )

        # --- key_to_59: long LEFT KEY push @ y≈141 (trap: short push wastes key) ---
        if self.phase == "key_to_59":
            if self.phase_frames > KEY_5A_MAX_FRAMES:
                return self._fail("key_59_timeout")
            if snap.screen == ROOM_L3_WEST_DARKNUTS:
                if _is_room_scroll(snap) or snap.mode != PLAY_MODE:
                    return FrameAction(nes_action("LEFT"), "key_59_scroll")
                self._set_phase("spawn_59", "entered_0x59")
                return FrameAction(nes_idle_action(), "key_59_arrived")
            if _is_room_scroll(snap):
                return FrameAction(nes_action("LEFT"), "key_59_scroll")
            if snap.mode != PLAY_MODE:
                return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
            if snap.screen != ROOM_L3_COMPASS:
                return FrameAction(
                    nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}"
                )
            if self.keys_at_key_door is None:
                self.keys_at_key_door = int(snap.keys)
            # Optional compass: if standing near center item and free, step once.
            # Not required — skip if not aligned.
            if (
                abs(snap.link_x - 120) <= 8
                and abs(snap.link_y - 141) <= 8
                and snap.room_item_id == ROOM_ITEM_COMPASS
                and self.phase_frames < 30
            ):
                return FrameAction(nes_idle_action(), "optional_compass_touch")
            # Align y first, approach west wall, long push.
            if abs(snap.link_y - KEY_DOOR_Y) > KEY_DOOR_Y_TOL:
                direction = "UP" if snap.link_y > KEY_DOOR_Y else "DOWN"
                self.push_frames = 0
                return FrameAction(nes_action(direction), "key_59_align_y")
            if snap.link_x > 48:
                self.push_frames = 0
                return FrameAction(nes_action("LEFT"), "key_59_approach")
            # Long hold LEFT at door plane (critical residual).
            self.push_frames += 1
            if self.push_frames > KEY_DOOR_PUSH_FRAMES + 80 and snap.keys == 0:
                # Key spent without scroll — fail honestly.
                if self.keys_at_key_door is not None and snap.keys < self.keys_at_key_door:
                    return self._fail("key_spent_no_scroll")
            return FrameAction(nes_action("LEFT"), "key_59_long_push")

        # --- spawn_59: wait for Darknuts to materialize ---
        if self.phase == "spawn_59":
            live = _count_live_darknuts(snap)
            self.max_live_59 = max(self.max_live_59, live)
            if snap.screen != ROOM_L3_WEST_DARKNUTS:
                return FrameAction(
                    nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}"
                )
            if live >= 3 or self.phase_frames >= SPAWN_SETTLE_FRAMES:
                self._set_phase(
                    "clear_59",
                    f"spawn_59_live={live}_f{self.phase_frames}",
                )
                return self.clear_59.step(snap)
            return FrameAction(nes_idle_action(), "spawn_59_wait")

        # --- clear_59: sword patrol until type-0x0b gone AND DOWN opens ---
        if self.phase == "clear_59":
            live = _count_live_darknuts(snap)
            self.max_live_59 = max(self.max_live_59, live)
            if snap.screen != ROOM_L3_WEST_DARKNUTS and snap.mode == PLAY_MODE:
                # Accidental exit — try recover only if still on path.
                if snap.screen == ROOM_L3_SOUTH_DARKNUTS:
                    self._set_phase("spawn_69", "early_0x69")
                    return FrameAction(nes_idle_action(), "phase_handoff")
            action = self.clear_59.step(snap)
            down_open = bool(snap.cur_opened_doors & DoorDir.DOWN)
            # Kill-clear lag: live can hit 0 while doors still lack DOWN for ~40f
            # (room_all_dead ramps after last corpse). Do not leave early.
            if live == 0 and self.max_live_59 >= 3 and not down_open:
                return FrameAction(nes_idle_action(), "clear_59_wait_door")
            if (
                self.clear_59.success
                or (down_open and live == 0 and self.max_live_59 >= 3)
            ):
                self._set_phase(
                    "down_to_69",
                    f"cleared_59_doors={snap.cur_opened_doors}_alldead={snap.room_all_dead}",
                )
                return FrameAction(nes_idle_action(), "clear_59_done")
            if self.clear_59.phase is DungeonPhase.FAILED:
                # Soft fallback: if DOWN open and few live, still try exit.
                if down_open and live <= 1:
                    self._set_phase("down_to_69", "clear_59_partial_down_open")
                    return FrameAction(nes_idle_action(), "clear_59_partial")
                return self._fail("clear_59_failed")
            return action

        # --- down_to_69: south after kill-clear ---
        if self.phase == "down_to_69":
            if self.phase_frames > DOWN_69_MAX_FRAMES:
                return self._fail("down_69_timeout")
            if snap.screen == ROOM_L3_SOUTH_DARKNUTS:
                if _is_room_scroll(snap) or snap.mode != PLAY_MODE:
                    return FrameAction(nes_action("DOWN"), "down_69_scroll")
                self._set_phase("spawn_69", "entered_0x69")
                return FrameAction(nes_idle_action(), "down_69_arrived")
            if _is_room_scroll(snap):
                return FrameAction(nes_action("DOWN"), "down_69_scroll")
            if snap.mode != PLAY_MODE:
                return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
            if snap.screen != ROOM_L3_WEST_DARKNUTS:
                return FrameAction(
                    nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}"
                )
            # Live spine v1: combat ended east of the right diamond at
            # (192,157); LEFT cannot cross it. dest_6b-clear leftover
            # (48,141): RIGHT cannot leave the west mouth. Descend to the
            # lower aisle before aligning toward the south door column.
            west_of_diamond = snap.link_x <= DOWN_69_WEST_DIAMOND_X
            east_of_diamond = snap.link_x >= DOWN_69_EAST_DIAMOND_X
            if (
                (west_of_diamond or east_of_diamond)
                and snap.link_y < DOWN_69_LOWER_AISLE_Y
            ):
                return FrameAction(nes_action("DOWN"), "down_69_escape_diamond")
            # Align x≈120 then hold DOWN. Do not chase y=205 — past the door
            # plane Link thrash-oscillates align_y/push and never scrolls.
            if abs(snap.link_x - NORTH_DOOR_X) > NORTH_DOOR_X_TOL:
                direction = "LEFT" if snap.link_x > NORTH_DOOR_X else "RIGHT"
                return FrameAction(nes_action(direction), "down_69_align_x")
            return FrameAction(nes_action("DOWN"), "down_69_push_DOWN")

        # --- spawn_69 ---
        if self.phase == "spawn_69":
            live = _count_live_darknuts(snap)
            self.max_live_69 = max(self.max_live_69, live)
            if snap.screen != ROOM_L3_SOUTH_DARKNUTS:
                if snap.mode == RAFT_PASSAGE_MODE or snap.screen == ROOM_L3_RAFT_PASSAGE:
                    self._set_phase("passage_raft", "early_passage")
                    return FrameAction(nes_idle_action(), "phase_handoff")
                return FrameAction(
                    nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}"
                )
            if live >= 4 or self.phase_frames >= SPAWN_SETTLE_FRAMES:
                self._set_phase(
                    "clear_69",
                    f"spawn_69_live={live}_f{self.phase_frames}",
                )
                return self.clear_69.step(snap)
            return FrameAction(nes_idle_action(), "spawn_69_wait")

        # --- clear_69: 8 Darknuts then stairs ---
        if self.phase == "clear_69":
            live = _count_live_darknuts(snap)
            self.max_live_69 = max(self.max_live_69, live)
            if snap.mode == RAFT_PASSAGE_MODE or snap.screen == ROOM_L3_RAFT_PASSAGE:
                self._set_phase("passage_raft", "stairs_during_clear")
                return FrameAction(nes_idle_action(), "phase_handoff")
            action = self.clear_69.step(snap)
            if (
                self.clear_69.success
                or (
                    live == 0
                    and self.max_live_69 >= 4
                    and self.phase_frames > 80
                )
            ):
                self._set_phase("stairs_to_0f", f"cleared_69_maxlive={self.max_live_69}")
                return FrameAction(nes_idle_action(), "clear_69_done")
            if self.clear_69.phase is DungeonPhase.FAILED:
                # Try stairs anyway if clear-ish (stairs may need full clear).
                if live == 0:
                    self._set_phase("stairs_to_0f", "clear_69_timeout_try_stairs")
                    return FrameAction(nes_idle_action(), "clear_69_try_stairs")
                return self._fail("clear_69_failed")
            return action

        # --- stairs_to_0f: RIGHT @ y≈141 only ---
        if self.phase == "stairs_to_0f":
            if self.phase_frames > STAIRS_69_MAX_FRAMES:
                return self._fail("stairs_timeout")
            if snap.mode == RAFT_PASSAGE_MODE or snap.screen == ROOM_L3_RAFT_PASSAGE:
                if snap.mode not in (PLAY_MODE, RAFT_PASSAGE_MODE) and snap.mode != 10:
                    return FrameAction(nes_action("RIGHT"), "stairs_scroll")
                self._set_phase("passage_raft", "entered_passage")
                return FrameAction(nes_idle_action(), "stairs_arrived")
            if _is_room_scroll(snap) or snap.mode in (6, 7, 10):
                return FrameAction(nes_action("RIGHT"), "stairs_scroll")
            if snap.mode != PLAY_MODE:
                return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
            if snap.screen != ROOM_L3_SOUTH_DARKNUTS:
                return FrameAction(
                    nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}"
                )
            return dungeon_align_then_push(
                snap,
                push_dir="RIGHT",
                target_x=208,
                target_y=STAIRS_69_RIGHT_Y,
                y_tol=KEY_DOOR_Y_TOL,
                x_tol=NORTH_DOOR_X_TOL,
                door_plane=192,
                reason="stairs",
            )

        # --- passage_raft: mode-9 channel geometry ---
        if self.phase == "passage_raft":
            if self.phase_frames > PASSAGE_RAFT_MAX_FRAMES:
                return self._fail("passage_timeout")
            # level3_has_raft checked at top; keep walking to touch tile.
            return raft_passage_step(snap)

        if self.phase == "done":
            return FrameAction(nes_idle_action(), "done")
        return FrameAction(nes_idle_action(), self.phase)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "phase": self.phase,
            "frames": self.frames,
            "phase_frames": self.phase_frames,
            "notes": list(self.notes),
            "max_live_59": self.max_live_59,
            "max_live_69": self.max_live_69,
            "keys_at_key_door": self.keys_at_key_door,
            "clear_59": self.clear_59.report(),
            "clear_69": self.clear_69.report(),
            "phases": list(RAFT_PATH_PHASES),
            "stop": "level3_has_raft",
            "path": (
                "0x5b LEFT→0x5a LEFT KEY→0x59 clear DOWN→0x69 clear "
                "RIGHT@y141→0x0f channel→Raft"
            ),
            "intervention_class": "survival",
            "track": "assisted",
            "geometry": {
                "key_door_y": KEY_DOOR_Y,
                "stairs_y": STAIRS_69_RIGHT_Y,
                "channel_x": RAFT_CHANNEL_X,
                "pickup_xy": [RAFT_PICKUP_X, RAFT_PICKUP_Y],
                "south_y": RAFT_SOUTH_Y,
            },
        }
