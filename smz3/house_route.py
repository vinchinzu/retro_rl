"""Link's House OW porch → interior → open chest (map-driven).

Map sources (not in-game minimap):

- **Yaze warp export** (snes_editor):
  ``state_screenshots/route_probe/yaze_map_data/hyrule_castle_warps_0x1b.json``
  entrance_id ``0x01`` "Link's House Post-intro" absolute ``(2224, 2800)``
  on map ``0x2C``.
- **zelda3 overworld asset** (snes_editor):
  ``zelda3/assets/overworld/overworld-44.yaml`` — entrance tile (11, 15),
  exit door local xy (184, 232) → world (2232, 2792).
- **CSV features** (snes_editor ``data/overworld_features.csv``) and
  ``overworld_map.overworld_feature_rows(0x2C)``.
- **Interior** (snes_editor ``asset_editor/assets/rooms/room_004.json``):
  Chest object at (6, 16); door spawn measured ``(2424, 8664)``.
- **Vanilla open XY** (alttp ``FRESH_PROFILE_LAMP_CHEST_SCRIPT`` end):
  ``(2491, 8632)`` face UP + A — works on SMZ3 (item is randomized).

Verified approach from Fortune Teller outdoor end ``(2528, 2920)``:

1. DOWN clear → LEFT along south band to west flank ``x≈2112``.
2. UP west ramp to approach Y ``≈2846`` (under house south face).
3. RIGHT along that Y to door X ``≈2224``, then UP into entrance gap.
4. Indoors: walk to chest open XY, face UP, A; wait for chest flag /
   inventory (test seed 1337: heart container, max HP 24→32).

Porch is a one-way ledge from the deep south (y≥2936 cannot climb back);
west ramp + under-house latitude is the natural re-entry corridor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from adventure_common.nav import Waypoint
from alttp.ram import (
    EQUIP_SWORD,
    LINK_ACTION,
    LINK_ACTION_HOLD_UP_ITEM,
    LINK_ITEM_LAMP,
    LINKS_HOUSE_ROOM,
    wram_index,
    read_u8_safe,
)
from smz3.control import go_xy, hold, is_z3_dead, wait_z3_control
from smz3.ram import ComboSnapshot, snapshot_env
from smz3.segment import SegmentResult

# --- Map-derived constants -------------------------------------------------

ENTRANCE_X = 2224
ENTRANCE_Y = 2800
APPROACH_Y = 2846
WEST_FLANK_X = 2112
CHEST_OPEN_X = 2491
CHEST_OPEN_Y = 8632

LINKS_HOUSE_INTERIOR_ROOM = LINKS_HOUSE_ROOM  # 0x0004
CHEST_OPEN_FLAG_ADDR = 0x0403

MAX_HOUSE_FRAMES = 8_000

# Approach waypoints (absolute world XY) before door scan.
_SOUTH_CLEAR_Y = 2955
HOUSE_APPROACH_WAYPOINTS: tuple[Waypoint, ...] = (
    Waypoint(WEST_FLANK_X, _SOUTH_CLEAR_Y, tolerance=12, label="south_clear"),
    Waypoint(WEST_FLANK_X, APPROACH_Y, tolerance=8, label="west_ramp"),
    Waypoint(ENTRANCE_X - 4, APPROACH_Y, tolerance=6, label="under_house"),
)

DOOR_X_OFFSETS: tuple[int, ...] = (0, 1, -1, 2, -2, 3, 4, -3)


@dataclass
class HouseSegmentResult(SegmentResult):
    """Outcome of OW porch → interior chest open."""

    goal: str = "links_house_chest"
    entered: bool = False
    chest_opened: bool = False
    max_hp_before: int | None = None
    max_hp_after: int | None = None
    lamp: int = 0
    sword: int = 0
    inventory_delta: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        snap = self.final_snapshot
        d.update(
            {
                "entered": self.entered,
                "chest_opened": self.chest_opened,
                "max_hp_before": self.max_hp_before,
                "max_hp_after": self.max_hp_after,
                "lamp": self.lamp,
                "sword": self.sword,
                "inventory_delta": self.inventory_delta,
                "map": {
                    "entrance_xy": [ENTRANCE_X, ENTRANCE_Y],
                    "approach_y": APPROACH_Y,
                    "chest_open_xy": [CHEST_OPEN_X, CHEST_OPEN_Y],
                    "approach_waypoints": [
                        {"x": w.x, "y": w.y, "label": w.label} for w in HOUSE_APPROACH_WAYPOINTS
                    ],
                    "sources": [
                        "yaze warp 0x01 @ (2224,2800) map 0x2C",
                        "overworld-44.yaml entrance tile (11,15)",
                        "room_004.json chest + vanilla open XY",
                    ],
                },
                "snapshot": snap.to_dict() if snap is not None else None,
            }
        )
        d.pop("final_snapshot", None)
        return d


def _z3_inv(env: Any) -> dict[str, Any]:
    ram = np.asarray(env.get_ram(), dtype=np.uint8)
    return {
        "hold": int(ram[LINK_ACTION]) == LINK_ACTION_HOLD_UP_ITEM,
        "action": int(ram[LINK_ACTION]),
        "lamp": int(read_u8_safe(ram, wram_index(LINK_ITEM_LAMP))),
        "sword": int(read_u8_safe(ram, wram_index(EQUIP_SWORD))),
        "max_hp": int(read_u8_safe(ram, wram_index(0xF36C))),
        "hp": int(read_u8_safe(ram, wram_index(0xF36D))),
        "chest_flag": int(ram[CHEST_OPEN_FLAG_ADDR]),
        "mod": int(ram[0x10]),
        "sub": int(ram[0x11]),
        "bytes": {
            hex(a): int(read_u8_safe(ram, wram_index(a)))
            for a in range(0xF340, 0xF370)
        },
    }


def _holding_item(env: Any) -> bool:
    return bool(_z3_inv(env)["hold"])


def _wait(env: Any, *, start_frame: int = 0, max_frames: int = 400) -> tuple[int, ComboSnapshot]:
    return wait_z3_control(
        env,
        start_frame=start_frame,
        max_frames=max_frames,
        clear_hold_up=True,
        hold_up_check=_holding_item,
    )


def _budget_ok(frame: int, start_frame: int, max_frames: int) -> bool:
    return frame - start_frame <= max_frames


def _walk_south_clear(
    env: Any, frame: int, *, start_frame: int, max_frames: int
) -> tuple[int, ComboSnapshot | None]:
    while snapshot_env(env, frame=frame).z3_link_y < _SOUTH_CLEAR_Y:
        if not _budget_ok(frame, start_frame, max_frames):
            return frame, snapshot_env(env, frame=frame)
        frame = hold(env, ("DOWN",), 4, frame=frame)
        if is_z3_dead(snapshot_env(env, frame=frame)):
            return frame, snapshot_env(env, frame=frame)
    return frame, None


def _walk_west_flank(
    env: Any, frame: int, *, start_frame: int, max_frames: int
) -> tuple[int, ComboSnapshot | None]:
    while snapshot_env(env, frame=frame).z3_link_x > WEST_FLANK_X + 8:
        if not _budget_ok(frame, start_frame, max_frames):
            return frame, snapshot_env(env, frame=frame)
        snap = snapshot_env(env, frame=frame)
        if is_z3_dead(snap):
            return frame, snap
        if not snap.z3_controllable:
            frame, snap = _wait(env, start_frame=frame, max_frames=60)
            continue
        frame = hold(env, ("LEFT",), 4, frame=frame)
    return frame, None


def _walk_west_ramp(
    env: Any, frame: int, *, start_frame: int, max_frames: int
) -> tuple[int, ComboSnapshot | None]:
    """North on west ramp to approach Y (do not overshoot north of band)."""
    for _ in range(200):
        if not _budget_ok(frame, start_frame, max_frames):
            return frame, snapshot_env(env, frame=frame)
        snap = snapshot_env(env, frame=frame)
        if is_z3_dead(snap):
            return frame, snap
        if not snap.z3_controllable:
            frame, snap = _wait(env, start_frame=frame, max_frames=60)
            continue
        if abs(snap.z3_link_y - APPROACH_Y) <= 4:
            break
        buttons: list[str]
        if snap.z3_link_y > APPROACH_Y + 4:
            buttons = ["UP"]
        else:
            buttons = ["DOWN"]
        if snap.z3_link_x > WEST_FLANK_X + 16:
            buttons = ["LEFT", buttons[0]]
        elif snap.z3_link_x < WEST_FLANK_X - 16:
            buttons = ["RIGHT", buttons[0]]
        frame = hold(env, tuple(buttons), 3, frame=frame)
    return frame, None


def _walk_under_house(
    env: Any, frame: int, *, start_frame: int, max_frames: int
) -> tuple[int, bool, ComboSnapshot]:
    """East under house to entrance X; return (frame, entered, snap)."""
    stuck = 0
    prev_xy: tuple[int, int] | None = None
    for _ in range(220):
        if not _budget_ok(frame, start_frame, max_frames):
            break
        snap = snapshot_env(env, frame=frame)
        if snap.z3_indoors:
            frame, snap = _wait(env, start_frame=frame)
            return frame, True, snap
        if is_z3_dead(snap):
            return frame, False, snap
        if not snap.z3_controllable:
            frame, snap = _wait(env, start_frame=frame, max_frames=80)
            if snap.z3_indoors:
                return frame, True, snap
            continue
        if snap.z3_link_x >= ENTRANCE_X - 1 and abs(snap.z3_link_y - APPROACH_Y) <= 8:
            break
        # Drifted onto house west wall band — back out west then south.
        if (
            snap.z3_link_x >= 2160
            and snap.z3_link_x < ENTRANCE_X - 8
            and snap.z3_link_y < APPROACH_Y - 12
        ):
            frame = hold(env, ("LEFT",), 12, frame=frame)
            frame = hold(env, ("DOWN",), 20, frame=frame)
            continue
        buttons = ["RIGHT"]
        if snap.z3_link_y < APPROACH_Y - 2:
            buttons = ["DOWN", "RIGHT"]
        elif snap.z3_link_y > APPROACH_Y + 2:
            buttons = ["UP", "RIGHT"]
        prev_x = snap.z3_link_x
        frame = hold(env, tuple(buttons), 2, frame=frame)
        snap2 = snapshot_env(env, frame=frame)
        if snap2.z3_link_x <= prev_x:
            frame = hold(env, ("DOWN", "RIGHT"), 8, frame=frame)
            if snapshot_env(env, frame=frame).z3_link_x <= prev_x:
                frame = hold(env, ("UP", "RIGHT"), 8, frame=frame)
        xy = (
            snapshot_env(env, frame=frame).z3_link_x,
            snapshot_env(env, frame=frame).z3_link_y,
        )
        if xy == prev_xy:
            stuck += 1
        else:
            stuck = 0
            prev_xy = xy
        if stuck >= 10:
            frame = hold(env, ("LEFT",), 16, frame=frame)
            frame = hold(env, ("DOWN",), 12, frame=frame)
            stuck = 0

    return frame, False, snapshot_env(env, frame=frame)


def _try_door_entry(
    env: Any, frame: int, *, start_frame: int, max_frames: int
) -> tuple[int, bool, ComboSnapshot]:
    """Scan X offsets and walk UP into entrance gap."""
    for x_off in DOOR_X_OFFSETS:
        if not _budget_ok(frame, start_frame, max_frames):
            break
        target = ENTRANCE_X + x_off
        for _ in range(24):
            snap = snapshot_env(env, frame=frame)
            if abs(snap.z3_link_x - target) <= 1:
                break
            frame = hold(
                env,
                ("RIGHT",) if snap.z3_link_x < target else ("LEFT",),
                1,
                frame=frame,
            )
        for _ in range(20):
            snap = snapshot_env(env, frame=frame)
            if snap.z3_link_y >= APPROACH_Y:
                break
            frame = hold(env, ("DOWN",), 1, frame=frame)
        for step in range(60):
            snap = snapshot_env(env, frame=frame)
            buttons: list[str] = ["UP"]
            if snap.z3_link_x < target - 1:
                buttons = ["RIGHT", "UP"]
            elif snap.z3_link_x > target + 1:
                buttons = ["LEFT", "UP"]
            frame = hold(env, tuple(buttons), 1, frame=frame)
            snap = snapshot_env(env, frame=frame)
            if snap.z3_indoors or snap.z3_module not in (0x09, 0x0B):
                frame, snap = _wait(env, start_frame=frame, max_frames=150)
                if snap.z3_indoors:
                    return frame, True, snap
                break
            if snap.z3_link_y <= 2824 and step > 20:
                frame = hold(env, ("DOWN",), 20, frame=frame)
                break

    snap = snapshot_env(env, frame=frame)
    return frame, bool(snap.z3_indoors), snap


def enter_links_house(
    env: Any,
    *,
    start_frame: int = 0,
    max_frames: int = MAX_HOUSE_FRAMES,
) -> tuple[int, bool, ComboSnapshot]:
    """From OW $2C near porch, walk west-ramp path into Link's House.

    Returns ``(frame, entered, snapshot)``. Phases:
    south clear → west flank → west ramp → under house → door scan.
    """
    frame = start_frame
    frame, snap = _wait(env, start_frame=frame)
    if not snap.z3_controllable or snap.z3_indoors:
        return frame, bool(snap.z3_indoors), snap

    for walker in (_walk_south_clear, _walk_west_flank, _walk_west_ramp):
        frame, early = walker(env, frame, start_frame=start_frame, max_frames=max_frames)
        if early is not None:
            return frame, bool(early.z3_indoors), early

    frame, entered, snap = _walk_under_house(
        env, frame, start_frame=start_frame, max_frames=max_frames
    )
    if entered or is_z3_dead(snap):
        return frame, entered, snap

    return _try_door_entry(env, frame, start_frame=start_frame, max_frames=max_frames)


def open_links_house_chest(
    env: Any,
    *,
    start_frame: int = 0,
    max_frames: int = 2_000,
) -> tuple[int, bool, dict[str, Any]]:
    """From interior control, open the Link's House chest at map-verified XY.

    Acceptance: chest flag ``$0403`` flips and/or max HP / inventory changes
    (item is seed-randomized; test seed 1337 grants a heart container).
    """
    frame = start_frame
    frame, snap = _wait(env, start_frame=frame)
    if not snap.z3_indoors:
        return frame, False, _z3_inv(env)

    before = _z3_inv(env)
    frame = go_xy(
        env,
        frame,
        CHEST_OPEN_X,
        CHEST_OPEN_Y,
        tol=1,
        max_steps=200,
        clear_hold_up=True,
        hold_up_check=_holding_item,
    )

    frame = hold(env, ("UP",), 3, frame=frame)
    frame = hold(env, ("A",), 4, frame=frame)

    opened = False
    flag_frame: int | None = None
    for j in range(max(0, max_frames // 2)):
        inv = _z3_inv(env)
        snap = snapshot_env(env, frame=frame)
        if inv["chest_flag"] != before["chest_flag"] and flag_frame is None:
            flag_frame = j
            opened = True
        if (
            inv["hold"]
            or inv["lamp"] != before["lamp"]
            or inv["max_hp"] != before["max_hp"]
            or inv["sword"] != before["sword"]
        ):
            opened = True
        if inv["hold"]:
            frame = hold(env, ("LEFT",), 1, frame=frame)
        elif inv["mod"] == 0x0E:
            btn = ("A",) if (j // 4) % 2 == 0 else ("B",)
            frame = hold(env, btn, 2, frame=frame)
        else:
            frame = hold(env, None, 2, frame=frame)
        inventory_settled = (
            inv["max_hp"] != before["max_hp"]
            or inv["lamp"] != before["lamp"]
            or inv["sword"] != before["sword"]
            or inv["hold"]
            or inv["mod"] == 0x0E
        )
        if (
            opened
            and snap.z3_controllable
            and not inv["hold"]
            and inv["mod"] in (0x07, 0x09)
            and flag_frame is not None
            and (inventory_settled or j >= flag_frame + 40)
        ):
            break

    after = _z3_inv(env)
    if after["chest_flag"] != before["chest_flag"] or after["max_hp"] != before["max_hp"]:
        opened = True
    return frame, opened, after


def run_links_house_chest(
    env: Any,
    *,
    start_frame: int = 0,
    max_frames: int = MAX_HOUSE_FRAMES,
    on_frame: Callable[[int, ComboSnapshot], None] | None = None,
) -> HouseSegmentResult:
    """Drive porch → house interior → open chest. Expects OW $2C controllable."""
    frame = start_frame
    frame, snap = _wait(env, start_frame=frame)
    if on_frame is not None:
        on_frame(frame, snap)

    if not snap.z3_controllable and not snap.z3_indoors:
        return HouseSegmentResult(
            ok=False,
            frames=frame - start_frame,
            detail=f"not controllable module=${snap.z3_module:02X}",
            final_snapshot=snap,
        )

    inv0 = _z3_inv(env)
    max_hp_before = inv0["max_hp"]

    if not snap.z3_indoors:
        frame, entered, snap = enter_links_house(
            env, start_frame=frame, max_frames=max_frames
        )
        if on_frame is not None:
            on_frame(frame, snap)
    else:
        entered = True

    if not entered:
        return HouseSegmentResult(
            ok=False,
            frames=frame - start_frame,
            detail=(
                f"failed to enter house xy=({snap.z3_link_x},{snap.z3_link_y}) "
                f"screen=${snap.z3_screen_id:02X}"
            ),
            entered=False,
            max_hp_before=max_hp_before,
            final_snapshot=snap,
        )

    frame, chest_ok, after = open_links_house_chest(env, start_frame=frame)
    snap = snapshot_env(env, frame=frame)
    if on_frame is not None:
        on_frame(frame, snap)

    delta = {
        k: after["bytes"][k]
        for k in after["bytes"]
        if after["bytes"][k] != inv0["bytes"].get(k)
        and k not in ("0xf36c", "0xf36d", "0xf36f")
    }
    if after["max_hp"] != max_hp_before:
        delta["max_hp"] = after["max_hp"]

    detail = (
        f"entered room=${snap.z3_room_id:04X} chest_ok={chest_ok} "
        f"flag={after['chest_flag']} max_hp={max_hp_before}->{after['max_hp']} "
        f"xy=({snap.z3_link_x},{snap.z3_link_y})"
    )
    return HouseSegmentResult(
        ok=entered and chest_ok,
        frames=frame - start_frame,
        detail=detail,
        entered=entered,
        chest_opened=chest_ok,
        max_hp_before=max_hp_before,
        max_hp_after=after["max_hp"],
        lamp=after["lamp"],
        sword=after["sword"],
        inventory_delta=delta,
        final_snapshot=snap,
    )


def indoors_links_house(snap: ComboSnapshot) -> bool:
    return bool(snap.z3_indoors) and (snap.z3_room_id & 0xFF) == LINKS_HOUSE_INTERIOR_ROOM
