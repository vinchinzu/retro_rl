"""Take the centre 0x68 stairs in play 0x1F and get Magical Key naturally.

rr-6o7.2, one boundary past ``probe_l8_1e_gohma.py``.  Starts from the frontier
pin that probe saved (``Level8Interior1FReconFixture``, settled 0x1F of the
same fixture-only chain, ``natural_entry=false``).

PREDICTION (written before the first run, graded in the report):

  The hypothesis edge is blue_gohma -> magic_key_stairs RIGHT kill_clear,
  already confirmed as destination 0x1F. Magical Key is a cellar item.
  0x68 at ~(96,144) is the stairs object (same sprite L7 used in 0x1A:
  walk to it and push). First attempt does NOT clear the mixed census --
  residual says do not assume a clear is required. Walk to the stairs
  stand (predict ~ (96,144) or one tile south of the sprite so a push UP
  slides it / enters mode 9), push UP, enter cellar (mode 9), pick up
  Magical Key by walking onto the item/NPC the way L7 Red Candle did,
  ADDR_MAGIC_KEY 0->1, TF stays 0x7F, keys 8->8, bombs 6->6.

  Declared contingencies:

  * if enemies or a sealed block prevent the 0x68 push, halt with RAM/PNG
    (do not silently start a full-room clear). Then ONE follow-up that
    sword-clears only as much as needed to free the stairs -- not a
    wander. L7 0x1A DID require a kill-clear before the 0x68 push; that
    is a hypothesis to test, not a default.
  * if mode 9 cellar is a different screen than expected, record $EB and
    the item; do not invent the cellar id.
  * if ADDR_MAGIC_KEY stays 0 after walking the cellar, halt -- do not poke.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l8_1f_magic_key.py \
        --from-state Level8Interior1FReconFixture \
        --tag 20260904_E1 --infinite-life
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.ids import (
    DARKNUT_OBJECT_TYPE,
    POLS_VOICE_OBJECT_TYPE,
    object_name,
)
from zelda_i.dungeon.ops import fight_clear, goto, idle
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOMBS,
    ADDR_BOOK,
    ADDR_BOW,
    ADDR_CANDLE,
    ADDR_COMPASS,
    ADDR_FOOD,
    ADDR_KEYS,
    ADDR_MAGIC_KEY,
    ADDR_MAP,
    ADDR_MAX_BOMBS,
    ADDR_RUPEES,
    ADDR_SELECTED_ITEM,
    ADDR_SWORD,
    ADDR_TRIFORCE,
    PASSAGE_MODE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)
from zelda_i.runner import add_common_args, make_assist, write_report
from zelda_i.screen_glance import leftover_from_snapshot

LEVEL8 = 8
STAIRS_ROOM = 0x1F
POST_L7_TRIFORCE = 0x7F
STAIRS_OBJECT = 0x68
STAIRS_SPRITE_XY = (96, 144)
TYPE_0C = 0x0C  # unregistered; HP128 in this room, not the 0x0B darknut
STAIRS_CENSUS_TYPES = (POLS_VOICE_OBJECT_TYPE, TYPE_0C, DARKNUT_OBJECT_TYPE)
# One tile south of the 0x68 so a push UP registers (L1 PUSH_SOUTH_OFFSET=13;
# L7 0x1A stood ~y=162). 16px is one dungeon tile.
PUSH_SOUTH_OFFSET = 16
BLOCK_SLIDE_PX = 8
STAIRS_STAND_CENTER = (128, 141)  # L1 CheckWarps pose after a west-block UP
CELLAR_FLOOR_Y = 189
CELLAR_EAST_X = 176
CELLAR_PEDESTAL = (136, 141)  # L7 Red Candle leftover band
CELLAR_WEST_STAIRS = (48, 93)
STAIRS_TILES = range(0x70, 0x74)
DEATH_MODE = 17


class ProbeStop(RuntimeError):
    """Expected fail-closed halt with a reportable reason."""


def _inventory(env: Any) -> dict[str, int]:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    return {
        "sword": int(read_u8(ram, ADDR_SWORD)),
        "bombs": int(read_u8(ram, ADDR_BOMBS)),
        "max_bombs": int(read_u8(ram, ADDR_MAX_BOMBS)),
        "bow": int(read_u8(ram, ADDR_BOW)),
        "arrows": int(read_u8(ram, ADDR_ARROWS)),
        "candle": int(read_u8(ram, ADDR_CANDLE)),
        "food": int(read_u8(ram, ADDR_FOOD)),
        "keys": int(read_u8(ram, ADDR_KEYS)),
        "rupees": int(read_u8(ram, ADDR_RUPEES)),
        "magic_key": int(read_u8(ram, ADDR_MAGIC_KEY)),
        "map": int(read_u8(ram, ADDR_MAP)),
        "compass": int(read_u8(ram, ADDR_COMPASS)),
        "book": int(read_u8(ram, ADDR_BOOK)),
        "selected_item": int(read_u8(ram, ADDR_SELECTED_ITEM)),
        "triforce": int(read_u8(ram, ADDR_TRIFORCE)),
        "health": int(snap.health),
        "heart_containers": int(snap.heart_containers),
    }


def _live_objects(snap: ZeldaSnapshot) -> list:
    return [
        obj
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and obj.type_id not in (0, 0xFF) and obj.hp > 0
    ]


def _all_typed(snap: ZeldaSnapshot) -> list:
    return [
        obj
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and obj.type_id not in (0, 0xFF)
    ]


def _obj_row(obj: Any) -> dict[str, Any]:
    return {
        "slot": int(obj.slot),
        "type": int(obj.type_id),
        "type_hex": f"0x{obj.type_id:02X}",
        "type_name": object_name(obj.type_id),
        "xy": [int(obj.x), int(obj.y)],
        "hp": int(obj.hp),
    }


def _blocks(snap: ZeldaSnapshot) -> list:
    return [
        obj
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and int(obj.type_id) == STAIRS_OBJECT
    ]


def _glance(env: Any) -> dict[str, Any]:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    result = leftover_from_snapshot(snap)
    live = _live_objects(snap)
    result.update(
        {
            "level": int(snap.level),
            "screen_hex": f"0x{snap.screen:02X}",
            "xy": [int(snap.link_x), int(snap.link_y)],
            "tile": int(snap.colliding_tile),
            "facing": int(snap.facing),
            "room_item_id": int(snap.room_item_id),
            "room_item_hex": f"0x{snap.room_item_id:02X}",
            "room_obj_count": int(snap.room_obj_count),
            "room_all_dead": int(snap.room_all_dead),
            "cur_opened_doors": int(snap.cur_opened_doors),
            "open_doorway_mask": int(snap.open_doorway_mask),
            "inventory": _inventory(env),
            "blocks_0x68": [_obj_row(obj) for obj in _blocks(snap)],
            "live_objects": [_obj_row(obj) for obj in live],
            "typed_objects": [_obj_row(obj) for obj in _all_typed(snap)],
        }
    )
    return result


@dataclass
class ObservedEnv:
    """Delegate an emulator env while collecting transition/stuck evidence."""

    env: Any
    assist: Any
    tag: str
    frame: int = 0

    def __post_init__(self) -> None:
        self.phase = "start"
        self.reason = "start"
        self.latest_obs: Any | None = None
        snap = read_snapshot(self.env.get_ram())
        self.last_key = (int(snap.level), int(snap.mode), int(snap.screen))
        self.last_xy = (int(snap.link_x), int(snap.link_y))
        self.stuck = 0
        self.samples: list[dict[str, Any]] = []
        self.screenshots: list[str] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def set_reason(self, phase: str, reason: str) -> None:
        self.phase = phase
        self.reason = reason

    def _sample(self, snap: ZeldaSnapshot, reason: str | None = None) -> None:
        live = _live_objects(snap)
        self.samples.append(
            {
                "frame": int(self.frame),
                "phase": self.phase,
                "reason": reason or self.reason,
                "level": int(snap.level),
                "screen": f"0x{snap.screen:02X}",
                "mode": int(snap.mode),
                "xy": [int(snap.link_x), int(snap.link_y)],
                "tile": int(snap.colliding_tile),
                "keys": int(snap.keys),
                "bombs": int(snap.bombs),
                "magic_key": int(read_u8(self.env.get_ram(), ADDR_MAGIC_KEY)),
                "doors": int(snap.cur_opened_doors),
                "mask": int(snap.open_doorway_mask),
                "live": len(live),
                "live_types": [f"0x{obj.type_id:02X}" for obj in live],
                "blocks": [
                    [int(obj.x), int(obj.y)] for obj in _blocks(snap)
                ],
            }
        )

    def save_shot(self, label: str) -> Path:
        snap = read_snapshot(self.env.get_ram())
        path = RECORDINGS_DIR / (
            f"l8_1f_magic_key_{self.tag}_{label}_f{self.frame}_"
            f"L{snap.level}_s{snap.screen:02x}_m{snap.mode}.png"
        )
        save_rgb_png(
            self.latest_obs if self.latest_obs is not None else self.env.render(),
            path,
        )
        self.screenshots.append(str(path))
        return path

    def step(self, action: Any) -> Any:
        result = self.env.step(action)
        self.latest_obs = result[0]
        self.frame += 1
        snap = read_snapshot(self.env.get_ram())
        key = (int(snap.level), int(snap.mode), int(snap.screen))
        xy = (int(snap.link_x), int(snap.link_y))
        if key != self.last_key:
            self._sample(snap, f"transition:{self.last_key}->{key}:{self.reason}")
            self.save_shot("transition")
            self.last_key = key
            self.stuck = 0
        elif xy == self.last_xy:
            self.stuck += 1
            if self.stuck % 250 == 0:
                self._sample(snap, f"stuck_{self.stuck}:{self.reason}")
                self.save_shot(f"stuck_{self.stuck}")
        else:
            self.stuck = 0
        if self.frame % 250 == 0:
            self._sample(snap, f"periodic:{self.reason}")
        self.last_xy = xy
        return result


def _idle(
    env: ObservedEnv, assist: Any, total: list[int], frames: int, phase: str
) -> None:
    env.set_reason(phase, "idle_census")
    idle(env, assist, total, frames)


def _guard_play(snap: ZeldaSnapshot, phase: str) -> None:
    if snap.mode == DEATH_MODE:
        raise ProbeStop(f"{phase}:death")
    if snap.level != LEVEL8:
        raise ProbeStop(f"{phase}:left_level8:L{snap.level}")


def _south_face(block_xy: tuple[int, int]) -> tuple[int, int]:
    return (int(block_xy[0]), int(block_xy[1]) + PUSH_SOUTH_OFFSET)


def _approach_south_face(
    env: ObservedEnv,
    assist: Any,
    total: list[int],
    *,
    block_xy: tuple[int, int],
) -> dict[str, Any]:
    """West mouth -> south of 0x68. Y first so we do not ram the west face."""
    stand = _south_face(block_xy)
    start = read_snapshot(env.get_ram())
    west_x = int(start.link_x)
    waypoints = (
        (west_x, stand[1]),  # drop south while still in the west mouth
        stand,
    )
    trace: list[dict[str, Any]] = []
    for wx, wy in waypoints:
        env.set_reason("approach_0x68", f"waypoint_{wx}_{wy}")
        ok = goto(env, assist, total, wx, wy, tol=4, max_f=500)
        snap = read_snapshot(env.get_ram())
        _guard_play(snap, "approach_0x68")
        if snap.mode == PASSAGE_MODE:
            trace.append(
                {
                    "waypoint": [wx, wy],
                    "reached": True,
                    "entered_cellar_early": True,
                    "xy": [int(snap.link_x), int(snap.link_y)],
                    "mode": int(snap.mode),
                    "screen": f"0x{snap.screen:02X}",
                }
            )
            return {"walk": trace, "entered_cellar": True, "stand": list(stand)}
        if snap.screen != STAIRS_ROOM and snap.mode == PLAY_MODE:
            raise ProbeStop(f"approach_0x68:left_room:0x{snap.screen:02X}")
        trace.append(
            {
                "waypoint": [wx, wy],
                "reached": bool(ok),
                "xy": [int(snap.link_x), int(snap.link_y)],
                "mode": int(snap.mode),
            }
        )
    snap = read_snapshot(env.get_ram())
    at = (int(snap.link_x), int(snap.link_y))
    return {
        "walk": trace,
        "entered_cellar": False,
        "stand": list(stand),
        "at": list(at),
        "at_stand": abs(at[0] - stand[0]) <= 6 and abs(at[1] - stand[1]) <= 6,
    }


def _push_0x68_up(
    env: ObservedEnv,
    assist: Any,
    total: list[int],
    *,
    block_xy0: tuple[int, int],
    max_frames: int = 700,
) -> dict[str, Any]:
    """Hold UP on the south face. Halt (do not clear) if the block never moves."""
    notes: list[str] = []
    samples: list[dict[str, Any]] = []
    stand0 = _south_face(block_xy0)
    for frame in range(max_frames):
        snap = read_snapshot(env.get_ram())
        _guard_play(snap, "push_0x68")
        if snap.mode == PASSAGE_MODE:
            return {
                "ok": True,
                "entered_cellar": True,
                "block_slid": False,
                "frames": frame,
                "notes": notes,
                "samples": samples,
                "cellar_screen": f"0x{snap.screen:02X}",
                "cellar_xy": [int(snap.link_x), int(snap.link_y)],
            }
        if snap.screen != STAIRS_ROOM and snap.mode == PLAY_MODE:
            raise ProbeStop(f"push_0x68:left_room:0x{snap.screen:02X}")
        blocks = _blocks(snap)
        bx = int(blocks[0].x) if blocks else block_xy0[0]
        by = int(blocks[0].y) if blocks else -1
        slid = (not blocks) or (by >= 0 and by <= block_xy0[1] - BLOCK_SLIDE_PX)
        if frame % 20 == 0:
            samples.append(
                {
                    "frame": frame,
                    "xy": [int(snap.link_x), int(snap.link_y)],
                    "block": [bx, by],
                    "tile": int(snap.colliding_tile),
                    "stuck": int(env.stuck),
                }
            )
        if slid:
            notes.append(f"block_slid_at_{frame}_to_{bx}_{by}")
            return {
                "ok": True,
                "entered_cellar": False,
                "block_slid": True,
                "frames": frame,
                "notes": notes,
                "samples": samples,
                "block_after": [bx, by] if blocks else None,
            }
        x, y = int(snap.link_x), int(snap.link_y)
        stand = _south_face((bx, by if by >= 0 else block_xy0[1]))
        if abs(x - stand[0]) > 4:
            env.set_reason("push_0x68", "reacquire_x")
            env.step(nes_action("RIGHT" if x < stand[0] else "LEFT"))
        elif y > stand[1] + 4:
            env.set_reason("push_0x68", "reacquire_y")
            env.step(nes_action("UP"))
        elif y < stand[1] - 4:
            env.set_reason("push_0x68", "reacquire_south")
            env.step(nes_action("DOWN"))
        else:
            env.set_reason("push_0x68", "push_up")
            env.step(nes_action("UP"))
        total[0] += 1
        assist.apply_env(env, frame=total[0])
    snap = read_snapshot(env.get_ram())
    blocks = _blocks(snap)
    return {
        "ok": False,
        "entered_cellar": False,
        "block_slid": False,
        "frames": max_frames,
        "notes": notes,
        "samples": samples,
        "halt": "0x68_push_blocked",
        "stand_predicted": list(stand0),
        "xy": [int(snap.link_x), int(snap.link_y)],
        "block": [_obj_row(obj) for obj in blocks],
        "tile": int(snap.colliding_tile),
        "live": [_obj_row(obj) for obj in _live_objects(snap)],
    }


def _walk_revealed_stairs(
    env: ObservedEnv,
    assist: Any,
    total: list[int],
    *,
    max_frames: int = 800,
) -> dict[str, Any]:
    """After a west 0x68 UP slide, walk the L1 centre stairs pose and hold."""
    env.set_reason("revealed_stairs", "goto_128_141")
    ok = goto(env, assist, total, *STAIRS_STAND_CENTER, tol=4, max_f=500)
    snap = read_snapshot(env.get_ram())
    _guard_play(snap, "revealed_stairs")
    if snap.mode == PASSAGE_MODE:
        return {
            "ok": True,
            "entered_cellar": True,
            "goto_center": bool(ok),
            "cellar_screen": f"0x{snap.screen:02X}",
            "cellar_xy": [int(snap.link_x), int(snap.link_y)],
        }
    for frame in range(max_frames):
        snap = read_snapshot(env.get_ram())
        _guard_play(snap, "revealed_stairs")
        if snap.mode == PASSAGE_MODE:
            return {
                "ok": True,
                "entered_cellar": True,
                "goto_center": bool(ok),
                "hold_frames": frame,
                "cellar_screen": f"0x{snap.screen:02X}",
                "cellar_xy": [int(snap.link_x), int(snap.link_y)],
            }
        if snap.screen != STAIRS_ROOM and snap.mode == PLAY_MODE:
            raise ProbeStop(f"revealed_stairs:left_room:0x{snap.screen:02X}")
        x, y = int(snap.link_x), int(snap.link_y)
        tx, ty = STAIRS_STAND_CENTER
        if abs(x - tx) > 4:
            env.set_reason("revealed_stairs", "align_x")
            env.step(nes_action("RIGHT" if x < tx else "LEFT"))
        elif abs(y - ty) > 4:
            env.set_reason("revealed_stairs", "align_y")
            env.step(nes_action("DOWN" if y < ty else "UP"))
        elif int(snap.colliding_tile) in STAIRS_TILES:
            env.set_reason("revealed_stairs", "stairs_tile_hold")
            env.step(nes_action("UP"))
        else:
            env.set_reason("revealed_stairs", "center_hold_up")
            env.step(nes_action("UP"))
        total[0] += 1
        assist.apply_env(env, frame=total[0])
    snap = read_snapshot(env.get_ram())
    return {
        "ok": False,
        "entered_cellar": False,
        "goto_center": bool(ok),
        "xy": [int(snap.link_x), int(snap.link_y)],
        "tile": int(snap.colliding_tile),
        "mode": int(snap.mode),
    }


def _cellar_walk(
    env: ObservedEnv,
    assist: Any,
    total: list[int],
    *,
    max_frames: int = 2500,
) -> dict[str, Any]:
    """L7 Red Candle cellar walk: drop south, east, climb, left onto the item.

    Do not invent the cellar $EB. Halt if ADDR_MAGIC_KEY stays 0 after the walk.
    """
    snap0 = read_snapshot(env.get_ram())
    cellar_eb = int(snap0.screen)
    notes: list[str] = []
    waypoints = (
        (int(snap0.link_x), CELLAR_FLOOR_Y),
        (CELLAR_EAST_X, CELLAR_FLOOR_Y),
        (CELLAR_EAST_X, CELLAR_PEDESTAL[1]),
        CELLAR_PEDESTAL,
        (120, CELLAR_PEDESTAL[1]),
        (96, CELLAR_PEDESTAL[1]),
        (64, CELLAR_PEDESTAL[1]),
        CELLAR_WEST_STAIRS,
        (128, 93),
        (208, 141),
        (128, 141),
    )
    trace: list[dict[str, Any]] = []
    env.set_reason("cellar_walk", "enter")
    env._sample(snap0, "cellar_enter")
    for frame in range(max_frames):
        snap = read_snapshot(env.get_ram())
        _guard_play(snap, "cellar_walk")
        mk = int(read_u8(env.get_ram(), ADDR_MAGIC_KEY))
        if mk >= 1:
            notes.append(f"magic_key_at_{frame}")
            return {
                "ok": True,
                "magic_key": mk,
                "frames": frame,
                "cellar_eb": f"0x{cellar_eb:02X}",
                "observed_eb": f"0x{snap.screen:02X}",
                "mode": int(snap.mode),
                "xy": [int(snap.link_x), int(snap.link_y)],
                "notes": notes,
                "trace": trace,
                "typed_objects": [_obj_row(obj) for obj in _all_typed(snap)],
            }
        if snap.mode != PASSAGE_MODE:
            notes.append(f"left_cellar_mode_{snap.mode}_mk_{mk}")
            return {
                "ok": mk >= 1,
                "magic_key": mk,
                "frames": frame,
                "cellar_eb": f"0x{cellar_eb:02X}",
                "observed_eb": f"0x{snap.screen:02X}",
                "mode": int(snap.mode),
                "xy": [int(snap.link_x), int(snap.link_y)],
                "notes": notes,
                "trace": trace,
                "left_cellar_before_pickup": True,
            }
        # E1c chased HP-0 keese for ~15k frames. Magical Key is a pedestal
        # NPC like L7 Red Candle; walk the waypoints, do not hunt corpses.
        wp_i = min(frame // 180, len(waypoints) - 1)
        wx, wy = waypoints[wp_i]
        x, y = int(snap.link_x), int(snap.link_y)
        if abs(x - wx) <= 5 and abs(y - wy) <= 5:
            if not trace or trace[-1].get("wp") != wp_i:
                trace.append({"wp": wp_i, "xy": [x, y], "frame": frame})
            wp_i = min(wp_i + 1, len(waypoints) - 1)
            wx, wy = waypoints[wp_i]
        if abs(y - wy) > 4:
            env.set_reason("cellar_walk", "y")
            env.step(nes_action("UP" if y > wy else "DOWN"))
        elif abs(x - wx) > 4:
            env.set_reason("cellar_walk", "x")
            env.step(nes_action("RIGHT" if x < wx else "LEFT"))
        else:
            env.set_reason("cellar_walk", "idle_pad")
            env.step(nes_idle_action())
        total[0] += 1
        assist.apply_env(env, frame=total[0])
    snap = read_snapshot(env.get_ram())
    mk = int(read_u8(env.get_ram(), ADDR_MAGIC_KEY))
    return {
        "ok": False,
        "magic_key": mk,
        "frames": max_frames,
        "cellar_eb": f"0x{cellar_eb:02X}",
        "observed_eb": f"0x{snap.screen:02X}",
        "mode": int(snap.mode),
        "xy": [int(snap.link_x), int(snap.link_y)],
        "notes": notes,
        "trace": trace,
        "halt": "magic_key_stayed_0",
        "typed_objects": [_obj_row(obj) for obj in _all_typed(snap)],
    }


def _cellar_return(
    env: ObservedEnv,
    assist: Any,
    total: list[int],
    *,
    max_frames: int = 1200,
) -> dict[str, Any]:
    """Two-ladder cellar: do not LEFT/RIGHT at y=141 (pit tile 250).

    E1c leftover (112,141) tile 250 after a naive west-ladder walk. L1 bow
    return is DOWN to the floor, under the pit, UP the west ladder. That
    return is the next live boundary (toward hypothesized Gleeok); this
    sitting keeps the Magical Key pad leftover.
    """
    snap0 = read_snapshot(env.get_ram())
    if snap0.mode != PASSAGE_MODE:
        return {
            "attempted": False,
            "already_play": True,
            "mode": int(snap0.mode),
            "screen": f"0x{snap0.screen:02X}",
        }
    return {
        "attempted": False,
        "ok": False,
        "deferred": "two_ladder_return",
        "reason": (
            "y=141 LEFT/RIGHT is the pit (tile 250); L1 DOWN-first west-ladder"
            " return is the next boundary, not this Magical Key sitting"
        ),
        "screen": f"0x{snap0.screen:02X}",
        "xy": [int(snap0.link_x), int(snap0.link_y)],
        "mode": int(snap0.mode),
        "tile": int(snap0.colliding_tile),
    }


def _save_fixture(
    raw_env: Any,
    *,
    fixture_name: str,
    census: dict[str, Any],
    cellar_eb: str | None,
    magic_key: int,
    keys: int,
    bombs: int,
    returned: bool,
) -> dict[str, Any]:
    from retro_harness.env import save_state, state_path
    from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance

    after = read_snapshot(raw_env.get_ram())
    mk = int(read_u8(raw_env.get_ram(), ADDR_MAGIC_KEY))
    if not (after.level == LEVEL8 and mk >= 1):
        raise ProbeStop("save_fixture_pin_mismatch")
    path = save_state(raw_env, GAME_DIR, GAME, fixture_name)
    source_path = state_path(GAME_DIR, GAME, "Level8Interior1FReconFixture")
    leftover = (
        "returned play" if returned and after.mode == PLAY_MODE else "cellar leftover"
    )
    result = {
        "ok": True,
        "source_state": "Level8Interior1FReconFixture",
        "fixture_state": fixture_name,
        "state": compact_snapshot(after),
        "census": census,
        "cellar_eb": cellar_eb,
        "magic_key": magic_key,
        "keys_unchanged": keys,
        "bombs_unchanged": bombs,
        "returned_to_play": bool(returned),
        "leftover": leftover,
        "fixture_writes": [],
    }
    write_state_provenance(
        path,
        source_state_path=source_path if source_path.exists() else None,
        request={
            "bead": "rr-6o7.2",
            "phase": "level8_interior_magic_key_recon",
            "track": "recon_fixture",
            "route_eligible": False,
            "fixture_only": True,
            "natural_entry": False,
            "fixture_writes": [],
            "notes": [
                "Continuation pin: Level8Interior1FReconFixture is the settled"
                " 0x1F frame of probe_l8_1e_gohma, itself a fixture-only chain"
                " from Level8InteriorReconFixture.",
                "Walk to the centre 0x68 in play 0x1F, push UP, enter the"
                " Magical Key cellar, pick up ADDR_MAGIC_KEY naturally.",
                "No RAM poke. First attempt did not clear the mixed 0x1F census.",
                f"Cellar $EB observed as: {cellar_eb}.",
                f"Leftover is {leftover}.",
                "Not route eligible; not on L8_THROUGH;"
                " topology.magic_key_room stays unset.",
            ],
        },
        selected_trial=result,
        natural_entry=False,
    )
    return {
        "state_path": str(path),
        "provenance": str(path.with_suffix(".provenance.json")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    add_common_args(
        parser,
        default_state="Level8Interior1FReconFixture",
        default_tag="20260904_E1",
    )
    parser.add_argument("--save-fixture", default=None)
    args = parser.parse_args()
    if not args.infinite_life:
        raise SystemExit("this fixture route probe requires --infinite-life")

    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    assist = make_assist(True)
    raw_env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    payload: dict[str, Any] = {
        "trial": "0x1f_sword_clear_then_0x68_up_to_magic_key_cellar",
        "from_state": args.from_state,
        "fixture_only": True,
        "natural_entry": False,
        "route_eligible": False,
        "infinite_life": True,
        "prediction": {
            "written_before_run": True,
            "claim": (
                "walk to 0x68 ~(96,144) or one tile south, push UP without"
                " clearing 0x1F, enter mode-9 cellar, walk onto Magical Key"
                " ADDR_MAGIC_KEY 0->1; TF 0x7F; keys 8->8; bombs 6->6"
            ),
            "source_room": "0x1F",
            "direction": "STAIRS_UP",
            "expected_mode": 9,
            "expected_keys": [8, 8],
            "expected_bombs": [6, 6],
            "expected_magic_key": [0, 1],
            "expected_triforce": POST_L7_TRIFORCE,
            "hypothesis_edge": "blue_gohma -> magic_key_stairs RIGHT kill_clear",
            "clear_required": False,
            "cellar_id_invented": False,
            "contingency": (
                "if enemies or a sealed block prevent the 0x68 push, halt with"
                " RAM/PNG (do not silently start a full-room clear). If mode 9"
                " cellar $EB is unexpected, record it. If ADDR_MAGIC_KEY stays"
                " 0 after the cellar walk, halt -- do not poke."
            ),
            "attempt_history": [
                {
                    "tag": "20260904_E1",
                    "result": "0x68_push_blocked",
                    "detail": (
                        "no-clear south-face UP at (96,156) vs 0x68 still"
                        " (96,144); 700 push frames, block never slid, mode"
                        " stayed 5. PNG: diamond of 4 white blocks around an"
                        " already-visible centre staircase; mixed census"
                        " crowded the west face (pols_voice on (96,150),"
                        " darknut (64,166)). 9 damage events / 16 HP in 0x1F."
                        " Same hitstun-resets-push as L7 0x1A."
                    ),
                    "fix": (
                        "ONE follow-up: sword-clear the mixed census"
                        " (0x16/0x0C/0x0B) to free the stairs, then the same"
                        " 0x68 UP push. Not a wander."
                    ),
                },
                {
                    "tag": "20260904_E1b",
                    "result": "0x68_south_face_unreached",
                    "detail": (
                        "sword-clear ok (4519f, room_all_dead 0->158, 0x68"
                        " still (96,144) at post-clear). Approach from the"
                        " north patrol leftover (112,109) walked DOWN the"
                        " west column and pushed 0x68 DOWN to (96,160)."
                        " Link leftover (96,141) is the vacated west slot;"
                        " centre stairs are immediately RIGHT. Halted on"
                        " at_stand because the south-face target was now"
                        " the block itself."
                    ),
                    "fix": (
                        "If 0x68 has already slid off (96,144), walk the"
                        " vacated gap to the L1 centre stairs pose (128,141)"
                        " instead of insisting on a south-face UP."
                    ),
                },
                {
                    "tag": "20260904_E1c",
                    "result": "magic_key_0_to_1",
                    "detail": (
                        "MK 0->1 in mode-9 cellar $EB=0x0F at (128,141)."
                        " keys 8, bombs 6, TF 0x7F, deaths 0. Two-ladder"
                        " cellar (west/east ladders, pit). Naive return"
                        " LEFT at y=141 fell in tile 250 (112,141)."
                    ),
                    "fix": (
                        "Keep the Magical Key pad leftover; do not hunt HP-0"
                        " keese; defer the L1 DOWN-first two-ladder return."
                    ),
                },
            ],
        },
        "runtime_controller_writes": {
            "room": 0,
            "door": 0,
            "position": 0,
            "inventory": 0,
            "triforce": 0,
            "magic_key": 0,
            "capacity": 0,
        },
    }
    env: ObservedEnv | None = None
    total = [0]
    try:
        obs, _ = reset_obs(raw_env)
        env = ObservedEnv(raw_env, assist, args.tag)
        env.latest_obs = obs
        start = _glance(env)
        payload["start"] = start
        env._sample(read_snapshot(env.get_ram()), "fixture_start")
        env.save_shot("start")
        inv0 = start["inventory"]
        blocks0 = start["blocks_0x68"]
        if not (
            start["level"] == LEVEL8
            and start["screen"] == STAIRS_ROOM
            and start["mode"] == PLAY_MODE
            and inv0["triforce"] == POST_L7_TRIFORCE
            and inv0["sword"] == 3
            and inv0["keys"] == 8
            and inv0["bombs"] == 6
            and inv0["bow"] == 1
            and inv0["arrows"] == 1
            and inv0["candle"] == 2
            and inv0["magic_key"] == 0
            and blocks0
        ):
            raise ProbeStop("fixture_start_mismatch")

        assist.apply_env(env, frame=0)
        keys_in = int(inv0["keys"])
        bombs_in = int(inv0["bombs"])
        block_xy = (int(blocks0[0]["xy"][0]), int(blocks0[0]["xy"][1]))
        payload["stairs_object"] = {
            "type_hex": "0x68",
            "xy": list(block_xy),
            "predicted_xy": list(STAIRS_SPRITE_XY),
        }

        # E1: south-face UP never slid 0x68 (hitstun). Clear only the mixed
        # census that crowded the west block; then the same push.
        env.set_reason("clear_0x1f", "fight_clear_sword_only_guard_departure")
        clear = fight_clear(
            env,
            assist,
            total,
            enemy_types=STAIRS_CENSUS_TYPES,
            max_frames=8000,
            use_bombs=False,
            level=LEVEL8,
        )
        payload["clear_0x1f"] = clear
        after_clear = read_snapshot(env.get_ram())
        if (
            not clear.get("ok")
            or clear.get("left_room")
            or after_clear.screen != STAIRS_ROOM
            or after_clear.mode != PLAY_MODE
        ):
            payload["failed"] = "clear_0x1f_first_departure_guard"
            raise ProbeStop("clear_0x1f_first_departure_guard")
        _idle(env, assist, total, 60, "post_clear_0x1f_census")
        post_clear = _glance(env)
        payload["post_clear_0x1f"] = post_clear
        env.save_shot("cleared_0x1f")
        blocks1 = post_clear["blocks_0x68"]
        if blocks1:
            block_xy = (int(blocks1[0]["xy"][0]), int(blocks1[0]["xy"][1]))

        approach = _approach_south_face(
            env, assist, total, block_xy=block_xy
        )
        payload["approach_0x68"] = approach
        env.save_shot("at_south_face")
        entered = bool(approach.get("entered_cellar"))

        if not entered:
            now = read_snapshot(env.get_ram())
            blocks_now = _blocks(now)
            block_now = (
                (int(blocks_now[0].x), int(blocks_now[0].y))
                if blocks_now
                else None
            )
            slid = block_now is None or (
                abs(block_now[1] - block_xy[1]) >= BLOCK_SLIDE_PX
                or abs(block_now[0] - block_xy[0]) >= BLOCK_SLIDE_PX
            )
            payload["approach_0x68"]["block_after"] = (
                list(block_now) if block_now else None
            )
            payload["approach_0x68"]["block_slid"] = slid
            if approach.get("at_stand") and not slid:
                push = _push_0x68_up(env, assist, total, block_xy0=block_xy)
                payload["push_0x68"] = push
                env.save_shot("after_push_0x68")
                if push.get("entered_cellar"):
                    entered = True
                elif push.get("block_slid"):
                    slid = True
                else:
                    payload["failed"] = "0x68_push_blocked"
                    raise ProbeStop("0x68_push_blocked")
            elif not slid:
                payload["failed"] = "0x68_south_face_unreached"
                env.save_shot("south_face_unreached")
                raise ProbeStop("0x68_south_face_unreached")
            if not entered and slid:
                # E1b: north-column approach pushed 0x68 DOWN and vacated
                # (96,144). Walk the gap to the centre stairs like L1.
                revealed = _walk_revealed_stairs(env, assist, total)
                payload["revealed_stairs"] = revealed
                env.save_shot("after_revealed_stairs")
                if not revealed.get("entered_cellar"):
                    payload["failed"] = "revealed_stairs_no_mode9"
                    raise ProbeStop("revealed_stairs_no_mode9")
                entered = True

        cellar_in = _glance(env)
        payload["cellar_enter"] = cellar_in
        env.save_shot("cellar_enter")
        if cellar_in["mode"] != PASSAGE_MODE:
            payload["failed"] = "expected_mode9_cellar"
            raise ProbeStop("expected_mode9_cellar")

        pickup = _cellar_walk(env, assist, total)
        payload["cellar_walk"] = pickup
        env.save_shot("after_cellar_walk")
        mk = int(pickup.get("magic_key") or 0)
        if mk < 1:
            payload["failed"] = "magic_key_stayed_0"
            raise ProbeStop("magic_key_stayed_0")

        ret = _cellar_return(env, assist, total)
        payload["cellar_return"] = ret
        env.save_shot("after_cellar_return")
        _idle(env, assist, total, 60, "final_census")
        final = _glance(env)
        settled = read_snapshot(env.get_ram())
        env.save_shot("final_census")
        keys_out = int(settled.keys)
        bombs_out = int(settled.bombs)
        mk_out = int(final["inventory"]["magic_key"])
        tf_out = int(final["inventory"]["triforce"])
        misses: list[str] = []
        if settled.level != LEVEL8:
            misses.append(f"left L8 -> L{settled.level}")
        if mk_out < 1:
            misses.append("magic_key still 0")
        if tf_out != POST_L7_TRIFORCE:
            misses.append(f"triforce {tf_out:#x} != 0x7F")
        if keys_out != keys_in:
            misses.append(f"keys changed {keys_in}->{keys_out}")
        if bombs_out != bombs_in:
            misses.append(f"bombs changed {bombs_in}->{bombs_out}")
        payload["magic_key_grade"] = {
            "pass": not misses,
            "misses": misses,
            "magic_key": [0, mk_out],
            "keys": [keys_in, keys_out],
            "bombs": [bombs_in, bombs_out],
            "triforce": tf_out,
            "cellar_eb": pickup.get("cellar_eb"),
            "settled_room": f"0x{settled.screen:02X}",
            "settled_xy": [int(settled.link_x), int(settled.link_y)],
            "settled_mode": int(settled.mode),
            "returned_to_play": bool(ret.get("ok")),
        }
        payload["final"] = final
        payload["success"] = not misses
        if misses:
            payload["failed"] = "magic_key_grade_miss"
        elif args.save_fixture:
            payload["saved_fixture"] = _save_fixture(
                raw_env,
                fixture_name=args.save_fixture,
                census=final,
                cellar_eb=pickup.get("cellar_eb"),
                magic_key=mk_out,
                keys=keys_out,
                bombs=bombs_out,
                returned=bool(ret.get("ok")),
            )
    except ProbeStop as exc:
        payload["success"] = False
        payload.setdefault("failed", str(exc))
        if env is not None:
            payload["final"] = _glance(env)
            env._sample(read_snapshot(env.get_ram()), f"halt:{exc}")
            env.save_shot("final_halt")
    finally:
        if env is not None:
            payload["frames"] = int(env.frame)
            payload["samples"] = env.samples[-96:]
            payload["screenshots"] = env.screenshots
            payload["assist"] = assist.report()
            payload["deaths"] = int(assist.telemetry.deaths)
            payload["progression_writes"] = int(assist.telemetry.progression_writes)
            payload["capacity_writes"] = int(assist.telemetry.capacity_writes)
            grade = payload.get("magic_key_grade", {})
            payload["runtime_integrity"] = {
                "deaths": int(assist.telemetry.deaths),
                "progression_writes": int(assist.telemetry.progression_writes),
                "capacity_writes": int(assist.telemetry.capacity_writes),
                "direct_ram_writes": 0,
                "state_loads_after_start": 0,
                "magic_key_poke": False,
                "keys_spent_at_new_boundary": (
                    None
                    if not grade
                    else int(grade.get("keys", [0, 0])[0])
                    - int(grade.get("keys", [0, 0])[1])
                ),
                "bombs_spent_at_new_boundary": (
                    None
                    if not grade
                    else int(grade.get("bombs", [0, 0])[0])
                    - int(grade.get("bombs", [0, 0])[1])
                ),
            }
        report = write_report("l8_1f_magic_key_fixture", payload, tag=args.tag)
        raw_env.close()

    print(f"report={report}")
    print(f"success={payload.get('success')} failed={payload.get('failed')}")
    print(f"grade={payload.get('magic_key_grade')}")
    print(f"cellar_walk={ {k: v for k, v in payload.get('cellar_walk', {}).items() if k != 'trace'} }")
    print(f"final={payload.get('final')}")
    print(f"assist_deaths={payload.get('deaths')}")
    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
