"""Replay the fixture-live L8 prefix to ``0x3E`` then try the north wall.

rr-6o7.2, one boundary past ``probe_l8_4e_north.py``.  Starts from the
disclosed ``Level8InteriorReconFixture``, reuses the confirmed
``0x7E -> clear 0x6E -> bomb-N 0x5E -> clear/key -> shutter-N 0x4E ->
key-N 0x3E`` policy unchanged, then makes ONE attempt at the north gate of
``0x3E`` toward predicted ``0x2E``.

PREDICTION (written before the first run, graded in the report):

  The six live ``0x0C`` HP128 bodies in ``0x3E`` are cleared with the sword
  only (no bombs, no arrows).  ``0x3E`` arrives with ``cur_opened_doors=0x04``
  / ``open_doorway_mask=0x04`` -- the DOWN (south) key door we came through --
  so the north side is a WALL, matching the ``blue_darknuts -> map_manhandla
  UP bomb`` hypothesis edge.  After the clear we place ONE bomb from the north
  stand ``(120, 105)`` facing UP and push through.  Expected first settled play
  room: ``0x2E`` (row-4 north neighbour, same low nibble), entry pose x=120
  near y=205 hmm -- entry pose is graded loosely as "settled L8 play room" --
  with keys 9 -> 9 (no key spent) and bombs 7 -> 6 (exactly one bomb spent).

  Declared contingency, recorded rather than hidden: if the clear itself
  raises the UP bit (``0x08``) in ``open_doorway_mask`` then the north gate is
  a kill-clear shutter, not a bomb wall -- the hypothesis edge gate is refuted
  and we walk UP instead of bombing (bombs 7 -> 7).  The destination claim
  ``0x2E`` is unchanged either way; the gate kind is graded separately.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l8_3e_north.py \
        --from-state Level8InteriorReconFixture \
        --tag 20260904_v1 --infinite-life
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.bomb_wall import BombWallController
from zelda_i.dungeon.ids import MANHANDLA_OBJECT_TYPE, object_name
from zelda_i.dungeon.ops import exit_door, fight_clear, goto, room_fields
from zelda_i.level5.whistle_path import select_b_item_menu
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOMBS,
    ADDR_BOW,
    ADDR_CANDLE,
    ADDR_FOOD,
    ADDR_KEYS,
    ADDR_MAGIC_KEY,
    ADDR_MAX_BOMBS,
    ADDR_SELECTED_ITEM,
    ADDR_SWORD,
    ADDR_TRIFORCE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)
from zelda_i.runner import add_common_args, make_assist, write_report
from zelda_i.screen_glance import leftover_from_snapshot

LEVEL8 = 8
ENTRY_ROOM = 0x7E
MANHANDLA_ROOM = 0x6E
DARKNUT_KEY_ROOM = 0x5E
SHUTTER_ROOM = 0x4E
BLUE_DARKNUT_ROOM = 0x3E
PREDICTED_BOMB_NORTH_ROOM = 0x2E
POST_L7_TRIFORCE = 0x7F
BLUE_DARKNUT_OBJECT_TYPE = 0x0C
SMALL_KEY_ITEM = 0x19
B_ITEM_BOMBS = 1
# door_graph.core bit layout: RIGHT 0x01, LEFT 0x02, DOWN 0x04, UP 0x08.
DOWN_DOOR_BIT = 0x04
UP_DOOR_BIT = 0x08
BOMB_NORTH_STAND_0X3E = (120, 105)
# Live A1 (tag 20260904_A1): the naive BombWallController walk from the
# post-clear pose (40,189) wedged at (80,141) pushing RIGHT into the statue
# block at x~96,y~141 and burned ``stand_timeout`` without ever placing a
# bomb.  Approach the north band first (y=109 is clear of the statue row),
# then run east along it to the stand column.
BOMB_NORTH_APPROACH_0X3E: tuple[tuple[int, int], ...] = ((120, 109),)


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
        "magic_key": int(read_u8(ram, ADDR_MAGIC_KEY)),
        "selected_item": int(read_u8(ram, ADDR_SELECTED_ITEM)),
        "triforce": int(read_u8(ram, ADDR_TRIFORCE)),
        "health": int(snap.health),
        "heart_containers": int(snap.heart_containers),
    }


def _glance(env: Any) -> dict[str, Any]:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    result = leftover_from_snapshot(snap)
    live = [
        obj
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and obj.type_id not in (0, 0xFF) and obj.hp > 0
    ]
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
            "live_objects": [
                {
                    "slot": int(obj.slot),
                    "type": int(obj.type_id),
                    "type_hex": f"0x{obj.type_id:02X}",
                    "type_name": object_name(obj.type_id),
                    "xy": [int(obj.x), int(obj.y)],
                    "hp": int(obj.hp),
                }
                for obj in live
            ],
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
        live = [
            obj
            for obj in snap.objects
            if 1 <= obj.slot <= 12
            and obj.type_id not in (0, 0xFF)
            and obj.hp > 0
        ]
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
                "doors": int(snap.cur_opened_doors),
                "mask": int(snap.open_doorway_mask),
                "live": len(live),
                "live_types": [f"0x{obj.type_id:02X}" for obj in live],
            }
        )

    def save_shot(self, label: str) -> Path:
        snap = read_snapshot(self.env.get_ram())
        path = RECORDINGS_DIR / (
            f"l8_3e_north_{self.tag}_{label}_f{self.frame}_"
            f"L{snap.level}_s{snap.screen:02x}_m{snap.mode}.png"
        )
        save_rgb_png(self.latest_obs if self.latest_obs is not None else self.env.render(), path)
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


def _idle(env: ObservedEnv, assist: Any, total: list[int], frames: int, phase: str) -> None:
    env.set_reason(phase, "idle_census")
    for _ in range(frames):
        env.step(nes_idle_action())
        total[0] += 1
        assist.apply_env(env, frame=total[0])


def _drive_north_to_room(
    env: ObservedEnv,
    assist: Any,
    total: list[int],
    *,
    source_room: int,
    phase: str,
    max_frames: int = 1200,
) -> ZeldaSnapshot:
    """Hold UP and stop before acting in the first settled destination."""
    for _ in range(max_frames):
        snap = read_snapshot(env.get_ram())
        if snap.mode == 17:
            raise ProbeStop(f"{phase}:death")
        if snap.level != LEVEL8:
            raise ProbeStop(f"{phase}:left_level8:L{snap.level}")
        if snap.screen != source_room and snap.mode == PLAY_MODE and not snap.transitioning:
            return snap
        env.set_reason(phase, "push_up" if not snap.transitioning else "scroll_up")
        env.step(nes_action("UP"))
        total[0] += 1
        assist.apply_env(env, frame=total[0])
    raise ProbeStop(f"{phase}:timeout")


def _run_bomb_wall(
    env: ObservedEnv,
    assist: Any,
    total: list[int],
    *,
    room: int,
    stand: tuple[int, int],
    opens_to: int,
    phase: str,
    max_frames: int = 4000,
    stop_on_room_change: bool = False,
    approach_waypoints: tuple[tuple[int, int], ...] = (),
) -> dict[str, Any]:
    """Stand / face / place one bomb / push through the opened wall.

    ``stop_on_room_change`` ends the attempt at the first settled play room
    that is not the source, so a *wrong* destination is captured instead of
    burning the controller timeout.  The confirmed 0x6E replay keeps the
    original policy (``False``).
    """
    wall = SimpleNamespace(room=room, stand=stand, face="UP", opens_to=opens_to)
    controller = BombWallController(
        wall=wall,
        level=LEVEL8,
        max_frames=max_frames,
        approach_waypoints=approach_waypoints,
    )
    for _ in range(max_frames):
        snap = read_snapshot(env.get_ram())
        if snap.mode == 17:
            raise ProbeStop(f"{phase}:death")
        if (
            stop_on_room_change
            and snap.screen != room
            and snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.level == LEVEL8
        ):
            break
        action = controller.step(snap)
        env.set_reason(phase, action.reason)
        env.step(action.action)
        total[0] += 1
        assist.apply_env(env, frame=total[0])
        if controller.success or controller.phase.name == "FAILED":
            break
    return controller.report()


def _save_2e_fixture(
    raw_env: Any,
    *,
    fixture_name: str,
    gate: str,
    bombs_before: int,
    bombs_after: int,
    keys_before: int,
    keys_after: int,
    census: dict[str, Any],
) -> dict[str, Any]:
    """Save the settled-0x2E state + an auditable provenance sidecar.

    Disclosed writes: NONE.  The only inventory delta from
    ``Level8InteriorReconFixture`` is the natural key spend at the 0x4E->0x3E
    north key door plus the natural bomb spend at the 0x3E north wall.
    """
    from retro_harness.env import save_state, state_path
    from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance

    ram = raw_env.get_ram()
    after = read_snapshot(ram)
    if not (
        after.level == LEVEL8
        and after.mode == PLAY_MODE
        and after.screen == PREDICTED_BOMB_NORTH_ROOM
    ):
        raise ProbeStop("save_fixture_pin_mismatch")
    path = save_state(raw_env, GAME_DIR, GAME, fixture_name)
    source_path = state_path(GAME_DIR, GAME, "Level8InteriorReconFixture")
    result = {
        "ok": True,
        "source_state": "Level8InteriorReconFixture",
        "fixture_state": fixture_name,
        "state": compact_snapshot(after),
        "census": census,
        "gate": gate,
        "natural_key_spend": {"keys_before": keys_before, "keys_after": keys_after},
        "natural_bomb_spend": {"bombs_before": bombs_before, "bombs_after": bombs_after},
        "fixture_writes": [],
    }
    write_state_provenance(
        path,
        source_state_path=source_path if source_path.exists() else None,
        request={
            "bead": "rr-6o7.2",
            "phase": "level8_interior_0x2e_recon",
            "track": "recon_fixture",
            "route_eligible": False,
            "fixture_only": True,
            "natural_entry": False,
            "fixture_writes": [],
            "notes": [
                "Guarded replay from Level8InteriorReconFixture: reuse the confirmed"
                " 0x7E -> clear 0x6E -> bomb-N 0x5E -> clear/key -> shutter-N 0x4E"
                " -> key-N 0x3E policy, then ONE north gate 0x3E -> 0x2E.",
                "No RAM poke. Inventory deltas are the natural key spend"
                f" {keys_before}->{keys_after} at the 0x4E north key door and the"
                f" natural bomb spend {bombs_before}->{bombs_after} at 0x3E.",
                f"0x3E north gate observed as: {gate}.",
                "Not route eligible; not on L8_THROUGH.",
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
        default_state="Level8InteriorReconFixture",
        default_tag="20260904_v1",
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
        "trial": "fixture_replay_to_0x3e_then_one_north_gate_attempt",
        "from_state": args.from_state,
        "fixture_only": True,
        "natural_entry": False,
        "route_eligible": False,
        "infinite_life": True,
        "prediction": {
            "claim": "L8 play 0x5E x=120±4 UP through open shutter -> first settled play room 0x4E",
            "source_room": "0x5E",
            "direction": "UP",
            "x_band": [116, 124],
            "expected_destination": "0x4E",
            "keys_unchanged": True,
            "bombs_unchanged": True,
        },
        "prediction_next": {
            "written_before_run": True,
            "claim": (
                "clear the six 0x0C HP128 bodies in 0x3E with the sword, then ONE"
                " bomb from (120,105) facing UP opens the north wall -> first"
                " settled L8 play room 0x2E"
            ),
            "source_room": "0x3E",
            "direction": "UP",
            "expected_destination": "0x2E",
            "expected_gate": "bomb_north_wall",
            "bomb_stand": list(BOMB_NORTH_STAND_0X3E),
            "expected_keys": [9, 9],
            "expected_bombs": [7, 6],
            "hypothesis_edge": "blue_darknuts -> map_manhandla UP bomb",
            "contingency": (
                "if the clear raises open_doorway_mask UP bit 0x08 the north gate"
                " is a kill-clear shutter (hypothesis gate 'bomb' refuted); walk UP"
                " instead, bombs 7->7. Destination claim 0x2E unchanged."
            ),
            "attempt_history": [
                {
                    "tag": "20260904_A1",
                    "result": "no_go_navigation",
                    "detail": (
                        "clear ok (6/6 dead, cur_opened_doors 0x04 -> 0x05: the"
                        " clear opens the EAST shutter, not the north);"
                        " BombWallController stand walk wedged at (80,141)"
                        " against the statue block and hit stand_timeout."
                        " No bomb placed, so the north gate stayed untested."
                    ),
                    "fix": (
                        "approach_waypoints=((120,109),): ride the clear north"
                        " band east to the stand column before facing UP."
                    ),
                }
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
        if not (
            start["level"] == LEVEL8
            and start["screen"] == ENTRY_ROOM
            and start["mode"] == PLAY_MODE
            and inv0["triforce"] == POST_L7_TRIFORCE
            and inv0["sword"] == 3
            and inv0["bombs"] == 8
            and inv0["bow"] == 1
            and inv0["arrows"] == 1
            and inv0["keys"] == 9
            and inv0["magic_key"] == 0
        ):
            raise ProbeStop("fixture_start_mismatch")

        assist.apply_env(env, frame=0)
        _drive_north_to_room(
            env,
            assist,
            total,
            source_room=ENTRY_ROOM,
            phase="free_north_0x7e",
        )
        _idle(env, assist, total, 112, "settle_0x6e")
        entered_6e = _glance(env)
        payload["entered_0x6e"] = entered_6e
        env.save_shot("settled_0x6e")
        if not (
            entered_6e["screen"] == MANHANDLA_ROOM
            and entered_6e["mode"] == PLAY_MODE
            and any(obj["type"] == MANHANDLA_OBJECT_TYPE for obj in entered_6e["live_objects"])
        ):
            raise ProbeStop("settle_0x6e_census_mismatch")

        env.set_reason("clear_0x6e", "fight_clear_sword_only")
        clear_6e = fight_clear(
            env,
            assist,
            total,
            enemy_types=(MANHANDLA_OBJECT_TYPE,),
            max_frames=4000,
            use_bombs=False,
            level=LEVEL8,
        )
        payload["clear_0x6e"] = clear_6e
        after_clear_6e = read_snapshot(env.get_ram())
        if (
            not clear_6e.get("ok")
            or clear_6e.get("left_room")
            or after_clear_6e.screen != MANHANDLA_ROOM
            or after_clear_6e.mode != PLAY_MODE
        ):
            raise ProbeStop("clear_0x6e_first_departure_guard")
        env.save_shot("cleared_0x6e")

        before_menu = _inventory(env)
        env.set_reason("select_bombs", "pause_menu_input")
        menu = select_b_item_menu(env, assist, total, B_ITEM_BOMBS)
        after_menu = _inventory(env)
        payload["pause_menu_selection"] = {
            "result": menu,
            "before": before_menu,
            "after": after_menu,
            "ram_selection_write": False,
        }
        if after_menu["selected_item"] != B_ITEM_BOMBS:
            raise ProbeStop("pause_menu_failed_to_select_bombs")

        bombs_before_wall = after_menu["bombs"]
        payload["bomb_north_0x6e"] = _run_bomb_wall(
            env,
            assist,
            total,
            room=MANHANDLA_ROOM,
            stand=(120, 105),
            opens_to=DARKNUT_KEY_ROOM,
            phase="bomb_north_0x6e",
            max_frames=16000,
        )
        if not payload["bomb_north_0x6e"].get("success"):
            raise ProbeStop("bomb_north_0x6e_failed")
        _idle(env, assist, total, 2, "activate_0x5e")
        entered_5e = _glance(env)
        payload["entered_0x5e"] = entered_5e
        env.save_shot("spawn_0x5e")
        darknuts = [
            obj
            for obj in entered_5e["live_objects"]
            if obj["type"] == BLUE_DARKNUT_OBJECT_TYPE
        ]
        if not (
            entered_5e["screen"] == DARKNUT_KEY_ROOM
            and entered_5e["mode"] == PLAY_MODE
            and len(darknuts) == 5
            and all(obj["hp"] == 128 for obj in darknuts)
            and entered_5e["inventory"]["bombs"] == bombs_before_wall - 1
        ):
            raise ProbeStop("settle_0x5e_census_or_bomb_delta_mismatch")

        env.set_reason("clear_0x5e", "fight_clear_sword_only_guard_departure")
        clear_5e = fight_clear(
            env,
            assist,
            total,
            enemy_types=(BLUE_DARKNUT_OBJECT_TYPE,),
            max_frames=5000,
            use_bombs=False,
            level=LEVEL8,
        )
        payload["clear_0x5e"] = clear_5e
        after_clear_5e = read_snapshot(env.get_ram())
        if (
            not clear_5e.get("ok")
            or clear_5e.get("left_room")
            or after_clear_5e.screen != DARKNUT_KEY_ROOM
            or after_clear_5e.mode != PLAY_MODE
        ):
            raise ProbeStop("clear_0x5e_first_departure_guard")
        env.save_shot("cleared_0x5e")

        keys_before_pickup = int(after_clear_5e.keys)
        env.set_reason("pickup_center_key_0x5e", "goto_120_141")
        if not goto(env, assist, total, 120, 141, tol=3, max_f=400):
            raise ProbeStop("center_key_occupancy_miss")
        for _ in range(100):
            snap = read_snapshot(env.get_ram())
            if snap.keys == keys_before_pickup + 1:
                break
            env.set_reason("pickup_center_key_0x5e", "idle_item_freeze")
            env.step(nes_idle_action())
            total[0] += 1
            assist.apply_env(env, frame=total[0])
        after_pickup = _glance(env)
        payload["natural_key_pickup"] = {
            "target": [120, 141],
            "item": f"0x{SMALL_KEY_ITEM:02X}",
            "keys_before": keys_before_pickup,
            "keys_after": after_pickup["inventory"]["keys"],
            "state": after_pickup,
        }
        env.save_shot("key_picked_0x5e")
        if after_pickup["inventory"]["keys"] != keys_before_pickup + 1:
            raise ProbeStop("natural_key_pickup_miss")
        if not (after_pickup["open_doorway_mask"] & DOWN_DOOR_BIT):
            raise ProbeStop("north_shutter_not_open")

        env.set_reason("align_north_shutter_0x5e", "x_120")
        snap = read_snapshot(env.get_ram())
        if not goto(env, assist, total, 120, snap.link_y, tol=4, max_f=200):
            raise ProbeStop("north_shutter_x_occupancy_miss")
        before_north = _glance(env)
        payload["prediction"]["observed_source"] = before_north
        if not (
            before_north["screen"] == DARKNUT_KEY_ROOM
            and before_north["mode"] == PLAY_MODE
            and 116 <= before_north["xy"][0] <= 124
        ):
            raise ProbeStop("prediction_source_pose_miss")

        first_dest = _drive_north_to_room(
            env,
            assist,
            total,
            source_room=DARKNUT_KEY_ROOM,
            phase="north_shutter_0x5e",
        )
        env._sample(first_dest, "first_settled_destination")
        env.save_shot("first_settled_0x4e")
        _idle(env, assist, total, 120, "destination_idle_census")
        arrived_4e = _glance(env)
        payload["arrived_0x4e"] = arrived_4e
        if first_dest.screen != SHUTTER_ROOM:
            payload["failed"] = "replay_miss_0x4e"
            raise ProbeStop(
                f"replay_miss_0x4e:0x{first_dest.screen:02X}"
            )

        # ---- confirmed: ONE natural north key door 0x4E -> 0x3E ----
        keys_before_door = int(read_snapshot(env.get_ram()).keys)
        bombs_before_door = int(read_snapshot(env.get_ram()).bombs)
        env.set_reason("key_north_0x4e", "exit_door_up")
        door = exit_door(env, assist, total, "UP", push=180)
        env.save_shot("key_north_pushed")
        _idle(env, assist, total, 150, "keydoor_dest_idle_census")
        arrived_3e = _glance(env)
        settled_3e = read_snapshot(env.get_ram())
        payload["key_north_0x4e"] = {
            "door": door,
            "keys_before": keys_before_door,
            "keys_after": int(settled_3e.keys),
            "bombs_before": bombs_before_door,
            "bombs_after": int(settled_3e.bombs),
            "settled_room": f"0x{settled_3e.screen:02X}",
            "settled_xy": [int(settled_3e.link_x), int(settled_3e.link_y)],
            "census": arrived_3e,
            "room_fields": room_fields(settled_3e, env.get_ram()),
        }
        payload["arrived_0x3e"] = arrived_3e
        if not (
            settled_3e.level == LEVEL8
            and settled_3e.mode == PLAY_MODE
            and settled_3e.screen == BLUE_DARKNUT_ROOM
        ):
            payload["failed"] = "replay_miss_0x3e"
            raise ProbeStop(f"replay_miss_0x3e:0x{settled_3e.screen:02X}")
        live_3e = [
            obj
            for obj in arrived_3e["live_objects"]
            if obj["type"] == BLUE_DARKNUT_OBJECT_TYPE
        ]
        if not (
            len(live_3e) == 6
            and all(obj["hp"] == 128 for obj in live_3e)
            and arrived_3e["room_item_id"] == 0x03
            and arrived_3e["inventory"]["selected_item"] == B_ITEM_BOMBS
        ):
            raise ProbeStop("arrived_0x3e_census_mismatch")

        # ---- rr-6o7.2: ONE new boundary, 0x3E north -> predicted 0x2E ----
        keys_in = int(settled_3e.keys)
        bombs_in = int(settled_3e.bombs)
        env.set_reason("clear_0x3e", "fight_clear_sword_only_guard_departure")
        clear_3e = fight_clear(
            env,
            assist,
            total,
            enemy_types=(BLUE_DARKNUT_OBJECT_TYPE,),
            max_frames=8000,
            use_bombs=False,
            level=LEVEL8,
        )
        payload["clear_0x3e"] = clear_3e
        after_clear_3e = read_snapshot(env.get_ram())
        if (
            not clear_3e.get("ok")
            or clear_3e.get("left_room")
            or after_clear_3e.screen != BLUE_DARKNUT_ROOM
            or after_clear_3e.mode != PLAY_MODE
        ):
            payload["failed"] = "clear_0x3e_first_departure_guard"
            raise ProbeStop("clear_0x3e_first_departure_guard")
        _idle(env, assist, total, 60, "post_clear_0x3e_census")
        post_clear = _glance(env)
        payload["post_clear_0x3e"] = post_clear
        env.save_shot("cleared_0x3e")

        north_open_after_clear = bool(post_clear["open_doorway_mask"] & UP_DOOR_BIT)
        payload["north_gate_0x3e"] = {
            "north_open_after_clear": north_open_after_clear,
            "cur_opened_doors": post_clear["cur_opened_doors"],
            "open_doorway_mask": post_clear["open_doorway_mask"],
            "room_all_dead": post_clear["room_all_dead"],
            "keys_in": keys_in,
            "bombs_in": bombs_in,
        }
        if north_open_after_clear:
            gate = "north_shutter_on_clear"
            env.set_reason("north_open_0x3e", "align_x_120")
            snap = read_snapshot(env.get_ram())
            goto(env, assist, total, 120, snap.link_y, tol=4, max_f=200)
            settled = _drive_north_to_room(
                env,
                assist,
                total,
                source_room=BLUE_DARKNUT_ROOM,
                phase="north_open_0x3e",
            )
            payload["north_gate_0x3e"]["bomb_wall"] = None
        else:
            gate = "bomb_north_wall"
            payload["north_gate_0x3e"]["bomb_wall"] = _run_bomb_wall(
                env,
                assist,
                total,
                room=BLUE_DARKNUT_ROOM,
                stand=BOMB_NORTH_STAND_0X3E,
                opens_to=PREDICTED_BOMB_NORTH_ROOM,
                phase="bomb_north_0x3e",
                max_frames=4000,
                stop_on_room_change=True,
                approach_waypoints=BOMB_NORTH_APPROACH_0X3E,
            )
            settled = read_snapshot(env.get_ram())
        env._sample(settled, "first_settled_after_north_gate")
        env.save_shot("first_settled_after_north_gate")
        _idle(env, assist, total, 150, "north_gate_dest_idle_census")
        final = _glance(env)
        settled = read_snapshot(env.get_ram())
        env.save_shot("final_census")
        keys_out = int(settled.keys)
        bombs_out = int(settled.bombs)
        payload["north_gate_0x3e"].update(
            {
                "observed_gate": gate,
                "keys_out": keys_out,
                "bombs_out": bombs_out,
                "settled_room": f"0x{settled.screen:02X}",
                "settled_xy": [int(settled.link_x), int(settled.link_y)],
                "settled_mode": int(settled.mode),
                "settled_level": int(settled.level),
                "census": final,
                "room_fields": room_fields(settled, env.get_ram()),
            }
        )
        gate_misses: list[str] = []
        if not (settled.level == LEVEL8 and settled.mode == PLAY_MODE):
            gate_misses.append(f"settled L{settled.level} mode={settled.mode}")
        if settled.screen == BLUE_DARKNUT_ROOM:
            gate_misses.append("north gate did not open (still 0x3E)")
        elif settled.screen != PREDICTED_BOMB_NORTH_ROOM:
            gate_misses.append(
                f"settled 0x{settled.screen:02X} != predicted 0x2E"
            )
        if keys_out != keys_in:
            gate_misses.append(f"keys changed {keys_in}->{keys_out}")
        expected_bombs = bombs_in if north_open_after_clear else bombs_in - 1
        if bombs_out != expected_bombs:
            gate_misses.append(
                f"bombs {bombs_in}->{bombs_out}, expected {expected_bombs}"
            )
        payload["north_gate_0x3e"]["grade"] = {
            "pass": not gate_misses,
            "misses": gate_misses,
            "gate_hypothesis_pass": gate == "bomb_north_wall",
        }
        payload["final"] = final
        payload["success"] = not gate_misses
        if gate_misses:
            payload["failed"] = "north_gate_0x3e_miss"
        elif args.save_fixture:
            payload["saved_fixture"] = _save_2e_fixture(
                raw_env,
                fixture_name=args.save_fixture,
                gate=gate,
                bombs_before=bombs_in,
                bombs_after=bombs_out,
                keys_before=keys_before_door,
                keys_after=keys_out,
                census=final,
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
            gate_evidence = payload.get("north_gate_0x3e", {})
            keys_out = gate_evidence.get("keys_out")
            bombs_out = gate_evidence.get("bombs_out")
            payload["runtime_integrity"] = {
                "deaths": int(assist.telemetry.deaths),
                "progression_writes": int(assist.telemetry.progression_writes),
                "capacity_writes": int(assist.telemetry.capacity_writes),
                "direct_ram_writes": 0,
                "state_loads_after_start": 0,
                "combat_in_destination": False,
                "keys_spent_at_new_boundary": (
                    None
                    if keys_out is None
                    else int(gate_evidence.get("keys_in", 0)) - int(keys_out)
                ),
                "bombs_spent_at_new_boundary": (
                    None
                    if bombs_out is None
                    else int(gate_evidence.get("bombs_in", 0)) - int(bombs_out)
                ),
            }
        report = write_report("l8_3e_north_fixture", payload, tag=args.tag)
        raw_env.close()

    print(f"report={report}")
    print(f"success={payload.get('success')} failed={payload.get('failed')}")
    print(f"north_gate={payload.get('north_gate_0x3e', {}).get('grade')}")
    print(f"final={payload.get('final')}")
    print(f"assist={payload.get('assist')}")
    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
