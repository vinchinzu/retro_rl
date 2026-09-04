"""Replay the fixture-live L8 prefix to ``0x4E`` then try the north key door.

rr-6o7.2 groundwork.  Starts from the disclosed ``Level8InteriorReconFixture``,
reuses the confirmed ``0x7E -> clear 0x6E -> bomb-N 0x5E -> clear/key ->
shutter-N 0x4E`` policy unchanged, then makes ONE attempt at the north key
door from ``0x4E`` toward predicted ``0x3E``.  It does NOT clear the mixed
``0x4E`` census; the only new act is aligning to the north door and pushing
UP (natural key spend).  Stops at the first settled destination and censuses.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l8_4e_north.py \
        --from-state Level8InteriorReconFixture \
        --tag 20260903_v1 --infinite-life
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
PREDICTED_NORTH_ROOM = 0x4E
PREDICTED_KEYDOOR_ROOM = 0x3E
POST_L7_TRIFORCE = 0x7F
BLUE_DARKNUT_OBJECT_TYPE = 0x0C
SMALL_KEY_ITEM = 0x19
B_ITEM_BOMBS = 1
NORTH_OPEN_MASK = 0x04


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
            f"l8_5e_north_{self.tag}_{label}_f{self.frame}_"
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
    env: ObservedEnv, assist: Any, total: list[int]
) -> dict[str, Any]:
    wall = SimpleNamespace(
        room=MANHANDLA_ROOM,
        stand=(120, 105),
        face="UP",
        opens_to=DARKNUT_KEY_ROOM,
    )
    controller = BombWallController(wall=wall, level=LEVEL8)
    for _ in range(controller.max_frames):
        snap = read_snapshot(env.get_ram())
        action = controller.step(snap)
        env.set_reason("bomb_north_0x6e", action.reason)
        env.step(action.action)
        total[0] += 1
        assist.apply_env(env, frame=total[0])
        if controller.success or controller.phase.name == "FAILED":
            break
    report = controller.report()
    if not controller.success:
        raise ProbeStop(f"bomb_north_0x6e:{report['notes'][-1:]}")
    return report


def _save_3e_fixture(
    raw_env: Any,
    *,
    fixture_name: str,
    keys_before: int,
    keys_after: int,
    census: dict[str, Any],
) -> dict[str, Any]:
    """Save the settled-0x3E state + an auditable provenance sidecar.

    Disclosed writes: NONE. The only inventory delta from
    ``Level8InteriorReconFixture`` is the natural key spend at the 0x4E->0x3E
    north door (keys ``keys_before``->``keys_after``); no RAM was poked.
    """
    from retro_harness.env import save_state, state_path
    from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance

    ram = raw_env.get_ram()
    after = read_snapshot(ram)
    if not (after.level == LEVEL8 and after.mode == PLAY_MODE and after.screen == PREDICTED_KEYDOOR_ROOM):
        raise ProbeStop("save_fixture_pin_mismatch")
    path = save_state(raw_env, GAME_DIR, GAME, fixture_name)
    source_path = state_path(GAME_DIR, GAME, "Level8InteriorReconFixture")
    result = {
        "ok": True,
        "source_state": "Level8InteriorReconFixture",
        "fixture_state": fixture_name,
        "state": compact_snapshot(after),
        "census": census,
        "natural_key_spend": {"keys_before": keys_before, "keys_after": keys_after},
        "fixture_writes": [],
    }
    write_state_provenance(
        path,
        source_state_path=source_path if source_path.exists() else None,
        request={
            "bead": "rr-6o7.2",
            "phase": "level8_interior_0x3e_recon",
            "track": "recon_fixture",
            "route_eligible": False,
            "fixture_only": True,
            "natural_entry": False,
            "fixture_writes": [],
            "notes": [
                "Guarded replay from Level8InteriorReconFixture: reuse the confirmed"
                " 0x7E -> clear 0x6E -> bomb-N 0x5E -> clear/key -> shutter-N 0x4E"
                " policy, then ONE natural north key door 0x4E -> 0x3E.",
                "No RAM poke. Only inventory delta is the natural key spend"
                f" {keys_before}->{keys_after} at the north door.",
                "0x4E mixed census was NOT cleared.",
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
        default_tag="20260903_v1",
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
        "trial": "fixture_replay_to_0x4e_then_one_north_key_door_attempt",
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
        payload["bomb_north_0x6e"] = _run_bomb_wall(env, assist, total)
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
        if not (after_pickup["open_doorway_mask"] & NORTH_OPEN_MASK):
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

        before_dest_inv = _inventory(env)
        first_dest = _drive_north_to_room(
            env,
            assist,
            total,
            source_room=DARKNUT_KEY_ROOM,
            phase="north_shutter_0x5e",
        )
        env._sample(first_dest, "first_settled_destination")
        env.save_shot("first_settled_destination")
        _idle(env, assist, total, 120, "destination_idle_census")
        final = _glance(env)
        after_dest_inv = final["inventory"]
        env.save_shot("final_census")
        grade_misses: list[str] = []
        if first_dest.screen != PREDICTED_NORTH_ROOM:
            grade_misses.append(
                f"destination 0x{first_dest.screen:02X} != predicted 0x{PREDICTED_NORTH_ROOM:02X}"
            )
        if first_dest.level != LEVEL8 or first_dest.mode != PLAY_MODE:
            grade_misses.append(
                f"settled state L{first_dest.level} room=0x{first_dest.screen:02X} mode={first_dest.mode}"
            )
        if after_dest_inv["keys"] != before_dest_inv["keys"]:
            grade_misses.append(
                f"keys changed {before_dest_inv['keys']}->{after_dest_inv['keys']}"
            )
        if after_dest_inv["bombs"] != before_dest_inv["bombs"]:
            grade_misses.append(
                f"bombs changed {before_dest_inv['bombs']}->{after_dest_inv['bombs']}"
            )
        payload["prediction"]["grade"] = {
            "pass": not grade_misses,
            "misses": grade_misses,
            "first_settled": room_fields(first_dest, env.get_ram()),
        }
        payload["destination_census"] = final
        payload["arrived_0x4e"] = final

        # ---- rr-6o7.2: ONE attempt at the north key door 0x4E -> 0x3E ----
        # Do NOT clear the mixed 0x4E census. The only new act is aligning to
        # the north door and pushing UP (natural key spend).
        if grade_misses:
            payload["final"] = final
            payload["success"] = False
            payload["failed"] = "prediction_miss_0x4e"
            raise ProbeStop("prediction_miss_0x4e")

        keys_before_door = int(read_snapshot(env.get_ram()).keys)
        bombs_before_door = int(read_snapshot(env.get_ram()).bombs)
        env.set_reason("key_north_0x4e", "exit_door_up")
        door = exit_door(env, assist, total, "UP", push=180)
        env.save_shot("key_north_pushed")
        _idle(env, assist, total, 150, "keydoor_dest_idle_census")
        after_door = _glance(env)
        settled = read_snapshot(env.get_ram())
        keys_after_door = int(settled.keys)
        bombs_after_door = int(settled.bombs)
        payload["key_north_0x4e"] = {
            "door": door,
            "keys_before": keys_before_door,
            "keys_after": keys_after_door,
            "bombs_before": bombs_before_door,
            "bombs_after": bombs_after_door,
            "settled_room": f"0x{settled.screen:02X}",
            "settled_xy": [int(settled.link_x), int(settled.link_y)],
            "settled_mode": int(settled.mode),
            "settled_level": int(settled.level),
            "census": after_door,
            "room_fields": room_fields(settled, env.get_ram()),
        }
        door_misses: list[str] = []
        if not (settled.level == LEVEL8 and settled.mode == PLAY_MODE):
            door_misses.append(
                f"settled L{settled.level} mode={settled.mode}"
            )
        if settled.screen == PREDICTED_NORTH_ROOM:
            door_misses.append("key door did not open (still 0x4E)")
        elif settled.screen != PREDICTED_KEYDOOR_ROOM:
            door_misses.append(
                f"settled 0x{settled.screen:02X} != predicted 0x3E"
            )
        payload["key_north_0x4e"]["grade"] = {
            "pass": not door_misses,
            "misses": door_misses,
        }
        payload["final"] = after_door
        payload["success"] = not door_misses
        if door_misses:
            payload["failed"] = "key_north_0x4e_miss"
        elif args.save_fixture:
            fx = _save_3e_fixture(
                raw_env,
                fixture_name=args.save_fixture,
                keys_before=keys_before_door,
                keys_after=keys_after_door,
                census=after_door,
            )
            payload["saved_fixture"] = fx
    except ProbeStop as exc:
        payload["success"] = False
        payload["failed"] = str(exc)
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
            payload["runtime_integrity"] = {
                "deaths": int(assist.telemetry.deaths),
                "progression_writes": int(assist.telemetry.progression_writes),
                "capacity_writes": int(assist.telemetry.capacity_writes),
                "direct_ram_writes": 0,
                "state_loads_after_start": 0,
                "combat_in_destination": False,
                "key_or_bomb_use_in_destination": bool(
                    payload.get("key_north_0x4e", {}).get("keys_before")
                    != payload.get("key_north_0x4e", {}).get("keys_after")
                ),
            }
        report = write_report("l8_4e_north_fixture", payload, tag=args.tag)
        raw_env.close()

    print(f"report={report}")
    print(f"success={payload.get('success')} failed={payload.get('failed')}")
    print(f"final={payload.get('final')}")
    print(f"assist={payload.get('assist')}")
    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
