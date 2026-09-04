"""Clear ``0x2E`` then take ONE north gate toward predicted ``0x1E``.

rr-6o7.2, one boundary past ``probe_l8_3e_north.py``.  That probe replayed the
whole confirmed prefix from ``Level8InteriorReconFixture`` and settled 0x2E;
this one starts from the frontier pin it saved (``Level8Interior2EReconFixture``,
a live continuation of that same replay, ``natural_entry=false``) so a sitting
costs a few hundred frames instead of ~4800.

PREDICTION (written before the first run, graded in the report):

  ``0x2E`` holds ONE Manhandla -- five live ``0x3C`` HP64 slots (body + four
  heads) at ~(182,119) -- and arrives with ``cur_opened_doors`` /
  ``open_doorway_mask`` = ``0x04``, the DOWN bomb hole we came up through.
  The Manhandla is killed with the sword only: no bombs, no arrows.  The
  hypothesis edge ``map_manhandla -> blue_gohma UP key`` then says the north
  gate is a KEY door, so ONE north door push spends exactly one key and lands
  in the row-1 neighbour ``0x1E`` (high nibble -1, low nibble E, the same
  north-walk the whole 0x7E..0x2E chain has shown).

    expected destination 0x1E, keys 9 -> 8, bombs 6 -> 6.

  Declared contingencies, recorded rather than hidden:

  * if the clear itself raises the UP bit ``0x08`` the north gate is a
    kill-clear shutter, the hypothesis gate "key" is refuted, and we walk UP
    with keys 9 -> 9.  The destination claim ``0x1E`` is unchanged and graded
    separately from the gate kind.
  * ``room_item_id`` in 0x2E is ``0x17`` (map, walkthrough-correlated).  Map
    stays OMITTED from the route, so after the clear we approach the north
    door along the y=109 band instead of walking the centre column through a
    dropped item.  ``ADDR_MAP`` is read before and after and graded: a live
    0 -> 1 is reported as an incidental pickup, never a detour.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l8_2e_north.py \
        --from-state Level8Interior2EReconFixture \
        --tag 20260904_C1 --infinite-life
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.ids import MANHANDLA_OBJECT_TYPE, object_name
from zelda_i.dungeon.ops import fight_clear, goto, room_fields
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
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)
from zelda_i.runner import add_common_args, make_assist, write_report
from zelda_i.screen_glance import leftover_from_snapshot

LEVEL8 = 8
MAP_MANHANDLA_ROOM = 0x2E
PREDICTED_NORTH_ROOM = 0x1E
POST_L7_TRIFORCE = 0x7F
B_ITEM_BOMBS = 1
# door_graph.core bit layout: RIGHT 0x01, LEFT 0x02, DOWN 0x04, UP 0x08.
# 0x2E arrives with 0x04 set -- the DOWN bomb hole from 0x3E.
UP_DOOR_BIT = 0x08
# Dodge the centre column so a dropped 0x17 map is not walked over.  The north
# band at y=109 was the clear lane in 0x3E as well.
NORTH_DOOR_WAYPOINTS: tuple[tuple[int, int], ...] = ((88, 109), (120, 109))
NORTH_DOOR_TILE = (120, 93)


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
            if 1 <= obj.slot <= 12 and obj.type_id not in (0, 0xFF) and obj.hp > 0
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
            f"l8_2e_north_{self.tag}_{label}_f{self.frame}_"
            f"L{snap.level}_s{snap.screen:02x}_m{snap.mode}.png"
        )
        save_rgb_png(
            self.latest_obs if self.latest_obs is not None else self.env.render(), path
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


def _idle(env: ObservedEnv, assist: Any, total: list[int], frames: int, phase: str) -> None:
    env.set_reason(phase, "idle_census")
    for _ in range(frames):
        env.step(nes_idle_action())
        total[0] += 1
        assist.apply_env(env, frame=total[0])


def _north_gate_walk(
    env: ObservedEnv,
    assist: Any,
    total: list[int],
    *,
    source_room: int,
    phase: str,
    max_frames: int = 1200,
) -> dict[str, Any]:
    """Ride the y=109 band to the north door tile, then push UP.

    The waypoint lane keeps Link off the room centre so a dropped map item is
    not collected on the way out.  Returns the walk trace; the caller grades
    the settled room.
    """
    trace: list[dict[str, Any]] = []
    for wx, wy in NORTH_DOOR_WAYPOINTS:
        env.set_reason(phase, f"waypoint_{wx}_{wy}")
        ok = goto(env, assist, total, wx, wy, tol=5, max_f=400)
        snap = read_snapshot(env.get_ram())
        trace.append(
            {
                "waypoint": [wx, wy],
                "reached": bool(ok),
                "xy": [int(snap.link_x), int(snap.link_y)],
            }
        )
    env.set_reason(phase, "door_tile_120_93")
    ok_tile = goto(env, assist, total, *NORTH_DOOR_TILE, tol=6, max_f=400)
    snap = read_snapshot(env.get_ram())
    trace.append(
        {
            "waypoint": list(NORTH_DOOR_TILE),
            "reached": bool(ok_tile),
            "xy": [int(snap.link_x), int(snap.link_y)],
        }
    )
    at_door = room_fields(read_snapshot(env.get_ram()), env.get_ram())
    for _ in range(max_frames):
        snap = read_snapshot(env.get_ram())
        if snap.mode == 17:
            raise ProbeStop(f"{phase}:death")
        if snap.level != LEVEL8:
            raise ProbeStop(f"{phase}:left_level8:L{snap.level}")
        if (
            snap.screen != source_room
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            break
        env.set_reason(phase, "push_up" if not snap.transitioning else "scroll_up")
        env.step(nes_action("UP"))
        total[0] += 1
        assist.apply_env(env, frame=total[0])
    else:
        raise ProbeStop(f"{phase}:timeout")
    return {"walk": trace, "at_door": at_door}


def _save_fixture(
    raw_env: Any,
    *,
    fixture_name: str,
    gate: str,
    keys_before: int,
    keys_after: int,
    bombs_before: int,
    bombs_after: int,
    census: dict[str, Any],
) -> dict[str, Any]:
    """Save the settled-0x1E state + an auditable provenance sidecar.

    Disclosed writes: NONE.  The only inventory delta from
    ``Level8Interior2EReconFixture`` is the natural key spend at the 0x2E north
    gate (or nothing at all if the gate turned out to be a kill-clear shutter).
    """
    from retro_harness.env import save_state, state_path
    from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance

    after = read_snapshot(raw_env.get_ram())
    if not (
        after.level == LEVEL8
        and after.mode == PLAY_MODE
        and after.screen == PREDICTED_NORTH_ROOM
    ):
        raise ProbeStop("save_fixture_pin_mismatch")
    path = save_state(raw_env, GAME_DIR, GAME, fixture_name)
    source_path = state_path(GAME_DIR, GAME, "Level8Interior2EReconFixture")
    result = {
        "ok": True,
        "source_state": "Level8Interior2EReconFixture",
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
            "phase": "level8_interior_0x1e_recon",
            "track": "recon_fixture",
            "route_eligible": False,
            "fixture_only": True,
            "natural_entry": False,
            "fixture_writes": [],
            "notes": [
                "Continuation pin: Level8Interior2EReconFixture is the settled"
                " 0x2E frame of the probe_l8_3e_north replay, itself a"
                " fixture-only chain from Level8InteriorReconFixture.",
                "Clear the one Manhandla in 0x2E with the sword only, then ONE"
                " north gate 0x2E -> 0x1E.",
                "No RAM poke. The only inventory delta is the natural key spend"
                f" {keys_before}->{keys_after} at the 0x2E north gate.",
                f"0x2E north gate observed as: {gate}.",
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
        default_state="Level8Interior2EReconFixture",
        default_tag="20260904_C1",
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
        "trial": "clear_0x2e_then_one_north_gate_attempt",
        "from_state": args.from_state,
        "fixture_only": True,
        "natural_entry": False,
        "route_eligible": False,
        "infinite_life": True,
        "prediction": {
            "written_before_run": True,
            "claim": (
                "clear the one Manhandla in 0x2E (5 live 0x3C HP64 slots) with"
                " the sword only, then ONE north KEY door -> first settled L8"
                " play room 0x1E"
            ),
            "source_room": "0x2E",
            "direction": "UP",
            "expected_destination": "0x1E",
            "expected_gate": "north_key_door",
            "expected_keys": [9, 8],
            "expected_bombs": [6, 6],
            "hypothesis_edge": "map_manhandla -> blue_gohma UP key",
            "contingency": (
                "if the clear raises open_doorway_mask UP bit 0x08 the north"
                " gate is a kill-clear shutter (hypothesis gate 'key' refuted);"
                " walk UP instead, keys 9->9. Destination claim 0x1E unchanged."
            ),
            "map_policy": (
                "room_item_id 0x17 is the walkthrough map and stays OMITTED;"
                " the north walk rides the y=109 band instead of the centre"
                " column. ADDR_MAP graded before/after as incidental only."
            ),
            "attempt_history": [],
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
        manhandla0 = [
            obj
            for obj in start["live_objects"]
            if obj["type"] == MANHANDLA_OBJECT_TYPE
        ]
        if not (
            start["level"] == LEVEL8
            and start["screen"] == MAP_MANHANDLA_ROOM
            and start["mode"] == PLAY_MODE
            and inv0["triforce"] == POST_L7_TRIFORCE
            and inv0["sword"] == 3
            and inv0["keys"] == 9
            and inv0["bombs"] == 6
            and inv0["bow"] == 1
            and inv0["arrows"] == 1
            and inv0["magic_key"] == 0
            and inv0["map"] == 0
            and len(manhandla0) == 5
            and start["room_item_id"] == 0x17
        ):
            raise ProbeStop("fixture_start_mismatch")

        assist.apply_env(env, frame=0)
        keys_in = int(inv0["keys"])
        bombs_in = int(inv0["bombs"])
        map_in = int(inv0["map"])

        env.set_reason("clear_0x2e", "fight_clear_sword_only_guard_departure")
        clear = fight_clear(
            env,
            assist,
            total,
            enemy_types=(MANHANDLA_OBJECT_TYPE,),
            max_frames=8000,
            use_bombs=False,
            level=LEVEL8,
        )
        payload["clear_0x2e"] = clear
        after_clear = read_snapshot(env.get_ram())
        if (
            not clear.get("ok")
            or clear.get("left_room")
            or after_clear.screen != MAP_MANHANDLA_ROOM
            or after_clear.mode != PLAY_MODE
        ):
            payload["failed"] = "clear_0x2e_first_departure_guard"
            raise ProbeStop("clear_0x2e_first_departure_guard")
        _idle(env, assist, total, 60, "post_clear_0x2e_census")
        post_clear = _glance(env)
        payload["post_clear_0x2e"] = post_clear
        env.save_shot("cleared_0x2e")

        north_open_after_clear = bool(post_clear["open_doorway_mask"] & UP_DOOR_BIT)
        payload["north_gate_0x2e"] = {
            "north_open_after_clear": north_open_after_clear,
            "cur_opened_doors": post_clear["cur_opened_doors"],
            "open_doorway_mask": post_clear["open_doorway_mask"],
            "room_all_dead": post_clear["room_all_dead"],
            "keys_in": keys_in,
            "bombs_in": bombs_in,
            "map_in": map_in,
        }
        gate = "north_shutter_on_clear" if north_open_after_clear else "north_key_door"
        payload["north_gate_0x2e"]["walk"] = _north_gate_walk(
            env,
            assist,
            total,
            source_room=MAP_MANHANDLA_ROOM,
            phase="north_gate_0x2e",
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
        payload["north_gate_0x2e"].update(
            {
                "observed_gate": gate,
                "keys_out": keys_out,
                "bombs_out": bombs_out,
                "map_out": int(final["inventory"]["map"]),
                "settled_room": f"0x{settled.screen:02X}",
                "settled_xy": [int(settled.link_x), int(settled.link_y)],
                "settled_mode": int(settled.mode),
                "settled_level": int(settled.level),
                "census": final,
                "room_fields": room_fields(settled, env.get_ram()),
            }
        )
        misses: list[str] = []
        if not (settled.level == LEVEL8 and settled.mode == PLAY_MODE):
            misses.append(f"settled L{settled.level} mode={settled.mode}")
        if settled.screen == MAP_MANHANDLA_ROOM:
            misses.append("north gate did not open (still 0x2E)")
        elif settled.screen != PREDICTED_NORTH_ROOM:
            misses.append(f"settled 0x{settled.screen:02X} != predicted 0x1E")
        expected_keys = keys_in if north_open_after_clear else keys_in - 1
        if keys_out != expected_keys:
            misses.append(f"keys {keys_in}->{keys_out}, expected {expected_keys}")
        if bombs_out != bombs_in:
            misses.append(f"bombs changed {bombs_in}->{bombs_out}")
        payload["north_gate_0x2e"]["grade"] = {
            "pass": not misses,
            "misses": misses,
            "gate_hypothesis_pass": gate == "north_key_door",
            "map_stayed_omitted": int(final["inventory"]["map"]) == map_in,
        }
        payload["final"] = final
        payload["success"] = not misses
        if misses:
            payload["failed"] = "north_gate_0x2e_miss"
        elif args.save_fixture:
            payload["saved_fixture"] = _save_fixture(
                raw_env,
                fixture_name=args.save_fixture,
                gate=gate,
                keys_before=keys_in,
                keys_after=keys_out,
                bombs_before=bombs_in,
                bombs_after=bombs_out,
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
            gate_evidence = payload.get("north_gate_0x2e", {})
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
        report = write_report("l8_2e_north_fixture", payload, tag=args.tag)
        raw_env.close()

    print(f"report={report}")
    print(f"success={payload.get('success')} failed={payload.get('failed')}")
    print(f"north_gate={payload.get('north_gate_0x2e', {}).get('grade')}")
    print(f"final={payload.get('final')}")
    print(f"assist={payload.get('assist')}")
    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
