"""North dest hop from Level 9 entry leftover (play 0x76).

rr-sz8.6. Pin ``Level9EntranceReconFixture`` (composed full inventory,
live level==9 room 0x76). Glance leftover FIRST. Dest $EB is RAM
(hyp 0x66 Old Man full-TF gate). Fail if dest is not a north neighbor.
Do not poke TF / ADDR_ARROWS. Do not walk Compass, Map-Patra, or 0x07.

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/probe_l9_76_north.py \\
        --from-state Level9EntranceReconFixture \\
        --tag 20260904_P1 --infinite-life --no-video
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.ids import object_name
from zelda_i.level9.dungeon import FULL_TRIFORCE, LEVEL9
from zelda_i.level9.prefix import (
    NORTH_DEST_HYP,
    NORTH_DOOR,
    NORTH_ORIGIN,
    RED_RING,
    WEST_DEST_HYP,
    WEST_ORIGIN,
    is_north_neighbor,
    is_west_neighbor,
    make_north_76_controller,
    make_west_66_controller,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOMBS,
    ADDR_BOW,
    ADDR_COMPASS,
    ADDR_KEYS,
    ADDR_MAGIC_KEY,
    ADDR_MAP,
    ADDR_RING,
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

POSE_TOL = 8
DEATH_MODE = 17
CENSUS_IDLE = 60
SOURCE_STATE = "Level9EntranceReconFixture"

RAM_CLAIM_NORTH = (
    "From play 0x76 leftover (120,205) facing UP (glance 20260904_glance), "
    "hold UP through the ROM-open north door. Already on the x=120 band; "
    "no LEFT/RIGHT. First settled play $EB is RAM (hyp 0x66 Old Man "
    "full-TF gate, the -0x10 north neighbor). Fail if dest is not a north "
    "neighbor. Fail dest 0x07 Red Ring. Do not poke TF (fixture already "
    "0xFF). Do not poke ADDR_ARROWS=2 (fixture already 2). Do not walk "
    "Compass, Map-Patra, or Red Ring. One gate; 12 Keese optional "
    "(Survival refill, no sword). OccupancyWalker not used."
)


class ProbeStop(RuntimeError):
    """Expected fail-closed halt with a reportable reason."""


def _save_fixture(
    raw_env: Any,
    *,
    fixture_name: str,
    dest_eb: str | None,
    leftover: dict[str, Any],
) -> dict[str, Any]:
    from retro_harness.env import save_state, state_path
    from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance

    after = read_snapshot(raw_env.get_ram())
    if after.mode != PLAY_MODE or after.level != LEVEL9:
        raise ProbeStop("save_fixture_pin_mismatch")
    path = save_state(raw_env, GAME_DIR, GAME, fixture_name)
    source_path = state_path(GAME_DIR, GAME, SOURCE_STATE)
    result = {
        "ok": True,
        "source_state": SOURCE_STATE,
        "fixture_state": fixture_name,
        "state": compact_snapshot(after),
        "dest_eb": dest_eb,
        "leftover": leftover,
        "fixture_writes": [],
    }
    write_state_provenance(
        path,
        source_state_path=source_path if source_path.exists() else None,
        request={
            "bead": "rr-sz8.6",
            "phase": "level9_prefix_0x76_north_west",
            "track": "recon_fixture",
            "route_eligible": False,
            "fixture_only": True,
            "natural_entry": False,
            "fixture_writes": [],
            "notes": [
                "ZD §10.2 cut leftover, dest $EB live, not 10.3, not Red Ring 0x07",
            ],
        },
        selected_trial=result,
        natural_entry=False,
    )
    return {
        "state_path": str(path),
        "provenance": str(path.with_suffix(".provenance.json")),
    }


def _inventory(env: Any) -> dict[str, int]:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    u = lambda addr: int(read_u8(ram, addr))
    return {
        "sword": u(ADDR_SWORD),
        "bombs": u(ADDR_BOMBS),
        "bow": u(ADDR_BOW),
        "arrows": u(ADDR_ARROWS),
        "keys": u(ADDR_KEYS),
        "rupees": u(ADDR_RUPEES),
        "magic_key": u(ADDR_MAGIC_KEY),
        "selected_item": u(ADDR_SELECTED_ITEM),
        "triforce": u(ADDR_TRIFORCE),
        "ring": u(ADDR_RING),
        "compass": u(ADDR_COMPASS),
        "map": u(ADDR_MAP),
        "health": int(snap.health),
        "heart_containers": int(snap.heart_containers),
    }


def _typed(snap: ZeldaSnapshot, *, live: bool = False) -> list:
    return [
        obj
        for obj in snap.objects
        if 1 <= obj.slot <= 12
        and obj.type_id not in (0, 0xFF)
        and (not live or obj.hp > 0)
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


def _glance(env: Any) -> dict[str, Any]:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    result = leftover_from_snapshot(snap)
    result.update(
        level=int(snap.level),
        screen_hex=f"0x{snap.screen:02X}",
        next_screen_hex=f"0x{snap.next_screen:02X}",
        xy=[int(snap.link_x), int(snap.link_y)],
        tile=int(snap.colliding_tile),
        facing=int(snap.facing),
        transitioning=bool(snap.transitioning),
        room_item_hex=f"0x{snap.room_item_id:02X}",
        cur_opened_doors=int(snap.cur_opened_doors),
        open_doorway_mask=int(snap.open_doorway_mask),
        inventory=_inventory(env),
        live_objects=[_obj_row(obj) for obj in _typed(snap, live=True)],
        typed_objects=[_obj_row(obj) for obj in _typed(snap)],
    )
    return result


def _pin_ok(start: dict[str, Any]) -> bool:
    inv = start["inventory"]
    x, y = start["xy"]
    return (
        start["level"] == LEVEL9
        and start["mode"] == PLAY_MODE
        and int(start["screen"]) == int(NORTH_ORIGIN)
        and not start["transitioning"]
        and abs(int(x) - 120) <= POSE_TOL
        and abs(int(y) - 205) <= POSE_TOL
        and inv["triforce"] == FULL_TRIFORCE
        and inv["magic_key"] == 1
    )


def _dest_is_play_room(settled: ZeldaSnapshot) -> bool:
    return (
        settled.level == LEVEL9
        and settled.mode == PLAY_MODE
        and not settled.transitioning
        and int(settled.screen) != RED_RING
    )


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
        live = _typed(snap, live=True)
        self.samples.append({
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
            "live": len(live),
            "live_types": [f"0x{obj.type_id:02X}" for obj in live],
        })

    def save_shot(self, label: str) -> Path:
        snap = read_snapshot(self.env.get_ram())
        path = RECORDINGS_DIR / (
            f"l9_76_north_{self.tag}_{label}_f{self.frame}_"
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
            self.last_key, self.stuck = key, 0
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


def _drive(env: ObservedEnv, assist: Any, ctl: Any, total: list[int], phase: str) -> str:
    last_reason = ""
    for _ in range(ctl.max_frames):
        snap = read_snapshot(env.get_ram())
        if snap.mode == DEATH_MODE:
            raise ProbeStop("death")
        if snap.level != LEVEL9 and snap.mode == PLAY_MODE:
            raise ProbeStop("left_level9")
        act = ctl.step(snap)
        last_reason = str(act.reason)
        env.set_reason(phase, act.reason)
        env.step(act.action)
        total[0] += 1
        assist.apply_env(env, frame=total[0])
        if ctl.success or ctl.failed:
            break
    return last_reason


def _grade_north(settled: ZeldaSnapshot, arrival: dict[str, Any], start: dict[str, Any], assist: Any) -> dict[str, Any]:
    dest_int = int(settled.screen)
    dest_eb = f"0x{dest_int:02X}"
    inv0, inv1 = start["inventory"], arrival["inventory"]
    deaths = int(assist.telemetry.deaths)
    prog = int(assist.telemetry.progression_writes)
    cap = int(assist.telemetry.capacity_writes)
    misses = [
        msg
        for cond, msg in (
            (settled.level != LEVEL9, f"left L9 -> L{settled.level}"),
            (settled.mode != PLAY_MODE, f"mode {settled.mode} != PLAY 5"),
            (settled.mode == PASSAGE_MODE, f"cellar dest {dest_eb}"),
            (dest_int == RED_RING, "dest 0x07 Red Ring (fail closed)"),
            (dest_int == NORTH_ORIGIN, "still in origin 0x76"),
            (
                not is_north_neighbor(NORTH_ORIGIN, dest_int),
                f"dest {dest_eb} is not north neighbor of 0x76",
            ),
            (inv1["triforce"] != FULL_TRIFORCE, f"triforce {inv1['triforce']:#x}"),
            (inv1["magic_key"] != 1, f"magic_key {inv0['magic_key']}->{inv1['magic_key']}"),
            (deaths, f"deaths {deaths}"),
            (prog, f"progression_writes {prog}"),
            (cap, f"capacity_writes {cap}"),
        )
        if cond
    ]
    return {
        "pass": not misses,
        "misses": misses,
        "dest_eb": dest_eb,
        "dest_eb_int": dest_int,
        "dest_hyp": f"0x{NORTH_DEST_HYP:02X}",
        "north_neighbor": is_north_neighbor(NORTH_ORIGIN, dest_int),
        "settled_xy": [int(settled.link_x), int(settled.link_y)],
        "settled_mode": int(settled.mode),
        "settled_doors": int(settled.cur_opened_doors),
        "keys": [inv0["keys"], inv1["keys"]],
        "bombs": [inv0["bombs"], inv1["bombs"]],
        "arrows": [inv0["arrows"], inv1["arrows"]],
        "ring": [inv0["ring"], inv1["ring"]],
        "triforce": inv1["triforce"],
        "deaths": deaths,
        "progression_writes": prog,
        "capacity_writes": cap,
        "tf_arrow_poke": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_state=SOURCE_STATE, default_tag="20260904_P1")
    parser.add_argument("--no-video", action="store_true", help="JSON + PNG only")
    parser.add_argument(
        "--glance",
        action="store_true",
        help="Dump leftover + start PNG; do not walk",
    )
    parser.add_argument(
        "--also-west",
        action="store_true",
        help="After a north dest of 0x66, take one LEFT gate (hyp 0x65)",
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
        "trial": "0x76_north_gate",
        "from_state": args.from_state,
        "fixture_only": True,
        "natural_entry": False,
        "route_eligible": False,
        "infinite_life": True,
        "north_origin": int(NORTH_ORIGIN),
        "north_door": list(NORTH_DOOR),
        "dest_hyp": f"0x{NORTH_DEST_HYP:02X}",
        "prediction": {
            "written_before_run": True,
            "claim": RAM_CLAIM_NORTH,
            "contingency": (
                "Dest not a north neighbor of 0x76 is a miss. Dest 0x07 is a "
                "miss (halt, do not enter Red Ring). Cellar/passage is a miss. "
                "Still in 0x76 after timeout is a miss. Do not poke TF or "
                "ADDR_ARROWS. Do not walk Compass / Map-Patra / 0x07. "
                "OccupancyWalker not used."
            ),
            "do_not_assume_dest": [NORTH_DEST_HYP],
        },
        "runtime_controller_writes": {
            "room": 0, "door": 0, "position": 0, "inventory": 0,
            "triforce": 0, "arrows": 0, "capacity": 0,
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
        if not _pin_ok(start):
            raise ProbeStop("pin_mismatch")

        assist.apply_env(env, frame=0)
        if args.glance:
            payload["success"] = True
            payload["glance_only"] = True
            payload["final"] = start
            payload["leftover"] = {
                "xy": start["xy"],
                "screen": start["screen_hex"],
                "mode": start["mode"],
                "doors": start["cur_opened_doors"],
                "inventory": start["inventory"],
            }
            raise ProbeStop("glance_only")

        ctl = make_north_76_controller(dest=None)
        last_reason = _drive(env, assist, ctl, total, "north_gate")
        payload["controller"] = ctl.report()
        payload["done_reason"] = getattr(ctl, "done_reason", None)

        if ctl.failed or not ctl.success:
            payload["failed"] = (
                (ctl.notes[-1] if ctl.notes else "")
                or last_reason
                or "north_gate_failed"
            )
            payload["success"] = False
            payload["final"] = _glance(env)
            env.save_shot("final_halt")
        else:
            env.set_reason("arrived", "north_arrived")
            arrival = _glance(env)
            settled = read_snapshot(env.get_ram())
            env.save_shot("arrived")
            env.set_reason("final_census", "idle_census")
            for _ in range(CENSUS_IDLE):
                env.step(nes_idle_action())
                total[0] += 1
                assist.apply_env(env, frame=total[0])
            final = _glance(env)
            env.save_shot("final")
            grade = _grade_north(settled, arrival, start, assist)
            payload["north_gate_grade"] = grade
            payload["arrival"] = arrival
            payload["final"] = final
            payload["success"] = bool(grade["pass"])
            if not grade["pass"]:
                payload["failed"] = "north_gate_grade_miss"

            if (
                args.also_west
                and payload["success"]
                and int(settled.screen) == WEST_ORIGIN
            ):
                west_start = _glance(env)
                payload["west_start"] = west_start
                payload["west_prediction"] = {
                    "written_before_run": True,
                    "claim": (
                        f"From play 0x66 leftover {west_start['xy']}, y-align "
                        "to 141 then hold LEFT. First settled play $EB is RAM "
                        "(hyp 0x65, the -1 west neighbor). Fail if dest is "
                        "not a west neighbor. Fail dest 0x07. One gate."
                    ),
                }
                west = make_west_66_controller(dest=None)
                last_reason = _drive(env, assist, west, total, "west_gate")
                payload["west_controller"] = west.report()
                if west.failed or not west.success:
                    payload["success"] = False
                    payload["failed"] = (
                        (west.notes[-1] if west.notes else "")
                        or last_reason
                        or "west_gate_failed"
                    )
                    payload["final"] = _glance(env)
                    env.save_shot("west_final_halt")
                else:
                    west_arr = _glance(env)
                    west_set = read_snapshot(env.get_ram())
                    env.save_shot("west_arrived")
                    dest_int = int(west_set.screen)
                    dest_eb = f"0x{dest_int:02X}"
                    w_miss = [
                        msg
                        for cond, msg in (
                            (
                                not is_west_neighbor(WEST_ORIGIN, dest_int),
                                f"dest {dest_eb} is not west neighbor of 0x66",
                            ),
                            (dest_int == RED_RING, "dest 0x07 Red Ring"),
                            (dest_int == WEST_ORIGIN, "still in 0x66"),
                        )
                        if cond
                    ]
                    payload["west_gate_grade"] = {
                        "pass": not w_miss,
                        "misses": w_miss,
                        "dest_eb": dest_eb,
                        "dest_hyp": f"0x{WEST_DEST_HYP:02X}",
                        "settled_xy": [
                            int(west_set.link_x),
                            int(west_set.link_y),
                        ],
                    }
                    payload["final"] = west_arr
                    env.save_shot("final")
                    if w_miss:
                        payload["success"] = False
                        payload["failed"] = "west_gate_grade_miss"
                    elif args.save_fixture and payload["success"]:
                        payload["saved_fixture"] = _save_fixture(
                            raw_env,
                            fixture_name=args.save_fixture,
                            dest_eb=dest_eb,
                            leftover=west_arr,
                        )
            elif args.save_fixture and payload["success"]:
                payload["saved_fixture"] = _save_fixture(
                    raw_env,
                    fixture_name=args.save_fixture,
                    dest_eb=grade.get("dest_eb"),
                    leftover=payload.get("final") or {},
                )
    except ProbeStop as exc:
        if str(exc) == "glance_only":
            payload["success"] = True
            payload["failed"] = None
        else:
            payload["success"] = False
            payload.setdefault("failed", str(exc))
        if env is not None and "final" not in payload:
            payload["final"] = _glance(env)
            env._sample(read_snapshot(env.get_ram()), f"halt:{exc}")
            env.save_shot("final_halt")
    finally:
        if env is not None:
            tel = assist.telemetry
            payload["frames"] = int(env.frame)
            payload["samples"] = env.samples[-96:]
            payload["screenshots"] = env.screenshots
            payload["assist"] = assist.report()
            payload["deaths"] = int(tel.deaths)
            payload["progression_writes"] = int(tel.progression_writes)
            payload["capacity_writes"] = int(tel.capacity_writes)
            grade = payload.get("north_gate_grade", {})
            payload["runtime_integrity"] = {
                "deaths": int(tel.deaths),
                "progression_writes": int(tel.progression_writes),
                "capacity_writes": int(tel.capacity_writes),
                "direct_ram_writes": 0,
                "triforce_poke": False,
                "arrow_poke": False,
                "door_poke": False,
                "keys_delta": None
                if not grade
                else int(grade.get("keys", [0, 0])[0])
                - int(grade.get("keys", [0, 0])[1]),
                "bombs_delta": None
                if not grade
                else int(grade.get("bombs", [0, 0])[0])
                - int(grade.get("bombs", [0, 0])[1]),
            }
        report = write_report("l9_76_north_fixture", payload, tag=args.tag)
        raw_env.close()

    grade = payload.get("north_gate_grade") or {}
    final = payload.get("final") or {}
    print(f"report={report}")
    print(f"success={payload.get('success')} failed={payload.get('failed')}")
    print(
        f"dest_screen={grade.get('dest_eb') or final.get('screen_hex')} "
        f"dest_xy={grade.get('settled_xy') or final.get('xy')}"
    )
    print(f"grade={grade}")
    print(f"assist_deaths={payload.get('deaths')}")
    print(f"glance_xy={(payload.get('start') or {}).get('xy')}")
    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
