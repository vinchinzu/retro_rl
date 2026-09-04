"""South gate from cleared 0x1E leftover (208,141).

rr-6o7.2. Pin ``Level8Interior1EWestReconFixture`` (play 0x1E leftover).
Drives ``make_south_1e_controller(dest=None)``. No sword, no Gleeok fight.
Do not assume dest $EB is 0x2E or 0x3C. One gate only. Do not chain DOWN.

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/probe_l8_1e_south.py \\
        --from-state Level8Interior1EWestReconFixture \\
        --tag 20260904_H1 --infinite-life
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
from zelda_i.level8.path import (
    SOUTH_DOOR,
    SOUTH_ORIGIN,
    SOUTH_ORIGIN_POSE,
    make_south_1e_controller,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOMBS,
    ADDR_BOW,
    ADDR_CANDLE,
    ADDR_KEYS,
    ADDR_MAGIC_KEY,
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
POST_L7_TRIFORCE = 0x7F
POSE_TOL = 8
DEATH_MODE = 17
CENSUS_IDLE = 60
GLEEOK_ROOM = 0x3C
CELLAR_ROOM = 0x0F
SOURCE_STATE = "Level8Interior1EWestReconFixture"


class ProbeStop(RuntimeError):
    """Expected fail-closed halt with a reportable reason."""


def _inventory(env: Any) -> dict[str, int]:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    u = lambda addr: int(read_u8(ram, addr))
    return {
        "sword": u(ADDR_SWORD), "bombs": u(ADDR_BOMBS), "bow": u(ADDR_BOW),
        "arrows": u(ADDR_ARROWS), "candle": u(ADDR_CANDLE), "keys": u(ADDR_KEYS),
        "rupees": u(ADDR_RUPEES), "magic_key": u(ADDR_MAGIC_KEY),
        "selected_item": u(ADDR_SELECTED_ITEM), "triforce": u(ADDR_TRIFORCE),
        "health": int(snap.health), "heart_containers": int(snap.heart_containers),
    }


def _typed(snap: ZeldaSnapshot, *, live: bool = False) -> list:
    return [
        obj for obj in snap.objects
        if 1 <= obj.slot <= 12 and obj.type_id not in (0, 0xFF)
        and (not live or obj.hp > 0)
    ]


def _obj_row(obj: Any) -> dict[str, Any]:
    return {
        "slot": int(obj.slot), "type": int(obj.type_id),
        "type_hex": f"0x{obj.type_id:02X}", "type_name": object_name(obj.type_id),
        "xy": [int(obj.x), int(obj.y)], "hp": int(obj.hp),
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
    inv, (x, y) = start["inventory"], start["xy"]
    ox, oy = SOUTH_ORIGIN_POSE
    return (
        start["level"] == LEVEL8 and start["mode"] == PLAY_MODE
        and int(start["screen"]) == int(SOUTH_ORIGIN)
        and abs(int(x) - int(ox)) <= POSE_TOL
        and abs(int(y) - int(oy)) <= POSE_TOL
        and inv["magic_key"] == 1 and inv["keys"] == 8
        and inv["bombs"] == 6 and inv["triforce"] == POST_L7_TRIFORCE
    )


def _gate_spend_ok(before: int, after: int) -> bool:
    """Allow unchanged or a single natural gate spend (key/bomb)."""
    return after == before or after == before - 1


def _dest_is_play_room(settled: ZeldaSnapshot) -> bool:
    return (
        settled.level == LEVEL8
        and settled.mode == PLAY_MODE
        and not settled.transitioning
        and int(settled.screen) != GLEEOK_ROOM
        and int(settled.screen) != CELLAR_ROOM
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
            "frame": int(self.frame), "phase": self.phase,
            "reason": reason or self.reason, "level": int(snap.level),
            "screen": f"0x{snap.screen:02X}", "mode": int(snap.mode),
            "xy": [int(snap.link_x), int(snap.link_y)],
            "tile": int(snap.colliding_tile), "keys": int(snap.keys),
            "bombs": int(snap.bombs),
            "magic_key": int(read_u8(self.env.get_ram(), ADDR_MAGIC_KEY)),
            "doors": int(snap.cur_opened_doors),
            "live": len(live),
            "live_types": [f"0x{obj.type_id:02X}" for obj in live],
        })

    def save_shot(self, label: str) -> Path:
        snap = read_snapshot(self.env.get_ram())
        path = RECORDINGS_DIR / (
            f"l8_1e_south_{self.tag}_{label}_f{self.frame}_"
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


def _save_fixture(
    raw_env: Any, *, fixture_name: str, census: dict[str, Any],
    dest_eb: str | None, magic_key: int, keys: int, bombs: int,
) -> dict[str, Any]:
    from retro_harness.env import save_state, state_path
    from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance

    after = read_snapshot(raw_env.get_ram())
    mk = int(read_u8(raw_env.get_ram(), ADDR_MAGIC_KEY))
    if not _dest_is_play_room(after) or mk < 1:
        raise ProbeStop("save_fixture_pin_mismatch")
    path = save_state(raw_env, GAME_DIR, GAME, fixture_name)
    source_path = state_path(GAME_DIR, GAME, SOURCE_STATE)
    result = {
        "ok": True, "source_state": SOURCE_STATE,
        "fixture_state": fixture_name, "state": compact_snapshot(after),
        "census": census, "dest_eb": dest_eb, "magic_key": magic_key,
        "keys": keys, "bombs": bombs, "fixture_writes": [],
    }
    write_state_provenance(
        path,
        source_state_path=source_path if source_path.exists() else None,
        request={
            "bead": "rr-6o7.2",
            "phase": "level8_interior_0x1e_south_gate",
            "track": "recon_fixture",
            "route_eligible": False,
            "fixture_only": True,
            "natural_entry": False,
            "fixture_writes": [],
            "notes": [
                "south door from play 0x1E leftover, dest $EB live,"
                " do not assume 0x2E/0x3C, one gate, not on L8_THROUGH",
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
        default_state=SOURCE_STATE,
        default_tag="20260904_H1",
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
        "trial": "0x1e_south_gate",
        "from_state": args.from_state,
        "fixture_only": True, "natural_entry": False, "route_eligible": False,
        "infinite_life": True,
        "south_origin": int(SOUTH_ORIGIN),
        "south_origin_pose": list(SOUTH_ORIGIN_POSE),
        "south_door": list(SOUTH_DOOR),
        "prediction": {
            "written_before_run": True,
            "claim": (
                "From play 0x1E leftover (208,141), cardinal LEFT along y=141"
                " until x≈120, then DOWN to (120,205) and hold DOWN. First"
                " settled play $EB is RAM (hyp 0x2E, NOT Gleeok 0x3C, NOT"
                " cellar 0x0F). Keys 8->8 bombs 6->6 MK 1 TF 0x7F. No sword."
                " One gate; do not continue DOWN. Do not start Gleeok."
                " OccupancyWalker not used live: empty-grid BFS first dir is"
                " DOWN along x=208 into the SE statue, and 1px-grade"
                " false-misses 2px dungeon steps (west G1)."
            ),
            "contingency": (
                "Cardinal LEFT stuck at a statue or west wall is a miss."
                " Stairs warp to 0x0F is a miss. Dest 0x3C is a miss"
                " (halt, do not fight). Do not UP into the north door."
                " Do not chain a second DOWN into 0x3E."
            ),
            "attempt_history": [
                {
                    "tag": "offline",
                    "result": "occupancy_skipped",
                    "detail": (
                        "OccupancyGrid.shortest_path((208,141),(120,205))"
                        " len=153 first_dir=DOWN along x=208; SE statue is"
                        " on that wall. OccupancyWalker 1px-grade false-missed"
                        " 2px dungeon steps on the west hop. Cardinal x-align."
                    ),
                },
            ],
            "do_not_assume_dest": [0x2E, 0x3C],
        },
        "runtime_controller_writes": {
            "room": 0, "door": 0, "position": 0, "inventory": 0,
            "triforce": 0, "magic_key": 0, "capacity": 0,
        },
    }
    env: ObservedEnv | None = None
    total = [0]
    last_reason = ""
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
        inv0 = start["inventory"]
        keys_in, bombs_in = int(inv0["keys"]), int(inv0["bombs"])
        mk_in = int(inv0["magic_key"])

        ctl = make_south_1e_controller(dest=None)
        for _ in range(ctl.max_frames):
            snap = read_snapshot(env.get_ram())
            if snap.mode == DEATH_MODE:
                raise ProbeStop("death")
            if snap.level != LEVEL8 and snap.mode == PLAY_MODE:
                raise ProbeStop("left_level8")
            act = ctl.step(snap)
            last_reason = str(act.reason)
            env.set_reason("south_gate", act.reason)
            env.step(act.action)
            total[0] += 1
            assist.apply_env(env, frame=total[0])
            if ctl.success or ctl.failed:
                break

        payload["controller"] = ctl.report()
        payload["done_reason"] = getattr(ctl, "done_reason", None)

        if ctl.failed or not ctl.success:
            payload["failed"] = (
                (ctl.notes[-1] if ctl.notes else "")
                or last_reason or "south_gate_failed"
            )
            payload["success"] = False
            payload["final"] = _glance(env)
            env.save_shot("final_halt")
        else:
            env.set_reason("arrived", "south_arrived")
            arrival = _glance(env)
            settled = read_snapshot(env.get_ram())
            env.save_shot("arrived")
            saved = None
            dest_int = int(settled.screen)
            dest_eb = f"0x{dest_int:02X}"
            keys_out, bombs_out = int(settled.keys), int(settled.bombs)
            mk_out = int(arrival["inventory"]["magic_key"])
            tf_out = int(arrival["inventory"]["triforce"])
            if (
                args.save_fixture
                and _dest_is_play_room(settled)
                and dest_int != GLEEOK_ROOM
            ):
                saved = _save_fixture(
                    raw_env, fixture_name=args.save_fixture, census=arrival,
                    dest_eb=dest_eb, magic_key=mk_out, keys=keys_out,
                    bombs=bombs_out,
                )
            env.set_reason("final_census", "idle_census")
            for _ in range(CENSUS_IDLE):
                env.step(nes_idle_action())
                total[0] += 1
                assist.apply_env(env, frame=total[0])
            final = _glance(env)
            env.save_shot("final")
            deaths = int(assist.telemetry.deaths)
            prog = int(assist.telemetry.progression_writes)
            cap = int(assist.telemetry.capacity_writes)
            misses = [
                msg for cond, msg in (
                    (settled.level != LEVEL8, f"left L8 -> L{settled.level}"),
                    (settled.mode != PLAY_MODE, f"mode {settled.mode} != PLAY 5"),
                    (
                        settled.mode == PASSAGE_MODE
                        or dest_int == CELLAR_ROOM,
                        f"cellar/passage dest {dest_eb} mode {settled.mode}",
                    ),
                    (dest_int == GLEEOK_ROOM, "dest 0x3C Gleeok (fail closed)"),
                    (dest_int == SOUTH_ORIGIN, "still in origin 0x1E"),
                    (mk_out != 1, f"magic_key {mk_in}->{mk_out}"),
                    (tf_out != POST_L7_TRIFORCE, f"triforce {tf_out:#x} != 0x7F"),
                    (
                        not _gate_spend_ok(keys_in, keys_out),
                        f"keys changed {keys_in}->{keys_out}",
                    ),
                    (
                        not _gate_spend_ok(bombs_in, bombs_out),
                        f"bombs changed {bombs_in}->{bombs_out}",
                    ),
                    (deaths, f"deaths {deaths}"),
                    (prog, f"progression_writes {prog}"),
                    (cap, f"capacity_writes {cap}"),
                ) if cond
            ]
            payload["south_gate_grade"] = {
                "pass": not misses, "misses": misses,
                "magic_key": [mk_in, mk_out], "keys": [keys_in, keys_out],
                "bombs": [bombs_in, bombs_out], "triforce": tf_out,
                "dest_eb": dest_eb, "dest_eb_int": dest_int,
                "dest_compared_to_0x3c": False,
                "settled_xy": [int(settled.link_x), int(settled.link_y)],
                "settled_mode": int(settled.mode),
                "settled_doors": int(settled.cur_opened_doors),
                "deaths": deaths, "progression_writes": prog,
                "capacity_writes": cap, "mk_tf_door_poke": False,
            }
            payload["arrival"] = arrival
            payload["final"] = final
            payload["success"] = not misses and dest_int != GLEEOK_ROOM
            if misses or dest_int == GLEEOK_ROOM:
                payload["failed"] = (
                    "dest_0x3c_fail_closed" if dest_int == GLEEOK_ROOM
                    else "south_gate_grade_miss"
                )
            elif saved is not None:
                payload["saved_fixture"] = saved
            elif args.save_fixture:
                payload["save_fixture_skipped"] = (
                    f"dest {dest_eb} mode {settled.mode} not a saveable play room"
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
            tel = assist.telemetry
            payload["frames"] = int(env.frame)
            payload["samples"] = env.samples[-96:]
            payload["screenshots"] = env.screenshots
            payload["assist"] = assist.report()
            payload["deaths"] = int(tel.deaths)
            payload["progression_writes"] = int(tel.progression_writes)
            payload["capacity_writes"] = int(tel.capacity_writes)
            grade = payload.get("south_gate_grade", {})
            keys = grade.get("keys") or [0, 0]
            bombs = grade.get("bombs") or [0, 0]
            payload["runtime_integrity"] = {
                "deaths": int(tel.deaths),
                "progression_writes": int(tel.progression_writes),
                "capacity_writes": int(tel.capacity_writes),
                "direct_ram_writes": 0,
                "magic_key_poke": False, "triforce_poke": False, "door_poke": False,
                "keys_delta": None if not grade else int(keys[0]) - int(keys[1]),
                "bombs_delta": None if not grade else int(bombs[0]) - int(bombs[1]),
            }
        report = write_report("l8_1e_south_fixture", payload, tag=args.tag)
        raw_env.close()

    grade = payload.get("south_gate_grade") or {}
    final = payload.get("final") or {}
    print(f"report={report}")
    print(f"success={payload.get('success')} failed={payload.get('failed')}")
    print(f"dest_screen={grade.get('dest_eb') or final.get('screen_hex')} "
          f"dest_xy={grade.get('settled_xy') or final.get('xy')}")
    print(f"grade={grade}")
    print(f"assist_deaths={payload.get('deaths')}")
    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
