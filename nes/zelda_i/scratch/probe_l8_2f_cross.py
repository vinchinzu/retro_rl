"""Cellar-cross from settled 0x2F east ladder. One gate.

rr-6o7.2. Pin ``Level8Interior2FCellarReconFixture`` (settled mode-9
cellar 0x2F leftover, not the unloaded first frame). DOWN east, floor
LEFT, UP west. Never UP on the east source ladder.

RAM CLAIM (written before the first live trial):

  From settled mode-9 cellar 0x2F leftover (192,93) east/source ladder,
  DOWN to floor y=189, LEFT to west x=48, UP west ladder. Never UP on
  the east source ladder (returns to play 0x3F). First settled play $EB
  is RAM (hyp 0x4C, NOT source 0x3F, NOT Gleeok 0x3C). Keys 8->8 bombs
  6->6 MK 1 TF 0x7F. One gate.

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/probe_l8_2f_cross.py \\
        --from-state Level8Interior2FCellarReconFixture \\
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
from zelda_i.level8.passage import (
    RAM_CLAIM,
    SPAWN_XY,
    make_passage_2f_controller,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOMBS,
    ADDR_BOW,
    ADDR_CANDLE,
    ADDR_KEYS,
    ADDR_MAGIC_KEY,
    ADDR_MAP,
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
LOAD_IDLE = 60
GLEEOK_ROOM = 0x3C
SOURCE_ROOM = 0x3F
CELLAR_ROOM = 0x2F
SOURCE_STATE = "Level8Interior2FCellarReconFixture"


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
        "map": u(ADDR_MAP),
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
    ox, oy = SPAWN_XY
    return (
        start["level"] == LEVEL8 and int(start["mode"]) == PASSAGE_MODE
        and int(start["screen"]) == CELLAR_ROOM
        and abs(int(x) - int(ox)) <= POSE_TOL
        and abs(int(y) - int(oy)) <= POSE_TOL
        and inv["magic_key"] == 1 and inv["keys"] == 8
        and inv["bombs"] == 6 and inv["triforce"] == POST_L7_TRIFORCE
    )


def _gate_spend_ok(before: int, after: int) -> bool:
    return after == before or after == before - 1


def _dest_is_play_room(settled: ZeldaSnapshot) -> bool:
    return (
        settled.level == LEVEL8
        and settled.mode == PLAY_MODE
        and not settled.transitioning
        and int(settled.screen) != GLEEOK_ROOM
        and int(settled.screen) != SOURCE_ROOM
        and int(settled.screen) != CELLAR_ROOM
    )


def _parse_dest(raw: str | None) -> int | None:
    if raw is None or raw == "" or raw.lower() == "none":
        return None
    return int(raw, 0)


@dataclass
class ObservedEnv:
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
            "map": int(read_u8(self.env.get_ram(), ADDR_MAP)),
            "room_item": int(snap.room_item_id),
            "doors": int(snap.cur_opened_doors),
            "live": len(live),
            "live_types": [f"0x{obj.type_id:02X}" for obj in live],
        })

    def save_shot(self, label: str) -> Path:
        snap = read_snapshot(self.env.get_ram())
        path = RECORDINGS_DIR / (
            f"l8_2f_cross_{self.tag}_{label}_f{self.frame}_"
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
            "phase": "level8_interior_0x2f_cellar_cross",
            "track": "recon_fixture",
            "route_eligible": False,
            "fixture_only": True,
            "natural_entry": False,
            "fixture_writes": [],
            "notes": [
                "cellar-cross from settled 0x2F east ladder, dest $EB live,"
                " do not assume 0x4C/0x3C, one gate, not on L8_THROUGH,"
                " never UP on the source ladder",
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
        default_tag="20260904_P1",
    )
    parser.add_argument("--save-fixture", default=None)
    parser.add_argument("--dest", default=None)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if not args.infinite_life:
        raise SystemExit("this fixture route probe requires --infinite-life")

    dest = _parse_dest(args.dest)
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    assist = make_assist(True)
    raw_env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    payload: dict[str, Any] = {
        "trial": "0x2f_cellar_cross",
        "from_state": args.from_state,
        "fixture_only": True, "natural_entry": False, "route_eligible": False,
        "infinite_life": True,
        "spawn_xy": list(SPAWN_XY),
        "dest_arg": dest,
        "prediction": {
            "written_before_run": True,
            "claim": RAM_CLAIM,
            "contingency": (
                "UP on the east ladder returns to play 0x3F (miss)."
                " Dest 0x3C is a miss (halt, do not fight). Pit tile 250"
                " LEFT at y=141 is a miss. OccupancyWalker not used."
            ),
            "attempt_history": [
                {
                    "tag": "offline",
                    "result": "occupancy_skipped",
                    "detail": (
                        "L7 0x7B analog: DOWN east x=192, floor LEFT to"
                        " x=48, UP west. Never source UP."
                    ),
                },
            ],
            "do_not_assume_dest": [0x4C, 0x3C, 0x3F],
        },
        "runtime_controller_writes": {
            "room": 0, "door": 0, "position": 0, "inventory": 0,
            "triforce": 0, "magic_key": 0, "capacity": 0,
        },
    }
    env: ObservedEnv | None = None
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
        env.set_reason("cellar_load", "idle_load")
        for _ in range(LOAD_IDLE):
            env.step(nes_idle_action())
            assist.apply_env(env, frame=env.frame)
        payload["after_load"] = _glance(env)
        env.save_shot("after_load")

        inv0 = start["inventory"]
        keys_in, bombs_in = int(inv0["keys"]), int(inv0["bombs"])
        mk_in = int(inv0["magic_key"])

        ctl = make_passage_2f_controller(dest=dest)
        for _ in range(ctl.max_frames):
            snap = read_snapshot(env.get_ram())
            if snap.mode == DEATH_MODE:
                raise ProbeStop("death")
            if snap.level != LEVEL8 and snap.mode == PLAY_MODE:
                raise ProbeStop("left_level8")
            act = ctl.step(snap)
            last_reason = str(act.reason)
            env.set_reason("cellar_cross", act.reason)
            env.step(act.action)
            assist.apply_env(env, frame=env.frame)
            if ctl.success or ctl.failed:
                break

        payload["controller"] = ctl.report()
        payload["done_reason"] = getattr(ctl, "done_reason", None)

        if ctl.failed or not ctl.success:
            payload["failed"] = (
                (ctl.notes[-1] if ctl.notes else "")
                or last_reason or "cellar_cross_failed"
            )
            payload["success"] = False
            payload["final"] = _glance(env)
            env.save_shot("final_halt")
        else:
            env.set_reason("arrived", "cross_arrived")
            arrival = _glance(env)
            settled = read_snapshot(env.get_ram())
            env.save_shot("arrived")
            saved = None
            dest_int = int(settled.screen)
            dest_eb = f"0x{dest_int:02X}"
            keys_out, bombs_out = int(settled.keys), int(settled.bombs)
            mk_out = int(arrival["inventory"]["magic_key"])
            tf_out = int(arrival["inventory"]["triforce"])
            if args.save_fixture and _dest_is_play_room(settled):
                saved = _save_fixture(
                    raw_env, fixture_name=args.save_fixture, census=arrival,
                    dest_eb=dest_eb, magic_key=mk_out, keys=keys_out,
                    bombs=bombs_out,
                )
            env.set_reason("final_census", "idle_census")
            for _ in range(CENSUS_IDLE):
                env.step(nes_idle_action())
                assist.apply_env(env, frame=env.frame)
            final = _glance(env)
            env.save_shot("final")
            deaths = int(assist.telemetry.deaths)
            prog = int(assist.telemetry.progression_writes)
            cap = int(assist.telemetry.capacity_writes)
            misses = [
                msg for cond, msg in (
                    (settled.level != LEVEL8, f"left L8 -> L{settled.level}"),
                    (settled.mode != PLAY_MODE, f"mode {settled.mode} != PLAY 5"),
                    (dest_int == GLEEOK_ROOM, "dest 0x3C Gleeok (fail closed)"),
                    (dest_int == SOURCE_ROOM, "returned source 0x3F"),
                    (dest_int == CELLAR_ROOM, "still cellar 0x2F"),
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
            payload["cross_grade"] = {
                "pass": not misses, "misses": misses,
                "magic_key": [mk_in, mk_out], "keys": [keys_in, keys_out],
                "bombs": [bombs_in, bombs_out], "triforce": tf_out,
                "dest_eb": dest_eb, "dest_eb_int": dest_int,
                "dest_mode": int(settled.mode),
                "settled_xy": [int(settled.link_x), int(settled.link_y)],
                "settled_tile": int(settled.colliding_tile),
                "settled_doors": int(settled.cur_opened_doors),
                "deaths": deaths, "progression_writes": prog,
                "capacity_writes": cap, "mk_tf_door_poke": False,
                "position_writes": 0,
            }
            payload["arrival"] = arrival
            payload["final"] = final
            payload["success"] = not misses and dest_int != GLEEOK_ROOM
            if misses or dest_int == GLEEOK_ROOM:
                payload["failed"] = (
                    "dest_0x3c_fail_closed" if dest_int == GLEEOK_ROOM
                    else "cross_grade_miss"
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
            grade = payload.get("cross_grade", {})
            keys = grade.get("keys") or [0, 0]
            bombs = grade.get("bombs") or [0, 0]
            payload["runtime_integrity"] = {
                "deaths": int(tel.deaths),
                "progression_writes": int(tel.progression_writes),
                "capacity_writes": int(tel.capacity_writes),
                "direct_ram_writes": 0,
                "magic_key_poke": False, "triforce_poke": False, "door_poke": False,
                "position_writes": 0,
                "keys_delta": None if not grade else int(keys[0]) - int(keys[1]),
                "bombs_delta": None if not grade else int(bombs[0]) - int(bombs[1]),
            }
        report = write_report("l8_2f_cross_fixture", payload, tag=args.tag)
        raw_env.close()

    grade = payload.get("cross_grade") or {}
    final = payload.get("final") or {}
    print(f"report={report}")
    print(f"success={payload.get('success')} failed={payload.get('failed')}")
    print(f"dest_screen={grade.get('dest_eb') or final.get('screen_hex')} "
          f"dest_mode={grade.get('dest_mode') or final.get('mode')} "
          f"dest_xy={grade.get('settled_xy') or final.get('xy')}")
    print(f"grade={grade}")
    print(f"assist_deaths={payload.get('deaths')}")
    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
