"""L7 cellar 0x7B B→A floor-cross. Dest claim play 0x29 (rr-n91a).

Pin ``Level7Interior0DNoseCellarReconFixture`` (already mode-9 cellar 0x7B).
DOWN to floor y=189, LEFT to x=48, UP left ladder. NEVER UP on the
right/source ladder (that is the dead return-only miss to 0x0D).
OccupancyWalker banned. No new position pokes.

RAM claim (written before the first live trial): first settled play $EB
is 0x29 (ROM AttrA). Miss if dest is 0x0D.

If dest 0x29 is live, optionally continue the suffix (bomb-E 0x2A,
Aquamentus, E-shutter 0x2B, idle fanfare). Survival bomb top-up only at
the verified bomb gate. Chapter factories stay fail-closed.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l7_7b_cellar_cross.py \
        --from-state Level7Interior0DNoseCellarReconFixture \
        --tag 20260904_C1 --infinite-life --no-video
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from retro_harness.env import make_env, reset_obs, save_state, state_path
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.bomb_wall import BombWallController
from zelda_i.dungeon.ids import object_name
from zelda_i.dungeon.ops import apply_owned_inventory
from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance
from zelda_i.level1.finish import (
    AQUAMENTUS_MAX_FRAMES,
    AquamentusPhase,
    Level1AquamentusController,
    ROOM_AQUAMENTUS,
)
from zelda_i.level7.cellar import (
    CELLAR_ROOM,
    DEST_ROOM,
    EAST_X,
    PIT_TILE,
    RAM_CLAIM,
    SOURCE_ROOM,
    SPAWN_XY,
    make_nose_cellar_cross_controller,
)
from zelda_i.level7.stairs import AQUAMENTUS_ROM, TRIFORCE_ROM
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_BOW,
    ADDR_CANDLE,
    ADDR_KEYS,
    ADDR_RUPEES,
    ADDR_SELECTED_ITEM,
    ADDR_SWORD,
    ADDR_TRIFORCE,
    ADDR_WHISTLE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)
from zelda_i.runner import add_common_args, make_assist, write_report
from zelda_i.screen_glance import leftover_from_snapshot

LEVEL7 = 7
FROM = "Level7Interior0DNoseCellarReconFixture"
DEATH_MODE = 17
FANFARE_MODE = 18
CENSUS_IDLE = 60
LOAD_IDLE = 400
AQUA_TYPE = 0x3D
FIREBALL_TYPE = 0x55
TF_BIT_L7 = 0x40
BOMB_TOPUP = 8
EAST_BOMB_STAND = (208, 141)
EAST_BOMB_APPROACH = ((96, 189), (208, 189), (208, 141))


class ProbeStop(RuntimeError):
    """Expected fail-closed halt with a reportable reason."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Enum):
        return value.name
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


@dataclass(frozen=True)
class _EastBombWall:
    room: int = DEST_ROOM
    stand: tuple[int, int] = EAST_BOMB_STAND
    face: str = "RIGHT"
    opens_to: int = AQUAMENTUS_ROM


def _inventory(env: Any) -> dict[str, int]:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    u = lambda addr: int(read_u8(ram, addr))
    return {
        "sword": u(ADDR_SWORD),
        "bombs": u(ADDR_BOMBS),
        "bow": u(ADDR_BOW),
        "candle": u(ADDR_CANDLE),
        "keys": u(ADDR_KEYS),
        "rupees": u(ADDR_RUPEES),
        "selected_item": u(ADDR_SELECTED_ITEM),
        "triforce": u(ADDR_TRIFORCE),
        "whistle": u(ADDR_WHISTLE),
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
        room_all_dead=int(snap.room_all_dead),
        candle=int(snap.candle),
        inventory=_inventory(env),
        live_objects=[_obj_row(obj) for obj in _typed(snap, live=True)],
        typed_objects=[_obj_row(obj) for obj in _typed(snap)],
    )
    return result


def _pin_ok(start: dict[str, Any]) -> bool:
    inv, (x, y) = start["inventory"], start["xy"]
    return (
        start["level"] == LEVEL7
        and start["mode"] == 9
        and int(start["screen"]) == int(CELLAR_ROOM)
        and abs(int(x) - SPAWN_XY[0]) <= 12
        and inv["candle"] == 2
    )


def _alias_aqua(snap: ZeldaSnapshot, live_room: int) -> ZeldaSnapshot:
    if snap.screen == live_room:
        return replace(snap, screen=ROOM_AQUAMENTUS)
    return snap


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
                "candle": int(snap.candle),
                "doors": int(snap.cur_opened_doors),
                "live": len(live),
                "live_types": [f"0x{obj.type_id:02X}" for obj in live],
            }
        )

    def save_shot(self, label: str) -> Any:
        snap = read_snapshot(self.env.get_ram())
        path = RECORDINGS_DIR / (
            f"l7_7b_cellar_cross_{self.tag}_{label}_f{self.frame}_"
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


def _drive(env: ObservedEnv, assist: Any, ctl: Any, *, phase: str, limit: int) -> str:
    last = ""
    for _ in range(limit):
        snap = read_snapshot(env.get_ram())
        if snap.mode == DEATH_MODE:
            raise ProbeStop("death")
        act = ctl.step(snap)
        last = str(act.reason)
        env.set_reason(phase, act.reason)
        env.step(act.action)
        assist.apply_env(env, frame=env.frame)
        if getattr(ctl, "success", False) or getattr(ctl, "failed", False):
            break
        phase = getattr(ctl, "phase", None)
        if getattr(phase, "name", "") in ("FAILED", "DONE"):
            break
    return last


def _save_fixture(
    raw_env: Any,
    *,
    fixture_name: str,
    census: dict[str, Any],
    dest_eb: str,
    note: str,
) -> dict[str, Any]:
    after = read_snapshot(raw_env.get_ram())
    path = save_state(raw_env, GAME_DIR, GAME, fixture_name)
    source_path = state_path(GAME_DIR, GAME, FROM)
    result = {
        "ok": True,
        "source_state": FROM,
        "fixture_state": fixture_name,
        "state": compact_snapshot(after),
        "census": census,
        "dest_eb": dest_eb,
        "fixture_writes": [],
    }
    write_state_provenance(
        path,
        source_state_path=source_path if source_path.exists() else None,
        request={
            "bead": "rr-n91a",
            "phase": "level7_nose_cellar_cross",
            "track": "recon_fixture",
            "route_eligible": False,
            "fixture_only": True,
            "natural_entry": False,
            "fixture_writes": [],
            "notes": [note],
        },
        selected_trial=result,
        natural_entry=False,
    )
    return {
        "state_path": str(path),
        "provenance": str(path.with_suffix(".provenance.json")),
    }


def _run_suffix(env: ObservedEnv, assist: Any, payload: dict[str, Any]) -> None:
    """0x29 bomb-E → 0x2A Aquamentus → 0x2B TF. Halt on first red gate."""
    snap = read_snapshot(env.get_ram())
    if snap.screen != DEST_ROOM or snap.mode != PLAY_MODE:
        payload["suffix"] = {"skipped": "not_on_0x29"}
        return

    inv_before = _inventory(env)
    bombs_before = int(inv_before["bombs"])
    topup = apply_owned_inventory(
        env, bombs=BOMB_TOPUP, select_bomb=True
    )
    payload["inventory_assist"] = {
        "gate": "0x29_east_bomb",
        "bombs_before": bombs_before,
        "topup": topup,
        "max_bombs_written": False,
        "label": "Survival bomb-count top-up at verified 0x29 E-bomb gate. Never max_bombs.",
    }

    wall = BombWallController(
        wall=_EastBombWall(),
        level=LEVEL7,
        approach_waypoints=EAST_BOMB_APPROACH,
        approach_tol=4,
    )
    last = _drive(env, assist, wall, phase="bomb_east_0x29", limit=wall.max_frames)
    wall_phase = getattr(getattr(wall, "phase", None), "name", None)
    payload["bomb_east"] = {
        "success": bool(wall.success),
        "failed": wall_phase == "FAILED",
        "phase": wall_phase,
        "notes": [str(n) for n in wall.notes],
        "frames": int(wall.frames),
        "last_reason": last,
        "glance": _glance(env),
    }
    env.save_shot("after_bomb_east")
    if not wall.success:
        payload["suffix_halt"] = "bomb_east_red"
        return
    payload["saved_fixture_0x2a"] = _save_fixture(
        env.env,
        fixture_name="Level7Interior2AAquamentusReconFixture",
        census=_glance(env),
        dest_eb=f"0x{AQUAMENTUS_ROM:02X}",
        note="0x29 E-bomb dest play 0x2A. route_eligible=false.",
    )

    aqua_glance = _glance(env)
    live_types = [row["type"] for row in aqua_glance["typed_objects"]]
    payload["aquamentus_census"] = {
        "screen": aqua_glance["screen_hex"],
        "xy": aqua_glance["xy"],
        "has_0x3d": AQUA_TYPE in live_types,
        "has_0x55": FIREBALL_TYPE in live_types or any(
            row["type"] == FIREBALL_TYPE for row in aqua_glance["typed_objects"]
        ),
        "objects": aqua_glance["typed_objects"],
    }
    if int(aqua_glance["screen"]) != AQUAMENTUS_ROM:
        payload["suffix_halt"] = "aqua_not_0x2a"
        return

    aqua = Level1AquamentusController(
        phase=AquamentusPhase.ALIGN, tank_hits=True
    )
    last = ""
    for _ in range(AQUAMENTUS_MAX_FRAMES):
        snap = read_snapshot(env.get_ram())
        if snap.mode == DEATH_MODE:
            raise ProbeStop("death")
        aliased = _alias_aqua(snap, AQUAMENTUS_ROM)
        act = aqua.step(aliased)
        last = str(act.reason)
        env.set_reason("aquamentus_0x2a", act.reason)
        env.step(act.action)
        assist.apply_env(env, frame=env.frame)
        if aqua.success or aqua.phase is AquamentusPhase.FAILED:
            break
    payload["aquamentus"] = {
        **aqua.report(),
        "last_reason": last,
        "live_room": f"0x{AQUAMENTUS_ROM:02X}",
        "aliased_l1_room": f"0x{ROOM_AQUAMENTUS:02X}",
        "glance": _glance(env),
    }
    env.save_shot("after_aquamentus")
    if not aqua.success:
        payload["suffix_halt"] = "aquamentus_red"
        return

    hc_before = int(payload["aquamentus"]["initial_containers"] or 0)
    hc_after = int(_glance(env)["inventory"]["heart_containers"])
    payload["heart_container"] = {
        "before": hc_before,
        "after": hc_after,
        "plus_one": hc_after == hc_before + 1,
    }

    last = "push_east_shutter"
    for _ in range(1200):
        snap = read_snapshot(env.get_ram())
        if snap.mode == DEATH_MODE:
            raise ProbeStop("death")
        if (
            snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen == TRIFORCE_ROM
        ):
            last = "arrived_0x2b"
            break
        env.set_reason("east_shutter_0x2a", last)
        env.step(nes_action("RIGHT"))
        assist.apply_env(env, frame=env.frame)
    tf_glance = _glance(env)
    payload["triforce_room"] = {
        "arrived": int(tf_glance["screen"]) == TRIFORCE_ROM,
        "glance": tf_glance,
        "last_reason": last,
    }
    env.save_shot("after_shutter")
    if int(tf_glance["screen"]) != TRIFORCE_ROM:
        payload["suffix_halt"] = "tf_room_red"
        return
    payload["saved_fixture_0x2b"] = _save_fixture(
        env.env,
        fixture_name="Level7Interior2BTriforceReconFixture",
        census=tf_glance,
        dest_eb=f"0x{TRIFORCE_ROM:02X}",
        note="0x2A E-shutter dest play 0x2B TF. route_eligible=false.",
    )

    # Diamond floor: y=141 centre is blocked (C1 shutter leftover (16,141)).
    # RIGHT off the west mouth first (DOWN at x=16 does not move), then
    # south-around like L1 TF: (32,141) -> (32,189) -> (120,189) -> (128,141).
    tf_waypoints = ((32, 141), (32, 189), (120, 189), (128, 141))
    wp_i = 0
    last = "tf_south_around"
    ow = None
    for _ in range(3600):
        snap = read_snapshot(env.get_ram())
        if snap.mode == DEATH_MODE:
            raise ProbeStop("death")
        if snap.triforce & TF_BIT_L7:
            last = "tf_bit_set"
        if snap.mode == FANFARE_MODE:
            last = "fanfare"
            env.set_reason("tf_fanfare", last)
            env.step(nes_idle_action())
            assist.apply_env(env, frame=env.frame)
            continue
        if (
            snap.level == 0
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            ow = _glance(env)
            last = "ow_return"
            break
        if snap.transitioning or snap.mode != PLAY_MODE:
            env.set_reason("tf_fanfare", f"wait_mode_{snap.mode}")
            env.step(nes_idle_action())
            assist.apply_env(env, frame=env.frame)
            continue
        if wp_i < len(tf_waypoints):
            tx, ty = tf_waypoints[wp_i]
            dx, dy = tx - int(snap.link_x), ty - int(snap.link_y)
            if abs(dx) <= 3 and abs(dy) <= 3:
                wp_i += 1
                last = f"tf_wp_{wp_i}"
                env.step(nes_idle_action())
            elif abs(dy) > 3:
                last = "tf_align_y"
                env.step(nes_action("DOWN" if dy > 0 else "UP"))
            else:
                last = "tf_align_x"
                env.step(nes_action("RIGHT" if dx > 0 else "LEFT"))
        else:
            last = "idle_fanfare"
            env.step(nes_idle_action())
        env.set_reason("tf_collect", last)
        assist.apply_env(env, frame=env.frame)
    payload["post_l7_exit"] = {
        "measured": ow is not None,
        "verified": False,
        "leftover": ow,
        "last_reason": last,
        "note": (
            "MEASURED_POST_L7_EXIT.verified stays False until a filled "
            "packet is committed from a real leftover. Not invented."
        ),
    }
    env.save_shot("final_suffix")


def main() -> int:
    parser = argparse.ArgumentParser()
    add_common_args(
        parser,
        default_state=FROM,
        default_tag="20260904_C1",
    )
    parser.add_argument("--no-video", action="store_true", default=False)
    parser.add_argument("--save-fixture", default=None)
    parser.add_argument(
        "--continue-suffix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After dest 0x29, continue bomb-E / Aquamentus / TF (default on).",
    )
    args = parser.parse_args()
    if not args.infinite_life:
        raise SystemExit("this fixture route probe requires --infinite-life")

    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    assist = make_assist(True)
    raw_env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    payload: dict[str, Any] = {
        "trial": "0x7b_b_to_a_cellar_cross",
        "bead": "rr-n91a",
        "from_state": args.from_state,
        "fixture_only": True,
        "natural_entry": False,
        "route_eligible": False,
        "infinite_life": True,
        "cellar_room": int(CELLAR_ROOM),
        "pit_tile": int(PIT_TILE),
        "east_x": int(EAST_X),
        "ram_claim": {
            "written_before_run": True,
            "claim": RAM_CLAIM,
            "dest": f"0x{DEST_ROOM:02X}",
            "miss_if": f"0x{SOURCE_ROOM:02X}",
        },
        "runtime_controller_writes": {
            "room": 0,
            "door": 0,
            "position": 0,
            "inventory": 0,
            "triforce": 0,
            "capacity": 0,
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
        if int(start["tile"]) == PIT_TILE:
            raise ProbeStop("start_on_pit_tile_250")

        assist.apply_env(env, frame=0)
        inv0 = start["inventory"]
        keys_in, bombs_in = int(inv0["keys"]), int(inv0["bombs"])
        candle_in = int(inv0["candle"])
        tf_in = int(inv0["triforce"])

        env.set_reason("cellar_load", "idle_load")
        for _ in range(LOAD_IDLE):
            env.step(nes_idle_action())
            assist.apply_env(env, frame=env.frame)
        payload["after_load"] = _glance(env)
        env.save_shot("after_load")

        ctl = make_nose_cellar_cross_controller(dest=DEST_ROOM)
        last_reason = _drive(
            env, assist, ctl, phase="cellar_cross", limit=ctl.max_frames
        )
        payload["controller"] = ctl.report()
        payload["done_reason"] = getattr(ctl, "done_reason", None)

        if ctl.failed or not ctl.success:
            payload["failed"] = (
                (ctl.notes[-1] if ctl.notes else "")
                or last_reason
                or "cellar_cross_failed"
            )
            payload["success"] = False
            payload["final"] = _glance(env)
            env.save_shot("final_halt")
        else:
            env.set_reason("final_census", "idle_census")
            for _ in range(CENSUS_IDLE):
                env.step(nes_idle_action())
                assist.apply_env(env, frame=env.frame)
            final = _glance(env)
            settled = read_snapshot(env.get_ram())
            env.save_shot("final")
            dest_eb = f"0x{settled.screen:02X}"
            deaths = int(assist.telemetry.deaths)
            prog = int(assist.telemetry.progression_writes)
            cap = int(assist.telemetry.capacity_writes)
            misses = [
                msg
                for cond, msg in (
                    (settled.level != LEVEL7, f"left L7 -> L{settled.level}"),
                    (settled.mode != PLAY_MODE, f"mode {settled.mode} != PLAY 5"),
                    (int(settled.screen) != DEST_ROOM, f"dest {dest_eb} != 0x29"),
                    (int(settled.screen) == SOURCE_ROOM, "dest is source 0x0D"),
                    (int(final["inventory"]["candle"]) != candle_in, "candle changed"),
                    (int(final["inventory"]["triforce"]) != tf_in, "triforce changed"),
                    (int(settled.keys) != keys_in, f"keys {keys_in}->{settled.keys}"),
                    (deaths, f"deaths {deaths}"),
                    (prog, f"progression_writes {prog}"),
                    (cap, f"capacity_writes {cap}"),
                )
                if cond
            ]
            payload["cellar_cross_grade"] = {
                "pass": not misses,
                "misses": misses,
                "keys": [keys_in, int(settled.keys)],
                "bombs": [bombs_in, int(settled.bombs)],
                "candle": candle_in,
                "triforce": tf_in,
                "dest_eb": dest_eb,
                "dest_eb_int": int(settled.screen),
                "settled_xy": [int(settled.link_x), int(settled.link_y)],
                "settled_mode": int(settled.mode),
                "deaths": deaths,
                "progression_writes": prog,
                "capacity_writes": cap,
                "position_poke": False,
            }
            payload["final"] = final
            payload["success"] = not misses
            if misses:
                payload["failed"] = "cellar_cross_grade_miss"
            else:
                fixture_name = args.save_fixture or "Level7Interior29PreBossReconFixture"
                payload["saved_fixture"] = _save_fixture(
                    raw_env,
                    fixture_name=fixture_name,
                    census=final,
                    dest_eb=dest_eb,
                    note="0x7B B→A floor-cross dest play 0x29. route_eligible=false.",
                )
                if args.continue_suffix:
                    try:
                        _run_suffix(env, assist, payload)
                    except ProbeStop:
                        raise
                    except Exception as exc:
                        payload["suffix_halt"] = (
                            f"suffix_exception:{type(exc).__name__}:{exc}"
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
            grade = payload.get("cellar_cross_grade", {})
            payload["runtime_integrity"] = {
                "deaths": int(tel.deaths),
                "progression_writes": int(tel.progression_writes),
                "capacity_writes": int(tel.capacity_writes),
                "direct_ram_writes": 0,
                "position_poke": False,
                "triforce_poke": False,
                "door_poke": False,
                "max_bombs_written": False,
            }
            del grade
        report = write_report(
            "l7_7b_cellar_cross", _jsonable(payload), tag=args.tag
        )
        raw_env.close()

    grade = payload.get("cellar_cross_grade") or {}
    final = payload.get("final") or {}
    print(f"report={report}")
    print(f"success={payload.get('success')} failed={payload.get('failed')}")
    print(
        f"dest_screen={grade.get('dest_eb') or final.get('screen_hex')} "
        f"dest_xy={grade.get('settled_xy') or final.get('xy')}"
    )
    print(f"grade={grade}")
    print(f"suffix_halt={payload.get('suffix_halt')}")
    print(f"assist_deaths={payload.get('deaths')}")
    print(f"progression_writes={payload.get('progression_writes')} "
          f"capacity_writes={payload.get('capacity_writes')}")
    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
