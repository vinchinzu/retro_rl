"""East door dest hop from Level 9 room 0x15 leftover into room 0x16 (first Patra).

rr-sz8.6. Pin Level9Interior15ReconFixture (composed full inventory,
live level==9 room 0x15, west mouth arrival). Glance leftover FIRST.
Hold RIGHT along y=141 aisle from (16, 141) across Wizzrobe room into east door (224, 141).
Transition into room 0x16.
Fail if dest is not play 0x16. Fail dest 0x07 Red Ring.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l9_15_east.py \
        --from-state Level9Interior15ReconFixture \
        --tag 20260904_E15_1 --infinite-life --no-video
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from retro_harness.env import make_env, reset_obs
from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.hop_controller import (
    HopController,
    WAIT_SCROLL_B,
    dungeon_align_then_push,
)
from zelda_i.dungeon.ids import object_name
from zelda_i.dungeon.ops import DOOR_TARGETS
from zelda_i.level9.dungeon import FULL_TRIFORCE, LEVEL9
from zelda_i.level9.prefix import RED_RING
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
CENSUS_IDLE = 120
SOURCE_STATE = "Level9Interior15ReconFixture"
DEFAULT_SAVE_FIXTURE = "Level9Interior16PatraReconFixture"

EAST_15_ORIGIN = 0x15
EAST_15_DEST_HYP = 0x16
EAST_15_START_POSE = (16, 141)
EAST_15_DEST_POSE = (32, 141)  # west mouth of room 0x16
EAST_DOOR = DOOR_TARGETS["RIGHT"]  # (224, 141)

RAM_CLAIM_EAST_15 = (
    "From play 0x15 leftover (16, 141) facing RIGHT (glance 20260904_glance_15), "
    "y-align to 141, then push RIGHT across open Wizzrobe floor to east door at (224, 141). "
    "East doorway is open (doors=2); transition into room 0x16. "
    "Settle in play mode in room 0x16 at west mouth (32, 141). "
    "First settled play $EB is RAM (hyp 0x16). "
    "Fail if dest is not play 0x16. Fail dest 0x07 Red Ring. "
    "Zero direct RAM writes, zero progression writes, zero capacity writes."
)


class ProbeStop(RuntimeError):
    """Expected fail-closed halt with a reportable reason."""


def _save_fixture(
    raw_env: Any,
    *,
    source_state: str,
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
    source_path = state_path(GAME_DIR, GAME, source_state)
    result = {
        "ok": True,
        "source_state": source_state,
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
            "phase": "level9_prefix_0x15_east_to_0x16",
            "track": "recon_fixture",
            "route_eligible": False,
            "fixture_only": True,
            "natural_entry": False,
            "fixture_writes": [],
            "notes": [
                "ZD §10.2 cut leftover, dest $EB live, not 10.3, not Red Ring 0x07",
                "0x15 Wizzrobe room -> east open door to 0x16 (first Patra skippable)",
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
        and (not live or obj.hp > 0 or snap.mode == PASSAGE_MODE)
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
        and int(start["screen"]) == int(EAST_15_ORIGIN)
        and not start["transitioning"]
        and abs(int(x) - EAST_15_START_POSE[0]) <= POSE_TOL
        and abs(int(y) - EAST_15_START_POSE[1]) <= POSE_TOL
        and inv["triforce"] == FULL_TRIFORCE
        and inv["magic_key"] == 1
        and inv["sword"] == 3
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
            f"l9_15_east_{self.tag}_{label}_f{self.frame}_"
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


def east_15_step(snap: ZeldaSnapshot) -> FrameAction:
    """y-align to 141, then RIGHT to door."""
    return dungeon_align_then_push(
        snap,
        push_dir="RIGHT",
        target_y=EAST_DOOR[1],
        door_plane=EAST_DOOR[0],
        y_tol=4,
        reason="east_15",
    )


@dataclass(kw_only=True)
class Level9East15Controller(HopController):
    """0x15 leftover -> hold RIGHT -> play 0x16."""

    spec_id: str = "level9_east_15"
    max_frames: int = 4000
    require_level: int = LEVEL9
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    done_reason: str = "left_0x15_east"
    dest: int | None = None
    route_eligible: bool = False
    leftover: dict[str, Any] = field(default_factory=dict)
    writes: int = 0

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if snap.mode != PLAY_MODE or snap.transitioning:
            return False
        if snap.screen in (RED_RING, EAST_15_ORIGIN):
            return False
        if self.dest is not None:
            return snap.screen == self.dest
        return snap.screen == EAST_15_DEST_HYP

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"play_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        del snap
        return FrameAction(nes_action("RIGHT"), "east_scroll")

    def emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        if force or not self.leftover or self.frames % 12 == 0:
            self.leftover = {
                "x": int(snap.link_x),
                "y": int(snap.link_y),
                "mode": int(snap.mode),
                "screen": int(snap.screen),
                "tile": int(snap.colliding_tile),
                "keys": int(snap.keys),
                "bombs": int(snap.bombs),
                "triforce": int(snap.triforce),
            }
        return action

    def guard(self, snap: ZeldaSnapshot) -> FrameAction | None:
        blocked = HopController.guard(self, snap)
        if blocked is not None:
            return blocked
        if snap.screen == RED_RING:
            return self.mark_fail("red_ring_0x07")
        if snap.mode == PLAY_MODE and not snap.transitioning and snap.screen != EAST_15_ORIGIN:
            if self.dest is not None and snap.screen != self.dest:
                return self.mark_fail(f"unexpected_play_0x{snap.screen:02x}")
            if snap.screen != EAST_15_DEST_HYP:
                return self.mark_fail(f"unexpected_dest_0x{snap.screen:02x}")
        return None

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != EAST_15_ORIGIN:
            return FrameAction(nes_action("RIGHT"), "east_settle")

        # Periodically swing sword to clear wizzrobe projectile or wizzrobe in path
        if self.frames % 8 == 0:
            return FrameAction(nes_action("RIGHT", "A"), "east_15_slash")
        return east_15_step(snap)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": self.spec_id,
            "dest_screen": self.dest,
            "dest_hyp": EAST_15_DEST_HYP,
            "evidence": "fixture-live",
            "route_eligible": False,
            "natural_entry": False,
            "writes": int(self.writes),
            "door": "RIGHT",
            "leftover": dict(self.leftover),
        }


def make_east_15_controller(
    *, dest: int | None = None
) -> Level9East15Controller:
    return Level9East15Controller(dest=dest)


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


def _grade_east_15(
    settled: ZeldaSnapshot,
    arrival: dict[str, Any],
    start: dict[str, Any],
    assist: Any,
) -> dict[str, Any]:
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
            (dest_int == RED_RING, "dest 0x07 Red Ring (fail closed)"),
            (dest_int == EAST_15_ORIGIN, "still in origin 0x15"),
            (dest_int != EAST_15_DEST_HYP, f"dest {dest_eb} != hyp 0x16"),
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
        "dest_hyp": f"0x{EAST_15_DEST_HYP:02X}",
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
    add_common_args(parser, default_state=SOURCE_STATE, default_tag="20260904_E15_1")
    parser.add_argument("--no-video", action="store_true", help="JSON + PNG only")
    parser.add_argument(
        "--glance",
        action="store_true",
        help="Dump leftover + start PNG; do not walk",
    )
    parser.add_argument(
        "--save-fixture",
        default=DEFAULT_SAVE_FIXTURE,
        help="Checkpoint name to save on successful settlement",
    )
    args = parser.parse_args()
    if not args.infinite_life:
        raise SystemExit("this fixture route probe requires --infinite-life")

    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    assist = make_assist(True)
    raw_env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    payload: dict[str, Any] = {
        "trial": "0x15_east_door_gate",
        "from_state": args.from_state,
        "fixture_only": True,
        "natural_entry": False,
        "route_eligible": False,
        "infinite_life": True,
        "east_15_origin": int(EAST_15_ORIGIN),
        "dest_hyp": f"0x{EAST_15_DEST_HYP:02X}",
        "prediction": {
            "written_before_run": True,
            "claim": RAM_CLAIM_EAST_15,
            "contingency": (
                "Dest not play 0x16 is a miss. Dest 0x07 is a "
                "miss (halt, do not enter Red Ring). "
                "Do not poke TF or ADDR_ARROWS. "
                "Zero progression writes, zero capacity writes."
            ),
            "do_not_assume_dest": [EAST_15_DEST_HYP],
        },
        "runtime_controller_writes": {
            "room": 0,
            "door": 0,
            "position": 0,
            "inventory": 0,
            "triforce": 0,
            "arrows": 0,
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

        ctl = make_east_15_controller(dest=None)
        last_reason = _drive(env, assist, ctl, total, "east_15_gate")
        payload["controller"] = ctl.report()
        payload["done_reason"] = getattr(ctl, "done_reason", None)

        if ctl.failed or not ctl.success:
            payload["failed"] = (
                (ctl.notes[-1] if ctl.notes else "")
                or last_reason
                or "east_15_gate_failed"
            )
            payload["success"] = False
            payload["final"] = _glance(env)
            env.save_shot("final_halt")
        else:
            env.set_reason("arrived", "east_15_arrived")
            arrival = _glance(env)
            env.save_shot("arrived")
            env.set_reason("final_census", "idle_census")
            for _ in range(CENSUS_IDLE):
                env.step(nes_idle_action())
                total[0] += 1
                assist.apply_env(env, frame=total[0])
            settled = read_snapshot(env.get_ram())
            final = _glance(env)
            env.save_shot("final")
            grade = _grade_east_15(settled, arrival, start, assist)
            payload["east_15_gate_grade"] = grade
            payload["arrival"] = arrival
            payload["final"] = final
            payload["success"] = bool(grade["pass"])
            if not grade["pass"]:
                payload["failed"] = "east_15_gate_grade_miss"
            elif args.save_fixture and payload["success"]:
                payload["saved_fixture"] = _save_fixture(
                    raw_env,
                    source_state=args.from_state,
                    fixture_name=args.save_fixture,
                    dest_eb=grade.get("dest_eb"),
                    leftover=final,
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
            grade = payload.get("east_15_gate_grade", {})
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
        report = write_report("l9_15_east_fixture", payload, tag=args.tag)
        raw_env.close()

    grade = payload.get("east_15_gate_grade") or {}
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
