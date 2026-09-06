"""Bomb-north dest hop from Level 9 room 0x20 leftover (play 0x20) into room 0x10.

rr-sz8.6. Pin Level9Interior20ReconFixture (composed full inventory,
live level==9 room 0x20, emerged from cellar 0x75 at (96, 157)). Glance leftover FIRST.
Navigate perimeter to north bomb stand (120, 93), bomb north wall with B,
step back, wait for blast, push UP into room 0x10 (Silver Arrows).
Fail if dest is not 0x10. Fail dest 0x07 Red Ring.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l9_20_north.py \
        --from-state Level9Interior20ReconFixture \
        --tag 20260904_BN20_1 --infinite-life --no-video
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from retro_harness.env import make_env, reset_obs
from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.bomb_wall import BombWallController, BombWallPhase
from zelda_i.dungeon.hop_controller import HopController, WAIT_SCROLL_B
from zelda_i.dungeon.ids import object_name
from zelda_i.level9.dungeon import FULL_TRIFORCE, LEVEL9
from zelda_i.level9.prefix import RED_RING, is_north_neighbor
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
SOURCE_STATE = "Level9Interior20ReconFixture"
DEFAULT_SAVE_FIXTURE = "Level9Interior10SilverArrowsReconFixture"

BOMB_NORTH_20_ORIGIN = 0x20
BOMB_NORTH_20_DEST_HYP = 0x10
BOMB_NORTH_20_START_POSE = (96, 157)
BOMB_NORTH_20_DEST_POSE = (120, 189)  # south mouth of room 0x10
BOMB_NORTH_20_STAND = (120, 93)
BOMB_NORTH_20_APPROACH: tuple[tuple[int, int], ...] = (
    (96, 189),
    (176, 189),
    (176, 93),
    (120, 93),
)

RAM_CLAIM_BOMB_NORTH_20 = (
    "From play 0x20 leftover (96, 157) facing DOWN (glance 20260904_glance_20), "
    "navigate along perimeter: (96, 189) -> (176, 189) -> (176, 93) -> (120, 93). "
    "At north wall bomb stand (120, 93), face UP, place exactly one bomb with B, "
    "step back, wait for blast, push UP through blasted north doorway. "
    "Settle in play mode in room 0x10 at south mouth (120, 189). "
    "First settled play $EB is RAM (hyp 0x10). "
    "Fail if dest is not north neighbor 0x10. Fail dest 0x07 Red Ring. "
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
            "phase": "level9_prefix_0x20_bomb_north_to_0x10",
            "track": "recon_fixture",
            "route_eligible": False,
            "fixture_only": True,
            "natural_entry": False,
            "fixture_writes": [],
            "notes": [
                "ZD §10.2 cut leftover, dest $EB live, not 10.3, not Red Ring 0x07",
                "0x20 bomb-north -> play 0x10 (Silver Arrows room)",
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
    }


def _typed(snap: ZeldaSnapshot, *, live: bool = False) -> list[Any]:
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
        and int(start["screen"]) == int(BOMB_NORTH_20_ORIGIN)
        and not start["transitioning"]
        and abs(int(x) - BOMB_NORTH_20_START_POSE[0]) <= POSE_TOL
        and abs(int(y) - BOMB_NORTH_20_START_POSE[1]) <= POSE_TOL
        and inv["triforce"] == FULL_TRIFORCE
        and inv["magic_key"] == 1
        and inv["sword"] == 3
        and inv["bombs"] >= 1
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
            "facing": int(snap.facing),
            "transitioning": bool(snap.transitioning),
            "live_count": len(live),
            "objects": [_obj_row(o) for o in live[:4]],
        })

    def save_shot(self, label: str) -> str:
        tag_dir = RECORDINGS_DIR / self.tag
        tag_dir.mkdir(parents=True, exist_ok=True)
        path = tag_dir / f"{label}_{self.frame:05d}.png"
        save_rgb_png(self.latest_obs, path)
        self.screenshots.append(str(path))
        return str(path)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self.frame += 1
        obs, rew, term, trunc, info = self.env.step(action)
        self.latest_obs = obs
        ram = self.env.get_ram()
        snap = read_snapshot(ram)
        key = (int(snap.level), int(snap.mode), int(snap.screen))
        xy = (int(snap.link_x), int(snap.link_y))

        if key != self.last_key:
            self._sample(snap, f"trans_{self.last_key}->{key}")
            self.last_key = key
            self.last_xy = xy
            self.stuck = 0
        elif xy != self.last_xy:
            self.last_xy = xy
            self.stuck = 0
            if self.frame % 30 == 0:
                self._sample(snap)
        else:
            self.stuck += 1
            if self.stuck in (30, 90, 180, 300, 600):
                self._sample(snap, f"stuck_{self.stuck}")
            if self.stuck >= 900:
                self._sample(snap, "stuck_timeout")
                raise ProbeStop(f"stuck_at_{xy}_mode_{snap.mode}_screen_0x{snap.screen:02X}")
        return obs, rew, term, trunc, info


@dataclass(kw_only=True)
class Level9BombNorth20Controller(HopController):
    """0x20 leftover -> approach north stand (120, 93) -> bomb UP -> play 0x10."""

    spec_id: str = "level9_bomb_north_20"
    max_frames: int = 4000
    require_level: int = LEVEL9
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    done_reason: str = "settled_play_0x10"
    dest: int | None = None
    route_eligible: bool = False
    leftover: dict[str, Any] = field(default_factory=dict)
    writes: int = 0
    _bomb_wall: BombWallController = field(init=False, repr=False)

    def __post_init__(self) -> None:
        wall = SimpleNamespace(
            room=BOMB_NORTH_20_ORIGIN,
            stand=BOMB_NORTH_20_STAND,
            face="UP",
            opens_to=self.dest if self.dest is not None else BOMB_NORTH_20_DEST_HYP,
        )
        self._bomb_wall = BombWallController(
            wall=wall,
            level=self.require_level,
            approach_waypoints=BOMB_NORTH_20_APPROACH,
            approach_tol=4,
            stand_tol=4,
            face_frames=4,
            step_back=6,
            wait_blast=100,
            wait_hold_face=False,
            require_bomb_consumed=True,
            max_frames=self.max_frames,
        )

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if snap.mode != PLAY_MODE or snap.transitioning:
            return False
        if snap.screen in (RED_RING, BOMB_NORTH_20_ORIGIN):
            return False
        if self.dest is not None:
            return snap.screen == self.dest
        return snap.screen == BOMB_NORTH_20_DEST_HYP

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"play_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        del snap
        return FrameAction(nes_action("UP"), "north_enter_scroll")

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
        if snap.mode == PLAY_MODE and not snap.transitioning:
            if snap.screen != BOMB_NORTH_20_ORIGIN:
                if self.dest is not None and snap.screen != self.dest:
                    return self.mark_fail(f"unexpected_play_0x{snap.screen:02x}")
                if not is_north_neighbor(BOMB_NORTH_20_ORIGIN, snap.screen):
                    return self.mark_fail(f"not_north_neighbor_0x{snap.screen:02x}")
        return None

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != BOMB_NORTH_20_ORIGIN:
            return FrameAction(nes_action("UP"), "north_settle")
        act = self._bomb_wall.step(snap)
        self.notes.extend(n for n in self._bomb_wall.notes if n not in self.notes)
        if self._bomb_wall.phase == BombWallPhase.FAILED:
            return self.mark_fail(
                self._bomb_wall.notes[-1] if self._bomb_wall.notes else "bomb_wall_failed"
            )
        return act

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": self.spec_id,
            "dest_screen": self.dest,
            "dest_hyp": BOMB_NORTH_20_DEST_HYP,
            "evidence": "fixture-live",
            "route_eligible": False,
            "natural_entry": False,
            "writes": int(self.writes),
            "leftover": dict(self.leftover),
        }


def make_bomb_north_20_controller(
    *, dest: int | None = None
) -> Level9BombNorth20Controller:
    return Level9BombNorth20Controller(dest=dest)


def _drive(
    env: ObservedEnv,
    assist: Any,
    controller: Any,
    total_frames: list[int],
    phase_name: str,
) -> str:
    last_reason = "init"
    while not controller.success and not controller.failed:
        snap = read_snapshot(env.get_ram())
        if snap.mode == DEATH_MODE:
            controller.mark_fail("link_death")
            break
        action = controller.step(snap)
        last_reason = action.reason
        env.set_reason(phase_name, action.reason)
        env.step(action.action)
        total_frames[0] += 1
        assist.apply_env(env, frame=total_frames[0])
    return last_reason


def _census(env: ObservedEnv, assist: Any, total_frames: list[int], frames: int = CENSUS_IDLE) -> None:
    idle = nes_idle_action()
    for _ in range(frames):
        env.set_reason("census", "settled_census")
        env.step(idle)
        total_frames[0] += 1
        assist.apply_env(env, frame=total_frames[0])


def _assist_counts(assist: Any) -> dict[str, int]:
    pokes = getattr(assist, "pokes_applied", 0)
    prog = getattr(assist, "progression_writes", 0)
    cap = getattr(assist, "capacity_writes", 0)
    return {
        "pokes": pokes,
        "progression_writes": prog,
        "capacity_writes": cap,
        "tf_arrow_poke": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_state=SOURCE_STATE, default_tag="20260904_BN20_1")
    parser.add_argument("--no-video", action="store_true", help="JSON + PNG only")
    parser.add_argument(
        "--glance",
        action="store_true",
        help="Dump leftover + start PNG; do not walk",
    )
    parser.add_argument(
        "--dest",
        type=lambda v: int(v, 0),
        default=None,
        help="Optional destination room override",
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
        "trial": "0x20_bomb_north_gate",
        "from_state": args.from_state,
        "fixture_only": True,
        "natural_entry": False,
        "route_eligible": False,
        "infinite_life": True,
        "bomb_north_20_origin": int(BOMB_NORTH_20_ORIGIN),
        "dest_hyp": f"0x{BOMB_NORTH_20_DEST_HYP:02X}",
        "prediction": {
            "written_before_run": True,
            "claim": RAM_CLAIM_BOMB_NORTH_20,
            "contingency": (
                "Dest not play 0x10 is a miss. Dest 0x07 is a "
                "miss (halt, do not enter Red Ring). "
                "Do not poke TF or ADDR_ARROWS. "
                "Zero progression writes, zero capacity writes."
            ),
            "do_not_assume_dest": [BOMB_NORTH_20_DEST_HYP],
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

        ctl = make_bomb_north_20_controller(dest=args.dest)
        last_reason = _drive(env, assist, ctl, total, "bomb_north_20_gate")
        payload["controller"] = ctl.report()
        payload["done_reason"] = getattr(ctl, "done_reason", None)

        if ctl.failed or not ctl.success:
            payload["failed"] = (
                (ctl.notes[-1] if ctl.notes else "")
                or last_reason
                or "bomb_north_20_gate_failed"
            )
            payload["success"] = False
            payload["final"] = _glance(env)
            env.save_shot("final_halt")
        else:
            env.set_reason("arrived", "bomb_north_20_arrived")
            env.save_shot("arrival")
            _census(env, assist, total)
            env.save_shot("census_settled")
            final = _glance(env)
            payload["final"] = final
            payload["success"] = True
            payload["controller_frames"] = ctl.frames
            payload["total_frames"] = total[0]
            payload["dest_screen"] = final["screen_hex"]
            payload["leftover"] = {
                "xy": final["xy"],
                "screen": final["screen_hex"],
                "mode": final["mode"],
                "doors": final["cur_opened_doors"],
                "inventory": final["inventory"],
            }
            if args.save_fixture:
                payload["saved_fixture"] = _save_fixture(
                    raw_env,
                    source_state=args.from_state,
                    fixture_name=args.save_fixture,
                    dest_eb=final["screen_hex"],
                    leftover=payload["leftover"],
                )
    except ProbeStop as ex:
        payload["halt_reason"] = str(ex)
    finally:
        payload["assist"] = _assist_counts(assist)
        if env is not None:
            payload["samples"] = env.samples
            payload["screenshots"] = env.screenshots
            payload["stuck_events"] = env.stuck
        write_report(f"l9_20_north_{args.tag}", payload)
        raw_env.close()

    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
