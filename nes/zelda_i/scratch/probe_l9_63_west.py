"""West key door dest hop from Level 9 room 0x63 (5 Zols) into room 0x62 (8 Keese).

rr-sz8.6. Pin Level9Interior63ZolsReconFixture (composed full inventory,
live level==9 room 0x63, emerged from cellar 0x70). Glance leftover FIRST.
Walk west/align y=141, push LEFT through west key door into room 0x62.
Settle in play mode room 0x62 at east mouth (224, 141).
Fail if dest is not play 0x62. Fail dest 0x07 Red Ring.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l9_63_west.py \
        --from-state Level9Interior63ZolsReconFixture \
        --tag 20260904_W63_1 --infinite-life --no-video
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
from zelda_i.dungeon.hop_controller import HopController, WAIT_SCROLL_B
from zelda_i.dungeon.ids import object_name
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
SOURCE_STATE = "Level9Interior63ZolsReconFixture"
DEFAULT_SAVE_FIXTURE = "Level9Interior62KeeseReconFixture"

WEST_63_ORIGIN = 0x63
WEST_63_DEST_HYP = 0x62
WEST_63_START_POSE = (160, 157)
WEST_63_DEST_POSE = (224, 141)  # east mouth of room 0x62
WEST_DOOR_STAND = (32, 141)
ALIGN_TOL = 4

RAM_CLAIM_WEST_63 = (
    "From play 0x63 leftover (160, 157) (glance 20260904_glance_63), "
    "navigate west and align y=141, push LEFT through west key door opened by Magical Key. "
    "Settle in play mode in room 0x62 at east mouth (224, 141). "
    "First settled play $EB is RAM (hyp 0x62). "
    "Fail if dest is not play 0x62 (8 Keese). "
    "Fail dest 0x07 Red Ring. Zero direct RAM writes, zero progression writes, zero capacity writes."
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
            "phase": "level9_prefix_0x63_west_to_0x62",
            "track": "recon_fixture",
            "route_eligible": False,
            "fixture_only": True,
            "natural_entry": False,
            "fixture_writes": [],
            "notes": [
                "ZD §10.2 cut leftover, dest $EB live, not 10.3, not Red Ring 0x07",
                "0x63 west key door -> play 0x62 (8 Keese)",
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
        and int(start["screen"]) == int(WEST_63_ORIGIN)
        and not start["transitioning"]
        and abs(int(x) - WEST_63_START_POSE[0]) <= POSE_TOL
        and abs(int(y) - WEST_63_START_POSE[1]) <= POSE_TOL
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
            f"l9_63_west_{self.tag}_{label}_f{self.frame}_"
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


def west_63_step(snap: ZeldaSnapshot) -> FrameAction:
    """Align y=141, push LEFT through west door."""
    x, y = int(snap.link_x), int(snap.link_y)

    # First get off the emergence wall/column (move LEFT past x=144)
    if x > 144:
        return FrameAction(nes_action("LEFT"), "leave_stairs_exit")

    # Align y to door level 141
    if abs(y - 141) > 2:
        btn = "UP" if y > 141 else "DOWN"
        return FrameAction(nes_action(btn), "align_door_y")

    # Walk LEFT through door
    return FrameAction(nes_action("LEFT"), "push_west_door")


@dataclass(kw_only=True)
class Level9West63Controller(HopController):
    """0x63 leftover -> walk west / align y=141 -> push west door LEFT -> play 0x62."""

    spec_id: str = "level9_west_63"
    max_frames: int = 4000
    require_level: int = LEVEL9
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    done_reason: str = "settled_play_0x62"
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
        if snap.screen in (RED_RING, WEST_63_ORIGIN):
            return False
        if self.dest is not None:
            return snap.screen == self.dest
        return snap.screen == WEST_63_DEST_HYP

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"play_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        del snap
        return FrameAction(nes_action("LEFT"), "west_door_scroll")

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
            if self.dest is not None and snap.screen != self.dest:
                return self.mark_fail(f"unexpected_play_0x{snap.screen:02x}")
            if snap.screen not in (WEST_63_ORIGIN, WEST_63_DEST_HYP):
                return self.mark_fail(f"unexpected_dest_0x{snap.screen:02x}")
        if snap.mode == PASSAGE_MODE:
            return self.mark_fail(f"unexpected_cellar_0x{snap.screen:02x}")
        return None

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != WEST_63_ORIGIN:
            return FrameAction(nes_action("LEFT"), "west_settle")

        # Attack any Zol that gets close
        live_zols = [o for o in snap.objects if o.type_id == 0x13 and o.hp > 0]
        if live_zols:
            nearest = min(
                live_zols,
                key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
            )
            dist = abs(nearest.x - snap.link_x) + abs(nearest.y - snap.link_y)
            if dist <= 24:
                dx = nearest.x - snap.link_x
                dy = nearest.y - snap.link_y
                if abs(dx) > abs(dy):
                    d = "RIGHT" if dx > 0 else "LEFT"
                else:
                    d = "DOWN" if dy > 0 else "UP"
                return FrameAction(
                    nes_action(d, "A") if self.frames % 4 == 0 else nes_action(d),
                    "zol_slash",
                )

        return west_63_step(snap)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": self.spec_id,
            "dest_screen": self.dest,
            "dest_hyp": WEST_63_DEST_HYP,
            "evidence": "fixture-live",
            "route_eligible": False,
            "natural_entry": False,
            "writes": int(self.writes),
            "door": "LEFT",
            "leftover": dict(self.leftover),
        }


def make_west_63_controller(
    *, dest: int | None = None
) -> Level9West63Controller:
    return Level9West63Controller(dest=dest)


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
    add_common_args(parser, default_state=SOURCE_STATE, default_tag="20260904_W63_1")
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
        "trial": "west_0x63_gate",
        "from_state": args.from_state,
        "fixture_only": True,
        "natural_entry": False,
        "route_eligible": False,
        "infinite_life": True,
        "west_63_origin": int(WEST_63_ORIGIN),
        "dest_hyp": f"0x{WEST_63_DEST_HYP:02X}",
        "prediction": {
            "written_before_run": True,
            "claim": RAM_CLAIM_WEST_63,
            "contingency": (
                "Dest not play 0x62 is a miss. Dest 0x07 is a miss (halt, do not enter Red Ring). "
                "Do not poke TF or ADDR_ARROWS. "
                "Zero progression writes, zero capacity writes."
            ),
            "do_not_assume_dest": [WEST_63_DEST_HYP],
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

        ctl = make_west_63_controller(dest=None)
        last_reason = _drive(env, assist, ctl, total, "west_63_gate")
        payload["controller"] = ctl.report()
        payload["done_reason"] = getattr(ctl, "done_reason", None)

        if ctl.failed or not ctl.success:
            payload["failed"] = (
                (ctl.notes[-1] if ctl.notes else "")
                or last_reason
                or "west_63_gate_failed"
            )
            payload["success"] = False
            payload["final"] = _glance(env)
            env.save_shot("final_halt")
        else:
            env.set_reason("arrived", "west_63_arrived")
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
        write_report(f"l9_63_west_{args.tag}", payload)
        raw_env.close()

    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
