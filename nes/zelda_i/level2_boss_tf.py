"""Level 2 post-boss triforce collect (0x0d south-band waypoints).

Re-exported via ``level2_boss_path``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level2_boss_combat import (
    ROOM_0E,
    goto_action,
    sample_snapshot,
    triforce_bit_02,
)
from zelda_i.level2_puzzles import (
    DOOR_LEFT,
    DOOR_RIGHT,
    DOOR_UP,
    L2_BOSS_EXIT_DOOR_Y,
    L2_BOSS_HC_STAND,
    L2_TF_COLLECT_WAYPOINTS,
    L2_TF_PROBE_EVIDENCE,
    L2_TF_PUSH_BLOCK_STAND,
    L2_TF_PUSH_DIR,
    L2_TF_WAYPOINT_TOL,
    LEVEL2_TRIFORCE_BIT,
    POST_BOSS_TF_POLICY,
    ROOM_L2_TF,
)
from zelda_i.paths import RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_TRIFORCE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)

ROOM_TF: int = ROOM_L2_TF  # 0x0D
LEVEL2_TF_BIT: int = LEVEL2_TRIFORCE_BIT  # 0x02
TF_COLLECT_MAX_FRAMES: int = 4000
L2_TF_REACH_JSON: Path = RECORDINGS_DIR / Path(L2_TF_PROBE_EVIDENCE).name


def default_tf_waypoints() -> tuple[tuple[int, int], ...]:
    """LIVE south-band collect waypoints from ``POST_BOSS_TF_POLICY``."""
    return POST_BOSS_TF_POLICY.waypoints or L2_TF_COLLECT_WAYPOINTS


def load_tf_policy(
    json_path: Path | None = None,
) -> dict[str, Any]:
    """Resolve TF collect policy: JSON override if LIVE, else catalog.

    Hardcoded default is LIVE south-band maze so encode works even when
    ``l2_0d_tf_reach.json`` is missing.
    """
    default: dict[str, Any] = {
        "source": "level2_puzzles.POST_BOSS_TF_POLICY",
        "waypoints": [list(w) for w in default_tf_waypoints()],
        "push_stand": (
            list(L2_TF_PUSH_BLOCK_STAND) if L2_TF_PUSH_BLOCK_STAND else None
        ),
        "push_dir": L2_TF_PUSH_DIR,
        "tol": L2_TF_WAYPOINT_TOL,
        "notes": POST_BOSS_TF_POLICY.notes,
        "kind": "south_band_waypoints",
        "live": POST_BOSS_TF_POLICY.live,
    }
    path = json_path if json_path is not None else L2_TF_REACH_JSON
    if not path.is_file():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    result = str(data.get("result") or "").upper()
    if result not in ("LIVE", "OK", "TF_02", "SUCCESS"):
        return default
    pol = data.get("winning_policy") or data.get("policy")
    if not isinstance(pol, dict):
        return default
    merged = dict(default)
    merged["source"] = str(path)
    if pol.get("waypoints") or pol.get("collect_waypoints"):
        merged["waypoints"] = pol.get("waypoints") or pol.get("collect_waypoints")
    if "push_stand" in pol or "push_block_stand" in pol:
        merged["push_stand"] = pol.get("push_stand") or pol.get("push_block_stand")
    if pol.get("push_dir") or pol.get("push_face"):
        merged["push_dir"] = pol.get("push_dir") or pol.get("push_face")
    if pol.get("tol") is not None:
        merged["tol"] = int(pol["tol"])
    if pol.get("notes"):
        merged["notes"] = pol["notes"]
    if pol.get("kind"):
        merged["kind"] = pol["kind"]
    return merged


def policy_waypoints(pol: dict[str, Any] | None) -> list[tuple[int, int]]:
    """Extract waypoint list from a policy dict."""
    if not pol:
        return list(default_tf_waypoints())
    raw = pol.get("waypoints") or pol.get("collect_waypoints") or default_tf_waypoints()
    out: list[tuple[int, int]] = []
    for w in raw:
        if isinstance(w, (list, tuple)) and len(w) >= 2:
            out.append((int(w[0]), int(w[1])))
    return out or list(default_tf_waypoints())


def policy_push(
    pol: dict[str, Any] | None,
) -> tuple[tuple[int, int] | None, str | None]:
    """Extract optional push-block stand + face from policy."""
    if not pol:
        return None, None
    stand = pol.get("push_stand") or pol.get("push_block_stand")
    face = pol.get("push_dir") or pol.get("push_face") or pol.get("push")
    if isinstance(stand, (list, tuple)) and len(stand) >= 2:
        return (int(stand[0]), int(stand[1])), (
            str(face).upper() if face else None
        )
    return None, (str(face).upper() if face else None)

class PostBossTfPhase(Enum):
    """HC touch → LEFT exit → south-band waypoints → tf&0x02."""

    HEART = auto()
    EXIT_LEFT = auto()
    TF_PUSH = auto()
    TF_WP = auto()
    TF_WAIT = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level2PostBossTfController:
    """After Dodongo kill: heart container → LEFT 0x0d → TF bit 0x02.

    Reuses ``POST_BOSS_TF_POLICY`` waypoints. Optional JSON override via
    ``load_tf_policy`` at construction time.
    """

    phase: PostBossTfPhase = PostBossTfPhase.HEART
    frames: int = 0
    phase_frames: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)
    heart_touched: bool = False
    waypoint_index: int = 0
    push_done: bool = False
    stuck_frames: int = 0
    last_xy: tuple[int, int] | None = None
    max_frames: int = TF_COLLECT_MAX_FRAMES
    waypoints: list[tuple[int, int]] = field(default_factory=list)
    push_stand: tuple[int, int] | None = None
    push_dir: str | None = None
    tol: int = L2_TF_WAYPOINT_TOL
    policy_live: bool = True
    policy_source: str = "level2_puzzles.POST_BOSS_TF_POLICY"
    policy_notes: str = ""

    def __post_init__(self) -> None:
        if not self.waypoints:
            pol = load_tf_policy()
            self.waypoints = policy_waypoints(pol)
            self.push_stand, self.push_dir = policy_push(pol)
            self.tol = int(pol.get("tol") or L2_TF_WAYPOINT_TOL)
            self.policy_live = bool(self.waypoints)
            self.policy_source = str(pol.get("source") or self.policy_source)
            self.policy_notes = str(pol.get("notes") or POST_BOSS_TF_POLICY.notes)
        self.push_done = self.push_stand is None

    def _set_phase(self, phase: PostBossTfPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(PostBossTfPhase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def step(self, snap: ZeldaSnapshot, *, tf_value: int | None = None) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1

        if self.phase is PostBossTfPhase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is PostBossTfPhase.FAILED:
            return FrameAction(nes_idle_action(), "failed")

        if self.frames >= self.max_frames:
            return self._fail("timeout")

        tf = LEVEL2_TF_BIT if tf_value is None else int(tf_value)
        # When caller passes real tf_value, check bit; when None, rely on
        # mode 18 / external stop (controller is pure-geometry otherwise).
        if tf_value is not None and triforce_bit_02(tf):
            self.success = True
            self._set_phase(PostBossTfPhase.DONE, "tf_got")
            return FrameAction(nes_idle_action(), "done")

        if snap.mode == 17:
            return self._fail("link_death")

        if snap.mode != PLAY_MODE and snap.mode != 18:
            return FrameAction(nes_idle_action(), f"settle_mode_{snap.mode}")
        if snap.mode == 18:
            # Fanfare — idle; bit may lag one frame.
            return FrameAction(nes_idle_action(), "tf_fanfare")

        # Boss room: heart then LEFT.
        if snap.screen == ROOM_0E:
            doors = snap.cur_opened_doors
            if not self.heart_touched and self.frames < 400:
                self._set_phase(PostBossTfPhase.HEART)
                hx, hy = L2_BOSS_HC_STAND
                act, at = goto_action(snap, hx, hy, tol=8)
                if at:
                    self.heart_touched = True
                    self._set_phase(PostBossTfPhase.EXIT_LEFT, "heart_touched")
                return FrameAction(act, "heart")
            self._set_phase(PostBossTfPhase.EXIT_LEFT)
            door_y = L2_BOSS_EXIT_DOOR_Y
            if doors & DOOR_LEFT or not (doors & (DOOR_RIGHT | DOOR_UP)):
                if abs(snap.link_y - door_y) > 4:
                    return FrameAction(
                        nes_action("DOWN" if snap.link_y < door_y else "UP"),
                        "exit_align_y",
                    )
                return FrameAction(nes_action("LEFT"), "exit_left")
            if doors & DOOR_RIGHT:
                if abs(snap.link_y - door_y) > 4:
                    return FrameAction(
                        nes_action("DOWN" if snap.link_y < door_y else "UP"),
                        "exit_align_y_r",
                    )
                return FrameAction(nes_action("RIGHT"), "exit_right_residual")
            if doors & DOOR_UP:
                if abs(snap.link_x - 120) > 2:
                    return FrameAction(
                        nes_action("RIGHT" if snap.link_x < 120 else "LEFT"),
                        "exit_align_x",
                    )
                return FrameAction(nes_action("UP"), "exit_up")
            if abs(snap.link_y - door_y) > 4:
                return FrameAction(
                    nes_action("DOWN" if snap.link_y < door_y else "UP"),
                    "exit_align_y_default",
                )
            return FrameAction(nes_action("LEFT"), "exit_left_default")

        # TF room 0x0d.
        if snap.screen != ROOM_TF:
            return FrameAction(nes_idle_action(), f"wait_tf_room_0x{snap.screen:02x}")

        x, y = snap.link_x, snap.link_y
        if self.last_xy == (x, y):
            self.stuck_frames += 1
        else:
            self.stuck_frames = 0
            self.last_xy = (x, y)

        if not self.push_done and self.push_stand is not None:
            self._set_phase(PostBossTfPhase.TF_PUSH)
            sx, sy = self.push_stand
            act, at = goto_action(snap, sx, sy, tol=4)
            if not at:
                return FrameAction(act, "tf_push_approach")
            face = self.push_dir or "UP"
            if self.phase_frames > 90:
                self.push_done = True
                self._set_phase(PostBossTfPhase.TF_WP, "push_done")
            return FrameAction(nes_action(face), "tf_push_hold")

        if self.waypoint_index >= len(self.waypoints):
            self._set_phase(PostBossTfPhase.TF_WAIT)
            if y > 149 and abs(x - 128) <= 6:
                return FrameAction(nes_action("UP"), "tf_nudge")
            return FrameAction(nes_idle_action(), "tf_pickup_wait")

        self._set_phase(PostBossTfPhase.TF_WP)
        tx, ty = self.waypoints[self.waypoint_index]
        # Free east alcove x≈224 toward first WP column.
        if x >= 216 and self.waypoint_index == 0:
            return FrameAction(nes_action("LEFT"), "tf_free_alcove")
        act, at = goto_action(snap, tx, ty, tol=self.tol)
        if self.stuck_frames > 20 and not at:
            self.stuck_frames = 0
            if abs(x - tx) > abs(y - ty):
                return FrameAction(
                    nes_action("DOWN" if y < ty else "UP"), "tf_unstick"
                )
            return FrameAction(
                nes_action("RIGHT" if x < tx else "LEFT"), "tf_unstick"
            )
        if at:
            self.waypoint_index += 1
            self.stuck_frames = 0
            self.notes.append(f"tf_wp_{self.waypoint_index}")
            if self.waypoint_index >= len(self.waypoints):
                self._set_phase(PostBossTfPhase.TF_WAIT, "waypoints_done")
            return FrameAction(nes_idle_action(), f"tf_wp_hit_{self.waypoint_index}")
        return FrameAction(act, f"tf_wp_{self.waypoint_index}")

    def report(self) -> dict[str, Any]:
        return {
            "phase": self.phase.name,
            "frames": self.frames,
            "success": self.success,
            "notes": list(self.notes),
            "heart_touched": self.heart_touched,
            "waypoint_index": self.waypoint_index,
            "waypoints": [list(w) for w in self.waypoints],
            "policy_live": self.policy_live,
            "policy_source": self.policy_source,
            "tol": self.tol,
        }


def make_post_boss_tf_controller(
    *, policy: dict[str, Any] | None = None
) -> Level2PostBossTfController:
    """Factory for HC → LEFT → 0x0d TF collect."""
    if policy is None:
        return Level2PostBossTfController()
    wps = policy_waypoints(policy)
    push_stand, push_dir = policy_push(policy)
    return Level2PostBossTfController(
        waypoints=wps,
        push_stand=push_stand,
        push_dir=push_dir,
        tol=int(policy.get("tol") or L2_TF_WAYPOINT_TOL),
        policy_live=bool(wps),
        policy_source=str(policy.get("source") or "custom"),
        policy_notes=str(policy.get("notes") or ""),
    )

def collect_and_tf(
    env: Any,
    assist: Any | None = None,
    *,
    budget: int = TF_COLLECT_MAX_FRAMES,
    apply_assist: Callable[[Any, int], None] | None = None,
) -> dict[str, Any]:
    """HC → LEFT → 0x0d south-band maze → triforce & 0x02."""
    ctrl = make_post_boss_tf_controller()
    log: list[dict[str, Any]] = [
        {
            "event": "tf_policy_load",
            "source": ctrl.policy_source,
            "probe_path": str(L2_TF_REACH_JSON),
            "probe_present": L2_TF_REACH_JSON.is_file(),
            "policy_live": ctrl.policy_live,
            "n_waypoints": len(ctrl.waypoints),
            "waypoints": [list(w) for w in ctrl.waypoints],
            "tol": ctrl.tol,
            "push_stand": list(ctrl.push_stand) if ctrl.push_stand else None,
            "push_dir": ctrl.push_dir,
            "notes": ctrl.policy_notes or POST_BOSS_TF_POLICY.notes,
        }
    ]
    xy_hist: list[list[int]] = []
    last_wp = 0

    for f in range(budget):
        if assist is not None and f % 20 == 0:
            if apply_assist is not None:
                apply_assist(env, 20000 + f)
            else:
                assist.apply_env(env, frame=20000 + f)
        ram = env.get_ram()
        s = read_snapshot(ram)
        tf = int(read_u8(ram, ADDR_TRIFORCE))
        if triforce_bit_02(tf):
            log.append(
                sample_snapshot(s, ram, event="tf_got") | {"phase": ctrl.phase.name}
            )
            ctrl.success = True
            return {
                "ok": True,
                "frames": f + 1,
                "phase": ctrl.phase.name,
                "policy_live": ctrl.policy_live,
                "waypoints": [list(w) for w in ctrl.waypoints],
                "log": log,
                "final": log[-1],
                "controller": ctrl.report(),
            }
        action = ctrl.step(s, tf_value=tf)
        if f % 30 == 0 and s.screen == ROOM_TF:
            xy_hist.append([s.link_x, s.link_y])
            if len(xy_hist) > 40:
                xy_hist = xy_hist[-40:]
        if ctrl.waypoint_index > last_wp:
            last_wp = ctrl.waypoint_index
            log.append(
                sample_snapshot(s, ram, event=f"tf_wp_{last_wp}")
                | {"phase": ctrl.phase.name}
            )
        elif f % 80 == 0:
            log.append(
                sample_snapshot(s, ram, event=f"tf_{ctrl.phase.name.lower()}_f{f}")
                | {"phase": ctrl.phase.name, "wp_i": ctrl.waypoint_index}
            )
        env.step(action.action)
        if ctrl.phase is PostBossTfPhase.FAILED:
            break

    s = read_snapshot(env.get_ram())
    ram = env.get_ram()
    return {
        "ok": triforce_bit_02(read_u8(ram, ADDR_TRIFORCE)),
        "frames": budget,
        "phase": ctrl.phase.name,
        "policy_live": ctrl.policy_live,
        "waypoints": [list(w) for w in ctrl.waypoints],
        "xy_hist": xy_hist[-20:],
        "log": log[-40:],
        "final": sample_snapshot(s, ram, event="tf_fail"),
        "controller": ctrl.report(),
    }


__all__ = [
    "L2_TF_REACH_JSON",
    "Level2PostBossTfController",
    "PostBossTfPhase",
    "TF_COLLECT_MAX_FRAMES",
    "collect_and_tf",
    "default_tf_waypoints",
    "load_tf_policy",
    "make_post_boss_tf_controller",
    "policy_push",
    "policy_waypoints",
]
