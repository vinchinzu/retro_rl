"""Scene classification and recovery heuristics for golf autoplay."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hals_golf.core.ram import GameScene, GolfSnapshot, snapshot_from_ram


@dataclass(frozen=True)
class SceneDecision:
    """Classifier output used by the mission bot."""

    scene: GameScene
    needs_dismiss: bool
    wait_only: bool
    reason: str


def _mean_brightness(obs: np.ndarray | None) -> float:
    if obs is None or obs.size == 0:
        return 0.0
    return float(np.mean(obs))


def _edge_activity(obs: np.ndarray | None) -> float:
    """Cheap motion/UI proxy from frame edges."""
    if obs is None or obs.ndim < 2 or obs.shape[0] < 40:
        return 0.0
    band = 16
    top = obs[:band].astype(np.int16)
    bottom = obs[-band:].astype(np.int16)
    return float(np.mean(np.abs(top - bottom)))


def is_command_screen(obs: np.ndarray | None) -> bool:
    """Recognize the right-side ``SHOT / GREEN / HOLE`` command panel."""
    if obs is None or obs.ndim != 3 or obs.shape[0] < 205 or obs.shape[1] < 256:
        return False
    panel = obs[160:205, 200:256]
    blue = (
        (panel[:, :, 2] > 180)
        & (panel[:, :, 1] >= 50)
        & (panel[:, :, 0] < 20)
    )
    bright = np.max(panel, axis=2) > 220
    dark = np.max(panel, axis=2) < 16
    return (
        int(np.count_nonzero(blue)) >= 280
        and int(np.count_nonzero(bright)) >= 280
        and int(np.count_nonzero(dark)) >= 550
    )


def classify_scene(
    ram: np.ndarray,
    *,
    info: dict | None = None,
    obs: np.ndarray | None = None,
    previous: GolfSnapshot | None = None,
) -> SceneDecision:
    """Classify the current golf scene from RAM / frame cues.

    Early versions lean on stroke/hole stability plus brightness heuristics.
    Menu RAM fields are refined by ``scripts/probe_menus.py`` over time.
    """
    snap = snapshot_from_ram(ram, info=info)
    brightness = _mean_brightness(obs)
    edge = _edge_activity(obs)

    # Title / attract screens tend to be darker title art with little HUD.
    if snap.hole_number == 0 and snap.stroke_count in (0, 255, 85):
        if brightness < 40:
            return SceneDecision(
                GameScene.TITLE, needs_dismiss=False, wait_only=False, reason="dark_title"
            )
        return SceneDecision(
            GameScene.MODE_SELECT,
            needs_dismiss=False,
            wait_only=False,
            reason="pre_round_menu",
        )

    # In-round play: hole 1..18 and modest stroke counts.
    if 1 <= snap.hole_number <= 18 and snap.stroke_count <= 20:
        if previous is not None and previous.stroke_count != snap.stroke_count:
            return SceneDecision(
                GameScene.BALL_FLIGHT,
                needs_dismiss=False,
                wait_only=True,
                reason="stroke_changed",
            )
        if is_command_screen(obs) or edge > 35:
            return SceneDecision(
                GameScene.COMMAND,
                needs_dismiss=False,
                wait_only=False,
                reason="hud_active",
            )
        return SceneDecision(
            GameScene.TRANSITION,
            needs_dismiss=False,
            wait_only=True,
            reason="quiet_transition",
        )

    return SceneDecision(
        GameScene.UNKNOWN, needs_dismiss=False, wait_only=False, reason="unclassified"
    )


def snapshot_with_scene(
    ram: np.ndarray,
    *,
    info: dict | None = None,
    obs: np.ndarray | None = None,
    previous: GolfSnapshot | None = None,
) -> tuple[GolfSnapshot, SceneDecision]:
    """Return snapshot + scene decision together."""
    decision = classify_scene(ram, info=info, obs=obs, previous=previous)
    snap = snapshot_from_ram(ram, info=info, scene=decision.scene)
    return snap, decision
