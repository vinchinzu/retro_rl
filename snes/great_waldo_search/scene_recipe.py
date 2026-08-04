"""Shared scene-clear recipes for Great Waldo Search recordings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from great_waldo_search.targets import (
    CURSOR_X_ADDR,
    CURSOR_Y_ADDR,
    FOUND_FLAG_ADDR,
    SCENE1_CLEAR_SCORE,
    SCENE2_CLEAR_SCORE,
    SCENE2_SCROLL_SCORE,
    SCENE3_CLEAR_SCORE,
    SCENE3_SCROLL_SCORE,
    SCENE4_CLEAR_SCORE,
    SCENE4_SCROLL_SCORE,
    SCENE5_CLEAR_SCORE,
    SCENE5_SCROLL_SCORE,
    SCORE_HI_ADDR,
    SCORE_LO_ADDR,
    WALDO_POINTS,
    score_u16,
)
from retro_harness.actions import buttons_multi
from retro_harness.video import CaptureSession
from retro_harness.cursor import CursorPose, CursorTarget, step_toward_target


@dataclass(frozen=True)
class SceneBanner:
    """Footer labels for one search scene."""

    number: int
    name: str


@dataclass(frozen=True)
class SceneRecipe:
    """One proven scene clear with default coordinates."""

    banner: SceneBanner
    state: str
    scroll_p2a: int = 0
    scroll_x: int = 0
    scroll_y: int = 100
    pan_frames: int = 0
    waldo_p2a: int = 0
    waldo_x: int = 0
    waldo_y: int = 0
    scroll_score: int = 0
    clear_score: int = 0
    drive_scroll: bool = False
    settle_warm: int = 100
    settle_samples: int = 5
    settle_gap: int = 8


SCENE_RECIPES: tuple[SceneRecipe, ...] = (
    SceneRecipe(
        banner=SceneBanner(1, "Flying Carpets"),
        state="Scene1",
        scroll_p2a=300,
        scroll_x=32,
        scroll_y=100,
        pan_frames=80,
        waldo_x=36,
        waldo_y=28,
        scroll_score=1000,
        clear_score=SCENE1_CLEAR_SCORE,
    ),
    SceneRecipe(
        banner=SceneBanner(2, "Underground Hunters"),
        state="Scene2",
        scroll_x=224,
        scroll_y=100,
        waldo_p2a=500,
        waldo_x=32,
        waldo_y=120,
        scroll_score=SCENE2_SCROLL_SCORE,
        clear_score=SCENE2_CLEAR_SCORE,
        drive_scroll=True,
    ),
    SceneRecipe(
        banner=SceneBanner(3, "Battling Monks"),
        state="Scene3",
        scroll_p2a=300,
        scroll_x=160,
        scroll_y=100,
        waldo_p2a=200,
        # Continuous favorable layout clicks at 196; 198 misses.
        waldo_x=196,
        waldo_y=100,
        scroll_score=SCENE3_SCROLL_SCORE,
        clear_score=SCENE3_CLEAR_SCORE,
    ),
    SceneRecipe(
        banner=SceneBanner(4, "Unfriendly Giants"),
        state="Scene4",
        scroll_p2a=500,
        scroll_x=34,
        scroll_y=100,
        waldo_p2a=500,
        waldo_x=196,
        waldo_y=140,
        scroll_score=SCENE4_SCROLL_SCORE,
        clear_score=SCENE4_CLEAR_SCORE,
    ),
    SceneRecipe(
        banner=SceneBanner(5, "Land of Waldos"),
        state="Scene5",
        scroll_p2a=300,
        scroll_x=32,
        scroll_y=100,
        waldo_p2a=500,
        waldo_x=180,
        waldo_y=60,
        scroll_score=SCENE5_SCROLL_SCORE,
        clear_score=SCENE5_CLEAR_SCORE,
        settle_warm=220,
        settle_samples=8,
        settle_gap=12,
    ),
)


def _metrics(env: object) -> dict[str, int]:
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    return {
        "score": score_u16(ram[SCORE_LO_ADDR], ram[SCORE_HI_ADDR]),
        "found": int(ram[FOUND_FLAG_ADDR]),
        "x": int(ram[CURSOR_X_ADDR]),
        "y": int(ram[CURSOR_Y_ADDR]),
    }


def _settle(
    session: CaptureSession,
    env: object,
    *,
    warm: int,
    samples: int,
    gap: int,
) -> tuple[dict[str, int], bool]:
    session.idle(warm)
    scores: list[int] = []
    last = _metrics(env)
    for _ in range(samples):
        last = _metrics(env)
        scores.append(last["score"])
        session.idle(gap)
    return last, len(set(scores)) == 1


def _drive(
    session: CaptureSession,
    env: object,
    target: CursorTarget,
    *,
    frames: int = 800,
) -> CursorPose:
    for _ in range(frames):
        ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
        pose = CursorPose(int(ram[CURSOR_X_ADDR]), int(ram[CURSOR_Y_ADDR]))
        action = step_toward_target(pose, target, fast_button="Y")
        if action.reason == "confirm_at_target":
            return pose
        multi = buttons_multi()
        multi[:12] = list(action.action)
        session.step(multi)
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    return CursorPose(int(ram[CURSOR_X_ADDR]), int(ram[CURSOR_Y_ADDR]))


def _click_a(session: CaptureSession, *, hold: int = 6) -> None:
    for _ in range(hold):
        session.step(buttons_multi(p1=("A",)))


def _scroll_find(
    session: CaptureSession,
    env: object,
    recipe: SceneRecipe,
) -> dict[str, int]:
    """Run the scroll-objective half of a scene recipe."""
    if recipe.banner.number == 1:
        session.hold(buttons_multi(p2=("A",)), recipe.scroll_p2a)
        session.idle(4)
        _click_a(session)
        after_scroll, scroll_ok = _settle(
            session,
            env,
            warm=recipe.settle_warm,
            samples=recipe.settle_samples,
            gap=recipe.settle_gap,
        )
        if not (
            scroll_ok
            and after_scroll["score"] >= recipe.scroll_score
            and after_scroll["found"] == 2
        ):
            raise RuntimeError(
                f"scroll find failed in {recipe.state}: {after_scroll}"
            )
        if recipe.pan_frames:
            _drive(session, env, CursorTarget(x=240, y=100, deadzone=3))
            session.hold(buttons_multi(p1=("RIGHT", "Y")), recipe.pan_frames)
            session.idle(6)
        return after_scroll

    if recipe.drive_scroll:
        _drive(
            session,
            env,
            CursorTarget(x=recipe.scroll_x, y=recipe.scroll_y, deadzone=2),
        )
        _click_a(session)
    else:
        session.hold(buttons_multi(p2=("A",)), recipe.scroll_p2a)
        session.idle(6)
        _drive(
            session,
            env,
            CursorTarget(x=recipe.scroll_x, y=recipe.scroll_y, deadzone=2),
        )
        _click_a(session)

    after_scroll, scroll_ok = _settle(
        session,
        env,
        warm=recipe.settle_warm,
        samples=recipe.settle_samples,
        gap=recipe.settle_gap,
    )
    if not (
        scroll_ok
        and after_scroll["score"] >= recipe.scroll_score
        and after_scroll["found"] == 2
    ):
        raise RuntimeError(f"scroll find failed in {recipe.state}: {after_scroll}")
    return after_scroll


def _waldo_find(
    session: CaptureSession,
    env: object,
    recipe: SceneRecipe,
) -> dict[str, int]:
    """Run the Waldo-objective half of a scene recipe."""
    if recipe.waldo_p2a:
        session.hold(buttons_multi(p2=("A",)), recipe.waldo_p2a)
        session.idle(6)

    before_waldo = _metrics(env)
    _drive(
        session,
        env,
        CursorTarget(x=recipe.waldo_x, y=recipe.waldo_y, deadzone=2),
    )
    _click_a(session)
    after_waldo, waldo_ok = _settle(
        session,
        env,
        warm=recipe.settle_warm,
        samples=recipe.settle_samples,
        gap=recipe.settle_gap,
    )
    delta = after_waldo["score"] - before_waldo["score"]
    if not (
        waldo_ok
        and (
            after_waldo["score"] >= recipe.clear_score
            or delta >= WALDO_POINTS
        )
    ):
        raise RuntimeError(f"waldo clear failed in {recipe.state}: {after_waldo}")
    return after_waldo


def run_scene_recipe(
    session: CaptureSession,
    env: object,
    recipe: SceneRecipe,
) -> dict[str, object]:
    """Execute one scene clear while recording every stepped frame."""
    summary: dict[str, object] = {
        "scene": recipe.banner.number,
        "state": recipe.state,
        "name": recipe.banner.name,
    }
    session.idle(10)
    summary["load"] = _metrics(env)
    summary["after_scroll"] = _scroll_find(session, env, recipe)
    summary["after_waldo"] = _waldo_find(session, env, recipe)
    summary["cleared"] = True
    if recipe.banner.number == 5:
        after_waldo = summary["after_waldo"]
        assert isinstance(after_waldo, dict)
        summary["ending"] = after_waldo["score"] >= recipe.clear_score
    return summary
