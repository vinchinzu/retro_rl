"""Sweep punch/throw connect distance against the first Stage1 thug."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from final_fight.paths import GAME, GAME_DIR, RECORDINGS_DIR, STAGE1_STATE
from final_fight.ram import parse_game_state
from retro_harness.env import make_env
from snes_oneshot.actions import buttons, idle_action
from snes_oneshot.segment_runner import configure_headless, write_json_report


def _reset(env: Any) -> None:
    result = env.reset()
    if isinstance(result, tuple):
        return


def _enemy_hp(state: Any) -> int:
    living = state.living_enemies
    if not living:
        return 0
    return int(living[0].health)


def _align_and_place(
    env: Any,
    *,
    target_dx: int,
    invert_y: bool = True,
    max_frames: int = 240,
) -> tuple[Any, int]:
    """Walk until nearest enemy is roughly target_dx away and Y-aligned."""
    for _ in range(max_frames):
        state = parse_game_state(env.get_ram())
        enemy = state.nearest_enemy()
        if enemy is None:
            env.step(buttons("RIGHT"))
            continue
        dy = enemy.y - state.player_y
        if abs(dy) > 4:
            press_up = dy > 0 if invert_y else dy < 0
            env.step(buttons("UP" if press_up else "DOWN"))
            continue
        dx = enemy.x - state.player_x
        if abs(dx) > target_dx + 2:
            env.step(buttons("RIGHT" if dx > 0 else "LEFT"))
            continue
        if abs(dx) < target_dx - 2:
            env.step(buttons("LEFT" if dx > 0 else "RIGHT"))
            continue
        return state, abs(dx)
    return parse_game_state(env.get_ram()), -1


def run_sweep(
    *,
    distances: list[int] | None = None,
    hold: int = 2,
    observe: int = 10,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """For each distance, reload Stage1, approach, punch, record HP delta."""
    configure_headless()
    distances = distances or [12, 16, 20, 24, 28, 32, 36, 40]
    out = out_dir or (RECORDINGS_DIR / "punch_range_probe")
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for dist in distances:
        env = make_env(GAME, STAGE1_STATE, GAME_DIR, render_mode="rgb_array")
        try:
            _reset(env)
            # Let spawn settle.
            for _ in range(30):
                env.step(idle_action())
            state, actual = _align_and_place(env, target_dx=dist)
            enemy = state.nearest_enemy()
            if enemy is None or actual < 0:
                rows.append(
                    {
                        "target_dx": dist,
                        "actual_dx": actual,
                        "ok": False,
                        "hp_delta": 0,
                    }
                )
                continue
            hp0 = _enemy_hp(state)
            for _ in range(hold):
                env.step(buttons("Y"))
            for _ in range(observe):
                env.step(idle_action())
            state2 = parse_game_state(env.get_ram())
            hp1 = _enemy_hp(state2)
            # Kill / despawn counts as full remaining.
            if hp1 == 0 and hp0 > 0 and not state2.living_enemies:
                delta = hp0
            else:
                delta = max(0, hp0 - hp1)
            rows.append(
                {
                    "target_dx": dist,
                    "actual_dx": actual,
                    "ok": True,
                    "hp0": hp0,
                    "hp1": hp1,
                    "hp_delta": delta,
                    "hit": delta > 0,
                    "dy": enemy.y - state.player_y,
                    "player_x": state.player_x,
                    "enemy_x": enemy.x,
                }
            )
        finally:
            env.close()

    # Throw sweep at a few close distances.
    throw_rows: list[dict[str, Any]] = []
    for dist in [10, 14, 18, 22]:
        env = make_env(GAME, STAGE1_STATE, GAME_DIR, render_mode="rgb_array")
        try:
            _reset(env)
            for _ in range(30):
                env.step(idle_action())
            state, actual = _align_and_place(env, target_dx=dist)
            enemy = state.nearest_enemy()
            if enemy is None or actual < 0:
                throw_rows.append(
                    {"target_dx": dist, "actual_dx": actual, "hit": False}
                )
                continue
            hp0 = _enemy_hp(state)
            toward = "RIGHT" if enemy.x >= state.player_x else "LEFT"
            for _ in range(hold):
                env.step(buttons(toward, "Y"))
            for _ in range(observe + 8):
                env.step(idle_action())
            state2 = parse_game_state(env.get_ram())
            hp1 = _enemy_hp(state2)
            if hp1 == 0 and hp0 > 0 and not state2.living_enemies:
                delta = hp0
            else:
                delta = max(0, hp0 - hp1)
            throw_rows.append(
                {
                    "target_dx": dist,
                    "actual_dx": actual,
                    "hp_delta": delta,
                    "hit": delta > 0,
                }
            )
        finally:
            env.close()

    report = {
        "punch": rows,
        "throw": throw_rows,
        "punch_connect_max_dx": max(
            (r["actual_dx"] for r in rows if r.get("hit")),
            default=None,
        ),
        "throw_connect_max_dx": max(
            (r["actual_dx"] for r in throw_rows if r.get("hit")),
            default=None,
        ),
    }
    path = write_json_report(out / "punch_range.json", report)
    report["report_path"] = str(path)
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI for punch/throw range sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run_sweep(out_dir=args.out_dir)
    print("punch:")
    for r in report["punch"]:
        print(
            f"  dx~{r.get('actual_dx')} target={r['target_dx']} "
            f"hit={r.get('hit')} delta={r.get('hp_delta')}"
        )
    print("throw:")
    for r in report["throw"]:
        print(
            f"  dx~{r.get('actual_dx')} target={r['target_dx']} "
            f"hit={r.get('hit')} delta={r.get('hp_delta')}"
        )
    print(
        f"punch_connect_max_dx={report['punch_connect_max_dx']} "
        f"throw_connect_max_dx={report['throw_connect_max_dx']}"
    )
    print(f"report={report['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
