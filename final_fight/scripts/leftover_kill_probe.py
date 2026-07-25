"""Instrument what damages Area2 HP≤8 leftovers (e3/e8 stall).

Reloads preferred 1v1 saves and sweeps attack recipes: face-Y, throws,
jump kicks, B/X, wait-then-Y, grab-close. Logs HP deltas + geometry.

Confirmed kill (slot-tracked): 90f B+RIGHT then grounded toward+Y →
HP underflow (~251) at dx≈56 / cam3968 / sx≈149. Bare Y / X / wait
alone do not kill. Front Y @dx37 whiffs. Dual often spawns during JD.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from final_fight.paths import GAME, GAME_DIR, RECORDINGS_DIR
from final_fight.ram import (
    ENEMY_BASES,
    OFF_HP,
    OFF_STATUS,
    OFF_X,
    OFF_Y,
    parse_game_state,
    read_u8,
    read_u16le,
)
from retro_harness.env import make_env
from snes_oneshot.actions import buttons, idle_action
from snes_oneshot.segment_runner import configure_headless, write_json_report

RecipeFn = Callable[[Any, dict[str, int]], None]

DEFAULT_STATES: tuple[str, ...] = (
    "Stage2_Area2_1v1_p15_e3_cam3914",
    "Stage2_Area2_1v1_p54_e28_cam3969",
    "Stage2_Area2_1v1_p54_e69_cam3900",
)


def _reset(env: Any) -> None:
    result = env.reset()
    if isinstance(result, tuple):
        return


def _snap(env: Any) -> dict[str, Any]:
    ram = env.get_ram()
    state = parse_game_state(ram)
    living: list[dict[str, int]] = []
    for i, base in enumerate(ENEMY_BASES):
        status = read_u8(ram, base + OFF_STATUS)
        hp = read_u8(ram, base + OFF_HP)
        x = read_u16le(ram, base + OFF_X)
        y = read_u16le(ram, base + OFF_Y)
        if status == 3 and 0 < hp <= 192:
            living.append(
                {
                    "slot": i,
                    "status": status,
                    "hp": hp,
                    "x": x,
                    "y": y,
                    "dx": x - state.player_x,
                    "dy": y - state.player_y,
                    "sx": x - state.camera_x,
                }
            )
    enemy = state.nearest_enemy()
    return {
        "frame": state.frame,
        "health": state.health,
        "lives": state.lives,
        "cam": state.camera_x,
        "px": state.player_x,
        "py": state.player_y,
        "sx": state.player_x - state.camera_x,
        "enemy_hp": int(enemy.health) if enemy else 0,
        "dx": (enemy.x - state.player_x) if enemy else 0,
        "dy": (enemy.y - state.player_y) if enemy else 0,
        "living": living,
        "player_dead": state.player_dead,
        "living_count": len(state.living_enemies),
    }


def _hold(env: Any, action: list[int], frames: int) -> None:
    for _ in range(frames):
        env.step(action)


def _idle(env: Any, frames: int) -> None:
    _hold(env, idle_action(), frames)


def _geo(env: Any) -> dict[str, int]:
    s = _snap(env)
    return {
        "hp": int(s["enemy_hp"]),
        "dx": int(s["dx"]),
        "dy": int(s["dy"]),
        "sx": int(s["sx"]),
        "cam": int(s["cam"]),
        "php": int(s["health"]),
    }


def _align_y(env: Any, frames: int = 40) -> None:
    for _ in range(frames):
        s = _snap(env)
        if not s["living"]:
            return
        dy = int(s["dy"])
        if abs(dy) <= 4:
            return
        env.step(buttons("UP") if dy > 0 else buttons("DOWN"))


def _close_to(
    env: Any,
    *,
    target_adx: int,
    frames: int = 90,
    prefer_behind: bool = False,
) -> None:
    """Walk/JD toward punch/grab band without scrolling hard right."""
    for _ in range(frames):
        s = _snap(env)
        if not s["living"]:
            return
        dx = int(s["dx"])
        adx = abs(dx)
        sx = int(s["sx"])
        if prefer_behind and dx > 0 and adx > 20 and sx < 150:
            env.step(buttons("B", "RIGHT"))
            continue
        if adx <= target_adx + 2 and adx >= max(8, target_adx - 4):
            return
        if adx > target_adx + 2:
            toward = "RIGHT" if dx > 0 else "LEFT"
            if adx > 45:
                env.step(buttons("B", toward))
            else:
                env.step(buttons(toward))
            continue
        away = "LEFT" if dx > 0 else "RIGHT"
        if away == "LEFT" and sx < 70:
            away = "RIGHT"
        env.step(buttons(away))


def _recipe_face_y(env: Any, _g0: dict[str, int]) -> None:
    _align_y(env)
    _close_to(env, target_adx=32)
    for _ in range(8):
        s = _snap(env)
        toward = "RIGHT" if int(s["dx"]) > 0 else "LEFT"
        _hold(env, buttons(toward, "Y"), 2)
        _idle(env, 6)


def _recipe_bare_y(env: Any, _g0: dict[str, int]) -> None:
    _align_y(env)
    _close_to(env, target_adx=32)
    for _ in range(8):
        _hold(env, buttons("Y"), 2)
        _idle(env, 6)


def _recipe_throw_toward(env: Any, _g0: dict[str, int]) -> None:
    _align_y(env)
    _close_to(env, target_adx=14)
    for _ in range(10):
        s = _snap(env)
        toward = "RIGHT" if int(s["dx"]) > 0 else "LEFT"
        _hold(env, buttons(toward, "Y"), 3)
        _idle(env, 8)


def _recipe_throw_away(env: Any, _g0: dict[str, int]) -> None:
    _align_y(env)
    _close_to(env, target_adx=14)
    for _ in range(10):
        s = _snap(env)
        away = "LEFT" if int(s["dx"]) > 0 else "RIGHT"
        _hold(env, buttons(away, "Y"), 3)
        _idle(env, 8)


def _recipe_throw_up(env: Any, _g0: dict[str, int]) -> None:
    _align_y(env)
    _close_to(env, target_adx=14)
    for _ in range(10):
        _hold(env, buttons("UP", "Y"), 3)
        _idle(env, 8)


def _recipe_throw_down(env: Any, _g0: dict[str, int]) -> None:
    _align_y(env)
    _close_to(env, target_adx=14)
    for _ in range(10):
        _hold(env, buttons("DOWN", "Y"), 3)
        _idle(env, 8)


def _recipe_jump_kick(env: Any, _g0: dict[str, int]) -> None:
    _align_y(env)
    _close_to(env, target_adx=48)
    for _ in range(6):
        s = _snap(env)
        toward = "RIGHT" if int(s["dx"]) > 0 else "LEFT"
        _hold(env, buttons("B", toward), 4)
        _hold(env, buttons("Y"), 2)
        _idle(env, 10)


def _recipe_b_y(env: Any, _g0: dict[str, int]) -> None:
    _align_y(env)
    _close_to(env, target_adx=40)
    for _ in range(6):
        s = _snap(env)
        toward = "RIGHT" if int(s["dx"]) > 0 else "LEFT"
        _hold(env, buttons("B", toward, "Y"), 3)
        _idle(env, 10)


def _recipe_x(env: Any, _g0: dict[str, int]) -> None:
    _align_y(env)
    _close_to(env, target_adx=32)
    for _ in range(8):
        s = _snap(env)
        toward = "RIGHT" if int(s["dx"]) > 0 else "LEFT"
        _hold(env, buttons(toward, "X"), 2)
        _idle(env, 6)


def _recipe_a(env: Any, _g0: dict[str, int]) -> None:
    _align_y(env)
    _close_to(env, target_adx=32)
    for _ in range(8):
        s = _snap(env)
        toward = "RIGHT" if int(s["dx"]) > 0 else "LEFT"
        _hold(env, buttons(toward, "A"), 2)
        _idle(env, 6)


def _recipe_wait_standup_y(env: Any, _g0: dict[str, int]) -> None:
    """Idle for knockdown recovery, then face-Y spam."""
    _idle(env, 90)
    _align_y(env)
    _close_to(env, target_adx=30)
    for _ in range(10):
        s = _snap(env)
        toward = "RIGHT" if int(s["dx"]) > 0 else "LEFT"
        _hold(env, buttons(toward, "Y"), 2)
        _idle(env, 6)


def _recipe_behind_face_y(env: Any, _g0: dict[str, int]) -> None:
    """JD past into behind, then LEFT+Y (known ~7/hit chip)."""
    _align_y(env)
    _close_to(env, target_adx=28, prefer_behind=True)
    for _ in range(12):
        s = _snap(env)
        dx = int(s["dx"])
        if dx >= 0 and abs(dx) > 20:
            _hold(env, buttons("B", "RIGHT"), 3)
            continue
        _hold(env, buttons("LEFT", "Y"), 2)
        _idle(env, 5)


def _recipe_grab_mash(env: Any, _g0: dict[str, int]) -> None:
    """Walk into grab range and mash toward+Y / away+Y / UP+Y."""
    _align_y(env)
    _close_to(env, target_adx=12)
    for i in range(15):
        s = _snap(env)
        dx = int(s["dx"])
        toward = "RIGHT" if dx > 0 else "LEFT"
        away = "LEFT" if dx > 0 else "RIGHT"
        combo = (toward, "Y")
        if i % 3 == 1:
            combo = (away, "Y")
        elif i % 3 == 2:
            combo = ("UP", "Y")
        _hold(env, buttons(*combo), 3)
        _idle(env, 5)


def _recipe_jd90_toward_y(env: Any, g0: dict[str, int]) -> None:
    """Confirmed crumb kill: 90f B+RIGHT then grounded toward+Y.

    Tracks the starting enemy slot so dual spawn during JD does not
    retarget. Works on HP≤8; probe also tests mid-HP e28/e69 early-open.
    """
    start = _snap(env)
    if not start["living"]:
        return
    # Prefer nearest living at start (1v1).
    primary = min(
        start["living"],
        key=lambda e: abs(int(e["dx"])),
    )
    slot = int(primary["slot"])
    _hold(env, buttons("B", "RIGHT"), 90)
    for _ in range(40):
        s = _snap(env)
        living = s["living"]
        primary_now = next(
            (e for e in living if int(e["slot"]) == slot),
            None,
        )
        if primary_now is None:
            return
        hp = int(primary_now["hp"])
        if hp == 0 or hp > 192:
            return
        dx = int(primary_now["dx"])
        toward = "RIGHT" if dx >= 0 else "LEFT"
        _hold(env, buttons(toward, "Y"), 2)
        _idle(env, 6)


RECIPES: Sequence[tuple[str, RecipeFn]] = (
    ("jd90_toward_y", _recipe_jd90_toward_y),
    ("face_y", _recipe_face_y),
    ("bare_y", _recipe_bare_y),
    ("throw_toward", _recipe_throw_toward),
    ("throw_away", _recipe_throw_away),
    ("throw_up", _recipe_throw_up),
    ("throw_down", _recipe_throw_down),
    ("jump_kick", _recipe_jump_kick),
    ("b_y", _recipe_b_y),
    ("x", _recipe_x),
    ("a", _recipe_a),
    ("wait_standup_y", _recipe_wait_standup_y),
    ("behind_face_y", _recipe_behind_face_y),
    ("grab_mash", _recipe_grab_mash),
)


def run_probe(
    *,
    states: Sequence[str],
    recipes: Sequence[str] | None = None,
    out_dir: Path | None = None,
    observe: int = 20,
) -> dict[str, Any]:
    """Run each recipe from each start state; record HP deltas."""
    configure_headless()
    out = out_dir or (RECORDINGS_DIR / "leftover_kill_probe")
    out.mkdir(parents=True, exist_ok=True)
    wanted = set(recipes) if recipes else None
    trials: list[dict[str, Any]] = []

    for state_name in states:
        for name, fn in RECIPES:
            if wanted is not None and name not in wanted:
                continue
            env = make_env(
                GAME, state_name, GAME_DIR, render_mode="rgb_array"
            )
            try:
                _reset(env)
                g0 = _geo(env)
                if g0["hp"] <= 0 or g0["hp"] > 80:
                    trials.append(
                        {
                            "state": state_name,
                            "recipe": name,
                            "ok": False,
                            "reason": "no_low_hp_enemy",
                            "before": g0,
                        }
                    )
                    continue
                before = _snap(env)
                primary_slot = (
                    int(before["living"][0]["slot"])
                    if before["living"]
                    else -1
                )
                start_hp = g0["hp"]
                fn(env, g0)
                _idle(env, observe)
                g1 = _geo(env)
                after = _snap(env)
                primary_after = next(
                    (
                        e
                        for e in after["living"]
                        if int(e["slot"]) == primary_slot
                    ),
                    None,
                )
                # Slot gone or HP underflow (≥200) counts as kill.
                ram = env.get_ram()
                slot_hp = (
                    read_u8(ram, ENEMY_BASES[primary_slot] + OFF_HP)
                    if 0 <= primary_slot < len(ENEMY_BASES)
                    else 0
                )
                slot_st = (
                    read_u8(
                        ram, ENEMY_BASES[primary_slot] + OFF_STATUS
                    )
                    if 0 <= primary_slot < len(ENEMY_BASES)
                    else 0
                )
                end_hp = (
                    int(primary_after["hp"])
                    if primary_after is not None
                    else slot_hp
                )
                delta = max(0, start_hp - end_hp) if end_hp <= 192 else start_hp
                killed = (
                    primary_after is None
                    or end_hp == 0
                    or end_hp >= 200
                    or slot_st != 3
                    or after["living_count"] == 0
                )
                trials.append(
                    {
                        "state": state_name,
                        "recipe": name,
                        "ok": True,
                        "before": g0,
                        "after": g1,
                        "primary_slot": primary_slot,
                        "slot_hp": slot_hp,
                        "slot_status": slot_st,
                        "living_after": after["living"],
                        "hp_delta": delta,
                        "hit": delta > 0 or killed,
                        "killed": killed,
                        "player_dead": bool(after["player_dead"]),
                        "php_delta": g0["php"] - g1["php"],
                    }
                )
            finally:
                env.close()

    hits = [t for t in trials if t.get("hit")]
    kills = [t for t in trials if t.get("killed")]
    report: dict[str, Any] = {
        "states": list(states),
        "trials": trials,
        "hit_count": len(hits),
        "kill_count": len(kills),
        "hits": [
            {
                "state": t["state"],
                "recipe": t["recipe"],
                "hp_delta": t["hp_delta"],
                "before": t["before"],
                "after": t["after"],
                "killed": t["killed"],
            }
            for t in hits
        ],
        "kills": [
            {
                "state": t["state"],
                "recipe": t["recipe"],
                "before": t["before"],
                "after": t["after"],
            }
            for t in kills
        ],
    }
    path = write_json_report(out / "leftover_kill.json", report)
    report["report_path"] = str(path)
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI for Area2 leftover damage instrumentation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state",
        action="append",
        dest="states",
        default=None,
        help="Save state (repeatable); default preferred 1v1 trio",
    )
    parser.add_argument(
        "--recipe",
        action="append",
        dest="recipes",
        default=None,
        help="Recipe name filter (repeatable)",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run_probe(
        states=tuple(args.states or DEFAULT_STATES),
        recipes=args.recipes,
        out_dir=args.out_dir,
    )
    print(f"hit_count={report['hit_count']} kill_count={report['kill_count']}")
    for h in report["hits"]:
        b = h["before"]
        a = h["after"]
        print(
            f"  HIT {h['recipe']} @ {h['state']}: "
            f"hp {b['hp']}→{a['hp']} (Δ{h['hp_delta']}) "
            f"dx {b['dx']}→{a['dx']} sx {b['sx']}→{a['sx']} "
            f"killed={h['killed']}"
        )
    if not report["hits"]:
        print("  (no HP damage observed)")
        for t in report["trials"][:6]:
            if not t.get("ok"):
                continue
            b = t["before"]
            a = t["after"]
            print(
                f"  miss {t['recipe']} @ {t['state']}: "
                f"hp {b['hp']}→{a['hp']} dx {b['dx']}→{a['dx']} "
                f"sx {b['sx']}→{a['sx']}"
            )
    print(f"report={report['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
