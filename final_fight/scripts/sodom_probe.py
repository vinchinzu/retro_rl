"""Instrument / chip / kill Sodom from Boss2_Drawn / Mid states.

Sodom (``0x11E0=03``, HP44) one-shots with chains when ``a1>=8`` at
dx≈80–90. Working kill (repro 3/3 from ``Boss2_Drawn``):

1. Spaced ``UP+Y`` at dx≤65 chips ~1 HP (44→43→42→41).
2. Fourth connect is a grab/throw dealing **40** (41→1) at dx≈−118.
3. Immediately cycle grab dirs
   (``UP+Y`` / ``DOWN+Y`` / ``LEFT+Y`` / ``RIGHT+Y`` / ``Y`` / ``B+Y``)
   → HP underflow (~254) → save ``Stage2_Clear``.

Cold Drawn 2-hit+LEFT flee still saves ``Boss2_Mid_b42_p37``. Woken Mid
cannot LEFT-flee-save; prefer full UP+Y kill from Drawn.

Evidence: ``recordings/sodom_upy_finish/``, ``Stage2_Clear.state``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from final_fight.paths import GAME, GAME_DIR, RECORDINGS_DIR
from final_fight.ram import (
    ADDR_BOSS_DEAD_FLAG,
    ADDR_GAME_STATUS,
    BOSS_BASE,
    GameStatus,
    OFF_HP,
    OFF_STATUS,
    OFF_X,
    OFF_Y,
    parse_game_state,
    read_u8,
    read_u16le,
)
from retro_harness.env import get_available_states, make_env, save_state
from snes_oneshot.actions import buttons, idle_action
from snes_oneshot.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)

# Post-throw mash that finishes HP1 → underflow (deterministic).
_GRAB_CYCLE: tuple[list[int], ...] = (
    buttons("UP", "Y"),
    buttons("DOWN", "Y"),
    buttons("LEFT", "Y"),
    buttons("RIGHT", "Y"),
    buttons("Y"),
    buttons("B", "Y"),
)


def _boss(ram: Any) -> dict[str, int]:
    return {
        "st": read_u8(ram, BOSS_BASE + OFF_STATUS),
        "hp": read_u8(ram, BOSS_BASE + OFF_HP),
        "x": read_u16le(ram, BOSS_BASE + OFF_X),
        "y": read_u16le(ram, BOSS_BASE + OFF_Y),
        "a1": read_u8(ram, BOSS_BASE + 0x01),
        "a2": read_u8(ram, BOSS_BASE + 0x02),
        "cd2": read_u8(ram, ADDR_BOSS_DEAD_FLAG),
        "gs": read_u8(ram, ADDR_GAME_STATUS),
    }


def _is_boss_kill(b: dict[str, int], *, hits: int) -> bool:
    """True when Sodom HP underflow / zero after damage."""
    if b["hp"] > 200:
        return True
    return b["st"] == 3 and b["hp"] == 0 and hits > 0


def run_upy_kill(
    *,
    state_name: str = "Boss2_Drawn",
    hit_dx: int = 65,
    reset_dx: int = 115,
    max_frames: int = 8000,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Defeat Sodom with spaced UP+Y then grab-dir mash.

    Saves ``Stage2_Clear`` on HP underflow. Prefer cold ``Boss2_Drawn``.
    """
    configure_headless()
    available = get_available_states(GAME, GAME_DIR)
    if state_name not in available:
        raise SystemExit(f"missing state {state_name}; have {available[:8]}")
    out = out_dir or (RECORDINGS_DIR / "sodom_upy_finish")
    out.mkdir(parents=True, exist_ok=True)

    env = make_env(GAME, state_name, GAME_DIR, render_mode="rgb_array")
    env.reset()
    ram = env.get_ram()
    st = parse_game_state(ram)
    b0 = _boss(ram)
    php0, bhp0 = st.health, b0["hp"]
    prev_bhp = bhp0
    prev_php = php0
    phase = "close"
    hits = 0
    hit_log: list[dict[str, int]] = []
    php_log: list[dict[str, int]] = []
    min_bhp = bhp0
    min_php = php0
    big_hit = False
    grab_i = 0
    outcome = "timeout"
    saved: list[str] = []
    screenshots: list[str] = []

    for frame in range(1, max_frames + 1):
        ram = env.get_ram()
        st = parse_game_state(ram)
        b = _boss(ram)
        dx = b["x"] - st.player_x
        adx = abs(dx)
        dy = b["y"] - st.player_y
        sx = st.player_x - st.camera_x

        if big_hit:
            act = _GRAB_CYCLE[grab_i % len(_GRAB_CYCLE)]
            grab_i += 1
        elif b["a1"] >= 8 or (
            70 <= adx <= 100 and phase == "retreat"
        ):
            act = buttons("UP") if sx < 48 else buttons("UP", "LEFT")
            phase = "close"
        elif sx > 170:
            act = buttons("LEFT")
        elif phase == "close":
            if abs(dy) > 10:
                act = buttons("UP" if dy > 0 else "DOWN")
            elif adx <= hit_dx:
                phase = "hit"
                act = buttons("UP", "Y")
            else:
                act = buttons("RIGHT" if dx > 0 else "LEFT")
        elif phase == "hit":
            phase = "retreat"
            act = buttons("UP", "Y")
        else:
            if adx < reset_dx:
                act = buttons("LEFT" if dx > 0 else "RIGHT")
            else:
                phase = "close"
                act = idle_action()

        env.step(act)
        ram = env.get_ram()
        st = parse_game_state(ram)
        b = _boss(ram)
        dx = b["x"] - st.player_x

        if 0 < st.health <= 128 and st.health < prev_php:
            php_log.append(
                {
                    "f": frame,
                    "php": st.health,
                    "bhp": b["hp"],
                    "dx": dx,
                    "a1": b["a1"],
                }
            )
            min_php = min(min_php, st.health)
        prev_php = st.health if 0 < st.health <= 128 else prev_php

        if b["hp"] < prev_bhp:
            dmg = prev_bhp - b["hp"]
            hits += 1
            if 0 < b["hp"] <= 192:
                min_bhp = b["hp"]
            hit_log.append(
                {
                    "f": frame,
                    "bhp": b["hp"],
                    "dx": dx,
                    "php": st.health,
                    "a1": b["a1"],
                    "dmg": dmg,
                }
            )
            print(
                f"HIT#{hits} f{frame} {prev_bhp}->{b['hp']} "
                f"dmg={dmg} dx={dx} a1={b['a1']} php={st.health}"
            )
            if dmg >= 10 and 0 < b["hp"] <= 192:
                big_hit = True
                grab_i = 0
                mid = f"Boss2_Mid_b{b['hp']}_p{st.health}"
                saved.append(save_state(env, GAME_DIR, GAME, mid).name)
                print(f"  BIG throw saved {saved[-1]}")
        prev_bhp = b["hp"]

        if st.health > 128:
            outcome = "death"
            screenshots.append(
                save_rgb_png(env.render(), out / "upy_death.png").name
            )
            print(
                f"DEATH f{frame} dx={dx} a1={b['a1']} "
                f"bhp={b['hp']} hits={hits}"
            )
            break

        if _is_boss_kill(b, hits=hits):
            path = save_state(env, GAME_DIR, GAME, "Stage2_Clear")
            saved.append(path.name)
            screenshots.append(
                save_rgb_png(env.render(), out / "Stage2_Clear.png").name
            )
            outcome = "boss_uf"
            print(f"KILL UF {path.name} hp={b['hp']} php={st.health}")
            break
        if b["gs"] == int(GameStatus.CLEAR_ROUND) or b["cd2"] == 1:
            path = save_state(env, GAME_DIR, GAME, "Stage2_Clear")
            saved.append(path.name)
            screenshots.append(
                save_rgb_png(env.render(), out / "Stage2_Clear.png").name
            )
            outcome = "clear_round"
            print(f"CLEAR {path.name}")
            break

    env.close()
    report: dict[str, Any] = {
        "success": outcome in ("boss_uf", "clear_round"),
        "outcome": outcome,
        "method": "upy_throw_grab_cycle",
        "state": state_name,
        "php0": php0,
        "bhp0": bhp0,
        "hits": hits,
        "hit_log": hit_log,
        "php_log": php_log,
        "min_bhp": min_bhp,
        "min_php": min_php,
        "boss_dmg": bhp0 - min_bhp if min_bhp <= bhp0 else 0,
        "saved": saved,
        "screenshots": screenshots,
        "notes": (
            "UP+Y spaced chips then 40-dmg throw (41→1); grab-dir "
            "cycle finishes underflow. Chains a1>=8 still one-shot "
            "if mash is skipped after the throw."
        ),
    }
    write_json_report(out / "sodom_upy_kill.json", report)
    print(
        f"outcome={outcome} hits={hits} bhp {bhp0}->{min_bhp} "
        f"php {php0}->{min_php} saved={saved}"
    )
    return report


def run_drawn_mid_chip(
    *,
    state_name: str = "Boss2_Drawn",
    max_hits: int = 2,
    flee_frames: int = 280,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Chip Sodom then LEFT-flee; save Mid when dx≥150.

    Works from cold ``Boss2_Drawn``. Mid resumes usually cannot flee-save.
    """
    configure_headless()
    available = get_available_states(GAME, GAME_DIR)
    if state_name not in available:
        raise SystemExit(f"missing state {state_name}; have {available[:8]}")
    out = out_dir or (RECORDINGS_DIR / "sodom_probe")
    out.mkdir(parents=True, exist_ok=True)

    env = make_env(GAME, state_name, GAME_DIR, render_mode="rgb_array")
    env.reset()
    ram = env.get_ram()
    st = parse_game_state(ram)
    b0 = _boss(ram)
    php0, bhp0 = st.health, b0["hp"]
    phase = "close"
    prev_bhp = bhp0
    hits = 0
    hit_log: list[dict[str, int]] = []
    fleeing = False
    flee_i = 0
    outcome = "timeout"
    saved: list[str] = []
    screenshots: list[str] = []

    for frame in range(1, 5000):
        ram = env.get_ram()
        st = parse_game_state(ram)
        b = _boss(ram)
        dx = b["x"] - st.player_x
        adx = abs(dx)
        dy = b["y"] - st.player_y
        sx = st.player_x - st.camera_x

        if fleeing:
            flee_i += 1
            act = buttons("LEFT") if sx >= 48 else idle_action()
            if (
                flee_i >= flee_frames
                and adx >= 150
                and 0 < st.health <= 128
            ):
                name = f"Boss2_Mid_b{b['hp']}_p{st.health}"
                path = save_state(env, GAME_DIR, GAME, name)
                saved.append(path.name)
                screenshots.append(
                    save_rgb_png(env.render(), out / f"{name}.png").name
                )
                outcome = "mid_saved"
                break
        elif sx > 170:
            act = buttons("LEFT")
        elif phase == "close":
            if abs(dy) > 10:
                act = buttons("UP" if dy > 0 else "DOWN")
            elif adx <= 65:
                phase = "hit"
                act = buttons("Y")
            else:
                toward = "RIGHT" if dx > 0 else "LEFT"
                act = buttons(toward)
        elif phase == "hit":
            phase = "retreat"
            act = buttons("Y")
        else:
            if adx < 110:
                away = "LEFT" if dx > 0 else "RIGHT"
                act = buttons(away)
            else:
                phase = "close"
                act = idle_action()

        env.step(act)
        ram = env.get_ram()
        st = parse_game_state(ram)
        b = _boss(ram)
        dx = b["x"] - st.player_x

        if b["hp"] < prev_bhp and 0 < b["hp"] <= 192:
            hits += 1
            hit_log.append(
                {
                    "f": frame,
                    "bhp": b["hp"],
                    "dx": dx,
                    "php": st.health,
                    "a1": b["a1"],
                }
            )
            print(
                f"HIT#{hits} f{frame} {prev_bhp}->{b['hp']} "
                f"dx={dx} a1={b['a1']} php={st.health}"
            )
            prev_bhp = b["hp"]
            if hits >= max_hits:
                fleeing = True
                flee_i = 0
        else:
            prev_bhp = b["hp"]

        if st.health > 128:
            outcome = "death"
            screenshots.append(
                save_rgb_png(
                    env.render(), out / f"death_{frame:04d}.png"
                ).name
            )
            print(f"DEATH f{frame} dx={dx} a1={b['a1']} bhp={b['hp']}")
            break

        if _is_boss_kill(b, hits=hits):
            path = save_state(env, GAME_DIR, GAME, "Stage2_Clear")
            saved.append(path.name)
            outcome = "boss_uf"
            break
        if b["gs"] == int(GameStatus.CLEAR_ROUND) or b["cd2"] == 1:
            path = save_state(env, GAME_DIR, GAME, "Stage2_Clear")
            saved.append(path.name)
            outcome = "clear_round"
            break

    env.close()
    report: dict[str, Any] = {
        "success": outcome in ("mid_saved", "boss_uf", "clear_round"),
        "outcome": outcome,
        "state": state_name,
        "php0": php0,
        "bhp0": bhp0,
        "hits": hits,
        "hit_log": hit_log,
        "min_bhp": prev_bhp,
        "boss_dmg": bhp0 - prev_bhp if prev_bhp <= bhp0 else 0,
        "saved": saved,
        "screenshots": screenshots,
        "notes": (
            "Spaced Y at dx≈65 deals ~1 HP; prefer --mode kill "
            "(UP+Y throw) for Stage2_Clear."
        ),
    }
    write_json_report(out / "sodom_probe.json", report)
    print(
        f"outcome={outcome} hits={hits} bhp {bhp0}->{prev_bhp} "
        f"saved={saved}"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("kill", "chip"),
        default="kill",
    )
    parser.add_argument("--state", default="Boss2_Drawn")
    parser.add_argument("--hits", type=int, default=2)
    parser.add_argument("--flee", type=int, default=280)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.mode == "kill":
        run_upy_kill(state_name=args.state, out_dir=args.out_dir)
    else:
        run_drawn_mid_chip(
            state_name=args.state,
            max_hits=args.hits,
            flee_frames=args.flee,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
