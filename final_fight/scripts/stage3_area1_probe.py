"""Probe / develop West Side Area1 HP≈250 thug clear.

Loads ``Stage3_Area1_hp50_L1_cam2560`` (or a mid resume). The preferred
chip recipe is **face-then-Y**:

- Face LEFT briefly (do not hold LEFT every frame — that walks into the
  gutter and dies).
- Pulse bare ``Y`` (2/12 frames) while the thug sits behind at dx≈−40…−70.
- Continuous ``LEFT+Y`` deals **0** damage (animation lock).
- First hit lands ~f120–140 after spawn; first combo ~23 dmg, then more
  as the thug re-enters band.

Best observed chip with heal pokes: **250 → ~101** (~149 dmg) before
death. Full legit kill still open.

After a kill (living=0), plant-punch HP0 st=03 ghosts, then
``CLEAR_AREA`` advances to Area2 / **Boss3** entry (cam≈3072,
``0x11E0=01``) — same softlock bridge pattern as subway / area0 West.

Usage::

    uv run python final_fight/scripts/stage3_area1_probe.py
    uv run python final_fight/scripts/stage3_area1_probe.py \\
        --state Stage3_Area1_mid_p70_e101_cam2560 --heal-hp 70
    uv run python final_fight/scripts/stage3_area1_probe.py \\
        --force-enemy-hp 8   # map post-kill / Boss3 only (dev)
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
    ENEMY_BASES,
    ENTITY_HP_MAX,
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

DEFAULT_STATE = "Stage3_Area1_hp50_L1_cam2560"


def _snap(env: Any) -> tuple[Any, dict[str, int] | None, list[dict[str, int]], list[dict[str, int]]]:
    ram = env.get_ram()
    state = parse_game_state(ram)
    living: list[dict[str, int]] = []
    ghosts: list[dict[str, int]] = []
    for i, base in enumerate(ENEMY_BASES):
        st = read_u8(ram, base + OFF_STATUS)
        hp = read_u8(ram, base + OFF_HP)
        x = read_u16le(ram, base + OFF_X)
        y = read_u16le(ram, base + OFF_Y)
        dx = x - state.player_x
        dy = y - state.player_y
        if st in (1, 3) and 0 < hp <= ENTITY_HP_MAX:
            living.append(
                {"slot": i, "st": st, "hp": hp, "dx": dx, "dy": dy, "x": x, "y": y}
            )
        if st == 3 and (hp == 0 or hp > ENTITY_HP_MAX) and abs(dx) < 130:
            ghosts.append({"slot": i, "hp": hp, "dx": dx})
    enemy = max(living, key=lambda e: e["hp"]) if living else None
    return state, enemy, living, ghosts


def _face_y_action(
    frame: int,
    enemy: dict[str, int] | None,
    *,
    faced: bool,
) -> tuple[Any, str, bool]:
    """Return (action, reason, faced_flag)."""
    if enemy is None:
        return buttons("RIGHT"), "walk", faced
    if enemy["dx"] > -28:
        return buttons("RIGHT"), "space", faced
    if enemy["dx"] < -90:
        return buttons("LEFT"), "step", faced
    if not faced or frame < 20:
        return buttons("LEFT"), "face", True
    if frame % 60 < 3:
        return buttons("LEFT"), "reface", faced
    if frame % 12 < 2:
        return buttons("Y"), "y", faced
    return idle_action(), "gap", faced


def run_area1_probe(
    *,
    state_name: str = DEFAULT_STATE,
    max_frames: int = 12000,
    heal_hp: int | None = 80,
    reheal_below: int = 30,
    reheal_to: int = 70,
    force_enemy_hp: int | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Run Area1 face-Y chip; optional heal / force-enemy-HP for mapping."""
    configure_headless()
    available = get_available_states(GAME, GAME_DIR)
    if state_name not in available:
        raise SystemExit(f"missing state {state_name}; have {available[:12]}")

    out = out_dir or (RECORDINGS_DIR / "stage3_area1_probe")
    out.mkdir(parents=True, exist_ok=True)

    env = make_env(GAME, state_name, GAME_DIR, render_mode="rgb_array")
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    heals: list[int] = []
    if heal_hp is not None and 1 <= heal_hp <= 80:
        env.set_value("player_hp", heal_hp)
        heals.append(heal_hp)

    state, enemy, living, ghosts = _snap(env)
    start_ehp = enemy["hp"] if enemy else None
    force_applied = False
    if force_enemy_hp is not None and enemy is not None:
        env.set_value("enemy0_hp", force_enemy_hp)
        state, enemy, living, ghosts = _snap(env)
        start_ehp = enemy["hp"] if enemy else start_ehp
        force_applied = True
        print(f"force enemy0_hp→{force_enemy_hp} (now {enemy})")

    faced = False
    clear_hold = 0
    death = False
    saved: list[str] = []
    reasons: dict[str, int] = {}
    peak_dmg = 0
    screenshots: list[str] = []
    png = save_rgb_png(obs, out / "a1_0000_start.png")
    screenshots.append(png.name)

    for frame in range(1, max_frames + 1):
        state, enemy, living, ghosts = _snap(env)
        if enemy and start_ehp is None:
            if force_enemy_hp is not None and not force_applied:
                env.set_value("enemy0_hp", force_enemy_hp)
                state, enemy, living, ghosts = _snap(env)
                force_applied = True
                print(f"force enemy0_hp→{force_enemy_hp} after spawn")
            start_ehp = enemy["hp"] if enemy else None
            print(
                f"spawn f={frame} ehp={enemy['hp'] if enemy else None} "
                f"dx={enemy['dx'] if enemy else None}"
            )
        if enemy and start_ehp is not None:
            peak_dmg = max(peak_dmg, start_ehp - enemy["hp"])

        php = state.health
        if (
            reheal_to > 0
            and 0 < php <= reheal_below
            and 1 <= reheal_to <= 80
        ):
            env.set_value("player_hp", reheal_to)
            heals.append(reheal_to)
            state, enemy, living, ghosts = _snap(env)
            php = state.health
            print(f"reheal→{reheal_to} f={frame}")

        if php == 0 or php > 128 or state.player_dead:
            death = True
            png = save_rgb_png(obs, out / f"a1_{frame:04d}_death.png")
            screenshots.append(png.name)
            print(
                f"DEAD f={frame} ehp={enemy['hp'] if enemy else 0} "
                f"dmg={peak_dmg}"
            )
            break

        # Post-kill: plant ghosts, save clear, CLEAR_AREA → Area2/Boss3.
        if start_ehp is not None and not living:
            if ghosts:
                gdx = ghosts[0]["dx"]
                act = (
                    buttons("LEFT", "Y")
                    if gdx < 0
                    else buttons("RIGHT", "Y")
                )
                if frame % 6 >= 3:
                    act = idle_action()
                reasons["plant"] = reasons.get("plant", 0) + 1
                obs, *_ = env.step(act)
                continue

            clear_hold += 1
            if clear_hold == 12 and 0 < php <= 128:
                print(f"KILL f={frame} php={php} — CLEAR_AREA bridge")
                for name in (
                    f"Stage3_Area1_clear_hp{php}_L{state.lives}"
                    f"_cam{state.camera_x}",
                    "Stage3_Area1_Clear",
                ):
                    path = save_state(env, GAME_DIR, GAME, name)
                    if path.name not in saved:
                        saved.append(path.name)
                png = save_rgb_png(obs, out / f"a1_{frame:04d}_clear.png")
                screenshots.append(png.name)
                env.set_value(
                    "game_status", int(GameStatus.CLEAR_AREA)
                )

            if clear_hold > 12:
                if (
                    state.boss_active
                    and int(state.extras.get("boss_status") or 0) >= 1
                    and 0 < php <= 128
                ):
                    path = save_state(env, GAME_DIR, GAME, "Boss3")
                    if path.name not in saved:
                        saved.append(path.name)
                    png = save_rgb_png(obs, out / "a1_boss3.png")
                    screenshots.append(png.name)
                    print(
                        f"Boss3 f={frame} cam={state.camera_x} "
                        f"bst={state.extras.get('boss_status')} "
                        f"bhp={state.extras.get('boss_hp')}"
                    )
                    break
                if (
                    state.room >= 2
                    and state.mode.name == "PLAYING"
                    and 0 < php <= 128
                    and not any("Area2" in s for s in saved)
                ):
                    name = (
                        f"Stage3_Area{state.room}_hp{php}"
                        f"_cam{state.camera_x}"
                    )
                    path = save_state(env, GAME_DIR, GAME, name)
                    saved.append(path.name)
                    print(f"area{state.room} save → {path.name}")

            if clear_hold > 500:
                print(
                    f"post-clear timeout room={state.room} "
                    f"boss={state.boss_active}"
                )
                break

            act = (
                idle_action()
                if state.mode.name != "PLAYING"
                else buttons("RIGHT")
            )
            reasons["post"] = reasons.get("post", 0) + 1
            obs, *_ = env.step(act)
            continue

        clear_hold = 0

        if (
            enemy
            and start_ehp is not None
            and (start_ehp - enemy["hp"]) >= 50
            and 20 <= php <= 128
            and frame % 350 == 0
        ):
            name = (
                f"Stage3_Area1_mid_p{php}_e{enemy['hp']}"
                f"_cam{state.camera_x}"
            )
            path = save_state(env, GAME_DIR, GAME, name)
            if path.name not in saved:
                saved.append(path.name)
                print(f"mid → {path.name} dmg={start_ehp - enemy['hp']}")

        act, reason, faced = _face_y_action(
            frame, enemy, faced=faced
        )
        reasons[reason] = reasons.get(reason, 0) + 1
        if frame % 500 == 0:
            print(
                f"f={frame} php={php} "
                f"ehp={enemy['hp'] if enemy else 0} "
                f"dx={enemy['dx'] if enemy else None} "
                f"sx={state.player_x - state.camera_x} "
                f"dmg={peak_dmg}"
            )
        obs, *_ = env.step(act)

    state, enemy, living, ghosts = _snap(env)
    killed = start_ehp is not None and not living and not death
    outcome = (
        "boss3"
        if any(s.startswith("Boss3") for s in saved)
        else (
            "clear"
            if killed or any("Area1_clear" in s or "Area1_Clear" in s for s in saved)
            else ("death" if death else "timeout")
        )
    )
    report: dict[str, Any] = {
        "outcome": outcome,
        "start_state": state_name,
        "start_ehp": start_ehp,
        "peak_dmg": peak_dmg,
        "final_ehp": enemy["hp"] if enemy else 0,
        "final_php": state.health,
        "final_room": state.room,
        "final_cam": state.camera_x,
        "boss_active": state.boss_active,
        "boss_status": state.extras.get("boss_status"),
        "frames": frame,
        "heals": heals,
        "heal_count": len(heals),
        "force_enemy_hp": force_enemy_hp,
        "reasons": reasons,
        "saved_states": saved,
        "screenshots": screenshots,
        "notes": (
            "face-Y chip: brief LEFT face + pulsed Y. Continuous LEFT+Y "
            "whiffs. Heal pokes are Survival assists (document if used). "
            "force_enemy_hp is dev-only mapping. Post-kill CLEAR_AREA → "
            "Area2/Boss3."
        ),
    }
    write_json_report(out / "stage3_area1_probe.json", report)
    print(f"outcome={outcome} dmg={peak_dmg} saved={saved}")
    env.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default=DEFAULT_STATE)
    parser.add_argument("--max-frames", type=int, default=12000)
    parser.add_argument(
        "--heal-hp",
        type=int,
        default=80,
        help="Initial player_hp poke (1–80); 0 disables",
    )
    parser.add_argument(
        "--reheal-below",
        type=int,
        default=30,
        help="Re-poke player_hp when below this (0 disables)",
    )
    parser.add_argument("--reheal-to", type=int, default=70)
    parser.add_argument(
        "--force-enemy-hp",
        type=int,
        default=None,
        help="Dev-only: poke enemy0_hp after spawn (map Boss3 path)",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    heal = args.heal_hp if args.heal_hp > 0 else None
    reheal_below = args.reheal_below if args.reheal_below > 0 else 0
    run_area1_probe(
        state_name=args.state,
        max_frames=args.max_frames,
        heal_hp=heal,
        reheal_below=reheal_below,
        reheal_to=args.reheal_to,
        force_enemy_hp=args.force_enemy_hp,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
