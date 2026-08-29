"""Isolated: Level 2 bomb-north 0x4f → 0x3f from ``Level2Boom``.

Stand **(120, 101)** facing UP, B places bomb (natural inventory). Opens
north wall into traps+Keese room **0x3f** (Dodongo path, rr-n5i).

Stop: ``level2_room_3f_ready`` (level==2, screen==0x3f, mode==5).

Prefer ``--infinite-life`` for first-pass path past 0x3f; Clean when stable.

Examples::

    uv run python nes/zelda_i/scripts/run_level2_bomb_north_4f.py --trials 2
    uv run python nes/zelda_i/scripts/run_level2_bomb_north_4f.py \\
        --infinite-life --trials 1 --save-state
"""

from __future__ import annotations

import argparse

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level2_bomb_path import (
    BOMB_N_STAND,
    PostBoomBombNorthPhase,
    make_post_boom_bomb_north_controller,
)
from zelda_i.level2_dungeon import level2_room_3f_ready
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot, read_u8

ADDR_SELECTED_ITEM = 0x0656

def run_once(
    *,
    tag: str = "level2_bomb_north_4f",
    save_checkpoint: bool = False,
    start_state: str = "Level2Boom",
    checkpoint_name: str = "Level2_3F",
    infinite_life: bool = False,
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    controller = make_post_boom_bomb_north_controller()
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    try:
        obs, _ = reset_obs(env)
        obs, *_ = env.step(nes_idle_action())
        entry = read_snapshot(env.get_ram())
        entry_sel = read_u8(env.get_ram(), ADDR_SELECTED_ITEM)

        for f in range(controller.max_frames):
            if assist is not None:
                assist.apply_env(env, frame=f)
            action = controller.step(read_snapshot(env.get_ram()))
            obs, *_ = env.step(action.action)
            if (
                controller.success
                or controller.phase is PostBoomBombNorthPhase.FAILED
            ):
                break

        ram = env.get_ram()
        snap = read_snapshot(ram)
        ok = level2_room_3f_ready(ram)
        checkpoint = None
        provenance = None
        if ok and save_checkpoint:
            checkpoint_path = save_state(env, GAME_DIR, GAME, checkpoint_name)
            checkpoint = str(checkpoint_path)
            provenance = str(
                write_state_provenance(
                    checkpoint_path,
                    source_state_path=(
                        GAME_DIR
                        / "custom_integrations"
                        / GAME
                        / f"{start_state}.state"
                    ),
                    request={
                        "segment": "level2_bomb_north_4f",
                        "natural_entry": False,
                        "start_state": start_state,
                        "stand": list(BOMB_N_STAND),
                        "bead": "rr-n5i",
                    },
                    selected_trial=controller.report(),
                    natural_entry=False,
                )
            )

        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        png = RECORDINGS_DIR / f"{tag}_exit.png"
        save_rgb_png(obs, png)

        return {
            "ok": ok,
            "bead": "rr-n5i",
            "track": "assisted" if infinite_life else "clean",
            "start_state": start_state,
            "entry": {
                "sc": f"0x{entry.screen:02x}",
                "xy": [entry.link_x, entry.link_y],
                "bombs": entry.bombs,
                "keys": entry.keys,
                "selected": entry_sel,
            },
            "final": {
                "sc": f"0x{snap.screen:02x}",
                "mode": snap.mode,
                "xy": [snap.link_x, snap.link_y],
                "bombs": snap.bombs,
                "doors": snap.cur_opened_doors,
            },
            "controller": controller.report(),
            "checkpoint": checkpoint,
            "provenance": provenance,
            "screenshot": str(png),
        }
    finally:
        env.close()

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-state", default="Level2Boom")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--tag", default="level2_bomb_north_4f")
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument(
        "--infinite-life",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    trials = []
    for i in range(args.trials):
        r = run_once(
            tag=f"{args.tag}_t{i}",
            save_checkpoint=args.save_state and i == 0,
            start_state=args.from_state,
            infinite_life=args.infinite_life,
        )
        trials.append(r)
        print(
            f"t{i} ok={r['ok']} sc={r['final']['sc']} "
            f"phase={r['controller'].get('phase')} notes={r['controller'].get('notes')}"
        )

    ok_n = sum(1 for t in trials if t.get("ok"))
    report = {
        "bead": "rr-n5i",
        "segment": "level2_bomb_north_4f",
        "start_state": args.from_state,
        "trials_ok": ok_n,
        "trials_total": len(trials),
        "trials": trials,
        "stand": list(BOMB_N_STAND),
        "track": "assisted" if args.infinite_life else "clean",
    }
    out = RECORDINGS_DIR / f"{args.tag}_isolated.json"
    write_json_report(out, report)
    print(f"wrote {out} ({ok_n}/{len(trials)})")
    if ok_n < len(trials):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
