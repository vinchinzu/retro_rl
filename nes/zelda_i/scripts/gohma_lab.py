"""Fast Level 6 Gohma iteration from a live power-on checkpoint.

The full ``--through level6-gohma`` spine replays ~213k frames per trial. For
tuning the Gohma controller that is too slow, so this script pins the real
power-on state at Gohma-room entry (``--through level6-north2c``: play 0x1C
``(120,205)``) once, then loads it to run only the Gohma stage.

The pin captures the real walked-warp RNG phase, so it is NOT an isolated
BFS state. Re-pin after any upstream change and re-validate with a full
``--through level6-gohma`` / ``--through level6`` every ~10 iterations.

    # (re)build the checkpoint from power-on (~5 min):
    uv run python nes/zelda_i/scripts/gohma_lab.py --pin

    # iterate on the Gohma controller (~15 s each):
    uv run python nes/zelda_i/scripts/gohma_lab.py --tag l6_gohma_try

    # add --dump to also write per-frame WRAM to scratch/gohma_eye/ram_lab.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

from retro_harness.audit import AuditCapabilities, AuditedEnv
from retro_harness.env import make_env, reset_obs, resync_custom_state, save_state
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.level6.gohma import level6_gohma_success, make_gohma_controller
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot
from zelda_i.route.chain import run_controller_stage
from zelda_i.spine.survival import run_survival_spine

PIN_STATE = "GohmaEntryLive"
PIN_THROUGH = "level6-north2c"


def _glance(snap) -> dict:
    return {
        "mode": int(snap.mode),
        "level": int(snap.level),
        "screen": f"0x{int(snap.screen):02x}",
        "xy": [int(snap.link_x), int(snap.link_y)],
        "triforce": f"0x{int(snap.triforce):02x}",
        "keys": int(snap.keys),
        "bombs": int(snap.bombs),
        "rupees": int(snap.rupees),
        "bow": int(snap.bow),
        "rod": int(snap.rod),
        "health": f"0x{int(snap.health):02x}",
    }


def build_pin() -> int:
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    obs, _ = reset_obs(env)
    audited = AuditedEnv(env, capabilities=AuditCapabilities.all("zelda_i.gohma_lab"))
    run = run_survival_spine(audited, obs, assist=assist, through=PIN_THROUGH)
    snap = read_snapshot(audited.get_ram())
    path = save_state(env, GAME_DIR, GAME, PIN_STATE)
    print(f"pinned {path}")
    print(f"leftover glance: {_glance(snap)}  ok={run.report().get('ok')}")
    env.close()
    return 0


def run_gohma(tag: str, dump: bool = False) -> int:
    configure_headless()
    env = make_env(GAME, PIN_STATE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    obs, _ = reset_obs(env)
    resync_custom_state(env, GAME_DIR, GAME, PIN_STATE)
    obs, *_ = env.step(nes_idle_action())
    audited = AuditedEnv(env, capabilities=AuditCapabilities.all("zelda_i.gohma_lab"))
    entry = read_snapshot(audited.get_ram())

    ram_log: list = []
    on_frame = None
    if dump:
        import numpy as np

        def on_frame(e, _obs, _action, _frame):  # noqa: ANN001
            ram_log.append(np.asarray(e.get_ram(), dtype=np.uint8).copy())

    ctl = make_gohma_controller()
    obs, stage = run_controller_stage(
        audited,
        obs,
        name="level6_gohma_0x1c",
        controller=ctl,
        max_frames=ctl.max_frames,
        assist=assist,
        on_frame=on_frame,
    )
    if dump and ram_log:
        import numpy as np

        out = Path("nes/zelda_i/scratch/gohma_eye")
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "ram_lab.npy", np.stack(ram_log))
        print(f"dumped {len(ram_log)}f WRAM -> {out / 'ram_lab.npy'}")
    snap = read_snapshot(audited.get_ram())
    ok = bool(level6_gohma_success(snap))
    png = RECORDINGS_DIR / f"{tag}_final.png"
    save_rgb_png(obs, png)
    report = {
        "ok": ok,
        "entry_glance": _glance(entry),
        "leave_glance": _glance(snap),
        "stage": stage.report(),
        "screenshot": str(png),
    }
    write_json_report(RECORDINGS_DIR / f"{tag}.json", report)
    c = stage.report().get("controller", {})
    print(
        f"{tag}: ok={ok} frames={c.get('frames')} pulses={c.get('arrow_pulses')} "
        f"connect={c.get('connect_frame')} rupees={snap.rupees} "
        f"xy=({snap.link_x},{snap.link_y}) notes={c.get('notes')}"
    )
    env.close()
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", action="store_true", help="rebuild the checkpoint")
    parser.add_argument("--dump", action="store_true", help="save per-frame WRAM")
    parser.add_argument("--tag", default="gohma_lab")
    args = parser.parse_args(argv)
    return build_pin() if args.pin else run_gohma(args.tag, dump=args.dump)


if __name__ == "__main__":
    raise SystemExit(main())
