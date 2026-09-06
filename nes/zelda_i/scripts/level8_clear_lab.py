"""Fast Level 8 Gleeok-suffix iteration from a live power-on checkpoint.

Pins the post-Magical-Key ``0x1F`` frontier once (``Level8SuffixEntryLive`` --
built by running the now-green ``--through level8-magic-key`` spine), then
drives the ordered ``LEVEL8_SUFFIX_GATES`` from it stage by stage.  This is the
rr-6o7.3 iteration harness: the whole suffix is
``0x1F -> 0x1E -> 0x2E -> 0x3E -> 0x3F -> cellar 0x2F -> 0x4C -> 0x3C``
(Gleeok + heart) ``-> 0x2C`` (shard) -> post-fanfare OW leave.

    uv run python nes/zelda_i/scripts/level8_clear_lab.py --pin
    uv run python nes/zelda_i/scripts/level8_clear_lab.py --tag l8clr_try
    uv run python nes/zelda_i/scripts/level8_clear_lab.py --tag l8clr_try --start 3

``--start N`` skips the first N suffix gates (resume after a partial success);
combine with a hand-saved pin if you want to iterate one gate in isolation.
"""

from __future__ import annotations

import argparse

from retro_harness.audit import AuditCapabilities, AuditedEnv
from retro_harness.env import make_env, reset_obs, resync_custom_state, save_state
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.level8.suffix import LEVEL8_SUFFIX_GATES
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_MAGIC_KEY, read_snapshot, read_u8
from zelda_i.route.chain import run_controller_stage
from zelda_i.spine.survival import run_survival_spine

PIN_STATE = "Level8SuffixEntryLive"
PIN_THROUGH = "level8-magic-key"


def _glance(snap, ram=None) -> dict:
    return {
        "mode": int(snap.mode),
        "level": int(snap.level),
        "screen": f"0x{int(snap.screen):02x}",
        "xy": [int(snap.link_x), int(snap.link_y)],
        "triforce": f"0x{int(snap.triforce):02x}",
        "keys": int(snap.keys),
        "bombs": int(snap.bombs),
        "rupees": int(snap.rupees),
        "hearts": int(snap.heart_containers),
        "health": f"0x{int(snap.health):02x}",
        "magic_key": (
            int(read_u8(ram, ADDR_MAGIC_KEY)) if ram is not None else None
        ),
    }


def build_pin() -> int:
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    obs, _ = reset_obs(env)
    audited = AuditedEnv(
        env, capabilities=AuditCapabilities.all("zelda_i.level8_clear_lab")
    )
    run = run_survival_spine(audited, obs, assist=assist, through=PIN_THROUGH)
    ram = audited.get_ram()
    snap = read_snapshot(ram)
    rep = run.report()
    ok = (
        bool(rep.get("ok"))
        and snap.level == 8
        and snap.screen == 0x1F
        and snap.mode == 5
        and int(read_u8(ram, ADDR_MAGIC_KEY)) >= 1
    )
    if not ok:
        print(f"REFUSING to pin: {_glance(snap, ram)} ok={rep.get('ok')}")
        env.close()
        return 1
    path = save_state(env, GAME_DIR, GAME, PIN_STATE)
    print(f"pinned {path}\nleftover: {_glance(snap, ram)}")
    env.close()
    return 0


def run_suffix(tag: str, start: int) -> int:
    configure_headless()
    env = make_env(GAME, PIN_STATE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    obs, _ = reset_obs(env)
    resync_custom_state(env, GAME_DIR, GAME, PIN_STATE)
    obs, *_ = env.step(nes_idle_action())
    audited = AuditedEnv(
        env, capabilities=AuditCapabilities.all("zelda_i.level8_clear_lab")
    )
    entry = _glance(read_snapshot(audited.get_ram()), audited.get_ram())

    stages: list[dict] = []
    failed_at = None
    for i, gate in enumerate(LEVEL8_SUFFIX_GATES):
        if i < start:
            continue
        ctl = gate.factory()
        obs, stage = run_controller_stage(
            audited,
            obs,
            name=gate.stage,
            controller=ctl,
            max_frames=int(getattr(ctl, "max_frames", 4000)),
            assist=assist,
        )
        crep = stage.report().get("controller", {}) or (
            ctl.report() if hasattr(ctl, "report") else {}
        )
        s = read_snapshot(audited.get_ram())
        row = {
            "i": i,
            "stage": gate.stage,
            "dest_room": f"0x{gate.dest_room:02x}",
            "success": bool(crep.get("success")),
            "failed": bool(crep.get("failed")),
            "frames": crep.get("frames"),
            "glance": _glance(s, audited.get_ram()),
            "notes": crep.get("notes"),
        }
        stages.append(row)
        print(
            f"[{i}] {gate.stage}: succ={row['success']} failed={row['failed']} "
            f"f={row['frames']} -> {row['glance']['screen']} "
            f"{row['glance']['xy']} m{row['glance']['mode']} "
            f"hc={row['glance']['hearts']} notes={row['notes']}"
        )
        if not crep.get("success"):
            failed_at = gate.stage
            break

    # The final gate (level8_ow_leave_settle) already idles the fanfare to the
    # settled OW leave; a short extra idle just confirms it holds.
    ow_settle = None
    if failed_at is None:
        for f in range(1, 301):
            obs, *_ = env.step(nes_idle_action())
            assist.apply_env(audited, frame=f)
        ow_settle = _glance(read_snapshot(audited.get_ram()), audited.get_ram())
        print(f"OW settle (+300f hold): {ow_settle}")

    s = read_snapshot(audited.get_ram())
    png = RECORDINGS_DIR / f"{tag}_final.png"
    save_rgb_png(obs, png)
    write_json_report(
        RECORDINGS_DIR / f"{tag}.json",
        {
            "ok": failed_at is None,
            "failed_at": failed_at,
            "entry_glance": entry,
            "final_glance": _glance(s, audited.get_ram()),
            "ow_settle": ow_settle,
            "stages": stages,
            "screenshot": str(png),
        },
    )
    print(f"\nsummary: ok={failed_at is None} failed_at={failed_at}")
    print(f"final: {_glance(s, audited.get_ram())}")
    env.close()
    return 0 if failed_at is None else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", action="store_true")
    parser.add_argument("--tag", default="l8clr_lab")
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args(argv)
    return build_pin() if args.pin else run_suffix(args.tag, args.start)


if __name__ == "__main__":
    raise SystemExit(main())
