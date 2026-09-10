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

``--from-enter`` loads ``Level8InteriorReconFixture`` (play 0x7E) and runs
the magic-key chapter plus the suffix with no heart assist and no retopup
(rr-npv.4 fixture-live Clean glance). ``--from-state NAME`` overrides the pin.

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
from zelda_i.level8.path import (
    make_blue_gohma_controller,
    make_darknut_key_controller,
    make_magic_key_stairs_controller,
    make_north_manhandla_controller,
)
from zelda_i.level8.suffix import LEVEL8_SUFFIX_GATES
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_MAGIC_KEY, read_snapshot, read_u8
from zelda_i.route.chain import run_controller_stage

PIN_STATE = "Level8SuffixEntryLive"
PIN_THROUGH = "level8-magic-key"
ENTER_PIN = "Level8InteriorReconFixture"


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
    from zelda_i.spine.survival import run_survival_spine

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


def _magic_key_rows() -> list:
    return [
        ("level8_north_manhandla_bomb", make_north_manhandla_controller()),
        ("level8_darknut_key_up", make_darknut_key_controller()),
        ("level8_blue_gohma", make_blue_gohma_controller()),
        ("level8_magic_key_stairs", make_magic_key_stairs_controller()),
    ]


def run_suffix(
    tag: str,
    start: int,
    *,
    from_state: str = PIN_STATE,
    from_enter: bool = False,
    infinite_life: bool = True,
) -> int:
    configure_headless()
    pin = ENTER_PIN if from_enter else from_state
    env = make_env(GAME, pin, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    obs, _ = reset_obs(env)
    resync_custom_state(env, GAME_DIR, GAME, pin)
    obs, *_ = env.step(nes_idle_action())
    audited = AuditedEnv(
        env, capabilities=AuditCapabilities.all("zelda_i.level8_clear_lab")
    )
    entry = _glance(read_snapshot(audited.get_ram()), audited.get_ram())

    rows: list[tuple[str, object]] = []
    if from_enter:
        rows.extend(_magic_key_rows())
    rows.extend((gate.stage, gate.factory()) for gate in LEVEL8_SUFFIX_GATES)

    stages: list[dict] = []
    failed_at = None
    for i, (name, ctl) in enumerate(rows):
        if i < start:
            continue
        obs, stage = run_controller_stage(
            audited,
            obs,
            name=name,
            controller=ctl,
            max_frames=int(getattr(ctl, "max_frames", 4000)),
            assist=assist,
        )
        crep = stage.report().get("controller", {}) or (
            ctl.report() if hasattr(ctl, "report") else {}
        )
        s = read_snapshot(audited.get_ram())
        dest = None
        if i >= (len(_magic_key_rows()) if from_enter else 0):
            gi = i - (len(_magic_key_rows()) if from_enter else 0)
            if 0 <= gi < len(LEVEL8_SUFFIX_GATES):
                dest = f"0x{LEVEL8_SUFFIX_GATES[gi].dest_room:02x}"
        row = {
            "i": i,
            "stage": name,
            "dest_room": dest,
            "success": bool(crep.get("success")),
            "failed": bool(crep.get("failed")),
            "frames": crep.get("frames"),
            "glance": _glance(s, audited.get_ram()),
            "notes": crep.get("notes"),
            "writes": crep.get("writes"),
        }
        stages.append(row)
        print(
            f"[{i}] {name}: succ={row['success']} failed={row['failed']} "
            f"f={row['frames']} -> {row['glance']['screen']} "
            f"{row['glance']['xy']} m{row['glance']['mode']} "
            f"hc={row['glance']['hearts']} notes={row['notes']}"
        )
        if not crep.get("success"):
            failed_at = name
            break

    # The final gate (level8_ow_leave_settle) already idles the fanfare to the
    # settled OW leave; a short extra idle just confirms it holds.
    ow_settle = None
    if failed_at is None:
        for f in range(1, 301):
            obs, *_ = env.step(nes_idle_action())
            if assist is not None:
                assist.apply_env(audited, frame=f)
        ow_settle = _glance(read_snapshot(audited.get_ram()), audited.get_ram())
        print(f"OW settle (+300f hold): {ow_settle}")

    s = read_snapshot(audited.get_ram())
    png = RECORDINGS_DIR / f"{tag}_final.png"
    save_rgb_png(obs, png)
    deaths = 0 if assist is None else int(getattr(assist.telemetry, "deaths", 0) or 0)
    if int(s.mode) == 17:
        deaths = max(deaths, 1)
    audit = audited.audit()
    write_json_report(
        RECORDINGS_DIR / f"{tag}.json",
        {
            "ok": failed_at is None,
            "failed_at": failed_at,
            "from_enter": from_enter,
            "infinite_life": infinite_life,
            "retopup": False,
            "entry_glance": entry,
            "final_glance": _glance(s, audited.get_ram()),
            "ow_settle": ow_settle,
            "deaths": deaths,
            "audit": {
                "progression_writes": getattr(audit, "progression_writes", None),
                "capacity_writes": getattr(audit, "capacity_writes", None),
                "direct_ram_writes": getattr(audit, "direct_ram_writes", None),
            },
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
    parser.add_argument("--from-state", default=PIN_STATE)
    parser.add_argument(
        "--from-enter",
        action="store_true",
        help="Load Level8InteriorReconFixture and run MK + suffix (no retopup).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="No infinite life, no inventory pokes. Fixture-live Clean glance.",
    )
    args = parser.parse_args(argv)
    if args.pin:
        return build_pin()
    return run_suffix(
        args.tag,
        args.start,
        from_state=args.from_state,
        from_enter=args.from_enter,
        infinite_life=not args.clean,
    )


if __name__ == "__main__":
    raise SystemExit(main())
