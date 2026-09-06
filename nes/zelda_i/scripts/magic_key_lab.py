"""Fast Level 8 Magical-Key-stairs iteration from a live power-on checkpoint.

The full ``--through level8-magic-key`` spine replays the whole L1-L8 spine per
trial.  For tuning ``Level8MagicKeyStairsController`` (0x1F clear -> 0x68 slide
-> centre-stairs drop -> cellar 0x0F key loop -> two-ladder return) that is too
slow, so this pins the real power-on state at the ``level8_magic_key_stairs``
stage boundary (play 0x1F, blue-gohma just landed) once, then loads it to run
only that stage.

The pin captures the real power-on census / RNG phase.  Re-pin after any
upstream change and re-validate with a full ``--through level8-magic-key``
every ~10 iterations (see ``[[gohma-iteration-pins]]``).

    # (re)build the checkpoint from power-on (~several min):
    uv run python nes/zelda_i/scripts/magic_key_lab.py --pin

    # iterate on the stairs controller (fast):
    uv run python nes/zelda_i/scripts/magic_key_lab.py --tag l8mk_stairs_try
"""

from __future__ import annotations

import argparse

from retro_harness.audit import AuditCapabilities, AuditedEnv
from retro_harness.env import (
    make_env,
    reset_obs,
    resync_custom_state,
    save_state,
)
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.level8.magic_key import make_magic_key_stairs_live_controller
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_MAGIC_KEY, read_snapshot, read_u8
from zelda_i.route.chain import run_controller_stage
from zelda_i.spine.survival import run_survival_spine

PIN_STATE = "MagicKeyStairsEntryLive"
PIN_THROUGH = "level8-magic-key"
STAIRS_ROOM_1F = 0x1F


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
        "bow": int(snap.bow),
        "arrows": int(snap.arrows),
        "magic_key": (
            int(read_u8(ram, ADDR_MAGIC_KEY)) if ram is not None else None
        ),
        "health": f"0x{int(snap.health):02x}",
    }


def build_pin() -> int:
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    obs, _ = reset_obs(env)
    audited = AuditedEnv(
        env, capabilities=AuditCapabilities.all("zelda_i.magic_key_lab")
    )
    # magic_key_stairs is the fail-closed stub, so this run "fails" at stage 4
    # but leaves the emulator exactly at the 0x1F frontier we want to pin.
    run = run_survival_spine(audited, obs, assist=assist, through=PIN_THROUGH)
    ram = audited.get_ram()
    snap = read_snapshot(ram)
    rep = run.report()
    if not (snap.level == 8 and snap.screen == STAIRS_ROOM_1F and snap.mode == 5):
        print(
            f"REFUSING to pin: not at L8 0x1F play. glance={_glance(snap, ram)} "
            f"failed_stage={rep.get('failed_stage')}"
        )
        env.close()
        return 1
    path = save_state(env, GAME_DIR, GAME, PIN_STATE)
    print(f"pinned {path}")
    print(
        f"leftover glance: {_glance(snap, ram)}  "
        f"ok={rep.get('ok')} failed_stage={rep.get('failed_stage')}"
    )
    env.close()
    return 0


def run_stairs(tag: str) -> int:
    configure_headless()
    env = make_env(GAME, PIN_STATE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    obs, _ = reset_obs(env)
    resync_custom_state(env, GAME_DIR, GAME, PIN_STATE)
    obs, *_ = env.step(nes_idle_action())
    audited = AuditedEnv(
        env, capabilities=AuditCapabilities.all("zelda_i.magic_key_lab")
    )
    entry = read_snapshot(audited.get_ram())
    entry_glance = _glance(entry, audited.get_ram())

    ctl = make_magic_key_stairs_live_controller()
    trace: list[dict] = []
    rooms: dict[str, str] = {}

    def on_frame(e, _obs, action, frame):  # noqa: ANN001
        from zelda_i.dungeon.tilemap import ascii_room, has_room_tile_map

        if ctl.phase not in rooms and has_room_tile_map(e.get_ram()):
            try:
                rooms[ctl.phase] = ascii_room(e.get_ram())
            except Exception:  # noqa: BLE001
                pass
        if frame % 60 and frame > 1:
            return
        s = read_snapshot(e.get_ram())
        blocks = [
            [int(o.x), int(o.y)]
            for o in s.objects
            if 1 <= o.slot <= 12 and int(o.type_id) == 0x68
        ]
        trace.append(
            {
                "f": int(frame),
                "phase": ctl.phase,
                "xy": [int(s.link_x), int(s.link_y)],
                "mode": int(s.mode),
                "screen": f"0x{s.screen:02x}",
                "tile": int(s.colliding_tile),
                "blocks": blocks,
                "push_frames": int(ctl.push_frames),
                "block_xy0": ctl.block_xy0,
                "note": ctl.notes[-1] if ctl.notes else None,
            }
        )

    obs, stage = run_controller_stage(
        audited,
        obs,
        name="level8_magic_key_stairs",
        controller=ctl,
        max_frames=ctl.max_frames,
        assist=assist,
        on_frame=on_frame,
    )
    ram = audited.get_ram()
    snap = read_snapshot(ram)
    crep = stage.report().get("controller", {}) or ctl.report()
    ok = bool(crep.get("success")) and int(read_u8(ram, ADDR_MAGIC_KEY)) >= 1
    png = RECORDINGS_DIR / f"{tag}_final.png"
    save_rgb_png(obs, png)
    report = {
        "ok": ok,
        "entry_glance": entry_glance,
        "leave_glance": _glance(snap, ram),
        "stage": stage.report(),
        "controller": crep,
        "trace": trace,
        "rooms": rooms,
        "screenshot": str(png),
    }
    write_json_report(RECORDINGS_DIR / f"{tag}.json", report)
    print(
        f"{tag}: ok={ok} success={crep.get('success')} failed={crep.get('failed')} "
        f"frames={crep.get('frames')} phase={crep.get('phase')} "
        f"mk={read_u8(ram, ADDR_MAGIC_KEY)} "
        f"xy=({snap.link_x},{snap.link_y}) screen=0x{snap.screen:02x} "
        f"mode={snap.mode} notes={crep.get('notes')}"
    )
    env.close()
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", action="store_true", help="rebuild the checkpoint")
    parser.add_argument("--tag", default="l8mk_stairs_lab")
    args = parser.parse_args(argv)
    return build_pin() if args.pin else run_stairs(args.tag)


if __name__ == "__main__":
    raise SystemExit(main())
