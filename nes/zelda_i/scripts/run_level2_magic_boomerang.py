"""Isolated pure/assisted: Magical Boomerang on L2 room 0x4f.

Default path (finish-easy geometry from ``Level2_5F``):

1. Optional gel clear on 0x5f
2. Bomb north @ (120,101) UP+B → 0x4f
3. Clear 3× type 0x05 (blue Goriya residual) via ``ROOM_4F_SPEC``
4. Collect fixed RoomItemId 0x1e → ``ADDR_MAGIC_BOOMERANG``

Alt path ``--via-4e`` from ``Level2_5E``:

1. Free UP → 0x4e, clear 5× Rope + key
2. Key RIGHT → 0x4f, clear + collect

Stop: ``level2_room_4f_magic_boomerang_success`` (``ADDR_MAGIC_BOOMERANG != 0``).

Prefer ``--infinite-life`` first for fireball (0x55) damage; Clean when
stable. Evidence: ``recordings/level2_magic_boomerang_isolated.json``.

Examples::

    uv run python nes/zelda_i/scripts/run_level2_magic_boomerang.py \\
        --trials 2 --infinite-life
    uv run python nes/zelda_i/scripts/run_level2_magic_boomerang.py \\
        --trials 2 --infinite-life --save-state
    uv run python nes/zelda_i/scripts/run_level2_magic_boomerang.py \\
        --via-4e --from-state Level2_5E --infinite-life --trials 1
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NES_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT, _NES_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from retro_harness.env import make_env, save_state
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon import (
    DungeonPhase,
    GenericDungeonRoomController,
    ROOM_4E_SPEC,
    ROOM_4F_SPEC,
)
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level2_dungeon import (
    BOOM_BOMB_N_STAND,
    BoomBombNorthPhase,
    make_boom_bomb_north_controller,
    ROOM_L2_BOOM_CANDIDATE,
    level2_room_4f_magic_boomerang_success,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_BOOMERANG,
    ADDR_MAGIC_BOOMERANG,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)


def _act(env, direction: str | None, *buttons: str) -> None:
    if direction is None and not buttons:
        env.step(nes_idle_action())
    else:
        env.step(nes_action(direction, *buttons) if direction else nes_action(*buttons))


def _idle(env, n: int = 1) -> None:
    for _ in range(n):
        env.step(nes_idle_action())


def _run_controller(env, controller, max_frames: int, assist=None, frame0: int = 0):
    obs = None
    for i in range(max_frames):
        if assist is not None:
            assist.apply_env(env, frame=frame0 + i)
        snap = read_snapshot(env.get_ram())
        if level2_room_4f_magic_boomerang_success(env.get_ram()):
            controller.success = True
            break
        action = controller.step(snap)
        obs, *_ = env.step(action.action)
        if controller.success or controller.phase is DungeonPhase.FAILED:
            break
        if getattr(controller, "phase", None) is BoomBombNorthPhase.FAILED:
            break
        if getattr(controller, "phase", None) is BoomBombNorthPhase.DONE:
            break
    return obs


def _enter_4e_from_5e(env, assist=None) -> bool:
    """Free UP from 0x5e mid → 0x4e."""

    def snap():
        return read_snapshot(env.get_ram())

    for f in range(900):
        if assist is not None:
            assist.apply_env(env, frame=f)
        s = snap()
        if s.screen == 0x4E and s.mode == PLAY_MODE:
            return True
        if s.mode != PLAY_MODE:
            _act(env, "UP")
            continue
        if abs(s.link_x - 120) > 6:
            _act(env, "RIGHT" if s.link_x < 120 else "LEFT")
        else:
            _act(env, "UP")
    s = snap()
    return s.screen == 0x4E and s.mode == PLAY_MODE


def _enter_4f_from_4e(env, assist=None) -> bool:
    """Key RIGHT y≈141 from 0x4e → 0x4f (consumes 1 key)."""

    def snap():
        return read_snapshot(env.get_ram())

    for f in range(900):
        if assist is not None:
            assist.apply_env(env, frame=f)
        s = snap()
        if s.screen == 0x4F and s.mode == PLAY_MODE:
            return True
        if s.mode != PLAY_MODE:
            _act(env, "RIGHT")
            continue
        if abs(s.link_y - 141) > 4:
            _act(env, "DOWN" if s.link_y < 141 else "UP")
        else:
            _act(env, "RIGHT")
    s = snap()
    return s.screen == 0x4F and s.mode == PLAY_MODE


def run_once(
    *,
    tag: str = "level2_magic_boomerang",
    save_checkpoint: bool = False,
    start_state: str = "Level2_5F",
    checkpoint_name: str = "Level2Boom",
    via_4e: bool = False,
    infinite_life: bool = False,
    clear_gels: bool = True,
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    bomb_ctrl = None
    room4e_ctrl: GenericDungeonRoomController | None = None
    room4f_ctrl = GenericDungeonRoomController(ROOM_4F_SPEC)
    prefix_ok = True
    prefix_error = None
    path = "via_4e" if via_4e else "bomb_5f"

    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)
        entry = read_snapshot(env.get_ram())
        entry_mboom = read_u8(env.get_ram(), ADDR_MAGIC_BOOMERANG)

        if entry_mboom:
            prefix_ok = True
            prefix_error = "already_had_magic_boomerang"
        elif via_4e:
            # Level2_5E → UP 0x4e → clear+key → RIGHT 0x4f → boom
            if entry.screen == 0x4F and entry.mode == PLAY_MODE:
                room4f_ctrl.phase = DungeonPhase.FIGHT
            elif entry.screen == 0x4E and entry.mode == PLAY_MODE:
                room4e_ctrl = GenericDungeonRoomController(ROOM_4E_SPEC)
                room4e_ctrl.phase = DungeonPhase.FIGHT
                obs = _run_controller(
                    env, room4e_ctrl, ROOM_4E_SPEC.max_frames, assist
                )
                if not room4e_ctrl.success:
                    # still try RIGHT if clear-only-ish
                    pass
                if not _enter_4f_from_4e(env, assist):
                    prefix_ok = False
                    prefix_error = "failed_4e_right_to_4f"
                else:
                    room4f_ctrl.phase = DungeonPhase.FIGHT
            elif entry.screen == 0x5E:
                _idle(env, 20)
                if not _enter_4e_from_5e(env, assist):
                    prefix_ok = False
                    prefix_error = "failed_5e_up_to_4e"
                else:
                    room4e_ctrl = GenericDungeonRoomController(ROOM_4E_SPEC)
                    room4e_ctrl.phase = DungeonPhase.FIGHT
                    obs = _run_controller(
                        env, room4e_ctrl, ROOM_4E_SPEC.max_frames, assist
                    )
                    keys = read_snapshot(env.get_ram()).keys
                    if keys < 1 and not room4e_ctrl.success:
                        prefix_ok = False
                        prefix_error = "no_key_after_4e"
                    elif not _enter_4f_from_4e(env, assist):
                        prefix_ok = False
                        prefix_error = "failed_4e_right_to_4f"
                    else:
                        room4f_ctrl.phase = DungeonPhase.FIGHT
            else:
                prefix_ok = False
                prefix_error = f"unsupported_via4e_start_0x{entry.screen:02x}"
        else:
            # Level2_5F bomb N → 0x4f
            if entry.screen == 0x4F and entry.mode == PLAY_MODE:
                room4f_ctrl.phase = DungeonPhase.FIGHT
            elif entry.screen == 0x5F:
                bomb_ctrl = make_boom_bomb_north_controller(clear_gels=clear_gels)
                for i in range(bomb_ctrl.max_frames):
                    if assist is not None:
                        assist.apply_env(env, frame=i)
                    action = bomb_ctrl.step(read_snapshot(env.get_ram()))
                    obs, *_ = env.step(action.action)
                    if (
                        bomb_ctrl.success
                        or bomb_ctrl.phase is BoomBombNorthPhase.FAILED
                    ):
                        break
                if not bomb_ctrl.success:
                    prefix_ok = False
                    prefix_error = f"bomb_failed_{bomb_ctrl.phase.name}"
                else:
                    room4f_ctrl.phase = DungeonPhase.FIGHT
            else:
                prefix_ok = False
                prefix_error = f"unsupported_start_room_0x{entry.screen:02x}"

        # Fight + collect on 0x4f
        if prefix_ok and not level2_room_4f_magic_boomerang_success(env.get_ram()):
            if room4f_ctrl.phase is not DungeonPhase.FIGHT:
                # ensure on 4f
                s = read_snapshot(env.get_ram())
                if s.screen == ROOM_L2_BOOM_CANDIDATE and s.mode == PLAY_MODE:
                    room4f_ctrl.phase = DungeonPhase.FIGHT
            obs = _run_controller(
                env, room4f_ctrl, ROOM_4F_SPEC.max_frames, assist
            )

        # Extra center wander if enemies dead but inventory not yet
        if prefix_ok and not level2_room_4f_magic_boomerang_success(env.get_ram()):
            targets = (
                (136, 135),
                (128, 141),
                (120, 141),
                (144, 125),
                (112, 125),
                (136, 157),
            )
            ti = 0
            for f in range(2500):
                if assist is not None:
                    assist.apply_env(env, frame=f)
                if level2_room_4f_magic_boomerang_success(env.get_ram()):
                    break
                s = read_snapshot(env.get_ram())
                if s.mode != PLAY_MODE:
                    _idle(env, 1)
                    continue
                tx, ty = targets[ti % len(targets)]
                dx, dy = tx - s.link_x, ty - s.link_y
                if abs(dx) <= 4 and abs(dy) <= 4:
                    ti += 1
                    _idle(env, 1)
                elif abs(dx) >= abs(dy):
                    _act(env, "RIGHT" if dx > 0 else "LEFT")
                else:
                    _act(env, "DOWN" if dy > 0 else "UP")

        ram = env.get_ram()
        snap = read_snapshot(ram)
        mboom = read_u8(ram, ADDR_MAGIC_BOOMERANG)
        ok = prefix_ok and level2_room_4f_magic_boomerang_success(ram)
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
                        "segment": "level2_magic_boomerang",
                        "natural_entry": False,
                        "start_state": start_state,
                        "path": path,
                        "stand": list(BOOM_BOMB_N_STAND),
                    },
                    selected_trial={
                        "bomb": bomb_ctrl.report() if bomb_ctrl else None,
                        "room4e": room4e_ctrl.report() if room4e_ctrl else None,
                        "room4f": room4f_ctrl.report(),
                    },
                    natural_entry=False,
                )
            )
        screenshot = RECORDINGS_DIR / f"{tag}_isolated.png"
        save_rgb_png(obs, screenshot)
        return {
            "ok": ok,
            "natural_entry": False,
            "start_state": start_state,
            "path": path,
            "prefix_ok": prefix_ok,
            "prefix_error": prefix_error,
            "intervention_class": "assisted" if infinite_life else "clean",
            "entry": {
                "room": entry.screen,
                "x": entry.link_x,
                "y": entry.link_y,
                "keys": entry.keys,
                "bombs": entry.bombs,
                "health": entry.health,
                "mboom": entry_mboom,
            },
            "bomb": bomb_ctrl.report() if bomb_ctrl else None,
            "room4e": room4e_ctrl.report() if room4e_ctrl else None,
            "room4f": room4f_ctrl.report(),
            "final": {
                "mode": snap.mode,
                "level": snap.level,
                "room": snap.screen,
                "x": snap.link_x,
                "y": snap.link_y,
                "keys": snap.keys,
                "bombs": snap.bombs,
                "health": snap.health,
                "room_item_id": snap.room_item_id,
                "room_all_dead": snap.room_all_dead,
                "cur_opened_doors": snap.cur_opened_doors,
                "boomerang": read_u8(ram, ADDR_BOOMERANG),
                "magical_boomerang": mboom,
                "live_blue_goriya": len(ROOM_4F_SPEC.live_enemies(snap)),
            },
            "assist": assist.telemetry.to_dict() if assist is not None else None,
            "checkpoint": checkpoint,
            "provenance": provenance,
            "screenshot": str(screenshot),
        }
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--from-state", default="Level2_5F")
    parser.add_argument(
        "--checkpoint-name",
        default="Level2Boom",
        help="Name for --save-state (default Level2Boom)",
    )
    parser.add_argument(
        "--via-4e",
        action="store_true",
        help="Path Level2_5E → UP 0x4e → RIGHT 0x4f (else bomb from 0x5f)",
    )
    parser.add_argument(
        "--infinite-life",
        action="store_true",
        help="Survival assist (fireballs 0x55); not Clean STATUS alone",
    )
    parser.add_argument(
        "--no-clear-gels",
        action="store_true",
        help="Skip 0x5f gel clear before bomb (default clears)",
    )
    args = parser.parse_args(argv)

    start = args.from_state
    if args.via_4e and start == "Level2_5F":
        start = "Level2_5E"

    reports = [
        run_once(
            tag=f"level2_magic_boomerang_t{trial}",
            save_checkpoint=args.save_state and trial == 0,
            start_state=start,
            checkpoint_name=args.checkpoint_name,
            via_4e=args.via_4e,
            infinite_life=args.infinite_life,
            clear_gels=not args.no_clear_gels,
        )
        for trial in range(args.trials)
    ]
    for trial, report in enumerate(reports):
        final = report.get("final") or {}
        r4f = report.get("room4f") or {}
        bomb = report.get("bomb") or {}
        print(
            f"trial={trial} ok={report.get('ok')} "
            f"path={report.get('path')} "
            f"prefix_ok={report.get('prefix_ok')} "
            f"room={final.get('room', 0):02X} "
            f"mboom={final.get('magical_boomerang')} "
            f"live={final.get('live_blue_goriya')} "
            f"xy=({final.get('x')},{final.get('y')}) "
            f"bomb_f={bomb.get('frames')} bomb_ph={bomb.get('phase')} "
            f"fight_f={r4f.get('frames')} fight_ph={r4f.get('phase')} "
            f"err={report.get('prefix_error')}"
        )

    output = RECORDINGS_DIR / "level2_magic_boomerang_isolated.json"
    successes = sum(1 for report in reports if report.get("ok"))
    track = "assisted" if args.infinite_life else "clean"
    write_json_report(
        output,
        {
            "segment": "level2_magic_boomerang",
            "bead": "rr-bsq+rr-ebe",
            "natural_entry": False,
            "start_state": start,
            "path": "via_4e" if args.via_4e else "bomb_5f",
            "runtime_class": "bronze",
            "intervention_class": track,
            "track": track,
            "trials": args.trials,
            "successes": successes,
            "stop_predicate": "level2_room_4f_magic_boomerang_success",
            "spec_id": ROOM_4F_SPEC.spec_id,
            "target_room": f"0x{ROOM_L2_BOOM_CANDIDATE:02x}",
            "stand": list(BOOM_BOMB_N_STAND),
            "room_item_id": "0x1e",
            "enemies": "3× type 0x05 TYPE_AND_HP (blue Goriya residual); 0x55 fireballs ignored",
            "policy": (
                "Level2_5F: gel clear optional; bomb N (120,101) UP+B → 0x4f; "
                "fight 0x05; collect magical_boomerang near (136,135). "
                "Alt: Level2_5E UP→0x4e clear+key RIGHT→0x4f."
            ),
            "reports": reports,
        },
    )
    print(f"wrote {output} successes={successes}/{args.trials}")
    return 0 if successes == args.trials else 1


if __name__ == "__main__":
    raise SystemExit(main())
