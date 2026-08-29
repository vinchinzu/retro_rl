"""Materialize L9 stair sources and find the live drop into Patra 0x52.

Backwards-development tool.  Fixture inventory + room-loader setup stays
route-ineligible.  After a continuous run starts there are no object / room /
door / progression / capacity writes.

Examples::

    uv run python nes/zelda_i/scripts/run_level9_stairs.py --probe
    uv run python nes/zelda_i/scripts/run_level9_stairs.py \
      --build-fixture --infinite-life --save-state --trials 2 \
      --tag l9_stairXX_patra_credits_recon
"""

from __future__ import annotations

import argparse
from typing import Any

from zelda_i.level9_room51 import dump_room_51
from zelda_i.level9_stair_session import (
    BEAD,
    TAG,
    _apply_loader,
    _exit_cellar,
    _hold_until_room,
    _walk_target,
    dump_play_rooms,
    dump_room_04,
    dump_room_13,
    dump_room_21,
    dump_room_30,
    dump_room_31,
    dump_room_40,
    dump_room_41,
    dump_room_tiles,
    materialize_stair_room,
    probe_cellar_dest_table,
    probe_sources,
    run_play_source_to_credits,
    run_room04_bomb_west_to_credits,
    run_room21_south_to_credits,
    run_room30_stairs_to_credits,
    run_room31_bomb_west_to_credits,
    run_room40_key_north_to_credits,
    take_stairs_from_source,
)
from zelda_i.level9_stair_suffix import (
    _trial_summary,
    build_winning_fixture,
    run_suffix_from_fixture,
)
from zelda_i.runner import add_common_args, write_report

# Probe scripts import these names from the CLI module.
__all__ = [
    "BEAD",
    "TAG",
    "_apply_loader",
    "_exit_cellar",
    "_hold_until_room",
    "_walk_target",
    "build_winning_fixture",
    "dump_play_rooms",
    "dump_room_04",
    "dump_room_13",
    "dump_room_21",
    "dump_room_30",
    "dump_room_31",
    "dump_room_40",
    "dump_room_41",
    "dump_room_51",
    "dump_room_tiles",
    "main",
    "materialize_stair_room",
    "probe_cellar_dest_table",
    "probe_sources",
    "run_play_source_to_credits",
    "run_room04_bomb_west_to_credits",
    "run_room21_south_to_credits",
    "run_room30_stairs_to_credits",
    "run_room31_bomb_west_to_credits",
    "run_room40_key_north_to_credits",
    "run_suffix_from_fixture",
    "take_stairs_from_source",
]


def _dump_tag(args: argparse.Namespace, default: str) -> str:
    return args.tag if args.tag != TAG else default


def _write(name: str, payload: dict[str, Any]):
    return write_report(name, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_state="", default_tag=TAG, default_trials=1)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--dest-table", action="store_true")
    parser.add_argument("--dump-play", action="store_true")
    parser.add_argument("--dump-13", action="store_true")
    parser.add_argument("--dump-04", action="store_true")
    parser.add_argument("--compose-04", action="store_true")
    parser.add_argument("--dump-30", action="store_true")
    parser.add_argument("--compose-30", action="store_true")
    parser.add_argument("--dump-40", action="store_true")
    parser.add_argument("--compose-40", action="store_true")
    parser.add_argument("--dump-31", action="store_true")
    parser.add_argument("--compose-31", action="store_true")
    parser.add_argument("--dump-21", action="store_true")
    parser.add_argument("--compose-21", action="store_true")
    parser.add_argument("--dump-41", action="store_true")
    parser.add_argument("--compose-41", action="store_true")
    parser.add_argument("--dump-51", action="store_true")
    parser.add_argument("--compose-51", action="store_true")
    parser.add_argument("--play-source", default="", help="hex play room, e.g. 03")
    parser.add_argument("--build-fixture", action="store_true")
    parser.add_argument("--source", default="", help="hex stair source, e.g. 60")
    parser.add_argument("--cellar-side", default="left", choices=("left", "right"))
    args = parser.parse_args()

    if args.probe:
        probed = probe_sources(tag=f"{args.tag}_probe")
        out = _write(f"{args.tag}_probe", probed)
        print("PROBE winner", probed.get("winner"))
        print("REPORT", out)
        return 0 if probed.get("ok") else 1

    if args.dest_table:
        table = probe_cellar_dest_table(tag=f"{args.tag}_dest_table")
        out = _write(f"{args.tag}_dest_table", table)
        print("DEST_TABLE winner", table.get("winner"))
        print("REPORT", out)
        return 0 if table.get("ok") else 1

    if args.dump_13:
        dumped = dump_room_13(tag=_dump_tag(args, "l9_room13_dump"))
        out = _write(_dump_tag(args, "l9_room13_dump"), dumped)
        print("DUMP_13", {
            "ok": dumped.get("ok"),
            "loaded": dumped.get("loaded"),
            "clean_walk": dumped.get("clean_walk"),
            "how_up_opens": dumped.get("how_up_opens"),
            "north_uncleared": dumped.get("north_probe_uncleared"),
            "north_cleared": dumped.get("north_probe_cleared"),
        })
        print("REPORT", out)
        return 0 if dumped.get("ok") else 1

    if args.dump_04:
        dumped = dump_room_04(tag=_dump_tag(args, "l9_room04_dump"))
        out = _write(_dump_tag(args, "l9_room04_dump"), dumped)
        print("DUMP_04", {
            "ok": dumped.get("ok"),
            "loaded": dumped.get("loaded"),
            "lands_0x03": dumped.get("lands_0x03"),
            "dest_screen": dumped.get("dest_screen"),
            "bomb_west": (dumped.get("bomb_west") or {}).get("controller"),
            "stair_tile_at_03": dumped.get("stair_tile_at_03"),
            "next_candidate": dumped.get("next_candidate"),
        })
        print("REPORT", out)
        return 0 if dumped.get("ok") else 1

    if args.compose_04:
        trials: list[dict[str, Any]] = []
        tag = _dump_tag(args, "l9_play04_bombwest_patra_credits_recon")
        for trial_i in range(max(1, args.trials)):
            result = run_room04_bomb_west_to_credits(
                infinite_life=args.infinite_life,
                save_checkpoints=args.save_state,
                tag=tag,
                trial_i=trial_i,
            )
            trials.append(result)
            print(
                "COMPOSE04_TRIAL",
                {
                    "trial": trial_i,
                    "ok": result.get("ok"),
                    "dest_screen": result.get("dest_screen"),
                    "patra": (result.get("walk") or {}).get("landed_final_patra"),
                    "credits": result.get("credits_reached"),
                    "error": result.get("error"),
                },
            )
        report = {
            "bead": BEAD,
            "segment": "play_0x04_bomb_west_to_final_screen",
            "track": "recon_fixture",
            "route_eligible": False,
            "fixture_only": True,
            "init_mode9": False,
            "source_room": "0x04",
            "via": "bomb_west",
            "ok": all(trial.get("ok") for trial in trials),
            "trials": trials,
        }
        out = _write(tag, report)
        print("REPORT", out)
        return 0 if report["ok"] else 1

    if args.dump_30:
        dumped = dump_room_30(tag=_dump_tag(args, "l9_room30_dump"))
        out = _write(_dump_tag(args, "l9_room30_dump"), dumped)
        print("DUMP_30", {
            "ok": dumped.get("ok"),
            "loaded": dumped.get("loaded"),
            "stair_hits": dumped.get("stair_hits"),
            "entered_cellar_67": dumped.get("entered_cellar_67"),
            "dest_screen": dumped.get("dest_screen"),
            "lands_0x04": dumped.get("lands_0x04"),
            "next_candidate": dumped.get("next_candidate"),
        })
        print("REPORT", out)
        return 0 if dumped.get("ok") else 1

    if args.compose_30:
        trials = []
        tag = _dump_tag(args, "l9_play30_cellar67_patra_credits_recon")
        for trial_i in range(max(1, args.trials)):
            result = run_room30_stairs_to_credits(
                infinite_life=args.infinite_life,
                save_checkpoints=args.save_state,
                tag=tag,
                trial_i=trial_i,
            )
            trials.append(result)
            print(
                "COMPOSE30_TRIAL",
                {
                    "trial": trial_i,
                    "ok": result.get("ok"),
                    "dest_screen": result.get("dest_screen"),
                    "cellar": result.get("cellar_room"),
                    "patra": (result.get("walk_04") or {}).get("landed_final_patra")
                    or result.get("credits_reached"),
                    "credits": result.get("credits_reached"),
                    "error": result.get("error"),
                },
            )
        report = {
            "bead": BEAD,
            "segment": "play_0x30_cellar_0x67_right_to_final_screen",
            "track": "recon_fixture",
            "route_eligible": False,
            "fixture_only": True,
            "init_mode9": False,
            "source_room": "0x30",
            "via": "cellar_0x67_right",
            "ok": all(trial.get("ok") for trial in trials),
            "trials": trials,
        }
        out = _write(tag, report)
        print("REPORT", out)
        return 0 if report["ok"] else 1

    if args.dump_40:
        dumped = dump_room_40(tag=_dump_tag(args, "l9_room40_dump"))
        out = _write(_dump_tag(args, "l9_room40_dump"), dumped)
        print("DUMP_40", {
            "ok": dumped.get("ok"),
            "loaded": dumped.get("loaded"),
            "lands_0x30": dumped.get("lands_0x30"),
            "dest_screen": dumped.get("dest_screen"),
            "how_up_opens": dumped.get("how_up_opens"),
            "stair_tile_at_30": dumped.get("stair_tile_at_30"),
            "next_candidate": dumped.get("next_candidate"),
        })
        print("REPORT", out)
        return 0 if dumped.get("ok") else 1

    if args.compose_40:
        trials = []
        tag = _dump_tag(args, "l9_play40_keynorth_patra_credits_recon")
        for trial_i in range(max(1, args.trials)):
            result = run_room40_key_north_to_credits(
                infinite_life=args.infinite_life,
                save_checkpoints=args.save_state,
                tag=tag,
                trial_i=trial_i,
            )
            trials.append(result)
            print(
                "COMPOSE40_TRIAL",
                {
                    "trial": trial_i,
                    "ok": result.get("ok"),
                    "dest_screen": result.get("dest_screen"),
                    "patra": (result.get("walk_04") or {}).get("landed_final_patra")
                    or result.get("credits_reached"),
                    "credits": result.get("credits_reached"),
                    "error": result.get("error"),
                },
            )
        report = {
            "bead": BEAD,
            "segment": "play_0x40_key_north_to_final_screen",
            "track": "recon_fixture",
            "route_eligible": False,
            "fixture_only": True,
            "init_mode9": False,
            "source_room": "0x40",
            "via": "key_north",
            "ok": all(trial.get("ok") for trial in trials),
            "trials": trials,
        }
        out = _write(tag, report)
        print("REPORT", out)
        return 0 if report["ok"] else 1

    if args.dump_31:
        dumped = dump_room_31(tag=_dump_tag(args, "l9_room31_dump"))
        out = _write(_dump_tag(args, "l9_room31_dump"), dumped)
        print("DUMP_31", {
            "ok": dumped.get("ok"),
            "loaded": dumped.get("loaded"),
            "lands_0x30": dumped.get("lands_0x30"),
            "dest_screen": dumped.get("dest_screen"),
            "bomb_west": (dumped.get("bomb_west") or {}).get("controller"),
            "stairs_still_work": dumped.get("stairs_still_work"),
            "next_candidate": dumped.get("next_candidate"),
        })
        print("REPORT", out)
        return 0 if dumped.get("ok") else 1

    if args.compose_31:
        trials = []
        tag = _dump_tag(args, "l9_play31_bombwest_patra_credits_recon")
        for trial_i in range(max(1, args.trials)):
            result = run_room31_bomb_west_to_credits(
                infinite_life=args.infinite_life,
                save_checkpoints=args.save_state,
                tag=tag,
                trial_i=trial_i,
            )
            trials.append(result)
            print(
                "COMPOSE31_TRIAL",
                {
                    "trial": trial_i,
                    "ok": result.get("ok"),
                    "dest_screen": result.get("dest_screen"),
                    "patra": (result.get("walk_04") or {}).get("landed_final_patra")
                    or result.get("credits_reached"),
                    "credits": result.get("credits_reached"),
                    "error": result.get("error"),
                },
            )
        report = {
            "bead": BEAD,
            "segment": "play_0x31_bomb_west_to_final_screen",
            "track": "recon_fixture",
            "route_eligible": False,
            "fixture_only": True,
            "init_mode9": False,
            "source_room": "0x31",
            "via": "bomb_west",
            "ok": all(trial.get("ok") for trial in trials),
            "trials": trials,
        }
        out = _write(tag, report)
        print("REPORT", out)
        return 0 if report["ok"] else 1

    if args.dump_21:
        dumped = dump_room_21(tag=_dump_tag(args, "l9_room21_dump"))
        out = _write(_dump_tag(args, "l9_room21_dump"), dumped)
        print("DUMP_21", {
            "ok": dumped.get("ok"),
            "loaded": dumped.get("loaded"),
            "lands_0x31": dumped.get("lands_0x31"),
            "dest_screen": dumped.get("dest_screen"),
            "how_south_opens": dumped.get("how_south_opens"),
            "west_bomb_still_works": dumped.get("west_bomb_still_works"),
            "next_candidate": dumped.get("next_candidate"),
        })
        print("REPORT", out)
        return 0 if dumped.get("ok") else 1

    if args.compose_21:
        trials = []
        tag = _dump_tag(args, "l9_play21_south_patra_credits_recon")
        for trial_i in range(max(1, args.trials)):
            result = run_room21_south_to_credits(
                infinite_life=args.infinite_life,
                save_checkpoints=args.save_state,
                tag=tag,
                trial_i=trial_i,
            )
            trials.append(result)
            print(
                "COMPOSE21_TRIAL",
                {
                    "trial": trial_i,
                    "ok": result.get("ok"),
                    "dest_screen": result.get("dest_screen"),
                    "patra": (result.get("walk_04") or {}).get("landed_final_patra")
                    or result.get("credits_reached"),
                    "credits": result.get("credits_reached"),
                    "error": result.get("error"),
                },
            )
        report = {
            "bead": BEAD,
            "segment": "play_0x21_south_to_final_screen",
            "track": "recon_fixture",
            "route_eligible": False,
            "fixture_only": True,
            "init_mode9": False,
            "source_room": "0x21",
            "via": "south_shutter",
            "ok": all(trial.get("ok") for trial in trials),
            "trials": trials,
        }
        out = _write(tag, report)
        print("REPORT", out)
        return 0 if report["ok"] else 1

    if args.compose_41:
        trials = []
        tag = _dump_tag(args, "l9_play41_north_patra_credits_recon")
        for trial_i in range(max(1, args.trials)):
            result = run_room31_bomb_west_to_credits(
                infinite_life=args.infinite_life,
                save_checkpoints=args.save_state,
                tag=tag,
                trial_i=trial_i,
                from_41=True,
            )
            trials.append(result)
            print(
                "COMPOSE41_TRIAL",
                {
                    "trial": trial_i,
                    "ok": result.get("ok"),
                    "dest_screen": result.get("dest_screen"),
                    "patra": (result.get("walk_04") or {}).get("landed_final_patra")
                    or result.get("credits_reached"),
                    "credits": result.get("credits_reached"),
                    "error": result.get("error"),
                },
            )
        report = {
            "bead": BEAD,
            "segment": "play_0x41_north_to_final_screen",
            "track": "recon_fixture",
            "route_eligible": False,
            "fixture_only": True,
            "init_mode9": False,
            "source_room": "0x41",
            "via": "north_then_bomb_west",
            "ok": all(trial.get("ok") for trial in trials),
            "trials": trials,
        }
        out = _write(tag, report)
        print("REPORT", out)
        return 0 if report["ok"] else 1

    if args.dump_41:
        dumped = dump_room_41(tag=_dump_tag(args, "l9_room41_dump"))
        out = _write(_dump_tag(args, "l9_room41_dump"), dumped)
        print("DUMP_41", {
            "ok": dumped.get("ok"),
            "loaded": dumped.get("loaded"),
            "lands_0x31": dumped.get("lands_0x31"),
            "dest_screen": dumped.get("dest_screen"),
            "how_north_opens": dumped.get("how_north_opens"),
            "next_candidate": dumped.get("next_candidate"),
        })
        print("REPORT", out)
        return 0 if dumped.get("ok") else 1

    if args.dump_51:
        dumped = dump_room_51(tag=_dump_tag(args, "l9_room51_dump"))
        out = _write(_dump_tag(args, "l9_room51_dump"), dumped)
        print("DUMP_51", {
            "ok": dumped.get("ok"),
            "loaded": dumped.get("loaded"),
            "lands_0x41": dumped.get("lands_0x41"),
            "dest_screen": dumped.get("dest_screen"),
            "how_north_opens": dumped.get("how_north_opens"),
            "next_candidate": dumped.get("next_candidate"),
            "route_eligible": dumped.get("route_eligible"),
        })
        print("REPORT", out)
        return 0 if dumped.get("ok") else 1

    if args.compose_51:
        print("COMPOSE51 dest not attached until dump dest-YES is recorded")
        return 2

    if args.dump_play:
        dumped = dump_play_rooms(tag=f"{args.tag}_play_tiles")
        out = _write(f"{args.tag}_play_tiles", dumped)
        print("DUMP_PLAY rooms", len(dumped.get("rooms") or []))
        print("REPORT", out)
        return 0 if dumped.get("ok") else 1

    if args.play_source:
        play = int(args.play_source, 16)
        trials = []
        for trial_i in range(max(1, args.trials)):
            result = run_play_source_to_credits(
                source=play,
                cellar_side=args.cellar_side,
                infinite_life=args.infinite_life,
                save_checkpoints=args.save_state,
                tag=args.tag,
                trial_i=trial_i,
            )
            trials.append(result)
            print(
                "PLAY_TRIAL",
                {
                    "trial": trial_i,
                    "ok": result.get("ok"),
                    "source": result.get("source_room"),
                    "stair_tile": result.get("stair_tile"),
                    "stair_xy": result.get("stair_xy"),
                    "patra": (result.get("walk") or {}).get("landed_final_patra"),
                    "credits": result.get("credits_reached"),
                    "error": result.get("error"),
                },
            )
        report = {
            "bead": BEAD,
            "segment": "play_room_stairs_to_final_screen",
            "track": "recon_fixture",
            "route_eligible": False,
            "fixture_only": True,
            "init_mode9": False,
            "source_room": f"0x{play:02X}",
            "ok": all(trial.get("ok") for trial in trials),
            "trials": trials,
        }
        out = _write(args.tag, report)
        print("REPORT", out)
        return 0 if report["ok"] else 1

    source = int(args.source, 16) if args.source else 0
    fixture_name = (
        f"Level9Stair{source:02X}PatraEnteredReconFixture" if source else ""
    )
    built = None
    if args.build_fixture:
        if not source:
            print("ERROR --source is required with --build-fixture")
            return 2
        built = build_winning_fixture(
            source=source,
            cellar_side=args.cellar_side,
            tag=args.tag,
            fixture_name=fixture_name,
        )
        print("FIXTURE", {"ok": built.get("ok"), "path": built.get("checkpoint_path")})
        if not built.get("ok"):
            _write(args.tag, {"fixture": built, "ok": False})
            return 1

    start_state = args.from_state or fixture_name
    if not start_state:
        print("ERROR provide --from-state or --build-fixture --source")
        return 2

    trials = []
    for trial_i in range(max(1, args.trials)):
        result = run_suffix_from_fixture(
            start_state=start_state,
            infinite_life=args.infinite_life,
            save_checkpoints=args.save_state,
            tag=args.tag,
            trial_i=trial_i,
        )
        trials.append(result)
        print("TRIAL", _trial_summary(result))

    report = {
        "bead": BEAD,
        "segment": "stair_source_to_final_screen",
        "track": "recon_fixture",
        "route_eligible": False,
        "fixture_only": True,
        "fixture": built,
        "ok": all(trial.get("ok") for trial in trials),
        "trials": trials,
    }
    out = _write(args.tag, report)
    print("REPORT", out)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
