"""Materialize Level 9 room ``0x62`` and test it as the Patra predecessor.

This is a backwards-development tool, not a route result.  ``--probe`` /
``--build-fixture`` load the live Level 9 entrance, ask the game room loader
to materialize uncleared room ``0x62``, and dump a live snapshot.

Live + ROM (rr-sz8.3) **retarget**: ``0x62`` north is a wall, ``0x52`` south
is a wall.  Do not claim a ``0x62`` → Patra credits run.  The next predecessor
is a stairs / underground-passage drop into ``0x52``.

Examples::

    uv run python nes/zelda_i/scripts/run_level9_room62.py --probe
    uv run python nes/zelda_i/scripts/run_level9_room62.py --build-fixture
    uv run python nes/zelda_i/scripts/run_level9_room62.py \
      --from-state Level9Room62ReconFixture --infinite-life \
      --save-state --trials 1 --tag l9_room62_patra_credits_recon
"""

from __future__ import annotations

import argparse
from typing import Any

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.dungeon_ids import object_name
from zelda_i.dungeon_trace import compact_snapshot
from zelda_i.level9_ganon import LEVEL9
from zelda_i.level9_patra import (
    PATRA_EYE_COUNT,
    final_patra_live,
    patra_eyes,
)
from zelda_i.level9_room62 import (
    LEVEL9_STAIR_SOURCES,
    LOADER_CANDIDATES,
    NORTH_DOOR,
    ROOM_LEVEL9_62,
    Room62LoaderCandidate,
    door_bits,
    in_room_62,
    room62_is_cardinal_predecessor_of_patra,
    room62_object_summary,
    room62_to_patra_step,
    uncleared_room62,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_CUR_OPENED_DOORS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_NEXT_SCREEN,
    ADDR_OPEN_DOORWAY_MASK,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.scripts.run_level9_ganon import (
    FIXTURE_SOURCE,
    FULL_LOADOUT,
    _assign,
    _idle,
    _save_checkpoint,
    _step,
)
from zelda_i.scripts.run_level9_patra import _inventory_snapshot

BEAD = "rr-sz8.3"
FIXTURE_NAME = "Level9Room62ReconFixture"
STITCH_PIN_NAME = "Level9Room62PatraEnteredReconFixture"
TAG = "l9_room62_patra_credits_recon"
SETTLE_IDLE_FRAMES = 20
NORTH_PROBE_FRAMES = 180


def _loader_write_rows(candidate: Room62LoaderCandidate) -> list[dict[str, Any]]:
    rows = [
        {
            "name": name,
            "address": address,
            "address_hex": f"0x{address:04X}",
            "value": value,
        }
        for name, address, value in FULL_LOADOUT
    ]
    rows.extend(
        [
            {
                "name": "loader_level",
                "address": ADDR_LEVEL,
                "address_hex": "0x0010",
                "value": LEVEL9,
            },
            {
                "name": "loader_mode",
                "address": ADDR_MODE,
                "address_hex": "0x0012",
                "value": PLAY_MODE,
            },
            {
                "name": "loader_current_room",
                "address": ADDR_SCREEN,
                "address_hex": "0x00EB",
                "value": candidate.from_room,
            },
            {
                "name": "loader_next_room",
                "address": ADDR_NEXT_SCREEN,
                "address_hex": "0x00EC",
                "value": ROOM_LEVEL9_62,
            },
            {
                "name": "loader_link_position",
                "addresses": [ADDR_LINK_X, ADDR_LINK_Y],
                "address_hex": ["0x0070", "0x0084"],
                "values": [candidate.link_x, candidate.link_y],
            },
            {
                "name": "loader_door_staging",
                "addresses": [ADDR_CUR_OPENED_DOORS, ADDR_OPEN_DOORWAY_MASK],
                "address_hex": ["0x00EE", "0x033F"],
                "values": [0x0F, 0x0F],
            },
            {
                "name": "loader_hold_direction",
                "value": candidate.direction,
                "from_room": candidate.from_room,
                "label": candidate.label,
            },
        ]
    )
    return rows


def _apply_loader(env: Any, candidate: Room62LoaderCandidate) -> None:
    for _, address, value in FULL_LOADOUT:
        _assign(env, address, value)
    for address, value in (
        (ADDR_LEVEL, LEVEL9),
        (ADDR_MODE, PLAY_MODE),
        (ADDR_SCREEN, candidate.from_room),
        (ADDR_NEXT_SCREEN, ROOM_LEVEL9_62),
        (ADDR_LINK_X, candidate.link_x),
        (ADDR_LINK_Y, candidate.link_y),
        (ADDR_CUR_OPENED_DOORS, 0x0F),
        (ADDR_OPEN_DOORWAY_MASK, 0x0F),
    ):
        _assign(env, address, value)


def _hold_until_room62(
    env: Any,
    candidate: Room62LoaderCandidate,
    *,
    total: list[int],
    max_frames: int = 500,
):
    obs = None
    for _ in range(max_frames):
        obs = _step(env, nes_action(candidate.direction), assist=None, total=total)
        if in_room_62(read_snapshot(env.get_ram())):
            return obs, True
    return obs, False


def _room_report(snap: Any, ram: Any) -> dict[str, Any]:
    return {
        "snapshot": compact_snapshot(snap),
        "objects": room62_object_summary(snap),
        "cur_opened_doors": door_bits(snap.cur_opened_doors),
        "open_doorway_mask": door_bits(snap.open_doorway_mask),
        "room_all_dead": int(snap.room_all_dead),
        "room_obj_count": int(snap.room_obj_count),
        "inventory": _inventory_snapshot(ram),
    }


def _probe_north(env: Any, *, total: list[int]) -> dict[str, Any]:
    """Controller-only north push to record the live transition, if any."""
    start = read_snapshot(env.get_ram())
    obs = None
    transition: dict[str, Any] | None = None
    for _ in range(NORTH_PROBE_FRAMES):
        snap = read_snapshot(env.get_ram())
        if snap.screen != ROOM_LEVEL9_62 and not snap.transitioning:
            transition = {
                "from_room": ROOM_LEVEL9_62,
                "direction": "UP",
                "to_room": int(snap.screen),
                "objects": [
                    {
                        "slot": obj.slot,
                        "type_id": obj.type_id,
                        "type_name": object_name(obj.type_id),
                        "hp": obj.hp,
                        "x": obj.x,
                        "y": obj.y,
                    }
                    for obj in snap.objects
                    if 1 <= obj.slot <= 12 and (obj.type_id or obj.hp)
                ],
                "final_patra_live": bool(final_patra_live(snap)),
                "patra_eye_count": len(patra_eyes(snap)),
                "north_door_closed": not bool(snap.cur_opened_doors & NORTH_DOOR),
            }
            break
        frame = room62_to_patra_step(snap)
        obs = _step(env, frame.action, assist=None, total=total)
    end = read_snapshot(env.get_ram())
    return {
        "start_room": int(start.screen),
        "end_room": int(end.screen),
        "end_mode": int(end.mode),
        "link": {"x": end.link_x, "y": end.link_y},
        "entered_other_room": transition is not None,
        "transition": transition,
        "still_in_62": in_room_62(end),
    }


def build_or_probe(
    *,
    tag: str = f"{TAG}_probe",
    fixture_name: str = FIXTURE_NAME,
    save_fixture: bool = False,
    try_all: bool = False,
    preferred_label: str | None = None,
) -> dict[str, Any]:
    """Materialize uncleared 0x62, dump snapshot/PNG, optionally save state."""
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    candidates = list(LOADER_CANDIDATES)
    if preferred_label:
        named = [c for c in candidates if c.label == preferred_label]
        if not named:
            return {"ok": False, "error": f"unknown loader label {preferred_label}"}
        candidates = named + [c for c in candidates if c.label != preferred_label]

    attempts: list[dict[str, Any]] = []
    chosen: Room62LoaderCandidate | None = None
    chosen_obs = None
    chosen_env = None
    chosen_total: list[int] | None = None
    chosen_writes: list[dict[str, Any]] = []

    for candidate in candidates:
        env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
        total = [0]
        writes = _loader_write_rows(candidate)
        attempt: dict[str, Any] = {
            "label": candidate.label,
            "from_room": candidate.from_room,
            "direction": candidate.direction,
            "ok": False,
        }
        try:
            obs, _ = reset_obs(env)
            _apply_loader(env, candidate)
            obs, loaded = _hold_until_room62(env, candidate, total=total)
            attempt["loaded"] = loaded
            attempt["frames"] = total[0]
            if not loaded:
                attempt["final"] = compact_snapshot(read_snapshot(env.get_ram()))
                attempts.append(attempt)
                if not try_all:
                    env.close()
                    continue
                env.close()
                continue
            obs = _idle(env, SETTLE_IDLE_FRAMES, assist=None, total=total)
            snap = read_snapshot(env.get_ram())
            attempt.update(_room_report(snap, env.get_ram()))
            attempt["ok"] = bool(in_room_62(snap))
            screenshot = RECORDINGS_DIR / f"{tag}_{candidate.label}.png"
            save_rgb_png(obs, screenshot)
            attempt["screenshot"] = str(screenshot)
            attempts.append(attempt)
            if attempt["ok"] and chosen is None:
                chosen = candidate
                chosen_obs = obs
                chosen_env = env
                chosen_total = total
                chosen_writes = writes
                if not try_all:
                    break
                env.close()
                chosen_env = None
            else:
                env.close()
        except Exception:
            env.close()
            raise

    report: dict[str, Any] = {
        "ok": chosen is not None,
        "bead": BEAD,
        "track": "recon_fixture",
        "route_eligible": False,
        "fixture_only": True,
        "source_state": FIXTURE_SOURCE,
        "checkpoint": fixture_name if save_fixture else None,
        "attempts": attempts,
    }
    if chosen is None:
        report["error"] = "game room loader did not settle in Level 9 room 0x62"
        return report

    report["loader"] = {
        "label": chosen.label,
        "from_room": chosen.from_room,
        "direction": chosen.direction,
    }
    report["fixture_writes"] = chosen_writes

    env = chosen_env
    if env is None:
        # try_all closed the winning env; rebuild the first success to snapshot.
        env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
        chosen_total = [0]
        try:
            chosen_obs, _ = reset_obs(env)
            _apply_loader(env, chosen)
            chosen_obs, loaded = _hold_until_room62(env, chosen, total=chosen_total)
            if not loaded:
                report["ok"] = False
                report["error"] = "winning loader failed on rebuild"
                return report
            chosen_obs = _idle(env, SETTLE_IDLE_FRAMES, assist=None, total=chosen_total)
        except Exception:
            env.close()
            raise

    assert chosen_total is not None
    try:
        snap = read_snapshot(env.get_ram())
        report["room_entry"] = _room_report(snap, env.get_ram())
        report["ok"] = bool(in_room_62(snap))
        start_png = RECORDINGS_DIR / f"{tag}_start.png"
        save_rgb_png(chosen_obs, start_png)
        report["screenshot"] = str(start_png)

        north = _probe_north(env, total=chosen_total)
        report["north_probe"] = north
        after = read_snapshot(env.get_ram())
        north_png = RECORDINGS_DIR / f"{tag}_north_probe.png"
        save_rgb_png(chosen_obs if chosen_obs is not None else env.render(), north_png)
        # Capture the actual current frame after the north probe.
        idle_obs = _idle(env, 1, assist=None, total=chosen_total)
        save_rgb_png(idle_obs, north_png)
        report["north_probe_screenshot"] = str(north_png)
        report["after_north_probe"] = compact_snapshot(after)

        if report["ok"] and save_fixture:
            # Re-materialize a clean uncleared settle for the start fixture.
            # The north probe may have left the room; rebuild from source.
            env.close()
            env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
            total = [0]
            reset_obs(env)
            _apply_loader(env, chosen)
            _, loaded = _hold_until_room62(env, chosen, total=total)
            if not loaded:
                report["ok"] = False
                report["error"] = "fixture rebuild failed to settle in 0x62"
                return report
            _idle(env, SETTLE_IDLE_FRAMES, assist=None, total=total)
            settled = read_snapshot(env.get_ram())
            if not in_room_62(settled):
                report["ok"] = False
                report["error"] = "fixture rebuild left 0x62"
                return report
            path = _save_checkpoint(
                env,
                fixture_name,
                source_state=FIXTURE_SOURCE,
                phase="fully_loaded_uncleared_room_62",
                result={
                    "ok": True,
                    "room": ROOM_LEVEL9_62,
                    "loader": chosen.label,
                    "objects": room62_object_summary(settled),
                    "cur_opened_doors": int(settled.cur_opened_doors),
                    "open_doorway_mask": int(settled.open_doorway_mask),
                    "frames": total[0],
                },
                fixture_writes=chosen_writes,
                bead=BEAD,
            )
            report["checkpoint_path"] = str(path)
        return report
    finally:
        env.close()


def run_once(
    *,
    start_state: str = FIXTURE_NAME,
    infinite_life: bool = True,
    save_checkpoints: bool = False,
    tag: str = TAG,
    trial_i: int = 0,
) -> dict[str, Any]:
    """Replay the uncleared 0x62 settle and record the north-wall retarget.

    Does not invent a door policy.  A credits suffix is not attempted because
    live+ROM disprove ``0x62`` as the cardinal predecessor of ``0x52``.
    """
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    total = [0]
    report: dict[str, Any] = {
        "ok": False,
        "bead": BEAD,
        "track": "recon_fixture",
        "route_eligible": False,
        "fixture_only": True,
        "start_state": start_state,
        "trial": trial_i,
        "tag": tag,
        "cardinal_predecessor_of_0x52": room62_is_cardinal_predecessor_of_patra(),
        "retarget": {
            "reason": (
                "0x62 north is ROM wall code 1; 0x52 south is ROM wall code 1; "
                "live north push after kill-clear and bomb stays in 0x62"
            ),
            "next_hypothesis": "stairs_drop_into_0x52",
            "stair_sources": [f"0x{room:02X}" for room in LEVEL9_STAIR_SOURCES],
        },
        "runtime_controller_writes": {
            "object": 0,
            "room": 0,
            "door": 0,
            "inventory": 0,
            "progression": 0,
            "capacity": 0,
        },
        "checkpoints": [],
    }
    try:
        obs, _ = reset_obs(env)
        start = read_snapshot(env.get_ram())
        report["start"] = _room_report(start, env.get_ram())
        if not in_room_62(start):
            report["error"] = (
                f"expected L9 room 0x62, got L{start.level} "
                f"room 0x{start.screen:02X} mode {start.mode}"
            )
            return report
        report["uncleared_start"] = bool(uncleared_room62(start))
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_room62_start.png")
        if save_checkpoints:
            path = _save_checkpoint(
                env,
                start_state,
                source_state=start_state,
                phase="uncleared_room_62_replay",
                result={"ok": True, "room": ROOM_LEVEL9_62, "frames": total[0]},
                fixture_writes=[],
                bead=BEAD,
            )
            report["checkpoints"].append(str(path))
        north = _probe_north(env, total=total)
        report["north_probe"] = north
        idle_obs = _idle(env, 1, assist=None, total=total)
        save_rgb_png(idle_obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_room62_north_wall.png")
        report["error"] = (
            "retarget: 0x62 is not the live cardinal predecessor of 0x52"
        )
        report["credits_reached"] = False
        report["total_frames"] = total[0]
        return report
    finally:
        env.close()


def _trial_summary(report: dict[str, Any]) -> dict[str, Any]:
    loader = report.get("loader") or {}
    entry = report.get("room_entry") or {}
    north = report.get("north_probe") or {}
    return {
        "ok": report.get("ok"),
        "loader": loader.get("label"),
        "from_room": loader.get("from_room"),
        "object_count": len(entry.get("objects") or []),
        "objects": [
            f"{row.get('type_name')}@s{row.get('slot')}"
            for row in (entry.get("objects") or [])
        ],
        "cur_opened_doors": (entry.get("cur_opened_doors") or {}).get("raw"),
        "open_doorway_mask": (entry.get("open_doorway_mask") or {}).get("raw"),
        "north_probe_to": (north.get("transition") or {}).get("to_room"),
        "final_patra_live": (north.get("transition") or {}).get("final_patra_live"),
        "error": report.get("error"),
        "checkpoint_path": report.get("checkpoint_path"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-state", default=FIXTURE_NAME)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--build-fixture", action="store_true")
    parser.add_argument("--try-all-loaders", action="store_true")
    parser.add_argument("--loader-label", default=None)
    parser.add_argument("--infinite-life", action="store_true")
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--tag", default=TAG)
    args = parser.parse_args()

    if args.probe or args.build_fixture:
        built = build_or_probe(
            tag=f"{args.tag}_probe" if args.probe else f"{args.tag}_fixture",
            fixture_name=FIXTURE_NAME,
            save_fixture=args.build_fixture,
            try_all=args.try_all_loaders,
            preferred_label=args.loader_label,
        )
        print("PROBE" if args.probe else "FIXTURE", _trial_summary(built))
        out = RECORDINGS_DIR / (
            f"{args.tag}_probe.json" if args.probe else f"{args.tag}_fixture.json"
        )
        write_json_report(out, built)
        print("REPORT", out)
        if args.probe or not built.get("ok"):
            return 0 if built.get("ok") else 1
        if not args.infinite_life and args.trials == 1 and not args.save_state:
            return 0 if built.get("ok") else 1

    if args.probe:
        return 0

    trials: list[dict[str, Any]] = []
    for trial_i in range(max(1, args.trials)):
        result = run_once(
            start_state=args.from_state,
            infinite_life=args.infinite_life,
            save_checkpoints=args.save_state,
            tag=args.tag,
            trial_i=trial_i,
        )
        trials.append(result)
        print(
            "TRIAL",
            {
                "ok": result.get("ok"),
                "error": result.get("error"),
                "uncleared_start": result.get("uncleared_start"),
                "retarget": (result.get("retarget") or {}).get("next_hypothesis"),
            },
        )

    report = {
        "bead": BEAD,
        "segment": "room62_to_final_screen",
        "track": "recon_fixture",
        "route_eligible": False,
        "fixture_only": True,
        "ok": all(trial.get("ok") for trial in trials),
        "trials": trials,
    }
    out = RECORDINGS_DIR / f"{args.tag}.json"
    write_json_report(out, report)
    print("REPORT", out)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
