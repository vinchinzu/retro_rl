"""Run the explicit Level 9 pre-Ganon recon fixture through the ending.

This is a backwards-development tool, not a route result.  ``--build-fixture``
loads the live Level 9 entrance, asks the game to load the final-Patra room,
then composes a full inventory, clears that one room, and opens its north door.
Every such write is reported and the resulting states are marked development
only / route-ineligible.

Examples::

    uv run python zelda_i/scripts/run_level9_ganon.py --build-fixture
    uv run python zelda_i/scripts/run_level9_ganon.py \
      --from-state Level9BeforeGanonReconFixture \
      --infinite-life --save-state --trials 1 --tag l9_ganon_credits_recon
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from retro_harness.env import make_env, reset_obs, save_state, state_path
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_trace import compact_snapshot, write_state_provenance
from zelda_i.level9_ganon import (
    ADDR_GANON_SCENE_PHASE,
    B_ITEM_ARROWS,
    GANON_SCENE_FIGHT,
    GanonFightController,
    LEVEL9,
    ROOM_BEFORE_GANON,
    ROOM_GANON,
    ROOM_ZELDA,
    credits_rolling,
    final_ending_screen,
    ganon_defeated,
    ganon_object,
    in_ganon_fight,
    in_room_before_ganon,
    in_zelda_room,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOMBS,
    ADDR_BOOK,
    ADDR_BOOMERANG,
    ADDR_BOW,
    ADDR_BRACELET,
    ADDR_CANDLE,
    ADDR_CUR_OPENED_DOORS,
    ADDR_FOOD,
    ADDR_HEALTH,
    ADDR_KEYS,
    ADDR_LADDER,
    ADDR_LETTER,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MAGIC_BOOMERANG,
    ADDR_MAGIC_KEY,
    ADDR_MAGIC_SHIELD,
    ADDR_MAX_BOMBS,
    ADDR_MODE,
    ADDR_NEXT_SCREEN,
    ADDR_OBJ_TYPE,
    ADDR_OPEN_DOORWAY_MASK,
    ADDR_POTION,
    ADDR_RAFT,
    ADDR_RING,
    ADDR_ROD,
    ADDR_ROOM_ALL_DEAD,
    ADDR_ROOM_OBJ_COUNT,
    ADDR_RUPEES,
    ADDR_SCREEN,
    ADDR_SELECTED_ITEM,
    ADDR_SWORD,
    ADDR_TRIFORCE,
    ADDR_WHISTLE,
    PLAY_MODE,
    read_snapshot,
)

BEAD = "rr-sz8.1"
FIXTURE_NAME = "Level9BeforeGanonReconFixture"
FIXTURE_SOURCE = "Level9EntranceReconFixture"
NORTH_DOOR = 0x08

# Fully loaded first-quest inventory.  These are fixture writes, never assist
# writes and never eligible for a route/STATUS claim.
FULL_LOADOUT: tuple[tuple[str, int, int], ...] = (
    ("selected_item_silver_arrows", ADDR_SELECTED_ITEM, B_ITEM_ARROWS),
    ("magical_sword", ADDR_SWORD, 3),
    ("bombs", ADDR_BOMBS, 16),
    ("silver_arrows", ADDR_ARROWS, 2),
    ("bow", ADDR_BOW, 1),
    ("red_candle", ADDR_CANDLE, 2),
    ("whistle", ADDR_WHISTLE, 1),
    ("food", ADDR_FOOD, 1),
    ("red_potion", ADDR_POTION, 2),
    ("magic_rod", ADDR_ROD, 1),
    ("raft", ADDR_RAFT, 1),
    ("book", ADDR_BOOK, 1),
    ("red_ring", ADDR_RING, 2),
    ("ladder", ADDR_LADDER, 1),
    ("magic_key", ADDR_MAGIC_KEY, 1),
    ("bracelet", ADDR_BRACELET, 1),
    ("letter", ADDR_LETTER, 1),
    ("rupees", ADDR_RUPEES, 255),
    ("keys", ADDR_KEYS, 9),
    ("health_16_full", ADDR_HEALTH, 255),
    ("triforce_all_8", ADDR_TRIFORCE, 255),
    ("wood_boomerang", ADDR_BOOMERANG, 1),
    ("magic_boomerang", ADDR_MAGIC_BOOMERANG, 1),
    ("magic_shield", ADDR_MAGIC_SHIELD, 1),
    ("max_bombs", ADDR_MAX_BOMBS, 16),
)


def _assign(env: Any, address: int, value: int) -> None:
    env.unwrapped.data.memory.assign(int(address), "|u1", int(value) & 0xFF)


def _step(
    env: Any,
    action: list[int],
    *,
    assist: UnlimitedHealthAssist | None,
    total: list[int],
):
    obs, *_ = env.step(action)
    total[0] += 1
    if assist is not None:
        assist.apply_env(env, frame=total[0])
    return obs


def _idle(
    env: Any,
    frames: int,
    *,
    assist: UnlimitedHealthAssist | None,
    total: list[int],
):
    obs = None
    for _ in range(frames):
        obs = _step(env, nes_idle_action(), assist=assist, total=total)
    return obs


def _fixture_write_rows() -> list[dict[str, Any]]:
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
                "value": 0x62,
            },
            {
                "name": "loader_next_room",
                "address": ADDR_NEXT_SCREEN,
                "address_hex": "0x00EC",
                "value": ROOM_BEFORE_GANON,
            },
            {
                "name": "loader_link_position",
                "addresses": [ADDR_LINK_X, ADDR_LINK_Y],
                "address_hex": ["0x0070", "0x0084"],
                "values": [0x78, 0x58],
            },
            {
                "name": "loader_door_staging",
                "addresses": [
                    ADDR_CUR_OPENED_DOORS,
                    ADDR_OPEN_DOORWAY_MASK,
                ],
                "address_hex": ["0x00EE", "0x033F"],
                "values": [0x0F, 0x0F],
            },
            {
                "name": "clear_final_patra_object_slots",
                "address_range": "0x0350..0x035B",
                "value": 0,
            },
            {
                "name": "clear_room_object_count",
                "address": ADDR_ROOM_OBJ_COUNT,
                "address_hex": "0x034E",
                "value": 0,
            },
            {
                "name": "mark_room_all_dead",
                "address": ADDR_ROOM_ALL_DEAD,
                "address_hex": "0x034D",
                "value": 1,
            },
            {
                "name": "open_north_door",
                "addresses": [
                    ADDR_CUR_OPENED_DOORS,
                    ADDR_OPEN_DOORWAY_MASK,
                ],
                "value_or": NORTH_DOOR,
            },
            {
                "name": "final_link_position",
                "addresses": [ADDR_LINK_X, ADDR_LINK_Y],
                "address_hex": ["0x0070", "0x0084"],
                "values": [0x78, 0x58],
            },
        ]
    )
    return rows


def _write_provenance(
    path: Path,
    *,
    source_state: str,
    phase: str,
    result: dict[str, Any],
    fixture_writes: list[dict[str, Any]],
) -> None:
    source = state_path(GAME_DIR, GAME, source_state)
    write_state_provenance(
        path,
        source_state_path=source if source.exists() else None,
        request={
            "bead": BEAD,
            "phase": phase,
            "track": "recon_fixture",
            "route_eligible": False,
            "fixture_only": True,
            "fixture_writes": fixture_writes,
        },
        selected_trial=result,
        natural_entry=False,
    )


def _save_checkpoint(
    env: Any,
    name: str,
    *,
    source_state: str,
    phase: str,
    result: dict[str, Any],
    fixture_writes: list[dict[str, Any]],
) -> Path:
    path = save_state(env, GAME_DIR, GAME, name)
    _write_provenance(
        path,
        source_state=source_state,
        phase=phase,
        result=result,
        fixture_writes=fixture_writes,
    )
    return path


def build_fixture(*, tag: str = "l9_before_ganon_fixture") -> dict[str, Any]:
    """Compose and verify the fully loaded room-0x52 checkpoint."""
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    total = [0]
    writes = _fixture_write_rows()
    report: dict[str, Any] = {
        "ok": False,
        "bead": BEAD,
        "track": "recon_fixture",
        "route_eligible": False,
        "fixture_only": True,
        "source_state": FIXTURE_SOURCE,
        "checkpoint": FIXTURE_NAME,
        "fixture_writes": writes,
    }
    try:
        obs, _ = reset_obs(env)
        for _, address, value in FULL_LOADOUT:
            _assign(env, address, value)

        # Ask the real dungeon scroll/room loader to materialize room 0x52.
        for address, value in (
            (ADDR_LEVEL, LEVEL9),
            (ADDR_MODE, PLAY_MODE),
            (ADDR_SCREEN, 0x62),
            (ADDR_NEXT_SCREEN, ROOM_BEFORE_GANON),
            (ADDR_LINK_X, 0x78),
            (ADDR_LINK_Y, 0x58),
            (ADDR_CUR_OPENED_DOORS, 0x0F),
            (ADDR_OPEN_DOORWAY_MASK, 0x0F),
        ):
            _assign(env, address, value)

        loaded = False
        for _ in range(500):
            obs = _step(env, nes_action("UP"), assist=None, total=total)
            snap = read_snapshot(env.get_ram())
            if in_room_before_ganon(snap):
                loaded = True
                break
        if not loaded:
            report["error"] = "game room loader did not settle in Level 9 room 0x52"
            report["final"] = compact_snapshot(read_snapshot(env.get_ram()))
            return report

        # Fixture-only skip of the final Patra; this is the exact boundary the
        # user requested, not a claim that the room was naturally cleared.
        for slot in range(1, 13):
            _assign(env, ADDR_OBJ_TYPE + slot, 0)
        _assign(env, ADDR_ROOM_OBJ_COUNT, 0)
        _assign(env, ADDR_ROOM_ALL_DEAD, 1)
        _assign(
            env,
            ADDR_CUR_OPENED_DOORS,
            read_snapshot(env.get_ram()).cur_opened_doors | NORTH_DOOR,
        )
        _assign(
            env,
            ADDR_OPEN_DOORWAY_MASK,
            read_snapshot(env.get_ram()).open_doorway_mask | NORTH_DOOR,
        )
        obs = _idle(env, 30, assist=None, total=total)
        _assign(env, ADDR_LINK_X, 0x78)
        _assign(env, ADDR_LINK_Y, 0x58)
        obs = _step(env, nes_idle_action(), assist=None, total=total)

        before = read_snapshot(env.get_ram())
        report["before_ganon"] = compact_snapshot(before)
        report["ok"] = bool(
            in_room_before_ganon(before)
            and not any(obj.type_id for obj in before.objects[1:])
            and before.cur_opened_doors & NORTH_DOOR
        )
        screenshot = RECORDINGS_DIR / f"{tag}.png"
        save_rgb_png(obs, screenshot)
        report["screenshot"] = str(screenshot)
        if report["ok"]:
            path = _save_checkpoint(
                env,
                FIXTURE_NAME,
                source_state=FIXTURE_SOURCE,
                phase="fully_loaded_room_before_ganon",
                result={
                    "ok": True,
                    "room": ROOM_BEFORE_GANON,
                    "north_door_open": True,
                    "frames": total[0],
                },
                fixture_writes=writes,
            )
            report["checkpoint_path"] = str(path)
        return report
    finally:
        env.close()


def _checkpoint_result(env: Any, total: list[int]) -> dict[str, Any]:
    return {
        "ok": True,
        "frame": total[0],
        "state": compact_snapshot(read_snapshot(env.get_ram())),
    }


def _enter_ganon(
    env: Any,
    *,
    assist: UnlimitedHealthAssist | None,
    total: list[int],
):
    obs = None
    for _ in range(900):
        snap = read_snapshot(env.get_ram())
        ram = env.get_ram()
        if in_ganon_fight(snap) and int(ram[ADDR_GANON_SCENE_PHASE]) == GANON_SCENE_FIGHT:
            return obs, True
        obs = _step(
            env,
            nes_idle_action() if snap.screen == ROOM_GANON else nes_action("UP"),
            assist=assist,
            total=total,
        )
    return obs, False


def _collect_power_triforce(
    env: Any,
    *,
    assist: UnlimitedHealthAssist | None,
    total: list[int],
):
    obs = None
    for _ in range(1400):
        snap = read_snapshot(env.get_ram())
        if snap.cur_opened_doors & NORTH_DOOR:
            return obs, True
        boss = ganon_object(snap)
        if boss is None:
            action = nes_idle_action()
        elif abs(snap.link_x - boss.x) > 4:
            action = nes_action("RIGHT" if snap.link_x < boss.x else "LEFT")
        elif abs(snap.link_y - boss.y) > 4:
            action = nes_action("DOWN" if snap.link_y < boss.y else "UP")
        else:
            action = nes_idle_action()
        obs = _step(env, action, assist=assist, total=total)
    return obs, False


def _enter_zelda(
    env: Any,
    *,
    assist: UnlimitedHealthAssist | None,
    total: list[int],
):
    obs = None
    for _ in range(1200):
        snap = read_snapshot(env.get_ram())
        if in_zelda_room(snap):
            return obs, True
        if snap.screen == ROOM_GANON and abs(snap.link_x - 0x78) > 4:
            action = nes_action("RIGHT" if snap.link_x < 0x78 else "LEFT")
        else:
            action = nes_action("UP")
        obs = _step(env, action, assist=assist, total=total)
    return obs, False


def _rescue_zelda(
    env: Any,
    *,
    assist: UnlimitedHealthAssist | None,
    total: list[int],
):
    obs = None
    for frame in range(3500):
        snap = read_snapshot(env.get_ram())
        if snap.mode == 0x13:
            return obs, True
        if snap.link_x < 0x70:
            direction = "RIGHT"
        elif snap.link_x > 0x80:
            direction = "LEFT"
        elif snap.link_y > 0x95:
            direction = "UP"
        elif snap.link_y < 0x95:
            direction = "DOWN"
        else:
            direction = "UP"
        # Pulse A through the two guard fires; release frames allow movement.
        action = (
            nes_action(direction, "A")
            if frame % 12 == 0
            else nes_action(direction)
        )
        obs = _step(env, action, assist=assist, total=total)
    return obs, False


def run_once(
    *,
    start_state: str = FIXTURE_NAME,
    infinite_life: bool = True,
    save_checkpoints: bool = False,
    tag: str = "l9_ganon_credits_recon",
    trial_i: int = 0,
) -> dict[str, Any]:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    total = [0]
    writes = _fixture_write_rows()
    report: dict[str, Any] = {
        "ok": False,
        "bead": BEAD,
        "track": "recon_fixture",
        "route_eligible": False,
        "fixture_only": True,
        "fixture_writes_inherited": writes,
        "runtime_fixture_write_policy": {
            "selected_item_fallback": {
                "address": ADDR_SELECTED_ITEM,
                "value": B_ITEM_ARROWS,
                "condition": "only if the loaded fixture did not preselect arrows",
                "route_eligible": False,
            }
        },
        "start_state": start_state,
        "trial": trial_i,
        "tag": tag,
        "checkpoints": [],
    }

    def checkpoint(name: str, phase: str) -> None:
        if not save_checkpoints:
            return
        path = _save_checkpoint(
            env,
            name,
            source_state=start_state,
            phase=phase,
            result=_checkpoint_result(env, total),
            fixture_writes=writes,
        )
        report["checkpoints"].append(str(path))

    try:
        obs, _ = reset_obs(env)
        start = read_snapshot(env.get_ram())
        report["start"] = compact_snapshot(start)
        if not in_room_before_ganon(start):
            report["error"] = (
                f"expected L9 room 0x52, got L{start.level} "
                f"room 0x{start.screen:02X} mode {start.mode}"
            )
            return report
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_before_ganon.png")

        obs, entered = _enter_ganon(env, assist=assist, total=total)
        report["ganon_entered"] = entered
        if not entered:
            report["error"] = "failed to enter live Ganon room 0x42"
            return report
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_ganon_start.png")
        checkpoint("Level9GanonReconFixture", "ganon_fight_start")

        fight = GanonFightController().run(env, assist=assist, total=total)
        report["fight"] = fight
        if not fight["ok"] or not ganon_defeated(env.get_ram()):
            report["error"] = "Ganon controller timed out before Silver Arrow kill"
            return report
        obs = _idle(env, 1, assist=assist, total=total)
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_ganon_arrow_kill.png")

        obs, power = _collect_power_triforce(env, assist=assist, total=total)
        report["power_triforce_collected"] = power
        if not power:
            report["error"] = "Ganon died but north door did not open"
            return report
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_ganon_defeated.png")
        checkpoint("Level9GanonDefeatedReconFixture", "ganon_defeated_north_open")

        obs, zelda_room = _enter_zelda(env, assist=assist, total=total)
        report["zelda_room_entered"] = zelda_room
        if not zelda_room:
            report["error"] = "failed to enter live Zelda room 0x32"
            return report
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_zelda_room.png")
        checkpoint("Level9ZeldaRoomReconFixture", "zelda_room_entry")

        obs, rescued = _rescue_zelda(env, assist=assist, total=total)
        report["zelda_rescued"] = rescued
        if not rescued:
            report["error"] = "failed to clear guard fires and trigger Zelda ending"
            return report
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_ending_start.png")
        checkpoint("Level9EndingStartReconFixture", "ending_mode_start")

        credits_frame = None
        credits_capture_frame = None
        final_frame = None
        for _ in range(12000):
            snap = read_snapshot(env.get_ram())
            if credits_frame is None and credits_rolling(snap):
                credits_frame = total[0]
                # Preserve a visibly scrolling staff-credit frame, not the
                # static peace text still on screen at the submode boundary.
                obs = _idle(env, 240, assist=assist, total=total)
                credits_capture_frame = total[0]
                save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_credits.png")
                checkpoint("Level9CreditsReconFixture", "credits_rolling")
            if final_ending_screen(snap):
                final_frame = total[0]
                # Let the final nametable and the $40-frame input guard settle
                # before preserving the user-facing final-screen artifact.
                obs = _idle(env, 90, assist=assist, total=total)
                save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_final_screen.png")
                checkpoint("Level9FinalScreenReconFixture", "final_press_start_screen")
                break
            obs = _step(env, nes_idle_action(), assist=assist, total=total)

        report["credits_frame"] = credits_frame
        report["credits_capture_frame"] = credits_capture_frame
        report["final_screen_frame"] = final_frame
        report["credits_reached"] = credits_frame is not None
        report["final_screen_reached"] = final_frame is not None
        report["final"] = compact_snapshot(read_snapshot(env.get_ram()))
        report["total_frames"] = total[0]
        report["assist"] = assist.report() if assist is not None else {"enabled": False}
        report["ok"] = bool(credits_frame is not None and final_frame is not None)
        return report
    finally:
        env.close()


def _trial_summary(report: dict[str, Any]) -> dict[str, Any]:
    fight = report.get("fight") or {}
    return {
        "trial": report.get("trial"),
        "ok": report.get("ok"),
        "ganon_entered": report.get("ganon_entered"),
        "ganon_defeated": fight.get("last_boss_defeated"),
        "brown_seen": fight.get("brown_seen"),
        "hp_changes": fight.get("hp_changes"),
        "sword_pulses": fight.get("sword_pulses"),
        "arrow_pulses": fight.get("arrow_pulses"),
        "selected_item_writes": fight.get("selected_item_writes"),
        "zelda_rescued": report.get("zelda_rescued"),
        "credits_frame": report.get("credits_frame"),
        "final_screen_frame": report.get("final_screen_frame"),
        "total_frames": report.get("total_frames"),
        "error": report.get("error"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-state", default=FIXTURE_NAME)
    parser.add_argument("--build-fixture", action="store_true")
    parser.add_argument("--infinite-life", action="store_true")
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--tag", default="l9_ganon_credits_recon")
    args = parser.parse_args()

    built = None
    if args.build_fixture:
        built = build_fixture(tag=f"{args.tag}_fixture")
        print(
            "FIXTURE",
            _trial_summary(built)
            | {"ok": built.get("ok"), "path": built.get("checkpoint_path")},
        )
        if not built.get("ok"):
            write_json_report(
                RECORDINGS_DIR / f"{args.tag}.json",
                {"fixture": built, "ok": False},
            )
            return 1

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
        print("TRIAL", _trial_summary(result))

    report = {
        "bead": BEAD,
        "segment": "pre_ganon_to_final_screen",
        "track": "recon_fixture",
        "route_eligible": False,
        "fixture_only": True,
        "fixture": built,
        "ok": all(trial.get("ok") for trial in trials),
        "trials": trials,
    }
    out = RECORDINGS_DIR / f"{args.tag}.json"
    write_json_report(out, report)
    print("REPORT", out)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
