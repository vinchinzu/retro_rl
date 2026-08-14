"""Advance Stage2_Clear → Stage3 (West Side) and clear early waves.

Sodom HP-underflow leaves TCRF ``0x0CD2=0`` (same as Damnd). Segment
bridge: ``set_value(game_status, CLEAR_AREA)`` runs clear-area →
clear-round → **Break Car bonus** (``round=06``) → open West Side
(``round=02``). ``Stage1Policy`` then clears early West Side waves.
"""

from __future__ import annotations

import argparse
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from final_fight.paths import GAME, GAME_DIR, RECORDINGS_DIR
from final_fight.policy import Stage1Policy
from final_fight.ram import (
    ADDR_AREA,
    ADDR_BOSS_DEAD_FLAG,
    ADDR_GAME_STATUS,
    ADDR_ROUND,
    ADDR_ROUNDS_CLEARED,
    BOSS_BASE,
    ENEMY_BASES,
    ENTITY_HP_MAX,
    GameStatus,
    OFF_HP,
    OFF_STATUS,
    OFF_X,
    OFF_Y,
    RoundId,
    parse_game_state,
    read_u8,
    read_u16le,
)
from retro_harness.env import get_available_states, make_env, reset_obs, save_state
from retro_harness.actions import buttons, idle_action
from retro_harness.ram_state import GameMode, GameState
from retro_harness.segment_runner import (
    SegmentOutcome,
    WaveChainTracker,
    configure_headless,
    save_rgb_png,
    snapshot_state,
    write_json_report,
)

WEST_SIDE = int(RoundId.WEST_SIDE)
BREAK_CAR = int(RoundId.BREAK_CAR)
MIN_SAVE_HP = 12
ENGAGE_DX = 110
WEST_CAM_LO = 630
WEST_CAM_HI = 720
STATE_DIR = GAME_DIR / "custom_integrations" / "FinalFight-Snes"

class W5Tactic(Enum):
    """Post-w4 dual 142+96 finish recipes (not pure alt60_3 chip)."""

    ALT = "alt"
    THROW = "throw"
    BAIT = "bait"
    KICK = "kick"
    SPLIT = "split"

def _maybe_save_state(
    env: Any,
    name: str,
    saved_states: list[str],
    *,
    overwrite: bool = False,
) -> bool:
    """Save ``name.state`` unless an existing resume should be kept."""
    dest = STATE_DIR / f"{name}.state"
    if dest.exists() and not overwrite:
        if dest.name not in saved_states:
            saved_states.append(dest.name)
            print(f"keep existing {dest.name}")
        return False
    path = save_state(env, GAME_DIR, GAME, name)
    if path.name not in saved_states:
        saved_states.append(path.name)
    return True

def _west_wave_offset(state_name: str, *, has_living: bool) -> int:
    """Segment-local wave index offset for Mid/Clear resumes."""
    if "Clear_w5" in state_name or "Mid_w6" in state_name:
        return 5
    if "Clear_w4" in state_name or "Mid_w5" in state_name:
        return 4
    if "Clear_w3" in state_name or "Mid_w4" in state_name:
        return 3
    if "Clear_w2" in state_name or "Mid_w3" in state_name:
        return 2
    if "Clear_w1" in state_name:
        return 1
    if "Mid_w2" in state_name:
        # Empty Mid_p31 is post-clear; Mid_p66 still in wave2.
        return 2 if not has_living else 1
    return 0

def _west_pack_target(
    living: list[Any],
    player_x: int,
    *,
    tactic: W5Tactic = W5Tactic.ALT,
) -> Any:
    """Prefer Andore; dual toughs kill weak-first; else nearest.

    Andore (HP≥200) is focus. Post-w4 dual 142+96 collapses faster by
    deleting the weaker first. Wave3 crumbs (HP≤55) stay weak-first.

    ``SPLIT``: while the weak is knockdown (st01), 1v1 the tough —
    verified chip window (142→105) before the weak wakes.
    """
    andores = [e for e in living if e.health >= 200]
    if andores:
        return max(andores, key=lambda e: e.health)
    behind = [e for e in living if (e.x - player_x) < 0]
    front = [e for e in living if (e.x - player_x) >= 0]
    if len(living) >= 2:
        if tactic is W5Tactic.SPLIT:
            weak = min(living, key=lambda e: e.health)
            tough = max(living, key=lambda e: e.health)
            if (
                weak is not tough
                and _entity_status(weak) == 1
                and _entity_status(tough) == 3
            ):
                return tough
        return min(
            living,
            key=lambda e: (e.health, abs(e.x - player_x)),
        )
    if behind and (
        not front
        or max(e.health for e in behind) >= max(e.health for e in front)
    ):
        return min(behind, key=lambda e: abs(e.x - player_x))
    return min(
        living,
        key=lambda e: (e.health, abs(e.x - player_x)),
    )

def _lane_vert(target_y: int, player_y: int) -> str:
    """D-pad toward target depth.

    Memory Y increases toward the foreground: UP raises Y, DOWN lowers
    Y. Older ``DOWN if target_y > player_y`` was inverted and parked
    Guy on the wrong lane for West Side duals.
    """
    return "UP" if target_y > player_y else "DOWN"

def _pack_with_spawns(
    living: list[Any], spawn_l: list[dict[str, int]]
) -> list[Any]:
    """Merge status-01 intros (off-screen left) into pack targeting."""
    pack: list[Any] = list(living)
    living_slots = {e.slot for e in living}
    for s in spawn_l:
        if s["slot"] in living_slots or s["hp"] <= 0:
            continue
        pack.append(
            SimpleNamespace(
                slot=s["slot"],
                x=s["x"],
                y=s["y"],
                health=s["hp"],
                animation=1,
            )
        )
    return pack

def _entity_status(entity: Any) -> int:
    """Combat status byte (``animation``); default drawn/fighting."""
    return int(getattr(entity, "animation", 3))

def _w5_setup_walk_frames(state_name: str, mid_tag: str) -> int:
    """Forced walk-past only from entry / Clear_w4 / fresh dual mid.

    Chip / 1v1 resumes already sat walk-past — re-running 55f lets the
    tough close and flips weak from behind back to ahead.
    """
    if mid_tag != "w5" and "Mid_w5" not in state_name:
        if "Clear_w4" in state_name:
            return 55
        return 0
    if any(
        tag in state_name
        for tag in ("chip", "1v1", "true1v1", "end_")
    ):
        return 0
    return 55

def _w5_dual_action(
    *,
    tactic: W5Tactic,
    sx: int,
    wdx: int,
    wadx: int,
    tdx: int,
    dy: int,
    target_hp: int,
    player_hp: int,
    player_y: int,
    target_y: int,
    weak_st: int,
    tough_close: bool,
    frame_j: int,
) -> tuple[Any, str]:
    """Choose a frame action for post-w4 dual (weak + tough).

    ``ALT`` is the prior JD-left 3/8 + LEFT+Y 5/8 recipe. ``THROW``
    finishes weak with UP+Y / toward+Y grab once in band (no wall
    chase). ``BAIT`` parks mid-left and only commits when the tough is
    far. ``KICK`` prefers jump-dash kick-band hits over punch chip.
    """
    toward = "RIGHT" if wdx > 0 else "LEFT"
    if sx > 155:
        return buttons("B", "LEFT"), "west_pack_flee"

    # Shared: knockdown / far weak — hold mid pocket, do not chase
    # into the wall. Crumb exception: if tough is far, ease LEFT so
    # we re-band when the weak stands (probe left_y kills w14/w22).
    # SPLIT tough window: do not idle — fall through to kick-band.
    if (
        tactic is not W5Tactic.SPLIT or target_hp < 100
    ) and (weak_st == 1 or wadx > 80):
        crumb = target_hp <= 30
        tough_adx = abs(tdx)
        if crumb and tough_adx > 70 and wdx < 0 and sx > 55:
            return buttons("LEFT"), "west_pack_mid"
        if tough_close and tdx > 0 and sx > 100:
            return buttons("B", "LEFT"), "west_pack_space"
        if sx < 65:
            return buttons("RIGHT"), "west_pack_mid"
        if sx > 130:
            return buttons("B", "LEFT"), "west_pack_flee"
        return idle_action(), "west_pack_gap"

    if tough_close and player_hp <= 25:
        if tdx > 0 and sx > 100:
            return buttons("B", "LEFT"), "west_pack_space"
        if sx < 70:
            return buttons("RIGHT"), "west_pack_mid"
        return idle_action(), "west_pack_space"

    if tactic is W5Tactic.BAIT:
        # Split: park left, engage only when tough is far or KD.
        tough_adx = abs(tdx)
        if sx > 110:
            return buttons("B", "LEFT"), "west_pack_flee"
        if sx < 55:
            return buttons("RIGHT"), "west_pack_mid"
        if tough_adx < 55 and weak_st != 1:
            # Tough in range — space / bait, do not trade dual.
            away = "LEFT" if tdx > 0 and sx > 80 else "RIGHT"
            if away == "LEFT" and sx < 70:
                away = "RIGHT"
            return buttons("B", away), "west_pack_space"
        if dy > 5 and wadx > 20:
            return (
                buttons(_lane_vert(target_y, player_y)),
                "west_pack_align",
            )
        if wadx <= 18 and dy <= 8 and wdx < 0:
            # Grab finish once isolated in band.
            cycle = frame_j % 3
            if cycle == 0:
                return buttons("UP", "Y"), "west_pack_throw"
            if cycle == 1:
                return buttons(toward, "Y"), "west_pack_throw"
            return buttons("LEFT", "Y"), "west_pack_face"
        if wadx <= 55 and wdx < 0:
            return buttons("LEFT", "Y"), "west_pack_face"
        if wdx > 15:
            return buttons("RIGHT"), "west_pack_mid"
        return idle_action(), "west_pack_gap"

    if tactic is W5Tactic.THROW:
        # Align in place, then grab/throw — finish without wall chase.
        crumb = target_hp <= 30
        if dy > 5 and not (target_hp <= 50 and wadx <= 70):
            return (
                buttons(_lane_vert(target_y, player_y)),
                "west_pack_align",
            )
        if wdx > 20 and not crumb:
            return buttons("RIGHT"), "west_pack_mid"
        if wadx < 12:
            # Overlap: UP+Y / away to trigger throw, not punch mash.
            if frame_j % 4 < 2:
                return buttons("UP", "Y"), "west_pack_throw"
            return buttons("RIGHT"), "west_pack_space"
        if wadx <= 28 and dy <= 12:
            cycle = frame_j % 4
            if cycle == 0:
                return buttons("UP", "Y"), "west_pack_throw"
            if cycle == 1:
                return buttons(toward, "Y"), "west_pack_throw"
            if cycle == 2:
                return buttons("LEFT", "Y"), "west_pack_face"
            return idle_action(), "west_pack_gap"
        if crumb and wdx < 0 and wadx <= 100:
            # Crumb weak: commit LEFT+Y (probe: finishes w14/w22).
            # Pure LEFT stalls forever while weak backpedals.
            return buttons("LEFT", "Y"), "west_pack_face"
        if target_hp <= 50 and wdx < 0 and wadx <= 70:
            # Close to grab band without JD (JD drops latch).
            if wadx > 28:
                return buttons("LEFT", "Y"), "west_pack_face"
            return buttons("UP", "Y"), "west_pack_throw"
        if sx < 65 or frame_j % 8 >= 3:
            return buttons("LEFT", "Y"), "west_pack_face"
        return buttons("B", "LEFT"), "west_pack_jd"

    if tactic is W5Tactic.KICK:
        # Door-style kick band: JD toward when dx 35–75, face when close.
        if dy > 8 and wadx > 25:
            vert = _lane_vert(target_y, player_y)
            return buttons(vert, toward), "west_pack_align"
        if wadx < 28:
            away = "LEFT" if wdx > 0 else "RIGHT"
            if away == "LEFT" and sx < 70:
                away = "RIGHT"
            return buttons("B", away), "west_pack_space"
        if 35 <= wadx <= 75:
            if frame_j % 10 < 6:
                return buttons("B", toward), "west_pack_jd"
            return buttons(toward, "Y"), "west_pack_face"
        if wadx <= 103:
            return buttons("B", toward), "west_pack_jd"
        if wdx > 0:
            return buttons("RIGHT"), "west_pack_mid"
        return buttons("B", "LEFT"), "west_pack_flee"

    if tactic is W5Tactic.SPLIT:
        # KD the weak with LEFT+Y, then during st01 window the caller
        # retargets to the tough — behind LEFT+Y (JD whiffs on HP142).
        if target_hp >= 100:
            if wadx < 18 and wdx >= 0:
                return buttons("RIGHT"), "west_pack_space"
            if wdx >= 0 and wadx <= 55:
                return buttons("RIGHT"), "west_pack_mid"
            if wdx < 0 and wadx <= 90:
                if frame_j % 3 == 2:
                    return idle_action(), "west_pack_gap"
                return buttons("LEFT", "Y"), "west_pack_face"
            if dy > 10 and wadx > 18:
                vert = _lane_vert(target_y, player_y)
                return buttons(vert, toward), "west_pack_align"
            return buttons(toward, "Y"), "west_pack_face"
        # Weak still up: LEFT+Y KD / crumb finish (gap every 3rd frame).
        if sx > 140:
            return buttons("B", "LEFT"), "west_pack_flee"
        if dy > 8 and wadx > 30:
            return (
                buttons(_lane_vert(target_y, player_y)),
                "west_pack_align",
            )
        if frame_j % 3 == 2:
            return idle_action(), "west_pack_gap"
        return buttons("LEFT", "Y"), "west_pack_face"

    # ALT (default): prior align-in-place + alt60_3 LEFT+Y commit.
    if dy > 5 and not (target_hp <= 50 and wadx <= 55):
        return (
            buttons(_lane_vert(target_y, player_y)),
            "west_pack_align",
        )
    if wdx > 20:
        return buttons("RIGHT"), "west_pack_mid"
    if wadx < 20:
        return buttons("RIGHT"), "west_pack_space"
    if target_hp <= 50 and wdx < 0 and wadx <= 70:
        return buttons("LEFT", "Y"), "west_pack_face"
    if sx < 65 or frame_j % 8 >= 3:
        return buttons("LEFT", "Y"), "west_pack_face"
    return buttons("B", "LEFT"), "west_pack_jd"


def _snap_ram(ram: Any) -> dict[str, int]:
    return {
        "game_status": read_u8(ram, ADDR_GAME_STATUS),
        "round": read_u8(ram, ADDR_ROUND),
        "area": read_u8(ram, ADDR_AREA),
        "rounds_cleared": read_u8(ram, ADDR_ROUNDS_CLEARED),
        "boss_status": read_u8(ram, BOSS_BASE + OFF_STATUS),
        "boss_hp": read_u8(ram, BOSS_BASE + OFF_HP),
        "boss_dead_flag": read_u8(ram, ADDR_BOSS_DEAD_FLAG),
    }

def _living_brief(ram: Any, cam: int, px: int) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    for i, base in enumerate(ENEMY_BASES):
        status = read_u8(ram, base + OFF_STATUS)
        hp = read_u8(ram, base + OFF_HP)
        x = read_u16le(ram, base + OFF_X)
        y = read_u16le(ram, base + OFF_Y)
        if status == 3 and 0 < hp <= ENTITY_HP_MAX:
            out.append(
                {
                    "slot": i,
                    "hp": hp,
                    "x": x,
                    "y": y,
                    "sx": x - cam,
                    "dx": x - px,
                }
            )
    return out

def _spawn_living_brief(
    ram: Any, cam: int, px: int
) -> list[dict[str, int]]:
    """Living status-01 intros (West Side right-edge chip zone)."""
    out: list[dict[str, int]] = []
    for i, base in enumerate(ENEMY_BASES):
        status = read_u8(ram, base + OFF_STATUS)
        hp = read_u8(ram, base + OFF_HP)
        x = read_u16le(ram, base + OFF_X)
        y = read_u16le(ram, base + OFF_Y)
        if status == 1 and 0 < hp <= ENTITY_HP_MAX:
            out.append(
                {
                    "slot": i,
                    "hp": hp,
                    "x": x,
                    "y": y,
                    "sx": x - cam,
                    "dx": x - px,
                }
            )
    return out

def _is_engage_ready(
    state: GameState, enemies: list[dict[str, int]]
) -> bool:
    if not enemies:
        return False
    if state.stage != WEST_SIDE:
        return False
    if state.mode is not GameMode.PLAYING:
        return False
    if not (0 < state.health <= 128 and state.lives > 0):
        return False
    nearest = min(enemies, key=lambda e: abs(e["dx"]))
    sx = state.player_x - state.camera_x
    return abs(nearest["dx"]) <= ENGAGE_DX and 40 <= sx <= 180

def _advance_action(
    status: int,
    round_id: int,
    state: GameState,
    enemies: list[dict[str, int]],
) -> Any:
    """Controller during bridge / bonus / walk-to-engage."""
    if status in (
        GameStatus.CLEAR_AREA,
        GameStatus.CLEAR_ROUND,
        GameStatus.OPEN_STAGE_A,
        GameStatus.OPEN_STAGE_B,
        GameStatus.CHARACTER_SELECT,
    ):
        return buttons("START")
    # Break Car bonus: mash Y (+ RIGHT) to finish the car faster.
    if (
        round_id == BREAK_CAR
        or status == GameStatus.BONUS_GAMEPLAY
    ):
        return buttons("RIGHT", "Y")
    if (
        round_id == WEST_SIDE
        and status == GameStatus.ACTIVE_GAMEPLAY
    ):
        if enemies:
            nearest = min(enemies, key=lambda e: abs(e["dx"]))
            sx = state.player_x - state.camera_x
            if nearest["dx"] > ENGAGE_DX and sx > 120:
                return buttons("LEFT")
            if nearest["dx"] > 80:
                return idle_action()
            return buttons("RIGHT")
        return buttons("RIGHT")
    return idle_action()

def run_stage3_advance(
    *,
    state_name: str = "Stage2_Clear",
    max_advance_frames: int = 6000,
    max_fight_frames: int = 12000,
    target_waves: int | None = 6,
    out_dir: Path | None = None,
    save_stage3: bool = True,
    bridge_clear_area: bool | None = None,
    w5_tactic: W5Tactic = W5Tactic.ALT,
    heal_hp: int | None = None,
) -> dict[str, Any]:
    """Load Stage2_Clear / Stage3*, bridge if needed, clear West Side waves.

    Mid-stage resumes (``Stage3*``) skip CLEAR_AREA when already in
    West Side play. ``heal_hp`` optionally pokes ``player_hp`` once at
    fight start (document in STATUS if used).
    """
    configure_headless()
    available = get_available_states(GAME, GAME_DIR)
    if state_name not in available:
        raise SystemExit(
            f"missing state {state_name}; have {available[:8]}"
        )
    out = out_dir or (RECORDINGS_DIR / "stage3_advance")
    out.mkdir(parents=True, exist_ok=True)

    if bridge_clear_area is None:
        bridge_clear_area = state_name.startswith("Stage2")

    env = make_env(GAME, state_name, GAME_DIR, render_mode="rgb_array")
    transitions: list[dict[str, Any]] = []
    screenshots: list[str] = []
    saved_states: list[str] = []
    heal_used: int | None = None
    obs, _ = reset_obs(env)
    ram = env.get_ram()
    state = parse_game_state(ram, frame=0)
    start = {**snapshot_state(state), **_snap_ram(ram)}
    transitions.append({"frame": 0, **start})
    png = save_rgb_png(obs, out / "s3_0000_start.png")
    screenshots.append(png.name)

    already_west = (
        start["round"] == WEST_SIDE
        and start["game_status"] == int(GameStatus.ACTIVE_GAMEPLAY)
        and 0 < state.health <= 128
        and state.lives > 0
    )

    bridged = False
    if bridge_clear_area and not already_west:
        # Underflow Sodom leaves 0x0CD2=0; CLEAR_AREA advances the round
        # (via Break Car bonus → West Side).
        env.set_value("game_status", int(GameStatus.CLEAR_AREA))
        bridged = True
        ram = env.get_ram()
        state = parse_game_state(ram, frame=0)
        print(
            "bridge CLEAR_AREA "
            f"(cd2 was {start.get('boss_dead_flag', 0)}; "
            f"status now 0x{read_u8(ram, ADDR_GAME_STATUS):02X})"
        )

    clear_round_frame: int | None = None
    bonus_frame: int | None = None
    west_frame: int | None = None
    fight_ready_frame: int | None = None
    stage3_snap: dict[str, Any] | None = None
    outcome = "advance_timeout"
    last_key: tuple[Any, ...] | None = None

    if already_west:
        fight_ready_frame = 0
        enemies = _living_brief(ram, state.camera_x, state.player_x)
        stage3_snap = {**start, "enemies": enemies}
        outcome = "stage3_resume"
        west_frame = 0
        print(
            f"resume {state_name} hp={state.health} lives={state.lives} "
            f"cam={state.camera_x} threats={len(state.threat_enemies)} "
            f"living={len(enemies)}"
        )
        png = save_rgb_png(obs, out / "s3_0000_resume.png")
        screenshots.append(png.name)
    else:
        for frame_i in range(1, max_advance_frames + 1):
            ram = env.get_ram()
            state = parse_game_state(ram, frame=frame_i)
            status = read_u8(ram, ADDR_GAME_STATUS)
            snap = {**snapshot_state(state), **_snap_ram(ram)}
            enemies = _living_brief(
                ram, state.camera_x, state.player_x
            )
            key = (
                snap["game_status"],
                snap["round"],
                snap["area"],
                snap["rounds_cleared"],
                len(enemies),
                state.camera_x // 64,
            )
            if key != last_key:
                last_key = key
                transitions.append(
                    {"frame": frame_i, **snap, "enemies": enemies}
                )
                print(
                    f"frame={frame_i} status=0x{snap['game_status']:02X} "
                    f"round={snap['round']} area={snap['area']} "
                    f"rc={snap['rounds_cleared']} "
                    f"hp={state.health} lives={state.lives} "
                    f"cam={state.camera_x} enemies={len(enemies)}"
                )
                tag = (
                    f"st{snap['game_status']:02X}"
                    f"_r{snap['round']}_a{snap['area']}"
                )
                png = save_rgb_png(
                    obs, out / f"s3_{frame_i:04d}_{tag}.png"
                )
                screenshots.append(png.name)

            if (
                clear_round_frame is None
                and snap["game_status"] == GameStatus.CLEAR_ROUND
            ):
                clear_round_frame = frame_i

            if bonus_frame is None and snap["round"] == BREAK_CAR:
                bonus_frame = frame_i
                png = save_rgb_png(
                    obs, out / f"s3_{frame_i:04d}_break_car.png"
                )
                screenshots.append(png.name)

            if west_frame is None and snap["round"] == WEST_SIDE:
                west_frame = frame_i
                png = save_rgb_png(
                    obs, out / f"s3_{frame_i:04d}_west_side.png"
                )
                screenshots.append(png.name)

            if _is_engage_ready(state, enemies):
                fight_ready_frame = frame_i
                stage3_snap = {**snap, "enemies": enemies}
                outcome = "stage3_ready"
                if save_stage3:
                    path = save_state(env, GAME_DIR, GAME, "Stage3")
                    saved_states.append(path.name)
                    print(
                        f"Stage3 saved frame={frame_i} "
                        f"hp={state.health} lives={state.lives} "
                        f"cam={state.camera_x} dx={enemies[0]['dx']}"
                    )
                png = save_rgb_png(
                    obs, out / f"s3_{frame_i:04d}_fight_ready.png"
                )
                screenshots.append(png.name)
                break

            if state.player_dead or state.health > 128:
                outcome = "advance_death"
                break

            action = _advance_action(
                status, snap["round"], state, enemies
            )
            obs, _r, _t, _tr, _info = env.step(action)

    fight_report: dict[str, Any] | None = None
    if fight_ready_frame is not None and target_waves != 0:
        policy = Stage1Policy()
        tracker = WaveChainTracker(
            max_frames=max_fight_frames,
            clear_hold_frames=30,
            target_waves=target_waves,
            stop_on_boss=True,
        )
        tracker.begin(state)
        last_cleared = 0
        # Tracker wave index is segment-local; Mid/Clear resumes bump
        # saved wave numbers and keep cam640 mid-screen pack AI.
        west_wave_offset = _west_wave_offset(
            state_name, has_living=bool(state.living_enemies)
        )
        if (
            heal_hp is not None
            and 1 <= heal_hp <= 80
            and 0 < state.health < heal_hp
        ):
            env.set_value("player_hp", heal_hp)
            heal_used = heal_hp
            ram = env.get_ram()
            state = parse_game_state(ram, frame=state.frame)
            print(
                f"heal poke player_hp→{heal_hp} "
                f"(was start hp={start.get('health')})"
            )
        print(f"w5_tactic={w5_tactic.value}")
        west_post_w1 = west_wave_offset >= 1
        pending_save_wave: int | None = None
        fight_outcome = SegmentOutcome.TIMEOUT
        far_cam_saved = False
        area_bridged = False
        area1_saved = False
        # After cam640 pack clear, player sits sx≈230 while st=01
        # intros at sx≈310+ chip the right edge. Force left park once.
        # Clear_w2 / empty Mid_p31 already parked mid-screen.
        start_sx = state.player_x - state.camera_x
        edge_park_done = west_wave_offset >= 2 or (
            west_post_w1
            and not state.living_enemies
            and start_sx <= 100
        )
        mid_tag = (
            "w5"
            if west_wave_offset >= 4
            else (
                "w4"
                if west_wave_offset >= 3
                else ("w3" if west_wave_offset >= 2 else "w2")
            )
        )
        # Clear_w4 sits sx≈232 with HP142 already spawning — must
        # JD-left before dual 142+96 engages. Mid_w5_entry / mid mids
        # already mid: skip entry.
        clear_w4_edge = (
            (
                west_wave_offset >= 4
                or "Clear_w4" in state_name
                or "Mid_w5" in state_name
            )
            and "entry" not in state_name
            and start_sx > 140
        )
        if clear_w4_edge:
            # Right-edge Clear_w4: run JD-left entry from 0.
            w2_entry_left = 0
        elif "Mid_w5_entry" in state_name or (
            "Mid_w5" in state_name and start_sx <= 130
        ):
            # Already mid for post-w4 dual.
            w2_entry_left = 90
        elif west_wave_offset >= 2 or (
            "Mid_w2" in state_name and state.living_enemies
        ):
            w2_entry_left = 90
        else:
            w2_entry_left = 0
        # Forced walk-past window for post-w4 dual (probe: ~55f).
        # Skip on chip/1v1 resumes — geometry is already walk-past.
        w5_walk_left = _w5_setup_walk_frames(state_name, mid_tag)
        w3_saw_dual = (
            "Mid_w3_e" in state_name
            or "Mid_w3_chip" in state_name
            or "Clear_w3" in state_name
            or "Clear_w4" in state_name
            or "Clear_w5" in state_name
            or "andore" in state_name
            or "Mid_w5" in state_name
        )
        early_clear_saved = False

        for frame_j in range(1, max_fight_frames + 1):
            sx = state.player_x - state.camera_x
            # Softlock guard: if scroll stalls with no living enemies,
            # CLEAR_AREA poke advances West Side sub-area (same as subway).
            if (
                not area_bridged
                and state.mode is GameMode.PLAYING
                and state.stage == WEST_SIDE
                and state.room == 0
                and state.camera_x >= 920
                and not state.living_enemies
                and 0 < state.health <= 128
                and state.lives > 0
            ):
                env.set_value(
                    "game_status", int(GameStatus.CLEAR_AREA)
                )
                area_bridged = True
                print(
                    f"west area0 bridge CLEAR_AREA "
                    f"cam={state.camera_x} hp={state.health} "
                    f"L={state.lives}"
                )
                png = save_rgb_png(
                    obs,
                    out
                    / (
                        f"s3_{fight_ready_frame + frame_j:04d}"
                        f"_area_bridge.png"
                    ),
                )
                screenshots.append(png.name)

            # Cam640: after pack clear, st=01 intros at sx≈310–400 chip
            # the right-edge park. JD-left escapes hitstun; LEFT alone
            # freezes at sx≈232. Prefer face-Y once mid-screen.
            spawn_l = _spawn_living_brief(
                env.get_ram(), state.camera_x, state.player_x
            )
            living = state.living_enemies
            # Keep mid-screen pack AI for wave2+ (continuous or resume).
            in_west_pack = (
                state.camera_x >= WEST_CAM_LO
                and state.camera_x <= WEST_CAM_HI
                and (
                    west_post_w1
                    or last_cleared >= 1
                )
            )
            wave3_plus = (
                west_wave_offset + last_cleared >= 2
                or mid_tag in ("w3", "w4")
            )
            pack = (
                _pack_with_spawns(list(living), spawn_l)
                if (in_west_pack and wave3_plus)
                else list(living)
            )
            # After wave1 clear: ~90f JD-left to sx≈90 before wave2
            # engage (st=01 chips the right edge through hitstun).
            if (
                in_west_pack
                and w2_entry_left < 90
                and state.mode is GameMode.PLAYING
                and state.stage == WEST_SIDE
                and sx > 95
                and 0 < state.health <= 128
            ):
                action = buttons("B", "LEFT")
                tracker.note_reason("west_w2_entry")
                w2_entry_left += 1
            # Wave2+ mid-screen pack combat. Wave3 sandwich (behind
            # HP112) uses behind-tough focus; wave2 keeps weak-first.
            # Include status-01 intros so leftovers are not abandoned.
            elif (
                in_west_pack
                and state.mode is GameMode.PLAYING
                and state.stage == WEST_SIDE
                and len(pack) >= 1
                and 0 < state.health <= 128
            ):
                if wave3_plus and len(pack) >= 2:
                    w3_saw_dual = True
                if wave3_plus:
                    target = _west_pack_target(
                        pack, state.player_x, tactic=w5_tactic
                    )
                else:
                    target = min(
                        living,
                        key=lambda e: (
                            e.health,
                            abs(e.x - state.player_x),
                        ),
                    )
                wdx = target.x - state.player_x
                wadx = abs(wdx)
                tough = max(pack, key=lambda e: e.health)
                tdx = tough.x - state.player_x
                front_close = any(
                    (e.x - state.player_x) > 0
                    and abs(e.x - state.player_x) < 36
                    and abs(e.y - state.player_y) < 18
                    for e in living
                )
                wave_mid = "w3" if wave3_plus else "w2"
                mid_prefix = f"Stage3_Mid_{wave_mid}"
                if (
                    MIN_SAVE_HP <= state.health <= 128
                    and state.lives > 0
                    and 70 <= sx <= 140
                    and not any(
                        abs(e.x - state.player_x) < 40 for e in living
                    )
                ):
                    name = (
                        f"{mid_prefix}_p{state.health}"
                        f"_cam{state.camera_x}"
                    )
                    already = any(
                        mid_prefix in s for s in saved_states
                    )
                    # Prefer Mid_w2_p66; never clobber any existing mid.
                    if already:
                        pass
                    elif _maybe_save_state(
                        env, name, saved_states
                    ):
                        print(f"{wave_mid} mid save → {name}.state")
                # After dual: true 1v1 leftover (pack merges st01).
                if (
                    wave3_plus
                    and w3_saw_dual
                    and len(pack) == 1
                    and pack[0].health <= 55
                    and MIN_SAVE_HP <= state.health <= 128
                    and state.lives > 0
                    and not any(
                        "Stage3_Mid_w3_e" in s for s in saved_states
                    )
                ):
                    crumb = pack[0]
                    name = (
                        f"Stage3_Mid_w3_e{int(crumb.health)}"
                        f"_p{state.health}_cam{state.camera_x}"
                    )
                    if _maybe_save_state(env, name, saved_states):
                        print(f"w3 crumb mid → {name}.state")
                pack_hp = sum(e.health for e in pack)
                if (
                    wave3_plus
                    and w3_saw_dual
                    and 0 < pack_hp <= 70
                    and MIN_SAVE_HP <= state.health <= 128
                    and state.lives > 0
                    and not any(
                        "Stage3_Mid_w3_chip" in s for s in saved_states
                    )
                ):
                    name = (
                        f"Stage3_Mid_w3_chip_p{state.health}"
                        f"_e{pack_hp}_cam{state.camera_x}"
                    )
                    if _maybe_save_state(env, name, saved_states):
                        print(f"w3 chip mid → {name}.state")
                # Low HP: prioritize space over punches — except wave5
                # finish when the weak is already crumb-low.
                weak_finish = (
                    mid_tag == "w5"
                    and len(pack) >= 1
                    and min(e.health for e in pack) <= 30
                )
                if (
                    wave3_plus
                    and state.health <= 12
                    and not weak_finish
                    and any(
                        abs(e.x - state.player_x) < 45 for e in pack
                    )
                ):
                    away = "LEFT" if sx > 100 else "RIGHT"
                    action = buttons("B", away)
                    tracker.note_reason("west_pack_panic")
                    obs, _r, _t, _tr, _info = env.step(action)
                    abs_frame = fight_ready_frame + frame_j
                    ram = env.get_ram()
                    state = parse_game_state(ram, frame=abs_frame)
                    stop = tracker.update(state)
                    if stop is not None:
                        fight_outcome = stop
                        break
                    continue
                # Post-w4 dual: forced setup walk matching probe (~55f).
                if (
                    w5_walk_left > 0
                    and mid_tag == "w5"
                    and len(pack) >= 2
                    and tough.health < 200
                    and 0 < state.health <= 128
                ):
                    w5_walk_left -= 1
                    # Same geometry gates as the working setup55 probe.
                    if wdx > -35:
                        if abs(target.y - state.player_y) > 3:
                            action = buttons(
                                _lane_vert(target.y, state.player_y),
                                "RIGHT",
                            )
                            tracker.note_reason("west_pack_align")
                        else:
                            action = buttons("RIGHT")
                            tracker.note_reason("west_pack_mid")
                    elif abs(target.y - state.player_y) > 5:
                        action = buttons(
                            _lane_vert(target.y, state.player_y)
                        )
                        tracker.note_reason("west_pack_align")
                    else:
                        action = idle_action()
                        tracker.note_reason("west_pack_gap")
                    obs, _r, _t, _tr, _info = env.step(action)
                    abs_frame = fight_ready_frame + frame_j
                    ram = env.get_ram()
                    state = parse_game_state(ram, frame=abs_frame)
                    stop = tracker.update(state)
                    if stop is not None:
                        fight_outcome = stop
                        break
                    continue
                # After walk budget, peel off the right-edge chip zone.
                if (
                    mid_tag == "w5"
                    and len(pack) >= 2
                    and sx > 155
                    and 0 < state.health <= 128
                ):
                    action = buttons("B", "LEFT")
                    tracker.note_reason("west_pack_flee")
                    obs, _r, _t, _tr, _info = env.step(action)
                    abs_frame = fight_ready_frame + frame_j
                    ram = env.get_ram()
                    state = parse_game_state(ram, frame=abs_frame)
                    stop = tracker.update(state)
                    if stop is not None:
                        fight_outcome = stop
                        break
                    continue
                if (
                    sx < 55
                    and wdx >= 0
                    and not (
                        mid_tag == "w5" and len(pack) >= 2
                    )
                    and not (
                        mid_tag == "w5"
                        and len(pack) == 1
                        and pack[0].health > 55
                    )
                ):
                    action = buttons("RIGHT")
                    tracker.note_reason("west_pack_nudge")
                elif (
                    sx < 40
                    and wdx < 0
                    and wadx > 55
                    and w3_saw_dual
                    and len(pack) == 1
                    and not (
                        len(living) == 1 and living[0].health <= 55
                    )
                    and not (
                        mid_tag == "w5" and pack[0].health > 55
                    )
                ):
                    # Chase far-behind leftover only — hugging the left
                    # wall into UF ghosts/chips is a death sentence.
                    # Wave5 tough 1v1: never chase KD flyaway.
                    action = buttons("LEFT", "Y")
                    tracker.note_reason("west_pack_chase")
                elif (
                    wave3_plus
                    and len(pack) >= 2
                    and wdx < -80
                    and sx < 70
                    and mid_tag != "w5"
                ):
                    # Dual + far-behind spawn: stay mid, do not hug wall.
                    # Wave5 finish-chase needs LEFT toward the weak KD —
                    # do not steal those frames into a RIGHT mid walk.
                    action = buttons("RIGHT")
                    tracker.note_reason("west_pack_mid")
                elif sx > 185 and wdx > 50:
                    # Only JD-left when the pack is still ahead on the
                    # chip edge — do not flee forever past mid-screen.
                    action = buttons("B", "LEFT")
                    tracker.note_reason("west_pack_flee")
                elif (
                    wave3_plus
                    and front_close
                    and wdx < 0
                    and mid_tag != "w5"
                ):
                    # Wave3 sandwich only — w5 dual uses alt JD/face.
                    if sx > 120:
                        action = buttons("B", "LEFT")
                    elif wadx <= 70:
                        action = buttons("LEFT", "Y")
                    else:
                        action = buttons("B", "LEFT")
                    tracker.note_reason("west_pack_behind")
                elif (
                    wave3_plus
                    and len(living) >= 1
                    and tough.health >= 200
                ):
                    # Andore (HP≈216): right-edge JD kick band (dx≈35–70).
                    toward = "RIGHT" if tdx > 0 else "LEFT"
                    away = "LEFT" if tdx > 0 else "RIGHT"
                    if away == "LEFT" and sx < 80:
                        away = "RIGHT"
                    if sx < 220 and abs(tdx) > 100:
                        # Only rush the kick edge while Andore is far.
                        action = buttons("B", "RIGHT")
                        tracker.note_reason("west_pack_andore_park")
                    elif abs(tdx) < 40:
                        action = buttons("B", away)
                        tracker.note_reason("west_pack_space")
                    elif abs(tough.y - state.player_y) > 12:
                        vert = _lane_vert(tough.y, state.player_y)
                        action = buttons(vert, toward)
                        tracker.note_reason("west_pack_align")
                    elif state.frame % 20 < 10:
                        action = buttons("B", toward)
                        tracker.note_reason("west_pack_jd")
                    else:
                        action = buttons(toward, "Y")
                        tracker.note_reason("west_pack_face")
                elif (
                    wave3_plus
                    and len(pack) >= 2
                    and tough.health < 200
                    and (mid_tag == "w5" or west_wave_offset >= 4)
                ):
                    # Post-w4 dual: tactic-driven finish (throw/bait/
                    # kick/alt). Wait mid on st01 / wadx>75 — do not
                    # chase-LEFT (weak outruns and wall-traps).
                    dy = abs(target.y - state.player_y)
                    weak_st = _entity_status(target)
                    tough_close = (
                        abs(tdx) < 32
                        and abs(tough.y - state.player_y) < 18
                    )
                    action, reason = _w5_dual_action(
                        tactic=w5_tactic,
                        sx=sx,
                        wdx=wdx,
                        wadx=wadx,
                        tdx=tdx,
                        dy=dy,
                        target_hp=int(target.health),
                        player_hp=state.health,
                        player_y=state.player_y,
                        target_y=int(target.y),
                        weak_st=weak_st,
                        tough_close=tough_close,
                        frame_j=frame_j,
                    )
                    tracker.note_reason(reason)
                    # Snapshot Mid_w5 while dual is still healthy / chip.
                    if (
                        mid_tag == "w5"
                        and MIN_SAVE_HP <= state.health <= 128
                        and state.lives > 0
                        and 55 <= sx <= 145
                    ):
                        weak_hp = min(e.health for e in pack)
                        if weak_hp <= 60:
                            name = (
                                f"Stage3_Mid_w5_chip_p{state.health}"
                                f"_w{int(weak_hp)}"
                                f"_t{int(tough.health)}"
                                f"_cam{state.camera_x}"
                            )
                            already = name + ".state" in saved_states
                            deeper = (
                                not any(
                                    "Stage3_Mid_w5_chip" in s
                                    for s in saved_states
                                )
                                or weak_hp <= 35
                            )
                            if (
                                deeper
                                and not already
                                and _maybe_save_state(
                                    env,
                                    name,
                                    saved_states,
                                    overwrite=weak_hp <= 35,
                                )
                            ):
                                print(f"w5 chip mid → {name}.state")
                        elif not any(
                            "Stage3_Mid_w5_p" in s
                            or "Stage3_Mid_w5_entry" in s
                            for s in saved_states
                        ):
                            name = (
                                f"Stage3_Mid_w5_p{state.health}"
                                f"_e{int(pack_hp)}_cam{state.camera_x}"
                            )
                            if _maybe_save_state(
                                env, name, saved_states
                            ):
                                print(f"w5 dual mid → {name}.state")
                elif (
                    wave3_plus
                    and mid_tag == "w5"
                    and len(pack) == 1
                    and pack[0].health > 55
                ):
                    # True 1v1 on remaining tough (HP≈142): grounded
                    # LEFT+Y chips through KD (142→102→65→…). Wait mid
                    # while st01 flies far (dx≲−80) — chase = death.
                    # Do NOT JD (whiffs).
                    lone = pack[0]
                    ldx = lone.x - state.player_x
                    ladx = abs(ldx)
                    lone_st = _entity_status(lone)
                    toward = "RIGHT" if ldx > 0 else "LEFT"
                    away = "LEFT" if ldx > 0 else "RIGHT"
                    if away == "LEFT" and sx < 80:
                        away = "RIGHT"
                    if state.health <= 12:
                        if ladx < 45:
                            action = buttons("B", away)
                        elif sx < 70:
                            action = buttons("RIGHT")
                        elif sx > 150:
                            action = buttons("B", "LEFT")
                        else:
                            action = idle_action()
                        tracker.note_reason("west_pack_space")
                    elif sx > 165:
                        action = buttons("B", "LEFT")
                        tracker.note_reason("west_pack_flee")
                    elif lone_st == 1 and (ldx < -70 or ladx > 75):
                        # KD flyaway — hold mid pocket until return.
                        if sx < 70:
                            action = buttons("RIGHT")
                            tracker.note_reason("west_pack_mid")
                        elif sx > 120:
                            action = buttons("B", "LEFT")
                            tracker.note_reason("west_pack_flee")
                        else:
                            action = idle_action()
                            tracker.note_reason("west_pack_gap")
                    elif abs(lone.y - state.player_y) > 12 and ladx > 24:
                        vert = _lane_vert(lone.y, state.player_y)
                        action = buttons(vert, toward)
                        tracker.note_reason("west_pack_align")
                    elif ladx <= 100:
                        if frame_j % 3 == 2:
                            action = idle_action()
                            tracker.note_reason("west_pack_gap")
                        else:
                            action = buttons("LEFT", "Y")
                            tracker.note_reason("west_pack_face")
                    else:
                        action = buttons(toward)
                        tracker.note_reason("west_pack_mid")
                    if (
                        MIN_SAVE_HP <= state.health <= 128
                        and state.lives > 0
                        and 55 <= sx <= 145
                    ):
                        name = (
                            f"Stage3_Mid_w5_true1v1_p{state.health}"
                            f"_e{int(lone.health)}"
                            f"_cam{state.camera_x}"
                        )
                        # One save per HP band — avoid per-frame spam.
                        band = f"_e{int(lone.health)}"
                        already_band = any(
                            "Stage3_Mid_w5_true1v1" in s
                            and band in s
                            for s in saved_states
                        )
                        if not already_band and _maybe_save_state(
                            env,
                            name,
                            saved_states,
                            overwrite=lone.health <= 80,
                        ):
                            print(f"w5 true1v1 → {name}.state")
                elif (
                    wave3_plus
                    and w3_saw_dual
                    and len(pack) == 1
                    and pack[0].health <= 55
                    and abs(pack[0].x - state.player_x) <= 90
                ):
                    # Patient 1v1 before align — pack may merge st01/UF.
                    crumb = pack[0]
                    cdx = crumb.x - state.player_x
                    cadx = abs(cdx)
                    toward = "RIGHT" if cdx > 0 else "LEFT"
                    if abs(crumb.y - state.player_y) > 12 and cadx > 24:
                        vert = _lane_vert(crumb.y, state.player_y)
                        action = buttons(vert, toward)
                        tracker.note_reason("west_pack_align")
                    elif cadx < 28:
                        away = "LEFT" if sx > 90 else "RIGHT"
                        action = buttons("B", away)
                        tracker.note_reason("west_pack_space")
                    elif cadx <= 42:
                        if state.frame % 4 < 2:
                            action = buttons(toward, "Y")
                            tracker.note_reason("west_pack_attack")
                        else:
                            action = idle_action()
                            tracker.note_reason("west_pack_gap")
                    else:
                        action = (
                            buttons(toward)
                            if state.frame % 2 == 0
                            else buttons(toward, "Y")
                        )
                        tracker.note_reason("west_pack_attack")
                elif (
                    wave3_plus
                    and len(pack) == 1
                    and abs(target.y - state.player_y) > 10
                    and wadx > 20
                ):
                    # Non-crumb 1v1 lane align (true leftover only).
                    vert = _lane_vert(target.y, state.player_y)
                    toward = "RIGHT" if wdx > 0 else "LEFT"
                    action = buttons(vert, toward)
                    tracker.note_reason("west_pack_align")
                    # Snapshot healthy 1v1 after dual.
                    if (
                        mid_tag == "w5"
                        and MIN_SAVE_HP <= state.health <= 128
                        and state.lives > 0
                        and 55 <= sx <= 145
                        and not any(
                            "Stage3_Mid_w5_1v1" in s
                            or "Stage3_Mid_w5_true1v1" in s
                            for s in saved_states
                        )
                    ):
                        name = (
                            f"Stage3_Mid_w5_1v1_p{state.health}"
                            f"_e{int(target.health)}"
                            f"_cam{state.camera_x}"
                        )
                        if _maybe_save_state(env, name, saved_states):
                            print(f"w5 1v1 mid → {name}.state")
                elif any(
                    abs(e.x - state.player_x) < 30
                    and abs(e.y - state.player_y) < 18
                    for e in living
                ):
                    away = "LEFT" if tdx > 0 else "RIGHT"
                    if away == "LEFT" and sx < 70:
                        away = "RIGHT"
                    action = buttons("B", away)
                    tracker.note_reason("west_pack_space")
                elif wadx <= 42:
                    toward = "RIGHT" if wdx > 0 else "LEFT"
                    if state.frame % 3 < 2:
                        action = buttons(toward, "Y")
                        tracker.note_reason("west_pack_attack")
                    else:
                        action = idle_action()
                        tracker.note_reason("west_pack_gap")
                elif wadx <= 103:
                    toward = "RIGHT" if wdx > 0 else "LEFT"
                    # Wave3 behind tough: grounded face-Y (subway HP148).
                    if (
                        wave3_plus
                        and wdx < 0
                        and state.frame % 4 < 3
                    ):
                        action = buttons(toward, "Y")
                        tracker.note_reason("west_pack_face")
                    elif state.frame % 30 < 16:
                        action = buttons("B", toward)
                        tracker.note_reason("west_pack_jd")
                    else:
                        action = buttons(toward, "Y")
                        tracker.note_reason("west_pack_attack")
                else:
                    toward = "RIGHT" if wdx > 0 else "LEFT"
                    action = buttons("B", toward)
                    tracker.note_reason("west_pack_jd")
            elif (
                state.mode is GameMode.PLAYING
                and state.stage == WEST_SIDE
                and state.camera_x >= WEST_CAM_LO
                and state.camera_x <= WEST_CAM_HI
                and sx > 170
                and 0 < state.health <= 128
                and not any(
                    abs(e.x - state.player_x) <= 38 for e in living
                )
            ):
                # Wave1 + inter-wave: JD-left off sx≈232 chip edge
                # unless a thug is already in punch band.
                action = buttons("B", "LEFT")
                tracker.note_reason("west_edge_jd")
            elif (
                (last_cleared >= 1 or west_post_w1)
                and state.mode is GameMode.PLAYING
                and state.stage == WEST_SIDE
                and state.camera_x >= WEST_CAM_LO
                and state.camera_x <= 700
                and not living
                and spawn_l
                and 0 < state.health <= 128
            ):
                nearest = min(spawn_l, key=lambda e: abs(e["dx"]))
                # Wave5 tough KD flyaway (st01 HP still high, far left):
                # wait mid — LEFT+Y chase is a false-clear death.
                if (
                    mid_tag == "w5"
                    and nearest["hp"] > 40
                    and nearest["dx"] < -70
                ):
                    if sx < 70:
                        action = buttons("RIGHT")
                    elif sx > 120:
                        action = buttons("B", "LEFT")
                    else:
                        action = idle_action()
                    tracker.note_reason("west_pack_gap")
                elif sx > 130:
                    action = buttons("B", "LEFT")
                    tracker.note_reason("west_spawn_plant")
                elif sx > 100:
                    action = buttons("LEFT", "Y")
                    tracker.note_reason("west_spawn_plant")
                else:
                    # Wave3 behind intro (sx≈-32): face-Y / JD-left.
                    if nearest["dx"] < -40:
                        action = buttons("LEFT", "Y")
                    elif abs(nearest["dx"]) < 45:
                        action = buttons("Y")
                    elif nearest["dx"] > 60:
                        action = buttons("B", "RIGHT")
                    elif nearest["dx"] > 0:
                        action = buttons("RIGHT", "Y")
                    else:
                        action = buttons("LEFT", "Y")
                    tracker.note_reason("west_spawn_plant")
            elif (
                not edge_park_done
                and (last_cleared >= 1 or west_post_w1)
                and state.mode is GameMode.PLAYING
                and state.stage == WEST_SIDE
                and state.camera_x >= WEST_CAM_LO
                and state.camera_x <= 700
                and not living
                and sx > 100
                and 0 < state.health <= 128
            ):
                action = (
                    buttons("B", "LEFT")
                    if sx > 130
                    else buttons("LEFT", "Y")
                )
                tracker.note_reason("west_edge_park")
            elif (
                not edge_park_done
                and (last_cleared >= 1 or west_post_w1)
                and state.mode is GameMode.PLAYING
                and not living
                and sx <= 100
                and 0 < state.health <= 128
            ):
                edge_park_done = True
                print(
                    f"west edge park done sx={sx} "
                    f"hp={state.health} cam={state.camera_x}"
                )
                if (
                    MIN_SAVE_HP <= state.health <= 128
                    and state.lives > 0
                ):
                    name = (
                        f"Stage3_Mid_{mid_tag}_p{state.health}"
                        f"_cam{state.camera_x}"
                    )
                    if _maybe_save_state(env, name, saved_states):
                        print(f"{mid_tag} mid save → {name}.state")
                tick = policy.tick(state)
                if tick.action is not None:
                    tracker.note_reason(tick.action.reason)
                    action = tick.action.action
                else:
                    action = idle_action()
                    tracker.note_reason("west_edge_park_done")
            elif area_bridged and state.mode is not GameMode.PLAYING:
                action = idle_action()
                tracker.note_reason("area_bridge_wait")
            else:
                tick = policy.tick(state)
                if tick.action is not None:
                    tracker.note_reason(tick.action.reason)
                    action = tick.action.action
                else:
                    tracker.note_reason(tick.reason or "no_action")
                    action = idle_action()
            obs, _r, _t, _tr, _info = env.step(action)
            abs_frame = fight_ready_frame + frame_j
            ram = env.get_ram()
            state = parse_game_state(ram, frame=abs_frame)

            if (
                area_bridged
                and not area1_saved
                and state.mode is GameMode.PLAYING
                and state.room >= 1
                and MIN_SAVE_HP <= state.health <= 128
                and state.lives > 0
            ):
                name = (
                    f"Stage3_Area{state.room}"
                    f"_hp{state.health}_L{state.lives}"
                    f"_cam{state.camera_x}"
                )
                path = save_state(env, GAME_DIR, GAME, name)
                saved_states.append(path.name)
                area1_saved = True
                print(
                    f"area{state.room} save hp={state.health} "
                    f"L={state.lives} cam={state.camera_x} "
                    f"→ {path.name}"
                )

            if (
                not far_cam_saved
                and state.mode is GameMode.PLAYING
                and state.camera_x >= 900
                and state.lives > 0
                and MIN_SAVE_HP <= state.health <= 128
            ):
                tag = "_threat" if state.threat_enemies else ""
                name = (
                    f"Stage3_Far_hp{state.health}"
                    f"_L{state.lives}_cam{state.camera_x}{tag}"
                )
                path = save_state(env, GAME_DIR, GAME, name)
                saved_states.append(path.name)
                far_cam_saved = True
                print(
                    f"far-cam save hp={state.health} L={state.lives} "
                    f"cam={state.camera_x} → {path.name}"
                )

            stop = tracker.update(state)

            # Snapshot clear only when ghost-free (UF corpses chip the hold).
            # Wave4+ (Andore): also require mid-screen — right-edge clears
            # load into dual 142+96 with player at sx≈232.
            pending_no = west_wave_offset + tracker.waves_cleared + 1
            clear_sx_ok = sx <= 130 or pending_no < 4
            if (
                not early_clear_saved
                and tracker._in_wave
                and tracker._wave_had_enemies
                and not state.living_enemies
                and not state.threat_enemies
                and not spawn_l
                and clear_sx_ok
                and state.mode is GameMode.PLAYING
                and MIN_SAVE_HP <= state.health <= 128
                and state.lives > 0
            ):
                # During hold, waves_cleared not yet incremented.
                name = (
                    f"Stage3_Clear_w{pending_no}"
                    f"_cam{state.camera_x}"
                )
                tagged = (
                    f"Stage3_Clear_w{pending_no}"
                    f"_hp{state.health}_cam{state.camera_x}"
                )
                # Prefer higher-HP untagged clears; always keep HP tag.
                untagged = STATE_DIR / f"{name}.state"
                allow_untagged = True
                if untagged.exists() and state.health < 38:
                    allow_untagged = False
                    print(
                        f"early clear-w{pending_no} hp={state.health} "
                        f"(kept existing {name}.state)"
                    )
                if allow_untagged and _maybe_save_state(
                    env,
                    name,
                    saved_states,
                    overwrite=state.health >= 38,
                ):
                    print(
                        f"early clear-w{pending_no} hp={state.health} "
                        f"sx={sx} → {name}.state"
                    )
                if _maybe_save_state(
                    env, tagged, saved_states, overwrite=True
                ):
                    print(
                        f"early clear tag → {tagged}.state"
                    )
                early_clear_saved = True

            if tracker.waves_cleared > last_cleared:
                last_cleared = tracker.waves_cleared
                wave = tracker.waves[-1]
                pending_save_wave = None
                wave_no = wave.index + west_wave_offset
                png = save_rgb_png(
                    obs,
                    out
                    / (
                        f"s3_{abs_frame:04d}"
                        f"_wave{wave_no}_clear.png"
                    ),
                )
                screenshots.append(png.name)
                if (
                    state.mode is GameMode.PLAYING
                    and MIN_SAVE_HP <= state.health <= 128
                    and state.lives > 0
                ):
                    tag = (
                        ""
                        if not state.threat_enemies
                        else "_threat"
                    )
                    name = (
                        f"Stage3_Clear_w{wave_no}"
                        f"_cam{state.camera_x}{tag}"
                    )
                    # Keep healthier early-clear / existing snapshots.
                    if early_clear_saved and state.health < 45:
                        print(
                            f"wave {wave_no} clear hp={state.health} "
                            "(kept healthier early clear)"
                        )
                    elif state.health < 38 and (
                        STATE_DIR / f"{name}.state"
                    ).exists():
                        print(
                            f"wave {wave_no} clear hp={state.health} "
                            f"(kept existing {name}.state)"
                        )
                        if name + ".state" not in saved_states:
                            saved_states.append(name + ".state")
                    elif _maybe_save_state(
                        env,
                        name,
                        saved_states,
                        overwrite=state.health >= 38,
                    ):
                        print(
                            f"wave {wave_no} clear hp={state.health} "
                            f"lives={state.lives} "
                            f"cam={state.camera_x} "
                            f"threats="
                            f"{len(state.threat_enemies)} "
                            f"→ {name}.state"
                        )
                    if state.threat_enemies:
                        pending_save_wave = wave_no
                early_clear_saved = False

            if (
                pending_save_wave is not None
                and state.mode is GameMode.PLAYING
                and MIN_SAVE_HP <= state.health <= 128
                and state.lives > 0
                and not state.threat_enemies
                and not state.living_enemies
            ):
                name = (
                    f"Stage3_Clear_w{pending_save_wave}"
                    f"_cam{state.camera_x}"
                )
                path = save_state(env, GAME_DIR, GAME, name)
                saved_states.append(path.name)
                print(
                    f"wave {pending_save_wave} ghost-free "
                    f"hp={state.health} lives={state.lives} "
                    f"cam={state.camera_x} → {path.name}"
                )
                pending_save_wave = None

            if stop is not None:
                fight_outcome = stop
                break

        end_tag = fight_outcome.name.lower()
        png = save_rgb_png(
            obs,
            out
            / (
                f"s3_{fight_ready_frame + tracker.frames:04d}"
                f"_{end_tag}.png"
            ),
        )
        screenshots.append(png.name)
        if (
            tracker.boss_reached
            and state.mode is GameMode.PLAYING
            and 12 <= state.health <= 128
            and state.lives > 0
        ):
            path = save_state(env, GAME_DIR, GAME, "Boss3")
            saved_states.append(path.name)
            print(
                f"Boss3 saved cam={state.camera_x} "
                f"hp={state.health} L={state.lives}"
            )
        fight_report = {
            "outcome": fight_outcome.name,
            "waves_cleared": tracker.waves_cleared,
            "frames": tracker.frames,
            "boss_reached": tracker.boss_reached,
            "waves": [w.to_dict() for w in tracker.waves],
            "reasons": dict(tracker.reason_counts),
            "final": snapshot_state(state),
            "ram": _snap_ram(env.get_ram()),
        }
        if fight_outcome is SegmentOutcome.SUCCESS:
            outcome = "stage3_waves_ok"
        elif fight_outcome is SegmentOutcome.DEATH:
            outcome = "stage3_death"
        else:
            outcome = f"stage3_{fight_outcome.name.lower()}"

    env.close()
    report: dict[str, Any] = {
        "success": outcome
        in ("stage3_ready", "stage3_waves_ok", "stage3_resume"),
        "outcome": outcome,
        "bridge_clear_area": bridged,
        "start_state": state_name,
        "w5_tactic": w5_tactic.value,
        "heal_hp_poke": heal_used,
        "start": start,
        "clear_round_frame": clear_round_frame,
        "bonus_frame": bonus_frame,
        "west_side_frame": west_frame,
        "fight_ready_frame": fight_ready_frame,
        "stage3": stage3_snap,
        "transitions": transitions,
        "fight": fight_report,
        "screenshots": screenshots,
        "saved_states": saved_states,
        "notes": (
            "Sodom HP underflow does not set 0x0CD2; segment uses "
            "set_value(game_status, CLEAR_AREA) → Break Car bonus "
            "(round=06) → West Side (round=02). Softlock cam≥990 "
            "area0: same CLEAR_AREA poke advances West Side area "
            "(stall threshold cam≥920)."
        ),
    }
    write_json_report(out / "stage3_advance.json", report)
    print(f"outcome={outcome} saved={saved_states}")
    return report

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default="Stage2_Clear")
    parser.add_argument("--max-advance", type=int, default=6000)
    parser.add_argument("--max-fight", type=int, default=12000)
    parser.add_argument(
        "--waves",
        type=int,
        default=6,
        help="Early West Side waves to clear (0=save Stage3 only)",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write Stage3.state",
    )
    parser.add_argument(
        "--no-bridge",
        action="store_true",
        help="Do not poke CLEAR_AREA",
    )
    parser.add_argument(
        "--bridge",
        action="store_true",
        help="Force CLEAR_AREA poke even from Stage3* states",
    )
    parser.add_argument(
        "--w5-tactic",
        choices=[t.value for t in W5Tactic],
        default=W5Tactic.ALT.value,
        help="Post-w4 dual recipe: alt|throw|bait|kick",
    )
    parser.add_argument(
        "--heal-hp",
        type=int,
        default=None,
        help="Optional set_value(player_hp) at fight start (1–80)",
    )
    args = parser.parse_args()
    bridge: bool | None
    if args.bridge:
        bridge = True
    elif args.no_bridge:
        bridge = False
    else:
        bridge = None
    run_stage3_advance(
        state_name=args.state,
        max_advance_frames=args.max_advance,
        max_fight_frames=args.max_fight,
        target_waves=args.waves,
        out_dir=args.out_dir,
        save_stage3=not args.no_save,
        bridge_clear_area=bridge,
        w5_tactic=W5Tactic(args.w5_tactic),
        heal_hp=args.heal_hp,
    )

if __name__ == "__main__":
    main()
