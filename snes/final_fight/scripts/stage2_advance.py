"""Advance Stage1_Clear → Stage2 (subway) and clear early waves.

Damnd HP-underflow alone does **not** set TCRF ``0x0CD2`` (boss-dead) or
``CLEAR_ROUND``. Segment bridge: ``set_value(game_status, CLEAR_AREA)``
runs the natural clear-area → clear-round → open-subway pipeline, then
``Stage1Policy`` (subway kick-band / door rules) clears early waves.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from final_fight.paths import GAME, GAME_DIR, RECORDINGS_DIR
from final_fight.policy import Stage1Policy, _living_all_far_behind
from final_fight.ram import (
    ADDR_AREA,
    ADDR_BOSS_DEAD_FLAG,
    ADDR_GAME_STATUS,
    ADDR_ROUND,
    BOSS_BASE,
    ENEMY_BASES,
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
from retro_harness.env import get_available_states, make_env, save_state
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

# TCRF 7E0CD2 — set when certain bosses die; Damnd UF never sets it.
SUBWAY_ROUND = int(RoundId.SUBWAY)
MIN_SAVE_HP = 25
ENGAGE_DX = 110


def _reset(env: Any) -> Any:
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def _snap_ram(ram: Any) -> dict[str, int]:
    return {
        "game_status": read_u8(ram, ADDR_GAME_STATUS),
        "round": read_u8(ram, ADDR_ROUND),
        "area": read_u8(ram, ADDR_AREA),
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
        if status == 3 and 0 < hp <= 192:
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


def _is_engage_ready(state: GameState, enemies: list[dict[str, int]]) -> bool:
    if not enemies:
        return False
    if state.stage != SUBWAY_ROUND:
        return False
    if state.mode is not GameMode.PLAYING:
        return False
    if not (0 < state.health <= 128 and state.lives > 0):
        return False
    nearest = min(enemies, key=lambda e: abs(e["dx"]))
    sx = state.player_x - state.camera_x
    return abs(nearest["dx"]) <= ENGAGE_DX and 50 <= sx <= 160


def run_stage2_advance(
    *,
    state_name: str = "Stage1_Clear",
    max_advance_frames: int = 3600,
    max_fight_frames: int = 15000,
    target_waves: int | None = 4,
    out_dir: Path | None = None,
    save_stage2: bool = True,
    bridge_clear_area: bool | None = None,
) -> dict[str, Any]:
    """Load a Stage1/Stage2 state, bridge if needed, clear subway waves.

    Mid-stage resumes (``Stage2*``) skip CLEAR_AREA and jump straight into
    the fight loop when already in subway play.
    """
    configure_headless()
    available = get_available_states(GAME, GAME_DIR)
    if state_name not in available:
        raise SystemExit(f"missing state {state_name}; have {available[:8]}")
    out = out_dir or (RECORDINGS_DIR / "stage2_advance")
    out.mkdir(parents=True, exist_ok=True)

    # Default: bridge only from Stage1_Clear-style kill frames.
    if bridge_clear_area is None:
        bridge_clear_area = not state_name.startswith("Stage2")

    env = make_env(GAME, state_name, GAME_DIR, render_mode="rgb_array")
    transitions: list[dict[str, Any]] = []
    screenshots: list[str] = []
    saved_states: list[str] = []
    obs = _reset(env)
    ram = env.get_ram()
    state = parse_game_state(ram, frame=0)
    start = {**snapshot_state(state), **_snap_ram(ram)}
    transitions.append({"frame": 0, **start})
    png = save_rgb_png(obs, out / "s2_0000_start.png")
    screenshots.append(png.name)

    already_subway = (
        start["round"] == SUBWAY_ROUND
        and start["game_status"] == int(GameStatus.ACTIVE_GAMEPLAY)
        and 0 < state.health <= 128
        and state.lives > 0
    )

    bridged = False
    if bridge_clear_area and not already_subway:
        # Underflow Damnd leaves 0x0CD2=0; CLEAR_AREA advances the round.
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
    subway_frame: int | None = None
    fight_ready_frame: int | None = None
    stage2_snap: dict[str, Any] | None = None
    outcome = "advance_timeout"
    last_key: tuple[Any, ...] | None = None

    if already_subway:
        fight_ready_frame = 0
        enemies = _living_brief(ram, state.camera_x, state.player_x)
        stage2_snap = {**start, "enemies": enemies}
        outcome = "stage2_resume"
        subway_frame = 0
        print(
            f"resume {state_name} hp={state.health} lives={state.lives} "
            f"cam={state.camera_x} threats={len(state.threat_enemies)} "
            f"living={len(enemies)}"
        )
        png = save_rgb_png(obs, out / "s2_0000_resume.png")
        screenshots.append(png.name)
    else:
        for frame_i in range(1, max_advance_frames + 1):
            ram = env.get_ram()
            state = parse_game_state(ram, frame=frame_i)
            status = read_u8(ram, ADDR_GAME_STATUS)
            snap = {**snapshot_state(state), **_snap_ram(ram)}
            enemies = _living_brief(ram, state.camera_x, state.player_x)
            key = (
                snap["game_status"],
                snap["round"],
                snap["area"],
                len(enemies),
            )
            if key != last_key:
                last_key = key
                transitions.append(
                    {"frame": frame_i, **snap, "enemies": enemies}
                )
                print(
                    f"frame={frame_i} status=0x{snap['game_status']:02X} "
                    f"round={snap['round']} area={snap['area']} "
                    f"hp={state.health} lives={state.lives} "
                    f"cam={state.camera_x} enemies={len(enemies)}"
                )
                tag = (
                    f"st{snap['game_status']:02X}"
                    f"_r{snap['round']}_a{snap['area']}"
                )
                png = save_rgb_png(
                    obs, out / f"s2_{frame_i:04d}_{tag}.png"
                )
                screenshots.append(png.name)

            if (
                clear_round_frame is None
                and snap["game_status"] == GameStatus.CLEAR_ROUND
            ):
                clear_round_frame = frame_i

            if subway_frame is None and snap["round"] == SUBWAY_ROUND:
                subway_frame = frame_i
                png = save_rgb_png(
                    obs, out / f"s2_{frame_i:04d}_subway.png"
                )
                screenshots.append(png.name)

            if _is_engage_ready(state, enemies):
                fight_ready_frame = frame_i
                stage2_snap = {**snap, "enemies": enemies}
                outcome = "stage2_ready"
                if save_stage2:
                    path = save_state(env, GAME_DIR, GAME, "Stage2")
                    saved_states.append(path.name)
                    print(
                        f"Stage2 saved frame={frame_i} "
                        f"hp={state.health} lives={state.lives} "
                        f"cam={state.camera_x} dx={enemies[0]['dx']}"
                    )
                png = save_rgb_png(
                    obs, out / f"s2_{frame_i:04d}_fight_ready.png"
                )
                screenshots.append(png.name)
                break

            if status in (
                GameStatus.CLEAR_AREA,
                GameStatus.CLEAR_ROUND,
                GameStatus.OPEN_STAGE_A,
                GameStatus.OPEN_STAGE_B,
                GameStatus.CHARACTER_SELECT,
            ):
                action = buttons("START")
            elif (
                snap["round"] == SUBWAY_ROUND
                and status == GameStatus.ACTIVE_GAMEPLAY
            ):
                # Wait for thug to walk into engage dx; don't overshoot.
                if enemies:
                    nearest = min(enemies, key=lambda e: abs(e["dx"]))
                    sx = state.player_x - state.camera_x
                    if nearest["dx"] > ENGAGE_DX and sx > 100:
                        action = buttons("LEFT")
                    elif nearest["dx"] > 80:
                        action = idle_action()
                    else:
                        action = buttons("RIGHT")
                else:
                    action = buttons("RIGHT")
            else:
                action = idle_action()
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
        pending_save_wave: int | None = None
        fight_outcome = SegmentOutcome.TIMEOUT
        behind_hp_log: list[dict[str, Any]] = []
        prev_behind_hp: dict[int, int] = {}
        mid_pack_saved = False
        far_cam_saved = False
        l2_clear_saved = False
        subway_area_bridged = False
        area1_saved = False
        cam994_mid_saved = False
        area1_clear_saved = False
        subway_area2_bridged = False
        subway_area3_bridged = False
        area2_saved = False
        area2_mid_saved = False
        area2_clear_saved = False
        # Area2 early JD90 open: mid-HP (e69) or HP≤8 crumb while
        # cam<3960. Skip at cam≥3960 (e28 behind peak — JD gutter-dies).
        # Slot-track primary so dual spawn during JD does not retarget.
        crumb_open_done = False
        if (
            state.stage >= 1
            and state.room >= 2
            and len(state.living_enemies) == 1
            and 0 < state.health <= 128
            and state.camera_x < 3960
        ):
            e0 = state.living_enemies[0]
            if 0 < e0.health <= 80:
                primary_slot = e0.slot
                print(
                    f"area2 early open JD90+Y "
                    f"e{e0.health} slot={primary_slot} "
                    f"cam={state.camera_x} php={state.health}"
                )
                for _ in range(90):
                    obs, _r, _t, _tr, _info = env.step(
                        buttons("B", "RIGHT")
                    )
                for _ in range(200):
                    ram = env.get_ram()
                    state = parse_game_state(ram)
                    if state.player_dead or state.health > 128:
                        break
                    primary = next(
                        (
                            e
                            for e in state.living_enemies
                            if e.slot == primary_slot
                        ),
                        None,
                    )
                    if primary is None:
                        break
                    # Underflow / dead primary — open done.
                    if primary.health <= 0 or primary.health >= 200:
                        break
                    toward = (
                        "RIGHT"
                        if primary.x >= state.player_x
                        else "LEFT"
                    )
                    # Prefer grounded toward+Y at kill cam; space if a
                    # tough dual overlaps punch range.
                    tough = max(
                        state.living_enemies,
                        key=lambda e: e.health,
                        default=None,
                    )
                    if (
                        tough is not None
                        and tough.slot != primary_slot
                        and tough.health > 40
                        and abs(tough.x - state.player_x) < 28
                        and (state.player_x - state.camera_x) > 100
                    ):
                        away = (
                            "LEFT"
                            if tough.x > state.player_x
                            else "RIGHT"
                        )
                        sx = state.player_x - state.camera_x
                        if away == "LEFT" and sx < 70:
                            away = "RIGHT"
                        for _ in range(4):
                            obs, _r, _t, _tr, _info = env.step(
                                buttons("B", away)
                            )
                        continue
                    for _ in range(2):
                        obs, _r, _t, _tr, _info = env.step(
                            buttons(toward, "Y")
                        )
                    for _ in range(6):
                        obs, _r, _t, _tr, _info = env.step(
                            idle_action()
                        )
                ram = env.get_ram()
                state = parse_game_state(ram)
                crumb_open_done = True
                living_hps = [e.health for e in state.living_enemies]
                print(
                    f"early open done living={len(living_hps)} "
                    f"hps={living_hps} php={state.health} "
                    f"cam={state.camera_x}"
                )
                # Plant UF leftovers from the open before next pack.
                for _ in range(90):
                    ram = env.get_ram()
                    state = parse_game_state(ram)
                    if state.player_dead or state.health > 128:
                        break
                    near_ghosts = tuple(
                        e
                        for e in state.threat_enemies
                        if e.health == 0
                        and abs(e.x - state.player_x) <= 160
                    )
                    if not near_ghosts and not state.living_enemies:
                        break
                    if near_ghosts:
                        g = min(
                            near_ghosts,
                            key=lambda e: abs(e.x - state.player_x),
                        )
                        gdx = g.x - state.player_x
                        if abs(gdx) < 16:
                            act = buttons("Y")
                        elif gdx > 0:
                            act = buttons("RIGHT", "Y")
                        else:
                            act = buttons("LEFT", "Y")
                        for _ in range(2):
                            obs, _r, _t, _tr, _info = env.step(act)
                        for _ in range(4):
                            obs, _r, _t, _tr, _info = env.step(
                                idle_action()
                            )
                        continue
                    break
                ram = env.get_ram()
                state = parse_game_state(ram)
                print(
                    f"post-open plant living="
                    f"{len(state.living_enemies)} "
                    f"threats={len(state.threat_enemies)} "
                    f"php={state.health} cam={state.camera_x}"
                )
        for frame_j in range(1, max_fight_frames + 1):
            # Cam994 area-0 softlock: scroll never advances. Same Damnd
            # bridge poke advances subway area 0→1 (train/sewer stretch).
            # Cam2561 area-1 softlock: same poke advances area 1→2
            # (Sodom stretch, cam≈3840).
            if (
                not subway_area_bridged
                and state.mode is GameMode.PLAYING
                and state.stage == int(RoundId.SUBWAY)
                and state.room == 0
                and state.camera_x >= 990
                and not state.living_enemies
                and 0 < state.health <= 128
                and state.lives > 0
            ):
                env.set_value(
                    "game_status", int(GameStatus.CLEAR_AREA)
                )
                subway_area_bridged = True
                print(
                    f"subway area bridge CLEAR_AREA "
                    f"cam={state.camera_x} hp={state.health} "
                    f"L={state.lives}"
                )
                png = save_rgb_png(
                    obs,
                    out
                    / (
                        f"s2_{fight_ready_frame + frame_j:04d}"
                        f"_area_bridge.png"
                    ),
                )
                screenshots.append(png.name)
            elif (
                not subway_area2_bridged
                and state.mode is GameMode.PLAYING
                and state.stage == int(RoundId.SUBWAY)
                and state.room == 1
                and state.camera_x >= 2550
                and (
                    not state.living_enemies
                    or _living_all_far_behind(state, min_dx=-100)
                )
                and 0 < state.health <= 128
                and state.lives > 0
            ):
                env.set_value(
                    "game_status", int(GameStatus.CLEAR_AREA)
                )
                subway_area2_bridged = True
                print(
                    f"subway area1→2 bridge CLEAR_AREA "
                    f"cam={state.camera_x} hp={state.health} "
                    f"L={state.lives}"
                )
                png = save_rgb_png(
                    obs,
                    out
                    / (
                        f"s2_{fight_ready_frame + frame_j:04d}"
                        f"_area2_bridge.png"
                    ),
                )
                screenshots.append(png.name)
            elif (
                not subway_area3_bridged
                and state.mode is GameMode.PLAYING
                and state.stage == int(RoundId.SUBWAY)
                and state.room == 2
                and state.camera_x >= 4120
                and (
                    not state.living_enemies
                    or _living_all_far_behind(state, min_dx=-100)
                )
                and 0 < state.health <= 128
                and state.lives > 0
            ):
                # Cam4130 area-2 softlock: scroll never past ~4130.
                # CLEAR_AREA → area 2→3 (Sodom door, cam≈4864, 0x11E0=01).
                env.set_value(
                    "game_status", int(GameStatus.CLEAR_AREA)
                )
                subway_area3_bridged = True
                print(
                    f"subway area2→3 bridge CLEAR_AREA "
                    f"cam={state.camera_x} hp={state.health} "
                    f"L={state.lives}"
                )
                png = save_rgb_png(
                    obs,
                    out
                    / (
                        f"s2_{fight_ready_frame + frame_j:04d}"
                        f"_area3_bridge.png"
                    ),
                )
                screenshots.append(png.name)

            # Idle through CLEAR_AREA / open-stage after the poke.
            if (
                (
                    subway_area_bridged
                    or subway_area2_bridged
                    or subway_area3_bridged
                )
                and state.mode is not GameMode.PLAYING
            ):
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
            # Save first healthy area≥1 resume after the bridge.
            if (
                subway_area_bridged
                and not area1_saved
                and state.mode is GameMode.PLAYING
                and state.room == 1
                and MIN_SAVE_HP <= state.health <= 128
                and state.lives > 0
            ):
                name = (
                    f"Stage2_Area{state.room}"
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
            # Save first healthy area≥2 resume (Sodom stretch).
            if (
                subway_area2_bridged
                and not area2_saved
                and state.mode is GameMode.PLAYING
                and state.room >= 2
                and MIN_SAVE_HP <= state.health <= 128
                and state.lives > 0
            ):
                name = (
                    f"Stage2_Area{state.room}"
                    f"_hp{state.health}_L{state.lives}"
                    f"_cam{state.camera_x}"
                )
                path = save_state(env, GAME_DIR, GAME, name)
                saved_states.append(path.name)
                area2_saved = True
                print(
                    f"area{state.room} save hp={state.health} "
                    f"L={state.lives} cam={state.camera_x} "
                    f"→ {path.name}"
                )
            # Evidence: living HP deltas while tough is behind (dx<0).
            living_now = _living_brief(ram, state.camera_x, state.player_x)
            for brief in living_now:
                if brief["hp"] <= 50 or brief["dx"] >= 0:
                    prev_behind_hp.pop(brief["slot"], None)
                    continue
                slot = brief["slot"]
                prev = prev_behind_hp.get(slot)
                if prev is not None and brief["hp"] < prev:
                    behind_hp_log.append(
                        {
                            "frame": abs_frame,
                            "slot": slot,
                            "hp_before": prev,
                            "hp_after": brief["hp"],
                            "dx": brief["dx"],
                            "player_hp": state.health,
                            "cam": state.camera_x,
                            "reason": (
                                tick.action.reason
                                if tick.action is not None
                                else None
                            ),
                        }
                    )
                prev_behind_hp[slot] = brief["hp"]
            # Healthy mid-pack snapshot during HP148 fight (not weak filler).
            if (
                not mid_pack_saved
                and state.mode is GameMode.PLAYING
                and state.camera_x >= 844
                and state.lives >= 2
                and 40 <= state.health <= 128
                and living_now
            ):
                tough_hp = max(e["hp"] for e in living_now)
                if 50 <= tough_hp <= 120:
                    name = (
                        f"Stage2_Mid_hp148_p{state.health}"
                        f"_e{tough_hp}_cam{state.camera_x}"
                    )
                    path = save_state(env, GAME_DIR, GAME, name)
                    saved_states.append(path.name)
                    mid_pack_saved = True
                    print(
                        f"mid-pack save hp={state.health} L={state.lives} "
                        f"tough={tough_hp} cam={state.camera_x} "
                        f"→ {path.name}"
                    )
            # Healthy far-cam (past 848 softlock toward train / Sodom).
            if (
                not far_cam_saved
                and state.mode is GameMode.PLAYING
                and state.camera_x >= 900
                and state.lives > 0
                and MIN_SAVE_HP <= state.health <= 128
            ):
                tag = "_threat" if state.threat_enemies else ""
                name = (
                    f"Stage2_Far_hp{state.health}"
                    f"_L{state.lives}_cam{state.camera_x}{tag}"
                )
                path = save_state(env, GAME_DIR, GAME, name)
                saved_states.append(path.name)
                far_cam_saved = True
                print(
                    f"far-cam save hp={state.health} L={state.lives} "
                    f"cam={state.camera_x} → {path.name}"
                )
            # Prefer an explicit L2 post-HP148 clear snapshot (once).
            if (
                not l2_clear_saved
                and state.mode is GameMode.PLAYING
                and state.camera_x >= 840
                and state.lives >= 2
                and MIN_SAVE_HP <= state.health <= 128
                and not state.living_enemies
            ):
                name = (
                    f"Stage2_Clear_L2_hp{state.health}"
                    f"_cam{state.camera_x}"
                )
                if state.threat_enemies:
                    name += "_threat"
                path = save_state(env, GAME_DIR, GAME, name)
                saved_states.append(path.name)
                l2_clear_saved = True
                print(
                    f"L2 clear save hp={state.health} "
                    f"cam={state.camera_x} → {path.name}"
                )
            # Mid-fight snapshot on the cam994 lock (train approach).
            # Only area-0 lock band — area1 cams 17xx matched "cam99" never.
            if (
                not cam994_mid_saved
                and state.mode is GameMode.PLAYING
                and 990 <= state.camera_x <= 1100
                and state.room == 0
                and state.lives > 0
                and 40 <= state.health <= 128
                and living_now
            ):
                tough_hp = max(e["hp"] for e in living_now)
                name = (
                    f"Stage2_Mid_cam{state.camera_x}"
                    f"_p{state.health}_e{tough_hp}"
                )
                path = save_state(env, GAME_DIR, GAME, name)
                saved_states.append(path.name)
                cam994_mid_saved = True
                print(
                    f"cam994 mid save hp={state.health} L={state.lives} "
                    f"tough={tough_hp} → {path.name}"
                )
            # Healthy area1 pack clear (dual-pack done; push to Sodom).
            if (
                not area1_clear_saved
                and state.mode is GameMode.PLAYING
                and state.room == 1
                and state.camera_x >= 1850
                and state.lives > 0
                and MIN_SAVE_HP <= state.health <= 128
                and not state.living_enemies
            ):
                name = (
                    f"Stage2_Area{state.room}_clear"
                    f"_hp{state.health}_L{state.lives}"
                    f"_cam{state.camera_x}"
                )
                path = save_state(env, GAME_DIR, GAME, name)
                saved_states.append(path.name)
                area1_clear_saved = True
                print(
                    f"area{state.room} clear save hp={state.health} "
                    f"L={state.lives} cam={state.camera_x} "
                    f"→ {path.name}"
                )
            # Area2 ultra dual mid (HP112/134 chipped; prefer 1v1).
            if (
                not area2_mid_saved
                and state.mode is GameMode.PLAYING
                and state.room >= 2
                and state.camera_x >= 3840
                and state.lives > 0
                and MIN_SAVE_HP <= state.health <= 128
                and living_now
            ):
                hps = sorted(e["hp"] for e in living_now)
                if (
                    len(living_now) == 1 and hps[0] <= 80
                ) or (
                    len(living_now) >= 2 and sum(hps) <= 160
                ):
                    tag = "_".join(f"e{h}" for h in hps)
                    name = (
                        f"Stage2_Area2_mid_p{state.health}"
                        f"_{tag}_cam{state.camera_x}"
                    )
                    path = save_state(env, GAME_DIR, GAME, name)
                    saved_states.append(path.name)
                    area2_mid_saved = True
                    print(
                        f"area2 mid save hp={state.health} "
                        f"L={state.lives} {tag} cam={state.camera_x} "
                        f"→ {path.name}"
                    )
            # Area2 pack clear (toward Sodom / Boss2).
            # Require no near UF threats — open leftovers poison clears.
            if (
                not area2_clear_saved
                and state.mode is GameMode.PLAYING
                and state.room >= 2
                and state.camera_x >= 3960
                and state.lives > 0
                and 12 <= state.health <= 128
                and not state.living_enemies
                and not state.threat_enemies
            ):
                name = (
                    f"Stage2_Area2_clear"
                    f"_hp{state.health}_L{state.lives}"
                    f"_cam{state.camera_x}"
                )
                path = save_state(env, GAME_DIR, GAME, name)
                saved_states.append(path.name)
                area2_clear_saved = True
                print(
                    f"area2 clear save hp={state.health} "
                    f"L={state.lives} cam={state.camera_x} "
                    f"→ {path.name}"
                )
            stop = tracker.update(state)

            if tracker.waves_cleared > last_cleared:
                last_cleared = tracker.waves_cleared
                wave = tracker.waves[-1]
                pending_save_wave = None
                png = save_rgb_png(
                    obs,
                    out
                    / f"s2_{abs_frame:04d}_wave{wave.index}_clear.png",
                )
                screenshots.append(png.name)
                # Snapshot healthy clears. Never save ghost-free at cam≥840
                # (scroll softlock). Prefer threat-tagged or skip.
                if (
                    state.mode is GameMode.PLAYING
                    and MIN_SAVE_HP <= state.health <= 128
                    and state.lives > 0
                ):
                    # Cam≥840 ghost-free = scroll softlock — do not save.
                    if (
                        state.camera_x >= 840
                        and not state.threat_enemies
                        and not state.living_enemies
                    ):
                        print(
                            f"wave {wave.index} skip ghost-free save "
                            f"cam={state.camera_x} (scroll softlock)"
                        )
                    else:
                        tag = (
                            ""
                            if not state.threat_enemies
                            else "_threat"
                        )
                        name = (
                            f"Stage2_Clear_w{wave.index}"
                            f"_cam{state.camera_x}{tag}"
                        )
                        path = save_state(env, GAME_DIR, GAME, name)
                        saved_states.append(path.name)
                        print(
                            f"wave {wave.index} clear hp={state.health} "
                            f"lives={state.lives} cam={state.camera_x} "
                            f"threats={len(state.threat_enemies)} "
                            f"→ {path.name}"
                        )
                    # Explicit: disable ghost-free gate at cam≥840.
                    if (
                        state.threat_enemies
                        and state.camera_x < 840
                    ):
                        pending_save_wave = wave.index

            # Prefer a second ghost-free snapshot when corpses despawn.
            # Skipped entirely once cam≥840 (scroll softlock).
            if (
                pending_save_wave is not None
                and state.camera_x < 840
                and state.mode is GameMode.PLAYING
                and MIN_SAVE_HP <= state.health <= 128
                and state.lives > 0
                and not state.threat_enemies
                and not state.living_enemies
            ):
                name = (
                    f"Stage2_Clear_w{pending_save_wave}"
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
            elif pending_save_wave is not None and state.camera_x >= 840:
                pending_save_wave = None

            if stop is not None:
                fight_outcome = stop
                break

        end_tag = fight_outcome.name.lower()
        png = save_rgb_png(
            obs,
            out
            / (
                f"s2_{fight_ready_frame + tracker.frames:04d}"
                f"_{end_tag}.png"
            ),
        )
        screenshots.append(png.name)
        # Prefer a boss / far-cam snapshot for Sodom (Boss2).
        # Allow low HP — wave2 clear arrives ~37.
        if (
            tracker.boss_reached
            and state.mode is GameMode.PLAYING
            and 12 <= state.health <= 128
            and state.lives > 0
        ):
            path = save_state(env, GAME_DIR, GAME, "Boss2")
            saved_states.append(path.name)
            print(
                f"Boss2 saved cam={state.camera_x} "
                f"hp={state.health} L={state.lives}"
            )
        fight_report = {
            "outcome": fight_outcome.name,
            "waves_cleared": tracker.waves_cleared,
            "frames": tracker.frames,
            "boss_reached": tracker.boss_reached,
            "waves": [w.to_dict() for w in tracker.waves],
            "reasons": dict(tracker.reason_counts),
            "behind_hp_deltas": behind_hp_log[:80],
            "behind_hp_delta_count": len(behind_hp_log),
            "final": snapshot_state(state),
            "ram": _snap_ram(env.get_ram()),
        }
        if fight_outcome is SegmentOutcome.SUCCESS:
            outcome = "stage2_waves_ok"
        elif fight_outcome is SegmentOutcome.DEATH:
            outcome = "stage2_death"
        else:
            outcome = f"stage2_{fight_outcome.name.lower()}"

    env.close()
    report: dict[str, Any] = {
        "success": outcome
        in ("stage2_ready", "stage2_waves_ok", "stage2_resume"),
        "outcome": outcome,
        "bridge_clear_area": bridged,
        "start_state": state_name,
        "start": start,
        "clear_round_frame": clear_round_frame,
        "subway_frame": subway_frame,
        "fight_ready_frame": fight_ready_frame,
        "stage2": stage2_snap,
        "transitions": transitions,
        "fight": fight_report,
        "screenshots": screenshots,
        "saved_states": saved_states,
        "notes": (
            "Damnd HP underflow does not set 0x0CD2; segment uses "
            "set_value(game_status, CLEAR_AREA) to enter subway. "
            "Cam994 area-0 softlock: same CLEAR_AREA poke advances "
            "subway area 0→1. Cam2561 area-1 softlock: same poke "
            "advances area 1→2 (cam≈3840 Sodom stretch). Cam4130 "
            "area-2 softlock: same poke advances area 2→3 (cam≈4864, "
            "0x11E0=01 Sodom). UF/HP0 subway ghosts are plant-punched "
            "before walk. Area2 early open: JD90+toward+Y on HP≤80 "
            "1v1 while cam<3960."
        ),
    }
    write_json_report(out / "stage2_advance.json", report)
    print(f"outcome={outcome} saved={saved_states}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default="Stage1_Clear")
    parser.add_argument("--max-advance", type=int, default=3600)
    parser.add_argument("--max-fight", type=int, default=15000)
    parser.add_argument(
        "--waves",
        type=int,
        default=4,
        help="Early subway waves to clear (0=save Stage2 only)",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write Stage2.state",
    )
    parser.add_argument(
        "--no-bridge",
        action="store_true",
        help="Do not poke CLEAR_AREA (expect natural 0x0CD2 path)",
    )
    parser.add_argument(
        "--bridge",
        action="store_true",
        help="Force CLEAR_AREA poke even from Stage2* states",
    )
    args = parser.parse_args()
    bridge: bool | None
    if args.bridge:
        bridge = True
    elif args.no_bridge:
        bridge = False
    else:
        bridge = None
    run_stage2_advance(
        state_name=args.state,
        max_advance_frames=args.max_advance,
        max_fight_frames=args.max_fight,
        target_waves=args.waves,
        out_dir=args.out_dir,
        save_stage2=not args.no_save,
        bridge_clear_area=bridge,
    )


if __name__ == "__main__":
    main()
