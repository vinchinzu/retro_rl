"""Door / Damnd clearer from Boss_ThugMid / PostThug*.

Kick-band rules: never idle at dx 40–103. Peak>50: rise to Y≈70, delay
~40f after spawn, then jump-dash. Peak≤50: park-bait / retreat (no hop_in).
Punch dx≈28–35; always space below 28 (corpse overlap chips). Ghosts:
status-03 with HP0 or underflow HP>128 — flee if dx<36 else plant-punch.
After pack clear, creep right (cam→~2675) to draw Damnd (`03`, HP44);
spam-Y finishes him (HP underflow). Saves PostThug* / Boss_Drawn /
Stage1_Clear with HP tables in the JSON report.
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
from final_fight.ram import (
    ADDR_GAME_STATUS,
    BOSS_BASE,
    ENEMY_BASES,
    OFF_HP,
    OFF_STATUS,
    OFF_X,
    OFF_Y,
    parse_game_state,
    read_u8,
    read_u16le,
)
from retro_harness.env import get_available_states, make_env, save_state
from snes_oneshot.actions import buttons, idle_action
from snes_oneshot.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)

PUNCH_LO = 28
PUNCH_HI = 35
NUDGE_HI = 39
KICK_HI = 103
ZERO_SAFE_DX = 160
Y_TARGET = 70
TOUGH_ENGAGE_DELAY = 40
DAMND_PUNCH_LO = 24
DAMND_PUNCH_HI = 40
# After Damnd draws on top of the player, kite left before spam-Y.
# 20f often leaves HP10; 60f clears at full HP40 from PostThug5.
DAMND_DRAW_KITE = 60


def _living(ram: Any, cam: int) -> list[dict[str, int]]:
    es: list[dict[str, int]] = []
    for i, base in enumerate(ENEMY_BASES):
        status = read_u8(ram, base + OFF_STATUS)
        hp = read_u8(ram, base + OFF_HP)
        x = read_u16le(ram, base + OFF_X)
        y = read_u16le(ram, base + OFF_Y)
        if status == 3 and 0 < hp <= 128 and cam - 128 <= x <= cam + 320:
            es.append({"slot": i, "hp": hp, "x": x, "y": y})
    return es


def _nearest_ghost(
    ram: Any, cam: int, px: int
) -> tuple[int, int, int, int, int] | None:
    """Nearest on-screen status-03 ghost (HP0 or underflow)."""
    best: tuple[int, int, int, int, int] | None = None
    for base in ENEMY_BASES:
        status = read_u8(ram, base + OFF_STATUS)
        hp = read_u8(ram, base + OFF_HP)
        x = read_u16le(ram, base + OFF_X)
        y = read_u16le(ram, base + OFF_Y)
        if status == 3 and (hp == 0 or hp > 128) and cam - 128 <= x <= cam + 320:
            d = abs(px - x)
            if best is None or d < best[0]:
                best = (d, x, y, x - px, hp)
    return best


def _fight_action(
    *,
    px: int,
    py: int,
    cam: int,
    enemy: dict[str, int],
    punch_cd: int,
    peak_hp: int,
) -> tuple[Any, str, int]:
    dx = enemy["x"] - px
    adx = abs(dx)
    dy = enemy["y"] - py
    sx = px - cam
    tough = peak_hp > 50
    if sx > 155:
        return buttons("LEFT"), "edge", punch_cd
    if abs(dy) > 10 and adx <= 45:
        act = buttons("UP") if dy > 0 else buttons("DOWN")
        return act, "align", punch_cd
    if PUNCH_LO <= adx <= PUNCH_HI:
        if punch_cd > 0:
            return idle_action(), "gap", punch_cd - 1
        return buttons("Y"), "punch", 7
    if adx < PUNCH_LO:
        # Always space — finishing on top leaves a damaging corpse.
        act = buttons("LEFT") if dx > 0 else buttons("RIGHT")
        return act, "space", punch_cd
    if adx <= NUDGE_HI:
        act = buttons("RIGHT") if dx > 0 else buttons("LEFT")
        return act, "nudge", punch_cd
    if not tough:
        # Peak≤50: park-bait. Retreat if they enter kick band.
        if adx <= KICK_HI:
            if sx > 55:
                return buttons("LEFT"), "retreat", punch_cd
            return idle_action(), "hold_left", punch_cd
        if sx > 60:
            return buttons("LEFT"), "park", punch_cd
        return idle_action(), "bait", punch_cd
    if adx <= KICK_HI:
        act = (
            buttons("B", "RIGHT") if dx > 0 else buttons("B", "LEFT")
        )
        return act, "jump_dash", punch_cd
    if sx > 65:
        return buttons("LEFT"), "park", punch_cd
    return idle_action(), "bait", punch_cd


def _damnd_action(
    *,
    px: int,
    py: int,
    cam: int,
    bx: int,
    by: int,
) -> tuple[Any, str]:
    """Spam-Y Damnd close; JD the kick band. HP44 dies via underflow."""
    dx = bx - px
    adx = abs(dx)
    dy = by - py
    sx = px - cam
    if sx > 155:
        return buttons("LEFT"), "b_edge"
    if abs(dy) > 10 and adx <= 48:
        act = buttons("UP") if dy > 0 else buttons("DOWN")
        return act, "b_align"
    if DAMND_PUNCH_LO <= adx <= DAMND_PUNCH_HI:
        return buttons("Y"), "b_yp"
    if adx < DAMND_PUNCH_LO:
        act = buttons("LEFT") if dx > 0 else buttons("RIGHT")
        return act, "b_space"
    if adx <= KICK_HI:
        act = buttons("B", "RIGHT") if dx > 0 else buttons("B", "LEFT")
        return act, "b_jd"
    act = buttons("RIGHT") if dx > 0 else buttons("LEFT")
    return act, "b_close"


def _ghost_action(
    *,
    zero: tuple[int, int, int, int, int],
    z_phase: int,
) -> tuple[Any, str, int]:
    """Flee overlapping corpse; else plant-punch."""
    d, _zx, _zy, dx, _hp = zero
    if d < 36:
        away = "RIGHT" if dx < 0 else "LEFT"
        return buttons("B", away), "z_flee", z_phase + 1
    if (z_phase % 8) < 3:
        return buttons("Y"), "z_punch", z_phase + 1
    return idle_action(), "z_gap", z_phase + 1


def run_door_jump_clear(
    *,
    state_name: str = "Boss_ThugMid",
    max_frames: int = 16000,
    out_dir: Path | None = None,
    trials: int = 1,
) -> dict[str, Any]:
    """Clear door thugs + Damnd; log HP tables + boss draw."""
    configure_headless()
    available = get_available_states(GAME, GAME_DIR)
    if state_name not in available:
        raise SystemExit(f"missing state {state_name}; have {available[:8]}")
    out = out_dir or (RECORDINGS_DIR / "door_jump_clear")
    out.mkdir(parents=True, exist_ok=True)

    best: dict[str, Any] | None = None
    for trial in range(trials):
        report = _one_trial(
            state_name=state_name,
            max_frames=max_frames,
            out=out,
            trial=trial,
        )
        print(
            f"trial{trial}: {report['outcome']} kills={report['kills']} "
            f"peaks={report['peak_enemy_hp']} drawn={report['drawn_frame']} "
            f"php={report['start_player_hp']}->{report['end_player_hp']}"
        )
        if best is None or _score(report) > _score(best):
            best = report
            write_json_report(out / "best.json", best)
            if report.get("success"):
                break
    assert best is not None
    write_json_report(out / "door_jump_clear.json", best)
    return best


def _score(report: dict[str, Any]) -> tuple[int, int, int, int]:
    end = report["end_player_hp"]
    end_ok = 0 if end > 128 or end <= 0 else end
    return (
        1 if report.get("success") else 0,
        report["kills"],
        1 if report["drawn_frame"] is not None else 0,
        end_ok,
    )


def _kill_offset(state_name: str) -> int:
    for n in (5, 4, 3, 2, 1):
        if f"PostThug{n}" in state_name:
            return n
    if state_name == "Boss_Drawn":
        return 5
    return 0


def _one_trial(
    *,
    state_name: str,
    max_frames: int,
    out: Path,
    trial: int,
) -> dict[str, Any]:
    env = make_env(GAME, state_name, GAME_DIR, render_mode="rgb_array")
    env.reset()
    punch_cd = 0
    z_phase = 0
    prev_enemy_hp: dict[int, int] = {}
    prev_php: int | None = None
    prev_bst = -1
    prev_bhp = -1
    kills = 0
    kill_off = _kill_offset(state_name)
    peaks: dict[int, int] = {}
    spawn_frame: dict[int, int] = {}
    events: list[dict[str, Any]] = []
    boss_hits: list[dict[str, Any]] = []
    status_changes: list[dict[str, Any]] = []
    drawn_frame: int | None = None
    damnd_kite = 0
    outcome = "timeout"
    start_hp = 0
    screenshots: list[str] = []
    saved_states: list[str] = []
    frame_i = 0
    pending_save: int | None = None

    try:
        for frame_i in range(1, max_frames + 1):
            ram = env.get_ram()
            state = parse_game_state(ram, frame=frame_i)
            px = state.player_x
            py = state.player_y
            cam = state.camera_x
            php = state.health
            if prev_php is None:
                start_hp = php
                prev_php = php
            bst = read_u8(ram, BOSS_BASE + OFF_STATUS)
            bhp = read_u8(ram, BOSS_BASE + OFF_HP)
            bx = read_u16le(ram, BOSS_BASE + OFF_X)
            by = read_u16le(ram, BOSS_BASE + OFF_Y)
            if prev_bst < 0:
                prev_bst, prev_bhp = bst, bhp
                if bst >= 3 and 0 < bhp <= 128:
                    drawn_frame = 0

            live = _living(ram, cam)
            for en in live:
                peaks[en["slot"]] = max(
                    peaks.get(en["slot"], 0), en["hp"]
                )
                spawn_frame.setdefault(en["slot"], frame_i)
            enemy = (
                min(live, key=lambda z: abs(z["x"] - px))
                if live
                else None
            )
            zero = _nearest_ghost(ram, cam, px)

            if php > 128 or php == 0:
                action = idle_action()
                reason = "ko"
            elif enemy is not None:
                peak = peaks.get(enemy["slot"], enemy["hp"])
                spawned_at = spawn_frame.get(enemy["slot"], frame_i)
                if py < Y_TARGET and peak > 50:
                    action, reason = buttons("UP"), "rise"
                elif (
                    peak > 50
                    and (frame_i - spawned_at) < TOUGH_ENGAGE_DELAY
                ):
                    sx = px - cam
                    if sx > 50:
                        action, reason = buttons("LEFT"), "kite"
                    else:
                        action, reason = idle_action(), "delay"
                else:
                    action, reason, punch_cd = _fight_action(
                        px=px,
                        py=py,
                        cam=cam,
                        enemy=enemy,
                        punch_cd=punch_cd,
                        peak_hp=peak,
                    )
                z_phase = 0
            elif zero is not None and zero[0] <= ZERO_SAFE_DX:
                action, reason, z_phase = _ghost_action(
                    zero=zero, z_phase=z_phase
                )
            elif bst >= 3 and 0 < bhp <= 128:
                if damnd_kite > 0:
                    damnd_kite -= 1
                    action, reason = buttons("LEFT"), "b_kite"
                else:
                    action, reason = _damnd_action(
                        px=px, py=py, cam=cam, bx=bx, by=by
                    )
            else:
                # Undrawn boss after pack: creep right to draw (cam~2675).
                sx = px - cam
                if py < Y_TARGET:
                    action, reason = buttons("UP"), "rise"
                elif sx < 150:
                    action, reason = buttons("RIGHT"), "creep"
                else:
                    action, reason = idle_action(), "hold_door"

            obs, _r, _t, _tr, _i = env.step(action)
            ram = env.get_ram()
            state = parse_game_state(ram, frame=frame_i)
            php = state.health
            bst = read_u8(ram, BOSS_BASE + OFF_STATUS)
            bhp = read_u8(ram, BOSS_BASE + OFF_HP)
            live = _living(ram, state.camera_x)
            cur = {en["slot"]: en["hp"] for en in live}

            for slot, hp in cur.items():
                peaks[slot] = max(peaks.get(slot, 0), hp)
                if slot in prev_enemy_hp and hp < prev_enemy_hp[slot]:
                    en = next(x for x in live if x["slot"] == slot)
                    events.append(
                        {
                            "frame": frame_i,
                            "hit": f"{prev_enemy_hp[slot]}->{hp}",
                            "slot": slot,
                            "dx": abs(state.player_x - en["x"]),
                            "player_hp": php,
                            "peak": peaks.get(slot),
                            "reason": reason,
                        }
                    )
            for slot, hp in list(prev_enemy_hp.items()):
                if (
                    slot not in cur
                    and hp > 0
                    and 0 < php <= 128
                ):
                    kills += 1
                    total = kills + kill_off
                    events.append(
                        {
                            "frame": frame_i,
                            "kill": total,
                            "slot": slot,
                            "peak": peaks.get(slot),
                            "player_hp": php,
                        }
                    )
                    zero_now = _nearest_ghost(
                        ram, state.camera_x, state.player_x
                    )
                    if zero_now is None or zero_now[0] > 50:
                        path = save_state(
                            env,
                            GAME_DIR,
                            GAME,
                            f"Boss_PostThug{total}",
                        )
                        saved_states.append(path.name)
                        png = save_rgb_png(
                            obs,
                            out / f"t{trial}_kill{total}_{frame_i}.png",
                        )
                        screenshots.append(png.name)
                        pending_save = None
                    else:
                        pending_save = total
                        events.append(
                            {
                                "frame": frame_i,
                                "tag": "defer_save",
                                "ghost_dx": zero_now[0],
                                "ghost_hp": zero_now[4],
                            }
                        )
                    punch_cd = 0
                    z_phase = 0
                    peaks.pop(slot, None)
                    spawn_frame.pop(slot, None)
            zero_after = _nearest_ghost(
                ram, state.camera_x, state.player_x
            )
            if zero is not None and zero_after is None:
                events.append(
                    {
                        "frame": frame_i,
                        "tag": "ghost_gone",
                        "player_hp": php,
                        "prev_dx": zero[0],
                    }
                )
                if pending_save is not None and 0 < php <= 128:
                    path = save_state(
                        env,
                        GAME_DIR,
                        GAME,
                        f"Boss_PostThug{pending_save}",
                    )
                    saved_states.append(path.name + "+late")
                    png = save_rgb_png(
                        obs,
                        out
                        / (
                            f"t{trial}_kill{pending_save}"
                            f"_late_{frame_i}.png"
                        ),
                    )
                    screenshots.append(png.name)
                    pending_save = None
            prev_enemy_hp = cur

            if (
                0 < php <= 128
                and prev_php is not None
                and 0 < prev_php <= 128
                and php < prev_php
            ):
                events.append(
                    {
                        "frame": frame_i,
                        "chip": f"{prev_php}->{php}",
                        "reason": reason,
                        "zero_dx": (
                            zero_after[0] if zero_after else None
                        ),
                    }
                )
            if 0 < php <= 128:
                prev_php = php

            if bhp != prev_bhp:
                boss_hits.append(
                    {
                        "frame": frame_i,
                        "from": prev_bhp,
                        "to": bhp,
                        "status": bst,
                        "player_hp": php,
                    }
                )
                if 0 < bhp <= 128 and bhp < prev_bhp:
                    png = save_rgb_png(
                        obs,
                        out / f"bosshit_{frame_i}_{bhp}.png",
                    )
                    screenshots.append(png.name)
                # Damnd kill: living HP → underflow while player alive.
                if (
                    prev_bhp > 0
                    and prev_bhp <= 128
                    and bhp > 128
                    and 0 < php <= 128
                ):
                    outcome = "boss_dead"
                    path = save_state(
                        env, GAME_DIR, GAME, "Stage1_Clear"
                    )
                    saved_states.append(path.name)
                    png = save_rgb_png(
                        obs, out / f"clear_{frame_i}.png"
                    )
                    screenshots.append(png.name)
                    events.append(
                        {
                            "frame": frame_i,
                            "tag": "boss_dead",
                            "boss_hp": bhp,
                            "player_hp": php,
                        }
                    )
                    prev_bhp = bhp
                    break
                prev_bhp = bhp

            if bst != prev_bst:
                status_changes.append(
                    {
                        "frame": frame_i,
                        "from": prev_bst,
                        "to": bst,
                        "hp": bhp,
                        "player_hp": php,
                    }
                )
                if bst == 3 and 0 < bhp <= 128:
                    drawn_frame = frame_i
                    damnd_kite = DAMND_DRAW_KITE
                    path = save_state(
                        env, GAME_DIR, GAME, "Boss_Drawn"
                    )
                    saved_states.append(path.name)
                    png = save_rgb_png(
                        obs, out / f"drawn_{frame_i}.png"
                    )
                    screenshots.append(png.name)
                    events.append(
                        {
                            "frame": frame_i,
                            "tag": "drawn",
                            "boss_hp": bhp,
                            "player_hp": php,
                        }
                    )
                if prev_bst >= 3 and bst < 3 and 0 < php <= 128:
                    outcome = "boss_dead"
                    path = save_state(
                        env, GAME_DIR, GAME, "Stage1_Clear"
                    )
                    saved_states.append(path.name)
                    events.append(
                        {
                            "frame": frame_i,
                            "tag": "boss_dead_st",
                            "player_hp": php,
                        }
                    )
                    prev_bst = bst
                    break
                prev_bst = bst

            if state.level_complete:
                outcome = "stage_clear"
                path = save_state(
                    env, GAME_DIR, GAME, "Stage1_Clear"
                )
                saved_states.append(path.name)
                break
            if php > 128 or php == 0:
                outcome = "death"
                break

        end = parse_game_state(env.get_ram(), frame=frame_i)
        return {
            "outcome": outcome,
            "success": outcome in ("stage_clear", "boss_dead"),
            "frames": frame_i,
            "start_state": state_name,
            "trial": trial,
            "kills": kills + kill_off,
            "seg_kills": kills,
            "peak_enemy_hp": peaks,
            "drawn_frame": drawn_frame,
            "start_player_hp": start_hp,
            "end_player_hp": end.health,
            "lives": end.lives,
            "boss_hits": boss_hits,
            "status_changes": status_changes,
            "events": events,
            "screenshots": screenshots,
            "saved_states": saved_states,
            "game_status": read_u8(env.get_ram(), ADDR_GAME_STATUS),
            "thug_kills": kills + kill_off,
            "thug_table": [
                e for e in events if "kill" in e or "hit" in e
            ],
        }
    finally:
        env.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default="Boss_ThugMid")
    parser.add_argument("--max-frames", type=int, default=16000)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI for jump-dash door clear."""
    args = _build_parser().parse_args(argv)
    report = run_door_jump_clear(
        state_name=args.state,
        max_frames=args.max_frames,
        out_dir=args.out_dir,
        trials=args.trials,
    )
    print(
        f"best outcome={report['outcome']} kills={report['kills']} "
        f"peaks={report['peak_enemy_hp']} drawn={report['drawn_frame']} "
        f"boss_hits={len(report['boss_hits'])}"
    )
    for event in report["events"]:
        if (
            "kill" in event
            or "chip" in event
            or event.get("tag")
            in ("drawn", "ghost_gone", "boss_dead", "boss_dead_st")
        ):
            print(f"  {event}")
    for hit in report["boss_hits"][:12]:
        print(f"  boss {hit}")
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
