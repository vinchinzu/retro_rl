"""Instrument production Slash KEEP (Raph jump-over behind-combo).

Live headless fight from ``RaphFullHardBoss5`` using ``Stage1Policy.tick``.
Full heal (HP=96 every drop) so the log is a DPS trace, not a survival
story. Never presses A. Writes histograms + hit events for the parent
to judge other algorithms against real waste.

Usage:
  SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
    uv run python -m tmnt_iv.lab.slash_trace \\
    --state RaphFullHardBoss5 --max-frames 40000 --heal full
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from retro_harness.env import make_env, reset_obs  # noqa: E402
from retro_harness.actions import idle_action  # noqa: E402
from retro_harness.segment_runner import configure_headless  # noqa: E402
from tmnt_iv.assist import apply_emergency_hp  # noqa: E402
from tmnt_iv.lab.slash_observe import read_slash, side  # noqa: E402
from tmnt_iv.observe import living_hp, policy_input  # noqa: E402
from tmnt_iv.paths import GAME, GAME_DIR, RECORDINGS_DIR  # noqa: E402
from tmnt_iv.policy import Stage1Policy  # noqa: E402
from tmnt_iv.ram import parse_game_state  # noqa: E402
from tmnt_iv.tactics.slash import SLASH_CHAR  # noqa: E402

_DEFAULT_STATE = "RaphFullHardBoss5"
_FULL_HP = 96
_PRE_HIT = 16
_PUNISH_WATCH = frozenset({0x3E, 0xB7, 0x2E})
_JUMP_THROUGH_REASONS = frozenset({"slash_jump_over", "slash_hop_away"})
_STATUS_KEYS = (0xEE, 0x09, 0x3E, 0x2E, 0xB7, 0x83)
_HP_MARKS = (0, 2000, 4000, 6000, 8000, 10000)
_SAMPLE_RADIUS = 100
_FPS = 60.0


@dataclass
class FrameRec:
    """One-frame KEEP snapshot (reason is the action chosen this tick)."""

    frame: int
    player_x: int
    player_y: int
    slash_x: int
    slash_y: int
    slash_hp: int
    slash_status: int
    adx: int
    dy: int
    reason: str
    iframes: int
    phase: str
    facing: int
    behind: bool
    side: str
    player_hp: int

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["slash_status_hex"] = f"0x{self.slash_status:02X}"
        return d


@dataclass
class BossHit:
    """Boss HP drop with 16-frame pre-hit trajectory."""

    frame: int
    hp_before: int
    hp_after: int
    damage: int
    reason: str
    adx: int
    dy: int
    side: str
    behind: bool
    facing: int
    phase: str
    player_x: int
    player_y: int
    slash_x: int
    slash_y: int
    iframes: int
    status_pre: str
    status_at: str
    status_traj: list[str] = field(default_factory=list)
    reason_traj: list[str] = field(default_factory=list)
    adx_traj: list[int] = field(default_factory=list)
    pre_hit: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlayerHit:
    """Player HP drop captured before the heal write."""

    frame: int
    damage: int
    slash_status: int
    slash_status_hex: str
    adx: int
    dy: int
    reason: str
    side: str
    iframes: int
    player_x: int
    player_y: int
    slash_x: int
    slash_y: int
    phase: str
    behind: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ComboStart:
    """Phase transition into grounded combo (KEEP behind-string)."""

    frame: int
    adx: int
    dy: int
    side: str
    behind: bool
    facing: int
    geom_left: bool
    slash_status: int
    slash_status_hex: str
    player_x: int
    player_y: int
    slash_x: int
    slash_y: int
    reason: str
    from_phase: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pct(xs: list[int], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return float(s[idx])


def _mean(xs: list[int] | list[float]) -> float | None:
    if not xs:
        return None
    return round(sum(xs) / len(xs), 2)


def _heal(env: Any, state: Any, mode: str) -> bool:
    """Write player HP. Return True if a write happened."""
    hp = state.health
    if mode == "full":
        if living_hp(hp) and hp < _FULL_HP:
            env.set_value("player_hp", _FULL_HP)
            return True
        return False
    return apply_emergency_hp(env, hp)


def _tactics_view(policy: Stage1Policy, state: Any, slash: Any) -> tuple[str, int, bool]:
    tactics = policy._slash  # noqa: SLF001 — KEEP diagnostic, not a rewrite
    phase = str(getattr(tactics, "_phase", ""))
    facing = int(getattr(tactics, "_facing", 0) or 0)
    behind = False
    if slash is not None:
        behind = bool(tactics._player_behind(slash, state.player_x))
    return phase, facing, behind


def _gap_stats(frames: list[int]) -> dict[str, Any]:
    if len(frames) < 2:
        gaps: list[int] = []
    else:
        gaps = [frames[i] - frames[i - 1] for i in range(1, len(frames))]
    return {
        "n": len(gaps),
        "min": min(gaps) if gaps else None,
        "max": max(gaps) if gaps else None,
        "mean": _mean(gaps),
        "p50": _pct(gaps, 0.5),
        "p90": _pct(gaps, 0.9),
        "short_lt_60": sum(1 for g in gaps if g < 60),
        "mid_60_200": sum(1 for g in gaps if 60 <= g < 200),
        "cycle_ge_200": sum(1 for g in gaps if g >= 200),
        "cycle_gaps": [g for g in gaps if g >= 200][:24],
        "all": gaps,
    }


def _summarize(
    *,
    frames_log: list[FrameRec],
    boss_hits: list[BossHit],
    player_hits: list[PlayerHit],
    combo_starts: list[ComboStart],
    reason_hist: Counter[str],
    status_hist: Counter[int],
    jumped: Counter[str],
    punish_n: int,
    adx_3e: list[int],
    adx_09: list[int],
    hp_marks: dict[str, int | None],
    outcome: str,
    final_frame: int,
    heals: int,
    damage_taken: int,
    state_name: str,
    heal_mode: str,
    start_hp: int | None,
    end_hp: int | None,
    blocked_a: int,
) -> dict[str, Any]:
    total_f = max(final_frame, 1)
    reason_frac = {
        k: round(v / total_f, 4) for k, v in reason_hist.most_common()
    }
    occupancy = {f"0x{k:02X}": status_hist.get(k, 0) for k in _STATUS_KEYS}
    occupancy["other"] = sum(
        v for k, v in status_hist.items() if k not in _STATUS_KEYS
    )
    occupancy["total"] = sum(status_hist.values())

    jump_n = sum(jumped.values())
    # Status × reason from the same-frame KEEP tick (what we did about it).
    reason_by_status: dict[str, Counter[str]] = {}
    wasted_punish = 0
    attack_on_punish = 0
    _attack_reasons = frozenset({"slash_back_attack", "slash_cross"})
    for rec in frames_log:
        if rec.slash_status < 0:
            continue
        key = f"0x{rec.slash_status:02X}"
        reason_by_status.setdefault(key, Counter())[rec.reason] += 1
        if rec.slash_status in _PUNISH_WATCH:
            if rec.reason in _attack_reasons:
                attack_on_punish += 1
            else:
                wasted_punish += 1
    reason_by_status_out = {
        k: dict(v.most_common()) for k, v in sorted(reason_by_status.items())
    }
    watch_reason: dict[str, dict[str, int]] = {}
    for st in (*_PUNISH_WATCH, 0xEE, 0x09, 0x83, 0xB6, 0xB8, 0x40):
        key = f"0x{st:02X}"
        if key in reason_by_status:
            watch_reason[key] = dict(reason_by_status[key].most_common())

    hit_frames = [h.frame for h in boss_hits]
    hit_ady = [abs(h.dy) for h in boss_hits]
    hit_y = [
        {
            "frame": h.frame,
            "player_y": h.player_y,
            "slash_y": h.slash_y,
            "dy": h.dy,
            "adx": h.adx,
            "side": h.side,
            "behind": h.behind,
        }
        for h in boss_hits
    ]
    combo_geom_left = sum(1 for c in combo_starts if c.geom_left)
    combo_behind = sum(1 for c in combo_starts if c.behind)
    combo_behind_matches_left = sum(
        1 for c in combo_starts if c.behind == c.geom_left
    )
    player_by_status: Counter[str] = Counter()
    player_dmg_by_status: Counter[str] = Counter()
    for h in player_hits:
        player_by_status[h.slash_status_hex] += 1
        player_dmg_by_status[h.slash_status_hex] += h.damage

    reason_buckets = {
        "jump_over": sum(
            reason_hist[k]
            for k in ("slash_jump_over", "slash_jump_kick")
        ),
        "slash_jump_over": reason_hist.get("slash_jump_over", 0),
        "slash_jump_kick": reason_hist.get("slash_jump_kick", 0),
        "hop_away": reason_hist.get("slash_hop_away", 0),
        "dodge": reason_hist.get("slash_dodge", 0),
        "approach": reason_hist.get("slash_approach", 0),
        "back_attack": reason_hist.get("slash_back_attack", 0),
        "space": reason_hist.get("slash_space", 0),
        "bait": reason_hist.get("slash_bait", 0),
        "cross": reason_hist.get("slash_cross", 0),
        "align": reason_hist.get("slash_align", 0),
        "wait": reason_hist.get("slash_wait", 0),
    }

    return {
        "state": state_name,
        "heal": heal_mode,
        "outcome": outcome,
        "frames": final_frame,
        "seconds": round(final_frame / _FPS, 3),
        "damage_taken": damage_taken,
        "heals": heals,
        "blocked_a": blocked_a,
        "boss_hp_start": start_hp,
        "boss_hp_end": end_hp,
        "n_boss_hits": len(boss_hits),
        "boss_damage_dealt": sum(h.damage for h in boss_hits),
        "reason_histogram": dict(reason_hist.most_common()),
        "reason_fractions": reason_frac,
        "reason_buckets": reason_buckets,
        "reason_bucket_fractions": {
            k: round(v / total_f, 4) for k, v in reason_buckets.items()
        },
        "status_occupancy": occupancy,
        "status_histogram_all": {
            f"0x{k:02X}": v for k, v in status_hist.most_common()
        },
        "jumped_through_punish": {
            "punish_frames_3e_b7_2e": punish_n,
            "jumped_frames": jump_n,
            "fraction": round(jump_n / punish_n, 4) if punish_n else None,
            "by_status_reason": dict(jumped.most_common()),
        },
        "mean_adx_status_0x3E": _mean(adx_3e),
        "mean_adx_status_0x09": _mean(adx_09),
        "n_adx_0x3E": len(adx_3e),
        "n_adx_0x09": len(adx_09),
        "first_connect_frame": hit_frames[0] if hit_frames else None,
        "first_connect_seconds": (
            round(hit_frames[0] / _FPS, 3) if hit_frames else None
        ),
        "inter_hit_gaps": _gap_stats(hit_frames),
        "boss_hp_vs_frame": hp_marks,
        "combo_starts": {
            "n": len(combo_starts),
            "behind_true": combo_behind,
            "geom_left": combo_geom_left,
            "behind_equals_geom_left": combo_behind_matches_left,
            "behind_false": len(combo_starts) - combo_behind,
            "geom_right": sum(1 for c in combo_starts if c.side == "right"),
            "mean_adx": _mean([c.adx for c in combo_starts]),
            "mean_ady": _mean([abs(c.dy) for c in combo_starts]),
            "status": dict(
                Counter(c.slash_status_hex for c in combo_starts).most_common()
            ),
            "from_phase": dict(
                Counter(c.from_phase for c in combo_starts).most_common()
            ),
        },
        "reason_by_watch_status": watch_reason,
        "reason_by_status": reason_by_status_out,
        "wasted_punish_window": {
            "punish_frames_3e_b7_2e": punish_n,
            "attacking_back_attack_or_cross": attack_on_punish,
            "not_attacking": wasted_punish,
            "not_attacking_fraction": (
                round(wasted_punish / punish_n, 4) if punish_n else None
            ),
        },
        "y_alignment_on_hits": {
            "n": len(hit_ady),
            "mean_abs_dy": _mean(hit_ady),
            "p50_abs_dy": _pct(hit_ady, 0.5),
            "max_abs_dy": max(hit_ady) if hit_ady else None,
            "aligned_le_8": sum(1 for d in hit_ady if d <= 8),
            "aligned_le_16": sum(1 for d in hit_ady if d <= 16),
            "hits": hit_y,
        },
        "side_at_hit": dict(Counter(h.side for h in boss_hits).most_common()),
        "behind_at_hit": {
            "true": sum(1 for h in boss_hits if h.behind),
            "false": sum(1 for h in boss_hits if not h.behind),
        },
        "player_damage": {
            "n_player_hits": len(player_hits),
            "dmg_taken": sum(h.damage for h in player_hits),
            "by_status_count": dict(player_by_status.most_common()),
            "by_status_damage": dict(player_dmg_by_status.most_common()),
            "mean_adx": _mean([h.adx for h in player_hits]),
        },
        "logged_frames": len(frames_log),
    }


def run_trace(
    *,
    state_name: str = _DEFAULT_STATE,
    max_frames: int = 40000,
    heal_mode: str = "full",
    stop_stage_gt: int = 4,
) -> dict[str, Any]:
    """Run KEEP policy with per-frame / per-hit tracing."""
    configure_headless()
    env = make_env(GAME, state_name, GAME_DIR, render_mode="rgb_array")
    policy = Stage1Policy()
    history: deque[FrameRec] = deque(maxlen=_PRE_HIT)
    frames_log: list[FrameRec] = []
    boss_hits: list[BossHit] = []
    player_hits: list[PlayerHit] = []
    combo_starts: list[ComboStart] = []
    reason_hist: Counter[str] = Counter()
    status_hist: Counter[int] = Counter()
    jumped: Counter[str] = Counter()
    punish_n = 0
    adx_3e: list[int] = []
    adx_09: list[int] = []
    hp_marks: dict[str, int | None] = {str(m): None for m in _HP_MARKS}
    prev_hp: int | None = None
    prev_player_hp: int | None = None
    prev_phase = "approach"
    prev_reason = "init"
    outcome = "timeout"
    start_hp: int | None = None
    end_hp: int | None = None
    final_frame = 0
    heals = 0
    damage_taken = 0
    blocked_a = 0
    start_lives = 0

    try:
        reset_obs(env)
        ram = env.get_ram()
        state0 = parse_game_state(ram, frame=0)
        start_lives = state0.lives
        slash0 = read_slash(ram)
        if slash0 is not None:
            sx, sy, shp, st, _char = slash0
            start_hp = shp
            prev_hp = shp
            hp_marks["0"] = shp
            rec0 = FrameRec(
                frame=0,
                player_x=state0.player_x,
                player_y=state0.player_y,
                slash_x=sx,
                slash_y=sy,
                slash_hp=shp,
                slash_status=st,
                adx=abs(sx - state0.player_x),
                dy=sy - state0.player_y,
                reason="init",
                iframes=int(state0.extras.get("iframes", 0)),
                phase="approach",
                facing=0,
                behind=False,
                side=side(state0.player_x, sx),
                player_hp=state0.health,
            )
            history.append(rec0)
            frames_log.append(rec0)
            status_hist[st] += 1
        prev_player_hp = state0.health if living_hp(state0.health) else None

        for frame in range(1, max_frames + 1):
            final_frame = frame
            ram = env.get_ram()
            state = parse_game_state(ram, frame=frame)
            slash_raw = read_slash(ram)
            slash_enemy = next(
                (e for e in state.living_enemies if e.kind == SLASH_CHAR),
                None,
            )

            # Player HP drop before any heal write.
            if (
                prev_player_hp is not None
                and living_hp(state.health)
                and prev_player_hp <= 0x60
                and state.health < prev_player_hp
            ):
                dmg = prev_player_hp - state.health
                damage_taken += dmg
                sx = sy = shp = st = adx = dy = 0
                hit_side = "none"
                if slash_raw is not None:
                    sx, sy, shp, st, _char = slash_raw
                    adx = abs(sx - state.player_x)
                    dy = sy - state.player_y
                    hit_side = side(state.player_x, sx)
                phase, facing, behind = _tactics_view(policy, state, slash_enemy)
                phit = PlayerHit(
                    frame=frame,
                    damage=dmg,
                    slash_status=st,
                    slash_status_hex=f"0x{st:02X}",
                    adx=adx,
                    dy=dy,
                    reason=prev_reason,
                    side=hit_side,
                    iframes=int(state.extras.get("iframes", 0)),
                    player_x=state.player_x,
                    player_y=state.player_y,
                    slash_x=sx,
                    slash_y=sy,
                    phase=phase,
                    behind=behind,
                )
                player_hits.append(phit)
                print(
                    f"PLAYER HIT f={frame} -{dmg} status=0x{st:02X} "
                    f"adx={adx} dy={dy} reason={prev_reason} "
                    f"side={hit_side} ifr={phit.iframes} phase={phase}"
                )

            if _heal(env, state, heal_mode):
                heals += 1
                ram = env.get_ram()
                state = parse_game_state(ram, frame=frame)
                slash_raw = read_slash(ram)
                slash_enemy = next(
                    (e for e in state.living_enemies if e.kind == SLASH_CHAR),
                    None,
                )
                prev_player_hp = (
                    state.health if living_hp(state.health) else prev_player_hp
                )
            elif living_hp(state.health):
                prev_player_hp = state.health

            if state.lives < start_lives:
                outcome = "life_loss"
                break
            if state.stage > stop_stage_gt:
                outcome = "stage_advance"
                end_hp = slash_raw[2] if slash_raw is not None else 0
                break
            event = int(state.extras.get("event", 0x0A))
            if event not in {0x0A, 0x0B, 0x00}:
                if boss_hits:
                    outcome = "stage_advance"
                break

            # Choose action on the post-heal snapshot (production tick).
            action, reason = policy_input(policy, state)
            if action[8]:
                action = idle_action()
                reason = "blocked_a"
                blocked_a += 1
            phase, facing, behind = _tactics_view(policy, state, slash_enemy)

            if slash_raw is not None:
                sx, sy, shp, st, _char = slash_raw
                adx = abs(sx - state.player_x)
                dy = sy - state.player_y
                rec = FrameRec(
                    frame=frame,
                    player_x=state.player_x,
                    player_y=state.player_y,
                    slash_x=sx,
                    slash_y=sy,
                    slash_hp=shp,
                    slash_status=st,
                    adx=adx,
                    dy=dy,
                    reason=reason,
                    iframes=int(state.extras.get("iframes", 0)),
                    phase=phase,
                    facing=facing,
                    behind=behind,
                    side=side(state.player_x, sx),
                    player_hp=state.health,
                )
                status_hist[st] += 1
                if st == 0x3E:
                    adx_3e.append(adx)
                elif st == 0x09:
                    adx_09.append(adx)
                if st in _PUNISH_WATCH:
                    punish_n += 1
                    if reason in _JUMP_THROUGH_REASONS:
                        jumped[f"0x{st:02X}:{reason}"] += 1

                if prev_hp is not None and shp < prev_hp:
                    pre = list(history)
                    pre_dicts = [p.as_dict() for p in pre]
                    hit = BossHit(
                        frame=frame,
                        hp_before=prev_hp,
                        hp_after=shp,
                        damage=prev_hp - shp,
                        reason=prev_reason,
                        adx=adx,
                        dy=dy,
                        side=rec.side,
                        behind=behind,
                        facing=facing,
                        phase=phase,
                        player_x=rec.player_x,
                        player_y=rec.player_y,
                        slash_x=sx,
                        slash_y=sy,
                        iframes=rec.iframes,
                        status_pre=(
                            pre_dicts[-1]["slash_status_hex"] if pre_dicts else ""
                        ),
                        status_at=f"0x{st:02X}",
                        status_traj=[p["slash_status_hex"] for p in pre_dicts],
                        reason_traj=[p.reason for p in pre],
                        adx_traj=[p.adx for p in pre],
                        pre_hit=pre_dicts,
                    )
                    boss_hits.append(hit)
                    traj = " ".join(hit.status_traj[-12:])
                    print(
                        f"BOSS HIT f={frame} hp {prev_hp}->{shp} (-{hit.damage}) "
                        f"pre={hit.status_pre} at={hit.status_at} "
                        f"adx={adx} dy={dy} reason={prev_reason} "
                        f"side={rec.side} behind={behind} facing={facing} "
                        f"py={rec.player_y} sy={sy}"
                    )
                    print(f"         traj[-12:]: {traj}")
                    print(f"         reasons: {' '.join(hit.reason_traj[-12:])}")
                    print(f"         adx: {hit.adx_traj[-12:]}")
                prev_hp = shp
                end_hp = shp
                history.append(rec)
                frames_log.append(rec)

                if shp <= 0:
                    outcome = "slash_ko"
                    reason_hist[reason] += 1
                    env.step(action)
                    break
            else:
                if prev_hp is not None and prev_hp > 0 and boss_hits:
                    outcome = "slash_gone"
                    end_hp = 0
                    break
                prev_hp = None
                rec = FrameRec(
                    frame=frame,
                    player_x=state.player_x,
                    player_y=state.player_y,
                    slash_x=-1,
                    slash_y=-1,
                    slash_hp=-1,
                    slash_status=-1,
                    adx=-1,
                    dy=0,
                    reason=reason,
                    iframes=int(state.extras.get("iframes", 0)),
                    phase=phase,
                    facing=facing,
                    behind=False,
                    side="none",
                    player_hp=state.health,
                )
                frames_log.append(rec)

            if phase == "combo" and prev_phase != "combo" and slash_raw is not None:
                sx, sy, shp, st, _char = slash_raw
                combo_starts.append(
                    ComboStart(
                        frame=frame,
                        adx=abs(sx - state.player_x),
                        dy=sy - state.player_y,
                        side=side(state.player_x, sx),
                        behind=behind,
                        facing=facing,
                        geom_left=state.player_x < sx,
                        slash_status=st,
                        slash_status_hex=f"0x{st:02X}",
                        player_x=state.player_x,
                        player_y=state.player_y,
                        slash_x=sx,
                        slash_y=sy,
                        reason=reason,
                        from_phase=prev_phase,
                    )
                )

            if frame in _HP_MARKS:
                hp_marks[str(frame)] = (
                    slash_raw[2] if slash_raw is not None else end_hp
                )

            reason_hist[reason] += 1
            prev_reason = reason
            prev_phase = phase
            env.step(action)
        else:
            outcome = "timeout"
            if slash_raw is not None:
                end_hp = slash_raw[2]
    finally:
        env.close()

    hp_marks["end"] = end_hp
    for mark in _HP_MARKS:
        if hp_marks[str(mark)] is None and final_frame >= mark:
            # Fight ended before the mark; leave None. Filled live above.
            pass

    summary = _summarize(
        frames_log=frames_log,
        boss_hits=boss_hits,
        player_hits=player_hits,
        combo_starts=combo_starts,
        reason_hist=reason_hist,
        status_hist=status_hist,
        jumped=jumped,
        punish_n=punish_n,
        adx_3e=adx_3e,
        adx_09=adx_09,
        hp_marks=hp_marks,
        outcome=outcome,
        final_frame=final_frame,
        heals=heals,
        damage_taken=damage_taken,
        state_name=state_name,
        heal_mode=heal_mode,
        start_hp=start_hp,
        end_hp=end_hp,
        blocked_a=blocked_a,
    )

    first = summary.get("first_connect_frame")
    sample: list[dict[str, Any]] = []
    if first is not None:
        lo = max(0, int(first) - _SAMPLE_RADIUS)
        hi = int(first) + _SAMPLE_RADIUS
        sample = [r.as_dict() for r in frames_log if lo <= r.frame < hi]

    return {
        "summary": summary,
        "boss_hits": [h.as_dict() for h in boss_hits],
        "player_hits": [h.as_dict() for h in player_hits],
        "combo_starts": [c.as_dict() for c in combo_starts],
        "first_connect_sample": sample,
    }


def _print_histograms(report: dict[str, Any]) -> None:
    s = report["summary"]
    print("\n=== Slash KEEP trace ===")
    print(
        f"state={s['state']} heal={s['heal']} outcome={s['outcome']} "
        f"frames={s['frames']} seconds={s['seconds']} "
        f"dmg_taken={s['damage_taken']} heals={s['heals']} "
        f"boss_hp={s['boss_hp_start']}->{s['boss_hp_end']} "
        f"blocked_a={s['blocked_a']}"
    )
    print(f"reason_buckets={s['reason_buckets']}")
    print(f"reason_bucket_fractions={s['reason_bucket_fractions']}")
    print(f"reason_histogram={s['reason_histogram']}")
    print(f"status_occupancy={s['status_occupancy']}")
    print(f"jumped_through_punish={s['jumped_through_punish']}")
    print(f"wasted_punish_window={s['wasted_punish_window']}")
    print(f"reason_by_watch_status={s['reason_by_watch_status']}")
    print(
        f"mean_adx 0x3E={s['mean_adx_status_0x3E']} "
        f"(n={s['n_adx_0x3E']})  0x09={s['mean_adx_status_0x09']} "
        f"(n={s['n_adx_0x09']})"
    )
    print(
        f"first_connect={s['first_connect_frame']} "
        f"({s['first_connect_seconds']}s)"
    )
    print(f"inter_hit_gaps={s['inter_hit_gaps']}")
    print(f"boss_hp_vs_frame={s['boss_hp_vs_frame']}")
    print(f"combo_starts={s['combo_starts']}")
    print(f"y_alignment_on_hits={s['y_alignment_on_hits']}")
    print(f"side_at_hit={s['side_at_hit']} behind_at_hit={s['behind_at_hit']}")
    print(f"player_damage={s['player_damage']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default=_DEFAULT_STATE)
    parser.add_argument("--max-frames", type=int, default=40000)
    parser.add_argument(
        "--heal",
        choices=("full", "emergency"),
        default="full",
        help="full=HP 96 every drop (DPS trace); emergency=HP<=16 → 80",
    )
    parser.add_argument("--stop-stage-gt", type=int, default=4)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="default: recordings/slash_lab/keep_trace_raph.json",
    )
    args = parser.parse_args(argv)

    report = run_trace(
        state_name=args.state,
        max_frames=args.max_frames,
        heal_mode=args.heal,
        stop_stage_gt=args.stop_stage_gt,
    )
    _print_histograms(report)

    out = args.json_out
    if out is None:
        name = (
            "keep_trace_raph.json"
            if args.heal == "full"
            else "keep_trace_raph_emergency.json"
        )
        out = RECORDINGS_DIR / "slash_lab" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\njson_out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
