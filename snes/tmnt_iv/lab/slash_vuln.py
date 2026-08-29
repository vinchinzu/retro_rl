"""Map Slash (char 0x50) vulnerability windows from RAM.

Loads FullHardBoss5 (or another Boss5* state), thrashes with the production
SlashTactics (or a simple toward+Y mash), continuously tops up player HP so
the probe survives, and logs every boss HP drop plus a short pre-hit status
trajectory.

Usage:
  SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
    uv run python -m tmnt_iv.lab.slash_vuln \\
    --state FullHardBoss5 --max-frames 20000 --pre-hit 16
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from retro_harness.env import make_env, reset_obs  # noqa: E402
from retro_harness.actions import buttons, idle_action  # noqa: E402
from retro_harness.segment_runner import configure_headless  # noqa: E402
from tmnt_iv.lab.slash_observe import read_slash, side  # noqa: E402
from tmnt_iv.observe import living_hp, policy_input  # noqa: E402
from tmnt_iv.paths import GAME, GAME_DIR, RECORDINGS_DIR  # noqa: E402
from tmnt_iv.policy import Stage1Policy  # noqa: E402
from tmnt_iv.ram import parse_game_state  # noqa: E402
from tmnt_iv.tactics.slash import SLASH_CHAR, SlashTactics  # noqa: E402

_DEFAULT_STATE = "FullHardBoss5"
# Probe survival (full bar). Not the Assist emergency restore (16 → 80).
_FULL_HEAL_HP = 96
_PRE_HIT_DEFAULT = 16

@dataclass(frozen=True)
class SlashSnap:
    """One-frame snapshot of player + Slash relevant RAM."""

    frame: int
    player_x: int
    player_y: int
    player_hp: int
    iframes: int
    slash_x: int
    slash_y: int
    slash_hp: int
    slash_status: int
    slash_char: int
    dx: int
    dy: int
    side: str  # "left" = player left of slash, "right", "overlap"
    adx: int
    ady: int

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["slash_status_hex"] = f"0x{self.slash_status:02X}"
        return d

@dataclass
class HitEvent:
    """Boss HP drop with pre-hit trajectory."""

    frame: int
    hp_before: int
    hp_after: int
    damage: int
    at_hit: dict[str, Any]
    pre_hit: list[dict[str, Any]] = field(default_factory=list)
    status_traj: list[str] = field(default_factory=list)
    # Status on the frame immediately before the drop (best vuln candidate).
    status_pre: str = ""
    # Status on the drop frame (often hitstun).
    status_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class PlayerHitEvent:
    """Player HP drop attributed to the current Slash status/geometry."""

    frame: int
    damage: int
    slash_status: int
    slash_status_hex: str
    adx: int
    dx: int
    side: str
    iframes: int
    player_x: int
    player_y: int
    slash_x: int
    slash_y: int
    status_traj: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

def _snap(ram: Any, frame: int) -> SlashSnap | None:
    state = parse_game_state(ram, frame=frame)
    slash = read_slash(ram)
    if slash is None:
        return None
    sx, sy, shp, status, char_id = slash
    dx = sx - state.player_x
    dy = sy - state.player_y
    return SlashSnap(
        frame=frame,
        player_x=state.player_x,
        player_y=state.player_y,
        player_hp=state.health,
        iframes=int(state.extras.get("iframes", 0)),
        slash_x=sx,
        slash_y=sy,
        slash_hp=shp,
        slash_status=status,
        slash_char=char_id,
        dx=dx,
        dy=dy,
        side=side(state.player_x, sx),
        adx=abs(dx),
        ady=abs(dy),
    )

def _heal_player(env: Any) -> None:
    """Full-bar probe survival so the window log is not cut by a KO."""
    env.set_value("player_hp", _FULL_HEAL_HP)

def _pick_action(
    *,
    mode: str,
    state: Any,
    policy: Stage1Policy,
    slash_tactics: SlashTactics,
) -> tuple[Any, str]:
    if mode == "thrash":
        # Production SlashTactics path via full stage policy.
        return policy_input(policy, state)
    if mode == "slash_only":
        fa = slash_tactics.next(state)
        if fa is not None:
            return fa.action, fa.reason
        return idle_action(), "slash_none"
    # Simple grounded mash: face Slash and hold Y (+ slight approach).
    slash = next(
        (e for e in state.living_enemies if e.kind == SLASH_CHAR),
        None,
    )
    if slash is None:
        return idle_action(), "no_slash"
    dx = slash.x - state.player_x
    toward = "RIGHT" if dx > 0 else "LEFT"
    adx = abs(dx)
    if abs(slash.y - state.player_y) > 12:
        return (
            buttons("UP" if slash.y < state.player_y else "DOWN"),
            "simple_align",
        )
    if adx > 40:
        return buttons(toward), "simple_approach"
    if adx < 8:
        return buttons("LEFT" if dx > 0 else "RIGHT"), "simple_spacing"
    return buttons(toward, "Y"), "simple_y"

def _summarize_player_hits(phits: list[PlayerHitEvent]) -> dict[str, Any]:
    if not phits:
        return {"n_player_hits": 0, "dmg_taken": 0}
    by_status: Counter[str] = Counter()
    dmg_by_status: Counter[str] = Counter()
    adx_by_status: dict[str, list[int]] = {}
    for h in phits:
        by_status[h.slash_status_hex] += 1
        dmg_by_status[h.slash_status_hex] += h.damage
        adx_by_status.setdefault(h.slash_status_hex, []).append(h.adx)
    adx_stats = {}
    for k, xs in adx_by_status.items():
        s = sorted(xs)
        adx_stats[k] = {
            "n": len(xs),
            "min": s[0],
            "max": s[-1],
            "mean": round(sum(xs) / len(xs), 1),
            "p50": s[len(s) // 2],
        }
    return {
        "n_player_hits": len(phits),
        "dmg_taken": sum(h.damage for h in phits),
        "by_status_count": dict(by_status.most_common()),
        "by_status_damage": dict(dmg_by_status.most_common()),
        "adx_by_status": adx_stats,
    }

def _summarize(
    hits: list[HitEvent],
    status_hist: Counter[int],
    phits: list[PlayerHitEvent] | None = None,
) -> dict[str, Any]:
    player_summary = _summarize_player_hits(phits or [])
    if not hits:
        return {
            "n_hits": 0,
            "status_histogram_all_frames": {
                f"0x{k:02X}": v for k, v in status_hist.most_common()
            },
            "player_damage": player_summary,
        }

    gaps = [hits[i].frame - hits[i - 1].frame for i in range(1, len(hits))]
    pre_status = Counter(h.status_pre for h in hits)
    at_status = Counter(h.status_at for h in hits)
    sides = Counter(h.at_hit["side"] for h in hits)
    dx_vals = [int(h.at_hit["dx"]) for h in hits]
    adx_vals = [int(h.at_hit["adx"]) for h in hits]
    iframe_vals = [int(h.at_hit["iframes"]) for h in hits]
    dmg_vals = [h.damage for h in hits]

    # Status present in the 4 frames immediately before each hit.
    near_pre: Counter[str] = Counter()
    for h in hits:
        for snap in h.pre_hit[-4:]:
            near_pre[snap["slash_status_hex"]] += 1

    # During spin (0xEE) frames that were in pre-hit windows, what adx?
    spin_adx: list[int] = []
    for h in hits:
        for snap in h.pre_hit:
            if snap["slash_status"] == 0xEE:
                spin_adx.append(int(snap["adx"]))

    def _pct(xs: list[int], p: float) -> float | None:
        if not xs:
            return None
        s = sorted(xs)
        idx = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
        return float(s[idx])

    return {
        "n_hits": len(hits),
        "total_damage": sum(dmg_vals),
        "dmg_per_hit": {
            "min": min(dmg_vals),
            "max": max(dmg_vals),
            "mean": round(sum(dmg_vals) / len(dmg_vals), 2),
        },
        "inter_hit_frames": {
            "n": len(gaps),
            "min": min(gaps) if gaps else None,
            "max": max(gaps) if gaps else None,
            "mean": round(sum(gaps) / len(gaps), 1) if gaps else None,
            "p50": _pct(gaps, 0.5),
            "p90": _pct(gaps, 0.9),
            # Cluster gaps into "cycle" (~long) vs combo (~short).
            "short_lt_60": sum(1 for g in gaps if g < 60),
            "mid_60_200": sum(1 for g in gaps if 60 <= g < 200),
            "cycle_ge_200": sum(1 for g in gaps if g >= 200),
            "cycle_gaps": [g for g in gaps if g >= 200][:20],
        },
        "status_at_hit_frame": dict(at_status.most_common()),
        "status_frame_before_hit": dict(pre_status.most_common()),
        "status_in_last_4_pre_frames": dict(near_pre.most_common()),
        "side_at_hit": dict(sides.most_common()),
        "dx_at_hit": {
            "min": min(dx_vals),
            "max": max(dx_vals),
            "mean": round(sum(dx_vals) / len(dx_vals), 1),
            "p25": _pct(dx_vals, 0.25),
            "p50": _pct(dx_vals, 0.5),
            "p75": _pct(dx_vals, 0.75),
        },
        "adx_at_hit": {
            "min": min(adx_vals),
            "max": max(adx_vals),
            "mean": round(sum(adx_vals) / len(adx_vals), 1),
            "p25": _pct(adx_vals, 0.25),
            "p50": _pct(adx_vals, 0.5),
            "p75": _pct(adx_vals, 0.75),
        },
        "iframes_at_hit": {
            "min": min(iframe_vals),
            "max": max(iframe_vals),
            "mean": round(sum(iframe_vals) / len(iframe_vals), 1),
            "zero_frac": round(
                sum(1 for v in iframe_vals if v == 0) / len(iframe_vals), 3
            ),
        },
        "spin_0xEE_adx_in_pre_hit": {
            "n": len(spin_adx),
            "min": min(spin_adx) if spin_adx else None,
            "max": max(spin_adx) if spin_adx else None,
            "mean": round(sum(spin_adx) / len(spin_adx), 1) if spin_adx else None,
            "p50": _pct(spin_adx, 0.5),
        },
        "status_histogram_all_frames": {
            f"0x{k:02X}": v for k, v in status_hist.most_common(24)
        },
        "first_hit_frame": hits[0].frame,
        "last_hit_frame": hits[-1].frame,
        "start_hp": hits[0].hp_before,
        "end_hp_after_last": hits[-1].hp_after,
        "player_damage": player_summary,
    }

def run_probe(
    *,
    state_name: str = _DEFAULT_STATE,
    max_frames: int = 20000,
    pre_hit: int = _PRE_HIT_DEFAULT,
    mode: str = "thrash",
    heal_every: int = 1,
    stop_on_ko: bool = True,
) -> dict[str, Any]:
    """Run thrash probe and return hits + summary."""
    configure_headless()
    env = make_env(GAME, state_name, GAME_DIR, render_mode="rgb_array")
    policy = Stage1Policy()
    slash_tactics = SlashTactics()
    history: deque[SlashSnap] = deque(maxlen=max(pre_hit, 4))
    hits: list[HitEvent] = []
    player_hits: list[PlayerHitEvent] = []
    status_hist: Counter[int] = Counter()
    reason_hist: Counter[str] = Counter()
    prev_hp: int | None = None
    prev_player_hp: int | None = None
    outcome = "timeout"
    start_snap: dict[str, Any] | None = None
    final_frame = 0
    heals = 0

    try:
        reset_obs(env)
        ram = env.get_ram()
        s0 = _snap(ram, 0)
        if s0 is not None:
            start_snap = s0.as_dict()
            prev_hp = s0.slash_hp
            prev_player_hp = s0.player_hp
            history.append(s0)
            status_hist[s0.slash_status] += 1

        for frame in range(1, max_frames + 1):
            final_frame = frame
            # Read first so player damage is visible, then top up.
            ram = env.get_ram()
            state = parse_game_state(ram, frame=frame)
            snap = _snap(ram, frame)

            if (
                prev_player_hp is not None
                and living_hp(state.health)
                and prev_player_hp <= 0x60
                and state.health < prev_player_hp
                and snap is not None
            ):
                pre = list(history)[-pre_hit:]
                player_hits.append(
                    PlayerHitEvent(
                        frame=frame,
                        damage=prev_player_hp - state.health,
                        slash_status=snap.slash_status,
                        slash_status_hex=f"0x{snap.slash_status:02X}",
                        adx=snap.adx,
                        dx=snap.dx,
                        side=snap.side,
                        iframes=snap.iframes,
                        player_x=snap.player_x,
                        player_y=snap.player_y,
                        slash_x=snap.slash_x,
                        slash_y=snap.slash_y,
                        status_traj=[p.as_dict()["slash_status_hex"] for p in pre],
                    )
                )

            if heal_every > 0 and frame % heal_every == 0:
                if living_hp(state.health) and state.health < _FULL_HEAL_HP:
                    _heal_player(env)
                    heals += 1
                    prev_player_hp = _FULL_HEAL_HP
                elif living_hp(state.health):
                    prev_player_hp = state.health
                else:
                    prev_player_hp = state.health
            elif living_hp(state.health):
                prev_player_hp = state.health

            # Re-read after optional heal so boss logging uses stable RAM.
            ram = env.get_ram()
            state = parse_game_state(ram, frame=frame)
            snap = _snap(ram, frame)

            if snap is not None:
                status_hist[snap.slash_status] += 1
                if prev_hp is not None and snap.slash_hp < prev_hp:
                    pre = list(history)[-pre_hit:]
                    pre_dicts = [p.as_dict() for p in pre]
                    status_traj = [p["slash_status_hex"] for p in pre_dicts]
                    status_pre = (
                        pre_dicts[-1]["slash_status_hex"] if pre_dicts else ""
                    )
                    hits.append(
                        HitEvent(
                            frame=frame,
                            hp_before=prev_hp,
                            hp_after=snap.slash_hp,
                            damage=prev_hp - snap.slash_hp,
                            at_hit=snap.as_dict(),
                            pre_hit=pre_dicts,
                            status_traj=status_traj,
                            status_pre=status_pre,
                            status_at=f"0x{snap.slash_status:02X}",
                        )
                    )
                prev_hp = snap.slash_hp
                history.append(snap)

                if stop_on_ko and snap.slash_hp <= 0:
                    outcome = "slash_ko"
                    break
            else:
                # Slash despawned / dead.
                if prev_hp is not None and prev_hp > 0 and hits:
                    outcome = "slash_gone"
                    break
                prev_hp = None

            # Stage/event leave.
            if state.stage > 4 or int(state.extras.get("event", 0x0A)) not in {
                0x0A,
                0x0B,
            }:
                if hits:
                    outcome = "stage_advance"
                break

            action, reason = _pick_action(
                mode=mode,
                state=state,
                policy=policy,
                slash_tactics=slash_tactics,
            )
            reason_hist[reason] += 1
            # Never allow A special.
            if action[8]:
                action = idle_action()
                reason_hist["blocked_a"] += 1
            env.step(action)
        else:
            outcome = "timeout"
    finally:
        env.close()

    summary = _summarize(hits, status_hist, player_hits)
    summary["outcome"] = outcome
    summary["frames"] = final_frame
    summary["state"] = state_name
    summary["mode"] = mode
    summary["heals"] = heals
    summary["top_reasons"] = reason_hist.most_common(16)
    summary["start"] = start_snap

    return {
        "summary": summary,
        "hits": [h.as_dict() for h in hits],
        "player_hits": [h.as_dict() for h in player_hits],
    }

def _print_human(report: dict[str, Any]) -> None:
    s = report["summary"]
    print("=== Slash vulnerability probe ===")
    print(
        f"state={s.get('state')} mode={s.get('mode')} "
        f"outcome={s.get('outcome')} frames={s.get('frames')} "
        f"hits={s.get('n_hits')} heals={s.get('heals')}"
    )
    if s.get("start"):
        st = s["start"]
        print(
            f"start: slash_hp={st['slash_hp']} "
            f"status=0x{st['slash_status']:02X} "
            f"px={st['player_x']} sx={st['slash_x']}"
        )
    print(f"total_damage={s.get('total_damage')} dmg/hit={s.get('dmg_per_hit')}")
    print(f"inter_hit_frames={s.get('inter_hit_frames')}")
    print(f"status_frame_before_hit={s.get('status_frame_before_hit')}")
    print(f"status_at_hit_frame={s.get('status_at_hit_frame')}")
    print(f"status_in_last_4_pre_frames={s.get('status_in_last_4_pre_frames')}")
    print(f"side_at_hit={s.get('side_at_hit')}")
    print(f"dx_at_hit={s.get('dx_at_hit')}")
    print(f"adx_at_hit={s.get('adx_at_hit')}")
    print(f"iframes_at_hit={s.get('iframes_at_hit')}")
    print(f"spin_0xEE_adx_in_pre_hit={s.get('spin_0xEE_adx_in_pre_hit')}")
    print(f"status_histogram_all_frames={s.get('status_histogram_all_frames')}")
    print(f"player_damage={s.get('player_damage')}")
    print(f"top_reasons={s.get('top_reasons')}")

    # Print a few annotated hits for the findings doc.
    hits = report["hits"]
    show = hits[:8] + (hits[-4:] if len(hits) > 12 else [])
    print("\n--- sample hits (first 8 + last 4) ---")
    for h in show:
        traj = " ".join(h["status_traj"][-12:])
        ah = h["at_hit"]
        print(
            f"f={h['frame']:5d} hp {h['hp_before']}->{h['hp_after']} "
            f"(-{h['damage']}) pre={h['status_pre']} at={h['status_at']} "
            f"dx={ah['dx']:+4d} side={ah['side']:5s} adx={ah['adx']:3d} "
            f"ifr={ah['iframes']:3d} "
            f"p=({ah['player_x']},{ah['player_y']}) "
            f"s=({ah['slash_x']},{ah['slash_y']})"
        )
        print(f"         traj[-12:]: {traj}")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default=_DEFAULT_STATE)
    parser.add_argument("--max-frames", type=int, default=20000)
    parser.add_argument(
        "--pre-hit",
        type=int,
        default=_PRE_HIT_DEFAULT,
        help="frames of status trajectory kept before each HP drop",
    )
    parser.add_argument(
        "--mode",
        choices=("thrash", "slash_only", "simple_y"),
        default="thrash",
        help="thrash=Stage1Policy (SlashTactics); slash_only; simple_y mash",
    )
    parser.add_argument(
        "--heal-every",
        type=int,
        default=1,
        help="restore player HP every N frames (0=never)",
    )
    parser.add_argument(
        "--no-stop-on-ko",
        action="store_true",
        help="keep running after Slash HP hits 0",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="write full hit log JSON (default: recordings/slash_vuln_probe/)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress human summary (JSON path still printed)",
    )
    args = parser.parse_args(argv)

    report = run_probe(
        state_name=args.state,
        max_frames=args.max_frames,
        pre_hit=args.pre_hit,
        mode=args.mode,
        heal_every=args.heal_every,
        stop_on_ko=not args.no_stop_on_ko,
    )

    out = args.json_out
    if out is None:
        out_dir = RECORDINGS_DIR / "slash_vuln_probe"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{args.state}_{args.mode}.json"
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not args.quiet:
        _print_human(report)
    print(f"\njson_out={out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
