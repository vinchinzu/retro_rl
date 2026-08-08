"""Focused fceumm probe for early HappyLee 8-3 jump/physics divergence.

Root cause (pinned by FCEUX 2.6.6 oracle vs stable-retro/fceumm):

- 8-3 **entry** matches exactly (``entry_diffs == []``), including timer /
  framerule phase and ``frame_counter``.
- Body offsets **0–100** match oracle pose (jumps 1–2 OK).
- First ``y`` / ``y_speed`` break is at body offset **101** (movie 13222):
  same ``x=248 y=152``, oracle ``ys=-3`` vs fceumm ``ys=-5``.
- Landmark **114** (first obstacle, movie 13235): oracle ``y=135 ys=-1`` vs
  fceumm baseline ``y=109 ys=-3`` (same ``x=280``, same timer/framerule).

This is **not** an 8-2→8-3 transition bug. Repair target is local jump-3
A-hold / input timing on FM2 ``13121–13235``. L+R is preserved; no
``natural_82`` mid-splice.

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.tas.oracle.probe_early_8_3
# v1 A-release sweep (default) or bounded local-search v3:
  uv run python -m smb.tas.oracle.probe_early_8_3 --search-v3 --export
```

Writes under ``recordings/tas_import/oracle_happylee_8_3/`` (distinct names;
never overwrites shared seeds). v3 → ``early83_local_search_v3_evidence.json``
+ ``smb_8_3_oracle_early_jump_repair_candidate_v3.json``.

Frame convention: body_offset N is the state **after** ``body[N-1]``; dense
row ``buttons`` is the input that just produced that state (FM2[movie-1]),
not FM2[movie].
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from smb.policy import compress_nes9_rle
from smb.ram import PLAYER_STATE_DYING, read_snapshot, rich_handoff_fingerprint
from smb.tas.chain import reach_stage_control
from smb.tas.fm2 import parse_fm2
from smb.tas.oracle.compare_fceumm_chain import (
    load_oracle_checkpoints,
    oracle_body_offsets,
)
from smb.tas.oracle.extract_fceux_checkpoints import (
    DEFAULT_FM2,
    ORACLE_EVIDENCE_DIR,
    load_jsonl,
)
from smb.tas.replay import get_state, make_level1_env, set_state, to_action9
from smb.tas.stages import is_8_4_control

# NES-9 slots (stable-retro layout; L+R at 6/7 — never sanitize).
NES_B = 0
NES_A = 8
NES_LEFT = 6
NES_RIGHT = 7

# Oracle pins (HappyLee #1715M under FCEUX 2.6.6).
ORACLE_CONTROL_FRAME = 13121
ORACLE_FIRST_OBSTACLE_FRAME = 13235
ORACLE_FIRST_DIVERGENCE_OFFSET = 101  # movie 13222
ORACLE_FIRST_OBSTACLE_OFFSET = 114

# Landmark gate order (exact pose compare; max_x alone is not success).
GATE_ORDER = (
    "early_8_3_after_first_obstacle",
    "mid_8_3_x900",
    "mid_8_3_x1600",
    "hammer_bro_nearby_8_3",
    "flag_approach_8_3",
    "flagpole_grab_8_3",
    "leave_8_3_to_8_4",
    "control_8_4",
)

POSE_KEYS = (
    "player_x",
    "player_y",
    "y_speed",
    "x_speed",
    "timer",
    "timer_mod21",
    "grounded",
)


@dataclass
class DenseRow:
    body_offset: int
    movie_frame: int
    oracle: dict[str, Any]
    fceumm: dict[str, Any]
    buttons: str
    y_div: bool
    any_div: bool


@dataclass
class ProbeReport:
    schema: str = "smb.oracle_early83_dense_probe.v1"
    success: bool = False
    entry_match: bool = False
    entry_diffs: list[dict[str, Any]] = field(default_factory=list)
    first_y_vy_divergence: dict[str, Any] | None = None
    first_any_divergence: dict[str, Any] | None = None
    landmark_114: dict[str, Any] = field(default_factory=dict)
    baseline_body: dict[str, Any] = field(default_factory=dict)
    gates: dict[str, Any] = field(default_factory=dict)
    repair: dict[str, Any] | None = None
    diagnosis: dict[str, Any] = field(default_factory=dict)
    dense_head: list[dict[str, Any]] = field(default_factory=list)
    output: str | None = None
    candidate: str | None = None


def _btn_fmt(frame: list[int]) -> str:
    names = ("B", "Sel", "St", "U", "D", "L", "R", "A")
    idxs = (0, 2, 3, 4, 5, 6, 7, 8)
    bits = [n for n, i in zip(names, idxs) if i < len(frame) and frame[i]]
    return "".join(bits) or "."


def _pose(fp: dict[str, Any]) -> dict[str, Any]:
    return {k: fp.get(k) for k in POSE_KEYS}


def _diff_pose(
    oracle_fp: dict[str, Any], fceumm_fp: dict[str, Any], *, x_tol: int = 0
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for k in POSE_KEYS:
        ov, fv = oracle_fp.get(k), fceumm_fp.get(k)
        if k == "player_x" and isinstance(ov, int) and isinstance(fv, int):
            if abs(ov - fv) <= x_tol:
                continue
        if ov != fv:
            out.append({"field": k, "oracle": ov, "fceumm": fv})
    return out


def clone_fm2_body(
    fm2_frames: list[list[int]],
    *,
    control_frame: int = ORACLE_CONTROL_FRAME,
    n: int = 2500,
) -> list[list[int]]:
    return [list(fm2_frames[control_frame + i][:9]) for i in range(n)]


def apply_a_release(body: list[list[int]], release_from: int, release_to: int = 120) -> list[list[int]]:
    """Zero A on body indices [release_from, release_to). Preserves L+R."""
    out = [list(fr) for fr in body]
    for i in range(release_from, min(release_to, len(out))):
        out[i][NES_A] = 0
    return out


def count_lr(body: list[list[int]]) -> int:
    return sum(1 for fr in body if fr[NES_LEFT] and fr[NES_RIGHT])


def lr_broken(base: list[list[int]], mut: list[list[int]]) -> bool:
    """True if L+R coexistence count drops or any L+R frame loses a side."""
    if count_lr(mut) < count_lr(base):
        return True
    n = min(len(base), len(mut))
    for i in range(n):
        bl, br = base[i][NES_LEFT], base[i][NES_RIGHT]
        if bl and br and not (mut[i][NES_LEFT] and mut[i][NES_RIGHT]):
            return True
    return False


def _clone_body(body: list[list[int]]) -> list[list[int]]:
    return [list(fr) for fr in body]


def mut_a_release_tail(body: list[list[int]], release_from: int, end: int = 140) -> list[list[int]]:
    """Zero A on [release_from, end). Preserves L+R and other buttons."""
    return apply_a_release(body, release_from, release_to=end)


def mut_a_clear_single(body: list[list[int]], index: int) -> list[list[int]]:
    out = _clone_body(body)
    if 0 <= index < len(out):
        out[index][NES_A] = 0
    return out


def mut_a_set_single(body: list[list[int]], index: int) -> list[list[int]]:
    out = _clone_body(body)
    if 0 <= index < len(out):
        out[index][NES_A] = 1
    return out


def mut_a_window_off(body: list[list[int]], start: int, length: int) -> list[list[int]]:
    """Zero A on [start, start+length); later base A bits remain."""
    out = _clone_body(body)
    for i in range(start, min(start + length, len(out))):
        out[i][NES_A] = 0
    return out


def mut_a_dual_edge(
    body: list[list[int]],
    release_from: int,
    rehold_at: int,
    rehold_len: int,
) -> list[list[int]]:
    """Zero A on [release_from, rehold_at); set A on [rehold_at, rehold_at+rehold_len).

    rehold_len=0 means leave A cleared through end of window (no re-hold).
    """
    out = _clone_body(body)
    end_clear = rehold_at if rehold_len > 0 else min(140, len(out))
    for i in range(release_from, min(end_clear, len(out))):
        out[i][NES_A] = 0
    if rehold_len > 0:
        for i in range(rehold_at, min(rehold_at + rehold_len, len(out))):
            out[i][NES_A] = 1
        # clear residual A after rehold window through natural end
        for i in range(rehold_at + rehold_len, min(140, len(out))):
            out[i][NES_A] = 0
    return out


def mut_b_clear_single(body: list[list[int]], index: int) -> list[list[int]]:
    out = _clone_body(body)
    if 0 <= index < len(out):
        out[index][NES_B] = 0
    return out


def mut_b_set_single(body: list[list[int]], index: int) -> list[list[int]]:
    out = _clone_body(body)
    if 0 <= index < len(out):
        out[index][NES_B] = 1
    return out


def mut_r_drop_single(body: list[list[int]], index: int) -> list[list[int]] | None:
    """Clear RIGHT at index only if R and not L (never touch L+R frames)."""
    if index < 0 or index >= len(body):
        return None
    fr = body[index]
    if not fr[NES_RIGHT] or fr[NES_LEFT]:
        return None
    out = _clone_body(body)
    out[index][NES_RIGHT] = 0
    return out


def mut_a_onset_shift(body: list[list[int]], onset: int, first_off: int) -> list[list[int]]:
    """Rewrite jump-3 A as hold on [onset, first_off); zero A on gap edges in 94–130."""
    out = _clone_body(body)
    for i in range(94, min(130, len(out))):
        out[i][NES_A] = 1 if onset <= i < first_off else 0
    return out


def body_window_sig(body: list[list[int]], lo: int = 90, hi: int = 140) -> tuple:
    """Hashable A/B/L/R pattern over [lo, hi) for dedupe."""
    parts = []
    for i in range(lo, min(hi, len(body))):
        fr = body[i]
        parts.append((fr[NES_B], fr[NES_LEFT], fr[NES_RIGHT], fr[NES_A]))
    return tuple(parts)


def exact_pose_114(s114: dict[str, Any], oracle_114: dict[str, Any]) -> bool:
    return not _diff_pose(oracle_114, s114, x_tol=0)


def yys_exact_114(s114: dict[str, Any], oracle_114: dict[str, Any]) -> bool:
    return (
        s114.get("player_x") == oracle_114.get("player_x")
        and s114.get("player_y") == oracle_114.get("player_y")
        and s114.get("y_speed") == oracle_114.get("y_speed")
    )


def rank_v3(row: dict[str, Any], oracle_114: dict[str, Any]) -> tuple:
    """Exact 114 pose first; max_x last (diagnostic only)."""
    g = row.get("gate_progress") or {}
    s = row.get("s114") or {}
    return (
        1 if row.get("exact_114") else 0,
        1 if row.get("yys_exact_114") else 0,
        1 if g.get("first_obstacle_xy") else 0,
        -abs(row.get("dy114") if row.get("dy114") is not None else 99),
        -abs(row.get("dys114") if row.get("dys114") is not None else 99),
        -abs(row.get("dx114") if row.get("dx114") is not None else 99),
        1 if s.get("timer") == oracle_114.get("timer") else 0,
        1 if s.get("timer_mod21") == oracle_114.get("timer_mod21") else 0,
        1 if row.get("ys101_match") else 0,
        1 if g.get("x900") else 0,
        int(row.get("x_at_x900_offset") or 0),
        1 if g.get("x1600") else 0,
        1 if g.get("flag_or_leave") else 0,
        1 if g.get("control_8_4") else 0,
        0 if row.get("lr_broken") else 1,
        int(row.get("max_x") or 0),
        -(row.get("death") or 10**9),
    )


def should_prune_p1(row: dict[str, Any], *, dy_max: int = 10) -> str | None:
    """Return prune reason or None if survivor."""
    if row.get("lr_broken"):
        return "lr_broken"
    death = row.get("death")
    if death is not None and int(death) < ORACLE_FIRST_OBSTACLE_OFFSET:
        return "death_before_114"
    if not row.get("s114"):
        return "missing_s114"
    dy = row.get("dy114")
    if dy is not None and abs(int(dy)) > dy_max:
        return f"dy114_gt_{dy_max}"
    return None


def enumerate_local_v3(
    base_body: list[list[int]],
    *,
    include_b_r: bool = True,
) -> list[tuple[str, list[list[int]], list[dict[str, Any]]]]:
    """Enumerate bounded jump-3 local mutations (deduped). Window ~90–140.

    Operators from design lane 3 + input-forensic ranked edges (onset / B-run /
    first-off@99 / R@96). Does not mutate outside the early-jump window.
    """
    out: list[tuple[str, list[list[int]], list[dict[str, Any]]]] = []
    seen: set[tuple] = set()
    base_lr = count_lr(base_body)

    def add(name: str, body: list[list[int]] | None, ops: list[dict[str, Any]]) -> None:
        if body is None:
            return
        sig = body_window_sig(body)
        if sig in seen:
            return
        seen.add(sig)
        out.append((name, body, ops))

    add("baseline_fm2", base_body, [{"op": "baseline"}])

    # a_release_tail: first A-off then clear (superset of v1 range 96–116)
    for r in range(95, 121):
        add(
            f"a_release_tail_{r}",
            mut_a_release_tail(base_body, r),
            [{"op": "a_release_tail", "r": r}],
        )

    # single-frame A clear where base holds A
    for i in range(95, 126):
        if i < len(base_body) and base_body[i][NES_A]:
            add(
                f"a_clear_single_{i}",
                mut_a_clear_single(base_body, i),
                [{"op": "a_clear_single", "i": i}],
            )

    # single-frame A set where base is off (post-jump re-tap)
    for i in range(110, 140):
        if i < len(base_body) and not base_body[i][NES_A]:
            add(
                f"a_set_single_{i}",
                mut_a_set_single(base_body, i),
                [{"op": "a_set_single", "i": i}],
            )

    # short A windows off inside hold
    for length in (2, 3, 4):
        for s in range(98, 116):
            add(
                f"a_window_off_{s}_L{length}",
                mut_a_window_off(base_body, s, length),
                [{"op": "a_window_off", "s": s, "L": length}],
            )

    # dual-edge: release then optional re-hold
    for r in range(99, 109):
        for p in range(r + 1, r + 5):
            for h in (0, 1, 2, 3):
                add(
                    f"a_dual_edge_r{r}_p{p}_h{h}",
                    mut_a_dual_edge(base_body, r, p, h),
                    [{"op": "a_dual_edge", "r": r, "p": p, "h": h}],
                )

    # onset × first_off compound (forensic #1/#3)
    for onset in (95, 96, 97):
        for first_off in range(99, 106):
            if first_off <= onset:
                continue
            add(
                f"a_onset_{onset}_off_{first_off}",
                mut_a_onset_shift(base_body, onset, first_off),
                [{"op": "a_onset_shift", "onset": onset, "first_off": first_off}],
            )

    if include_b_r:
        # B-run trim on pre-jump 90–92 (forensic #2)
        for i in range(90, 93):
            if i < len(base_body) and base_body[i][NES_B]:
                add(
                    f"b_clear_{i}",
                    mut_b_clear_single(base_body, i),
                    [{"op": "b_clear_single", "i": i}],
                )
        # B extend onto 93
        add(
            "b_set_93",
            mut_b_set_single(base_body, 93),
            [{"op": "b_set_single", "i": 93}],
        )
        # B + best A-release compound
        for r in (100, 101, 102, 103):
            body = mut_a_release_tail(base_body, r)
            for bi in range(90, 93):
                if bi < len(base_body) and base_body[bi][NES_B]:
                    b2 = mut_b_clear_single(body, bi)
                    add(
                        f"a_rel_{r}_b_clear_{bi}",
                        b2,
                        [
                            {"op": "a_release_tail", "r": r},
                            {"op": "b_clear_single", "i": bi},
                        ],
                    )
            b3 = mut_b_set_single(body, 93)
            add(
                f"a_rel_{r}_b_set_93",
                b3,
                [{"op": "a_release_tail", "r": r}, {"op": "b_set_single", "i": 93}],
            )
        # R drop at onset frame 96 only (forensic #4)
        rd = mut_r_drop_single(base_body, 96)
        add("r_drop_96", rd, [{"op": "r_drop_single", "i": 96}])
        for r in (100, 101, 102, 103):
            body = mut_a_release_tail(base_body, r)
            rd2 = mut_r_drop_single(body, 96)
            add(
                f"a_rel_{r}_r_drop_96",
                rd2,
                [{"op": "a_release_tail", "r": r}, {"op": "r_drop_single", "i": 96}],
            )

    # sanity: never emit lr-broken as preferred (still keep for residual log)
    _ = base_lr
    return out


def play_body(
    env: Any,
    body: list[list[int]],
    *,
    lives: int,
    sample_offsets: set[int],
    max_play: int | None = None,
    dense_until: int = 0,
) -> dict[str, Any]:
    """Play body from current env state; sample rich FPs; track leave/death."""
    samples: dict[int, dict[str, Any]] = {}
    dense: dict[int, dict[str, Any]] = {}
    max_x = 0
    death: int | None = None
    leave: int | None = None
    reached_84 = False
    control_84: int | None = None
    samples[0] = rich_handoff_fingerprint(env.get_ram(), frame=0)
    if dense_until >= 0:
        dense[0] = samples[0]
    limit = len(body) if max_play is None else min(len(body), max_play)
    for i in range(limit):
        env.step(to_action9(body[i]))
        off = i + 1
        ram = env.get_ram()
        snap = read_snapshot(ram, frame=off)
        px = int(snap.player_x)
        if 0 < px < 20000:
            max_x = max(max_x, px)
        if death is None and (
            int(snap.lives) < lives or int(snap.player_state) == PLAYER_STATE_DYING
        ):
            death = off
        if leave is None and int(snap.world) == 7 and int(snap.level) == 3:
            leave = off
        if not reached_84 and is_8_4_control(snap):
            reached_84 = True
            control_84 = off
        if off in sample_offsets:
            samples[off] = rich_handoff_fingerprint(ram, frame=off)
        if off <= dense_until:
            dense[off] = rich_handoff_fingerprint(ram, frame=off)
        if death is not None and off > death + 8:
            break
        if leave is not None and off > leave + 80:
            break
    return {
        "samples": samples,
        "dense": dense,
        "max_x": max_x,
        "death": death,
        "leave": leave,
        "reached_8_4_control": reached_84,
        "control_8_4_offset": control_84,
    }


def dense_compare_to_oracle(
    dense: dict[int, dict[str, Any]],
    oracle_trace: dict[int, dict[str, Any]],
    body: list[list[int]],
    *,
    control_frame: int = ORACLE_CONTROL_FRAME,
    until: int = 120,
) -> list[DenseRow]:
    rows: list[DenseRow] = []
    for off in range(0, until + 1):
        mf = control_frame + off
        o = oracle_trace.get(mf) or {}
        f = dense.get(off) or {}
        if not o and not f:
            continue
        y_div = bool(o) and bool(f) and (
            o.get("player_y") != f.get("player_y") or o.get("y_speed") != f.get("y_speed")
        )
        any_div = bool(o) and bool(f) and bool(
            _diff_pose(o, f, x_tol=0)
        )
        btn = "."
        if off > 0 and off - 1 < len(body):
            btn = _btn_fmt(body[off - 1])
        rows.append(
            DenseRow(
                body_offset=off,
                movie_frame=mf,
                oracle=_pose(o) if o else {},
                fceumm=_pose(f) if f else {},
                buttons=btn,
                y_div=y_div,
                any_div=any_div,
            )
        )
    return rows


def gate_landmarks(
    samples: dict[int, dict[str, Any]],
    oracle: dict[str, dict[str, Any]],
    offsets: dict[str, int],
    *,
    x_tol: int = 0,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in GATE_ORDER:
        if name not in oracle or name not in offsets:
            continue
        off = offsets[name]
        o = oracle[name]
        f = samples.get(off) or {}
        diffs = _diff_pose(o, f, x_tol=x_tol) if f else [
            {"field": k, "oracle": o.get(k), "fceumm": None} for k in POSE_KEYS
        ]
        out[name] = {
            "body_offset": off,
            "match": len(diffs) == 0 and bool(f),
            "diffs": diffs,
            "fceumm": _pose(f) if f else {},
            "oracle": _pose(o),
        }
    return out


def gate_progress(gates: dict[str, Any], *, xy_tol: int = 1) -> dict[str, bool]:
    """Ordered success flags. max_x alone is never treated as a pass."""
    early = gates.get("early_8_3_after_first_obstacle") or {}
    f = early.get("fceumm") or {}
    o = early.get("oracle") or {}
    xy_ok = (
        f.get("player_x") == o.get("player_x")
        and isinstance(f.get("player_y"), int)
        and isinstance(o.get("player_y"), int)
        and abs(int(f["player_y"]) - int(o["player_y"])) <= xy_tol
    )
    x900 = gates.get("mid_8_3_x900") or {}
    x1600 = gates.get("mid_8_3_x1600") or {}
    leave = gates.get("leave_8_3_to_8_4") or {}
    c84 = gates.get("control_8_4") or {}

    def _x_near(g: dict[str, Any], target: int, tol: int = 8) -> bool:
        fx = (g.get("fceumm") or {}).get("player_x")
        return isinstance(fx, int) and abs(fx - target) <= tol

    return {
        "first_obstacle_exact": bool(early.get("match")),
        "first_obstacle_xy": bool(xy_ok),
        "x900": _x_near(x900, 900),
        "x1600": _x_near(x1600, 1600),
        "flag_or_leave": bool(leave.get("match"))
        or bool((leave.get("fceumm") or {}).get("player_x") and
                abs(int((leave.get("fceumm") or {}).get("player_x") or 0) - 3554) <= 20),
        "control_8_4": bool(c84.get("match")),
    }


def search_a_release_repairs(
    env: Any,
    ctrl_state: Any,
    base_body: list[list[int]],
    *,
    lives: int,
    oracle: dict[str, dict[str, Any]],
    offsets: dict[str, int],
    sample_offsets: set[int],
    release_from_range: range = range(96, 116),
) -> dict[str, Any]:
    """Local state-gated A-release search on jump-3 window only (v1)."""
    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    o114 = oracle["early_8_3_after_first_obstacle"]

    def rank(row: dict[str, Any]) -> tuple:
        """Ordered gates: first obstacle → x900 → x1600 → leave/8-4.

        max_x is a tie-break only — never sole success.
        """
        g = row["gate_progress"]
        return (
            1 if g.get("first_obstacle_exact") else 0,
            1 if g.get("first_obstacle_xy") else 0,
            1 if g.get("x900") else 0,
            1 if g.get("x1600") else 0,
            1 if g.get("flag_or_leave") else 0,
            1 if g.get("control_8_4") else 0,
            -abs(row.get("dy114") or 99),
            -abs(row.get("dys114") or 99),
            row.get("max_x") or 0,
            -(row.get("death") or 10**9),
        )

    # baseline
    set_state(env, ctrl_state)
    base_res = play_body(
        env, base_body, lives=lives, sample_offsets=sample_offsets, max_play=2200
    )
    base_gates = gate_landmarks(base_res["samples"], oracle, offsets)
    s114 = base_res["samples"].get(ORACLE_FIRST_OBSTACLE_OFFSET) or {}
    base_row = {
        "name": "baseline_fm2",
        "release_from": None,
        "max_x": base_res["max_x"],
        "death": base_res["death"],
        "leave": base_res["leave"],
        "reached_8_4_control": base_res["reached_8_4_control"],
        "s114": _pose(s114),
        "dy114": int(s114.get("player_y") or 0) - int(o114["player_y"]),
        "dys114": int(s114.get("y_speed") or 0) - int(o114["y_speed"]),
        "gate_progress": gate_progress(base_gates),
        "lr_frames": count_lr(base_body),
    }
    trials.append(base_row)
    best = {**base_row, "body": base_body, "gates": base_gates}

    for rel in release_from_range:
        body = apply_a_release(base_body, rel)
        set_state(env, ctrl_state)
        res = play_body(
            env, body, lives=lives, sample_offsets=sample_offsets, max_play=2200
        )
        gates = gate_landmarks(res["samples"], oracle, offsets)
        s114 = res["samples"].get(ORACLE_FIRST_OBSTACLE_OFFSET) or {}
        row = {
            "name": f"A_release_from_{rel}",
            "release_from": rel,
            "max_x": res["max_x"],
            "death": res["death"],
            "leave": res["leave"],
            "reached_8_4_control": res["reached_8_4_control"],
            "s114": _pose(s114),
            "dy114": int(s114.get("player_y") or 0) - int(o114["player_y"]),
            "dys114": int(s114.get("y_speed") or 0) - int(o114["y_speed"]),
            "gate_progress": gate_progress(gates),
            "lr_frames": count_lr(body),
        }
        trials.append(row)
        if best is None or rank(row) > rank(best):
            best = {**row, "body": body, "gates": gates}

    trials_sorted = sorted(trials, key=rank, reverse=True)
    return {
        "n": len(trials),
        "baseline": base_row,
        "best": {k: best[k] for k in best if k not in ("body", "gates")} if best else None,
        "best_gates": (best or {}).get("gates"),
        "best_body": (best or {}).get("body"),
        "top": trials_sorted[:12],
        "rank_key": "gates then |dy114| then max_x (max_x never sole success)",
    }


def _trial_from_play(
    *,
    name: str,
    ops: list[dict[str, Any]],
    body: list[list[int]],
    base_body: list[list[int]],
    res: dict[str, Any],
    oracle: dict[str, dict[str, Any]],
    offsets: dict[str, int],
    o114: dict[str, Any],
    o101: dict[str, Any] | None,
    gates: dict[str, Any] | None = None,
    phase: str = "p1",
) -> dict[str, Any]:
    s114 = res["samples"].get(ORACLE_FIRST_OBSTACLE_OFFSET) or {}
    s101 = res["samples"].get(ORACLE_FIRST_DIVERGENCE_OFFSET) or {}
    if gates is None:
        gates = gate_landmarks(res["samples"], oracle, offsets)
    dx = (
        int(s114.get("player_x") or 0) - int(o114.get("player_x") or 0)
        if s114
        else 99
    )
    dy = (
        int(s114.get("player_y") or 0) - int(o114.get("player_y") or 0)
        if s114
        else 99
    )
    dys = (
        int(s114.get("y_speed") or 0) - int(o114.get("y_speed") or 0)
        if s114
        else 99
    )
    x900_off = offsets.get("mid_8_3_x900")
    x_at_900 = None
    if x900_off is not None:
        sx = res["samples"].get(x900_off) or {}
        x_at_900 = sx.get("player_x")
    ys101 = s101.get("y_speed")
    ys101_match = (
        o101 is not None
        and ys101 is not None
        and int(ys101) == int(o101.get("y_speed") or 0)
    )
    row = {
        "name": name,
        "ops": ops,
        "s101": {
            "player_x": s101.get("player_x"),
            "player_y": s101.get("player_y"),
            "y_speed": ys101,
        },
        "s114": _pose(s114) if s114 else {},
        "dx114": dx,
        "dy114": dy,
        "dys114": dys,
        "exact_114": exact_pose_114(s114, o114) if s114 else False,
        "yys_exact_114": yys_exact_114(s114, o114) if s114 else False,
        "ys101_match": bool(ys101_match),
        "gate_progress": gate_progress(gates),
        "x_at_x900_offset": x_at_900,
        "max_x": res["max_x"],
        "death": res["death"],
        "leave": res["leave"],
        "reached_8_4_control": res["reached_8_4_control"],
        "lr_frames": count_lr(body),
        "lr_broken": lr_broken(base_body, body),
        "phase": phase,
        "pruned": None,
    }
    return row


def search_local_v3(
    env: Any,
    ctrl_state: Any,
    base_body: list[list[int]],
    *,
    lives: int,
    oracle: dict[str, dict[str, Any]],
    offsets: dict[str, int],
    sample_offsets: set[int],
    include_b_r: bool = True,
    max_p1: int = 320,
    max_p2: int = 32,
    p1_max_play: int = 130,
    p2_max_play: int = 2200,
) -> dict[str, Any]:
    """Two-phase local search: exact 114 pose first, then mid-level gates.

    P1 plays to ~130 for every unique mutation; P2 extends survivors for
    x900/x1600/leave. max_x is diagnostic only.
    """
    import time

    t0 = time.time()
    o114 = oracle["early_8_3_after_first_obstacle"]
    # optional dense oracle row at first divergence for ys101 match
    o101: dict[str, Any] | None = None
    # if oracle checkpoints lack off-101, synthesize from known pin
    o101 = {
        "player_x": 248,
        "player_y": 152,
        "y_speed": -3,
    }

    p1_samples = {
        0,
        ORACLE_FIRST_DIVERGENCE_OFFSET,
        ORACLE_FIRST_OBSTACLE_OFFSET,
        116,
    }
    candidates = enumerate_local_v3(base_body, include_b_r=include_b_r)[:max_p1]
    p1_rows: list[dict[str, Any]] = []
    bodies: dict[str, list[list[int]]] = {}

    for name, body, ops in candidates:
        bodies[name] = body
        set_state(env, ctrl_state)
        res = play_body(
            env,
            body,
            lives=lives,
            sample_offsets=p1_samples,
            max_play=p1_max_play,
        )
        row = _trial_from_play(
            name=name,
            ops=ops,
            body=body,
            base_body=base_body,
            res=res,
            oracle=oracle,
            offsets=offsets,
            o114=o114,
            o101=o101,
            phase="p1",
        )
        row["pruned"] = should_prune_p1(row)
        p1_rows.append(row)

    # survivors: not pruned on dy/death; allow lr_broken only as residual log
    survivors = [
        r
        for r in p1_rows
        if r.get("pruned") is None or r.get("pruned") == "lr_broken"
    ]
    # always keep top-K by rank even if pruned on dy (for residual Pareto)
    p1_ranked = sorted(p1_rows, key=lambda r: rank_v3(r, o114), reverse=True)
    frontier: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for r in survivors + p1_ranked:
        if r["name"] in seen_names:
            continue
        if r.get("pruned") in ("death_before_114", "missing_s114"):
            continue
        if r.get("pruned") and r.get("pruned") != "lr_broken":
            # still allow top dy-near for residual, skip far for P2
            if abs(int(r.get("dy114") or 99)) > 10:
                continue
        seen_names.add(r["name"])
        frontier.append(r)
        if len(frontier) >= max_p2:
            break
    # prefer unpruned + best rank
    frontier = sorted(frontier, key=lambda r: rank_v3(r, o114), reverse=True)[:max_p2]

    p2_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_body: list[list[int]] | None = None
    best_gates: dict[str, Any] | None = None

    for seed in frontier:
        body = bodies[seed["name"]]
        set_state(env, ctrl_state)
        res = play_body(
            env,
            body,
            lives=lives,
            sample_offsets=sample_offsets,
            max_play=p2_max_play,
        )
        gates = gate_landmarks(res["samples"], oracle, offsets)
        row = _trial_from_play(
            name=seed["name"],
            ops=seed["ops"],
            body=body,
            base_body=base_body,
            res=res,
            oracle=oracle,
            offsets=offsets,
            o114=o114,
            o101=o101,
            gates=gates,
            phase="p2",
        )
        p2_rows.append(row)
        if best is None or rank_v3(row, o114) > rank_v3(best, o114):
            if not row.get("lr_broken"):
                best = row
                best_body = body
                best_gates = gates

    if best is None and p1_ranked:
        # fall back to best non-lr-broken p1
        for r in p1_ranked:
            if not r.get("lr_broken"):
                best = r
                best_body = bodies[r["name"]]
                best_gates = None
                break

    all_for_top = p2_rows if p2_rows else p1_rows
    top = sorted(all_for_top, key=lambda r: rank_v3(r, o114), reverse=True)[:16]
    baseline = next((r for r in p1_rows if r["name"] == "baseline_fm2"), p1_rows[0] if p1_rows else {})

    n_exact = sum(1 for r in p1_rows if r.get("exact_114"))
    n_yys = sum(1 for r in p1_rows if r.get("yys_exact_114"))
    n_xy = sum(1 for r in p1_rows if (r.get("gate_progress") or {}).get("first_obstacle_xy"))
    n_ys101 = sum(1 for r in p1_rows if r.get("ys101_match"))

    residual = {
        "schema": "smb.oracle_early83_local_search_residual.v3",
        "a_only_and_local_exhausted": n_exact == 0,
        "window": [90, 140],
        "best_name": (best or {}).get("name"),
        "best_ops": (best or {}).get("ops"),
        "oracle_114": _pose(o114),
        "best_114": (best or {}).get("s114"),
        "dx": (best or {}).get("dx114"),
        "dy": (best or {}).get("dy114"),
        "dys": (best or {}).get("dys114"),
        "s101": {
            "oracle_ys": -3,
            "fceumm_ys": ((best or {}).get("s101") or {}).get("y_speed"),
            "y": ((best or {}).get("s101") or {}).get("player_y"),
        },
        "n_exact_114": n_exact,
        "n_yys_exact": n_yys,
        "n_xy_tol1": n_xy,
        "n_ys101_match": n_ys101,
        "pareto": [
            {
                "name": r["name"],
                "dy": r.get("dy114"),
                "dys": r.get("dys114"),
                "max_x": r.get("max_x"),
                "ys101": (r.get("s101") or {}).get("y_speed"),
            }
            for r in top[:8]
        ],
        "next_narrow_class": [
            "instrument VerticalForce/jump-timer RAM on FCEUX vs fceumm at off 100-101",
            "pre-jump subpixel via earlier body 80-95 micro (still no entry reopen)",
            "do not promote max_x or xy-only as pure-FM2 success",
        ],
        "do_not": [
            "natural_82",
            "skills_leave",
            "hybrid",
            "fceumm_physics_patch",
            "reopen_entry",
        ],
    }

    return {
        "schema": "smb.oracle_early83_local_search.v3",
        "n_p1": len(p1_rows),
        "n_p2": len(p2_rows),
        "n_unique": len(candidates),
        "n_exact_114": n_exact,
        "n_yys_exact": n_yys,
        "n_xy_tol1": n_xy,
        "n_ys101_match": n_ys101,
        "exact_114_found": n_exact > 0,
        "baseline": baseline,
        "best": best,
        "best_gates": best_gates,
        "best_body": best_body,
        "top": top,
        "residual": residual,
        "rank_key": (
            "exact_114 > yys > xy > -|dy| > -|dys| > timer > ys101 > x900… > max_x last"
        ),
        "elapsed_s": round(time.time() - t0, 3),
        "operators": [
            "baseline",
            "a_release_tail",
            "a_clear_single",
            "a_set_single",
            "a_window_off",
            "a_dual_edge",
            "a_onset_shift",
            "b_clear/set_compound",
            "r_drop_96",
        ],
    }


def export_candidate(
    body: list[list[int]],
    *,
    path: Path,
    meta: dict[str, Any],
    route_id: str = "smb_8_3_oracle_early_jump_repair_candidate",
) -> Path:
    """Write a distinct candidate; never overwrite shared seeds under models/."""
    path = Path(path)
    if path.exists():
        path = path.with_name(path.stem + "_v2" + path.suffix)
    # drop trailing empty pad if death known
    n = int(meta.get("export_frames") or len(body))
    body = body[:n]
    payload = {
        "format": "nes9_rle",
        "route_id": route_id,
        "game_name": "SuperMarioBros-Nes",
        "num_frames": len(body),
        "source": (
            "Oracle-informed early-8-3 jump-3 A-hold repair from HappyLee FM2 "
            f"@ control {ORACLE_CONTROL_FRAME}; L+R preserved; not natural_82"
        ),
        "oracle_meta": {
            "kind": "state_gated_local_A_edge_repair",
            "fm2_control_frame": ORACLE_CONTROL_FRAME,
            "preserve_lr": True,
            "no_natural_82_splice": True,
            **meta,
        },
        "segments": compress_nes9_rle(body),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def run_probe(
    *,
    fm2_path: Path = DEFAULT_FM2,
    search_repair: bool = True,
    search_v3: bool = False,
    export: bool = True,
    dense_until: int = 120,
) -> ProbeReport:
    """Reach real 8-3 control, dense-compare early body, optional local repair."""
    ORACLE_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    report = ProbeReport()
    oracle = load_oracle_checkpoints()
    if "control_8_3" not in oracle or "early_8_3_after_first_obstacle" not in oracle:
        report.diagnosis = {"error": "missing oracle checkpoints — run extract first"}
        return report

    offsets = oracle_body_offsets(oracle)
    sample_offs = set(offsets.values()) | {
        0,
        ORACLE_FIRST_DIVERGENCE_OFFSET,
        ORACLE_FIRST_OBSTACLE_OFFSET,
        200,
        400,
        600,
        900,
        1200,
        1600,
        2000,
    }

    fm2 = parse_fm2(fm2_path).frames
    base_body = clone_fm2_body(fm2)
    # Oracle dense trace for frame-level compare
    trace_rows = load_jsonl(ORACLE_EVIDENCE_DIR / "fceux_ram_trace.jsonl")
    oracle_trace = {
        int(r["movie_frame"]): r for r in trace_rows if "movie_frame" in r
    }

    env = make_level1_env()
    try:
        gate = reach_stage_control(env, "8-3")
        if not gate.get("success"):
            report.diagnosis = {"error": "failed to reach 8-3 control", "gate": gate}
            return report
        lives = int(gate["control_snap"].lives)
        ctrl_fp = rich_handoff_fingerprint(env.get_ram(), frame=0)
        entry_diffs = _diff_pose(oracle["control_8_3"], ctrl_fp, x_tol=0)
        # also frame_counter / subpixel if present
        for k in ("frame_counter", "x_frac", "y_frac"):
            if oracle["control_8_3"].get(k) != ctrl_fp.get(k):
                entry_diffs.append(
                    {
                        "field": k,
                        "oracle": oracle["control_8_3"].get(k),
                        "fceumm": ctrl_fp.get(k),
                    }
                )
        report.entry_diffs = entry_diffs
        report.entry_match = len(entry_diffs) == 0
        ctrl_state = get_state(env)

        # Baseline dense + body
        set_state(env, ctrl_state)
        base_res = play_body(
            env,
            base_body,
            lives=lives,
            sample_offsets=sample_offs,
            max_play=2200,
            dense_until=dense_until,
        )
        rows = dense_compare_to_oracle(
            base_res["dense"], oracle_trace, base_body, until=dense_until
        )
        first_y = next((r for r in rows if r.body_offset > 0 and r.y_div), None)
        first_any = next((r for r in rows if r.body_offset > 0 and r.any_div), None)
        report.first_y_vy_divergence = asdict(first_y) if first_y else None
        report.first_any_divergence = asdict(first_any) if first_any else None
        report.dense_head = [
            asdict(r)
            for r in rows
            if r.body_offset in (0, 1, 3, 26, 97, 100, 101, 102, 114, 116)
            or r.y_div
            or r.body_offset % 20 == 0
        ][:40]

        s114 = base_res["samples"].get(ORACLE_FIRST_OBSTACLE_OFFSET) or {}
        o114 = oracle["early_8_3_after_first_obstacle"]
        report.landmark_114 = {
            "oracle": _pose(o114),
            "fceumm": _pose(s114),
            "dy": int(s114.get("player_y") or 0) - int(o114["player_y"]),
            "dys": int(s114.get("y_speed") or 0) - int(o114["y_speed"]),
            "match": not _diff_pose(o114, s114),
        }
        base_gates = gate_landmarks(base_res["samples"], oracle, offsets)
        report.baseline_body = {
            "max_x": base_res["max_x"],
            "death": base_res["death"],
            "leave": base_res["leave"],
            "reached_8_4_control": base_res["reached_8_4_control"],
            "gate_progress": gate_progress(base_gates),
            "lr_frames_in_body_prefix": count_lr(base_body[:200]),
        }
        report.gates = {"baseline": base_gates}

        repair_info: dict[str, Any] | None = None
        if search_v3:
            repair_info = search_local_v3(
                env,
                ctrl_state,
                base_body,
                lives=lives,
                oracle=oracle,
                offsets=offsets,
                sample_offsets=sample_offs,
            )
            repair_public = {
                k: repair_info[k]
                for k in (
                    "schema",
                    "n_p1",
                    "n_p2",
                    "n_unique",
                    "n_exact_114",
                    "n_yys_exact",
                    "n_xy_tol1",
                    "n_ys101_match",
                    "exact_114_found",
                    "baseline",
                    "best",
                    "top",
                    "residual",
                    "rank_key",
                    "elapsed_s",
                    "operators",
                )
                if k in repair_info
            }
            repair_public["best_gates"] = repair_info.get("best_gates")
            report.repair = repair_public
            report.gates["best_repair"] = repair_info.get("best_gates")

            # Distinct v3 evidence artifact (never overwrite v1/v2)
            v3_path = ORACLE_EVIDENCE_DIR / "early83_local_search_v3_evidence.json"
            v3_payload = {
                **repair_public,
                "entry_match": report.entry_match,
                "entry_diffs": report.entry_diffs,
                "first_y_vy_divergence": report.first_y_vy_divergence,
                "landmark_114_baseline": report.landmark_114,
                "full_port": False,
                "constraints": {
                    "preserve_lr": True,
                    "no_natural_82": True,
                    "no_hybrid": True,
                    "no_skills_leave_edit": True,
                    "no_global_physics_patch": True,
                    "entry_not_relitigated": True,
                    "max_x_diagnostic_only": True,
                },
                "pins": {
                    "oracle_control_frame": ORACLE_CONTROL_FRAME,
                    "oracle_first_obstacle_offset": ORACLE_FIRST_OBSTACLE_OFFSET,
                    "oracle_first_divergence_offset": ORACLE_FIRST_DIVERGENCE_OFFSET,
                    "window": [90, 140],
                    "fm2": str(fm2_path),
                },
                "gate_order": list(GATE_ORDER),
            }
            v3_path.write_text(
                json.dumps(v3_payload, indent=2, default=str) + "\n", encoding="utf-8"
            )

            if export and repair_info.get("best_body") is not None:
                best = repair_info["best"] or {}
                death = best.get("death")
                export_n = (death + 50) if death else min(len(base_body), 2250)
                cand_path = (
                    ORACLE_EVIDENCE_DIR
                    / "smb_8_3_oracle_early_jump_repair_candidate_v3.json"
                )
                # overwrite only the dedicated v3 path (distinct from v1/v2)
                if cand_path.exists():
                    cand_path.unlink()
                cand = export_candidate(
                    repair_info["best_body"],
                    path=cand_path,
                    route_id="smb_8_3_oracle_early_jump_repair_candidate_v3",
                    meta={
                        "kind": "state_gated_local_search_v3",
                        "search_schema": "smb.oracle_early83_local_search.v3",
                        "variant": best.get("name"),
                        "ops": best.get("ops"),
                        "dy114": best.get("dy114"),
                        "dys114": best.get("dys114"),
                        "exact_114": best.get("exact_114"),
                        "max_x": best.get("max_x"),
                        "death": best.get("death"),
                        "gate_progress": best.get("gate_progress"),
                        "export_frames": export_n,
                        "first_divergence_body_offset": (
                            first_y.body_offset if first_y else None
                        ),
                        "evidence": str(v3_path),
                    },
                )
                report.candidate = str(cand)
                v3_payload["candidate"] = str(cand)
                v3_path.write_text(
                    json.dumps(v3_payload, indent=2, default=str) + "\n",
                    encoding="utf-8",
                )

        elif search_repair:
            repair_info = search_a_release_repairs(
                env,
                ctrl_state,
                base_body,
                lives=lives,
                oracle=oracle,
                offsets=offsets,
                sample_offsets=sample_offs,
            )
            # strip non-json body from nested
            repair_public = {
                k: repair_info[k]
                for k in ("n", "baseline", "best", "top", "rank_key")
            }
            repair_public["best_gates"] = repair_info.get("best_gates")
            report.repair = repair_public
            report.gates["best_repair"] = repair_info.get("best_gates")

            if export and repair_info.get("best_body") is not None:
                best = repair_info["best"] or {}
                death = best.get("death")
                export_n = (death + 50) if death else min(len(base_body), 2250)
                cand = export_candidate(
                    repair_info["best_body"],
                    path=ORACLE_EVIDENCE_DIR
                    / "smb_8_3_oracle_early_jump_repair_candidate.json",
                    meta={
                        "variant": best.get("name"),
                        "release_from": best.get("release_from"),
                        "dy114": best.get("dy114"),
                        "dys114": best.get("dys114"),
                        "max_x": best.get("max_x"),
                        "death": best.get("death"),
                        "gate_progress": best.get("gate_progress"),
                        "export_frames": export_n,
                        "first_divergence_body_offset": (
                            first_y.body_offset if first_y else None
                        ),
                    },
                )
                report.candidate = str(cand)

        # Diagnosis summary (honest; no full-port claim)
        v3_note = ""
        if search_v3 and report.repair:
            v3_note = (
                f" Local-search v3: n_p1={report.repair.get('n_p1')} "
                f"exact_114={report.repair.get('exact_114_found')} "
                f"best={((report.repair.get('best') or {}).get('name'))} "
                f"dy={((report.repair.get('best') or {}).get('dy114'))} "
                f"dys={((report.repair.get('best') or {}).get('dys114'))}."
            )
        report.diagnosis = {
            "entry_proven_correct": report.entry_match,
            "first_y_vy_body_offset": first_y.body_offset if first_y else None,
            "first_y_vy_movie_frame": first_y.movie_frame if first_y else None,
            "landmark_114_baseline": report.landmark_114,
            "cause_class": (
                "jump3_A_hold_input_timing_or_fceumm_vs_fceux_variable_jump"
            ),
            "notes": (
                "Entry FP exact. Jumps 1–2 match. Jump 3: with identical FM2 A-hold, "
                "fceumm keeps ys=-5 at off 101 while FCEUX has ys=-3 (mid-hold force; "
                "A still pressed on FM2). Local input heals reach y≈136 (|dy|≤1) but "
                "not exact (y=135,ys=-1); x900/leave still fail. Not an 8-2→8-3 "
                "transition bug. L+R preserved. No natural_82 splice. max_x alone is "
                "not success."
                + v3_note
            ),
            "gate_order": list(GATE_ORDER),
            "full_port": False,
        }
        report.success = True
    finally:
        env.close()

    out_path = ORACLE_EVIDENCE_DIR / "early83_dense_probe_evidence.json"
    # JSON-safe dump (no body blobs)
    payload = {
        "schema": report.schema,
        "success": report.success,
        "entry_match": report.entry_match,
        "entry_diffs": report.entry_diffs,
        "first_y_vy_divergence": report.first_y_vy_divergence,
        "first_any_divergence": report.first_any_divergence,
        "landmark_114": report.landmark_114,
        "baseline_body": report.baseline_body,
        "gates_summary": {
            k: {
                name: {
                    "match": g.get("match"),
                    "fceumm": g.get("fceumm"),
                    "diffs": g.get("diffs"),
                }
                for name, g in (v or {}).items()
            }
            if isinstance(v, dict)
            else v
            for k, v in (report.gates or {}).items()
        },
        "repair": report.repair,
        "diagnosis": report.diagnosis,
        "dense_head": report.dense_head,
        "candidate": report.candidate,
        "pins": {
            "oracle_control_frame": ORACLE_CONTROL_FRAME,
            "oracle_first_obstacle_frame": ORACLE_FIRST_OBSTACLE_FRAME,
            "fm2": str(fm2_path),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    report.output = str(out_path)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fm2", type=Path, default=DEFAULT_FM2)
    ap.add_argument(
        "--search-repair",
        action="store_true",
        default=True,
        help="Search local A-release repairs v1 (default on unless --search-v3)",
    )
    ap.add_argument("--no-search-repair", action="store_true")
    ap.add_argument(
        "--search-v3",
        action="store_true",
        help="Bounded local-search v3 (exact-114 first; distinct artifacts)",
    )
    ap.add_argument("--export", action="store_true", default=True)
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--dense-until", type=int, default=120)
    args = ap.parse_args(argv)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    from smb.paths import REPO_ROOT

    os.chdir(REPO_ROOT.resolve())
    search_v3 = bool(args.search_v3)
    search_repair = (not args.no_search_repair) and not search_v3
    report = run_probe(
        fm2_path=args.fm2,
        search_repair=search_repair,
        search_v3=search_v3,
        export=not args.no_export,
        dense_until=args.dense_until,
    )
    summary = {
        "success": report.success,
        "entry_match": report.entry_match,
        "first_y_vy": report.first_y_vy_divergence,
        "landmark_114": report.landmark_114,
        "baseline": report.baseline_body,
        "repair_best": (report.repair or {}).get("best"),
        "exact_114_found": (report.repair or {}).get("exact_114_found"),
        "candidate": report.candidate,
        "output": report.output,
        "full_port": False,
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0 if report.success else 2


if __name__ == "__main__":
    sys.exit(main())
