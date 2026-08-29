"""Local NES-9 search operators.

1-1 polish (Evaluator loops):

- single-frame deletion sweep (stride-controllable)
- A/B button edge shifts (±N frames)
- multi-frame hold shrink at a given index

Oracle 8-3 jump-3 mutations (no emulator). L+R is preserved. Live probe
CLIs that used these are deleted; git restores the runner.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from retro_harness.platformer.evaluator import EvalResult, Evaluator
from retro_harness.platformer.frame_tools import clone_frames


@dataclass
class SearchReport:
    improvements: int = 0
    baseline_clear: int | None = None
    best_clear: int | None = None
    moves: list[dict[str, Any]] = field(default_factory=list)
    elapsed_s: float = 0.0


def _clear(result: EvalResult) -> int | None:
    if not result.completed:
        return None
    return int(result.total_frames)


def systematic_delete_sweep(
    frames: list[list[int]],
    evaluator: Evaluator,
    *,
    window: tuple[int, int] | None = None,
    stride: int = 1,
    max_tries: int | None = None,
    verbose: bool = True,
) -> tuple[list[list[int]], EvalResult, SearchReport]:
    """Try deleting one frame at a time in *window*; keep every improvement."""
    best = clone_frames(frames)
    base = evaluator.evaluate(best, early_terminate=False)
    report = SearchReport(baseline_clear=_clear(base), best_clear=_clear(base))
    if not base.completed:
        return best, base, report

    lo, hi = (0, len(best)) if window is None else window
    lo = max(0, lo)
    hi = min(len(best), hi)
    t0 = time.time()
    i = lo
    tries = 0
    while i < min(hi, len(best)):
        if max_tries is not None and tries >= max_tries:
            break
        cand = best[:i] + best[i + 1 :]
        r = evaluator.evaluate(cand, early_terminate=False)
        tries += 1
        c = _clear(r)
        if c is not None and c < (report.best_clear or c + 1):
            if verbose:
                print(f"[DEL] @{i} → clear {c} (−{(report.best_clear or c) - c})")
            best = cand
            base = r
            report.best_clear = c
            report.improvements += 1
            report.moves.append({"op": "delete", "at": i, "clear": c})
            # New frame sits at i; shrink hi; don't advance.
            hi = min(hi, len(best))
            continue
        i += max(1, stride)
    report.elapsed_s = time.time() - t0
    if verbose:
        print(
            f"[DEL] done imps={report.improvements} "
            f"clear {report.baseline_clear}→{report.best_clear} "
            f"tries={tries} in {report.elapsed_s:.1f}s"
        )
    return best, base, report


def _shift_button_edge(
    frames: list[list[int]],
    edge: int,
    button: int,
    shift: int,
) -> list[list[int]] | None:
    cand = clone_frames(frames)
    new_e = edge + shift
    if new_e <= 0 or new_e >= len(cand):
        return None
    val = cand[edge][button]
    prev = cand[edge - 1][button]
    if val == prev:
        return None
    if shift > 0:
        for j in range(edge, min(new_e, len(cand))):
            cand[j][button] = prev
    else:
        for j in range(new_e, edge):
            cand[j][button] = val
    return cand


def edge_shift_search(
    frames: list[list[int]],
    evaluator: Evaluator,
    *,
    buttons: Sequence[int] = (8, 0),  # A, B
    window: tuple[int, int] | None = None,
    shifts: Sequence[int] = (-3, -2, -1, 1, 2, 3),
    verbose: bool = True,
) -> tuple[list[list[int]], EvalResult, SearchReport]:
    """Shift each rising/falling edge of *buttons* by a few frames."""
    best = clone_frames(frames)
    base = evaluator.evaluate(best, early_terminate=False)
    report = SearchReport(baseline_clear=_clear(base), best_clear=_clear(base))
    if not base.completed:
        return best, base, report

    lo, hi = (1, len(best)) if window is None else window
    lo = max(1, lo)
    hi = min(len(best), hi)
    t0 = time.time()

    for btn in buttons:
        edges = [
            i
            for i in range(lo, hi)
            if i < len(best) and best[i][btn] != best[i - 1][btn]
        ]
        if verbose:
            print(f"[EDGE] button={btn} edges={len(edges)}")
        for edge in edges:
            for shift in shifts:
                cand = _shift_button_edge(best, edge, btn, shift)
                if cand is None:
                    continue
                r = evaluator.evaluate(cand, early_terminate=False)
                c = _clear(r)
                if c is not None and c < (report.best_clear or c + 1):
                    if verbose:
                        print(
                            f"[EDGE] btn={btn} edge={edge} shift={shift} → clear {c}"
                        )
                    best = cand
                    base = r
                    report.best_clear = c
                    report.improvements += 1
                    report.moves.append(
                        {
                            "op": "edge",
                            "button": btn,
                            "edge": edge,
                            "shift": shift,
                            "clear": c,
                        }
                    )
                    break  # next edge on improved seed

    report.elapsed_s = time.time() - t0
    if verbose:
        print(
            f"[EDGE] done imps={report.improvements} "
            f"clear {report.baseline_clear}→{report.best_clear} "
            f"in {report.elapsed_s:.1f}s"
        )
    return best, base, report


def polish_systematic(
    frames: list[list[int]],
    evaluator: Evaluator,
    *,
    flag_frame: int | None = None,
    delete_stride: int = 1,
    verbose: bool = True,
) -> tuple[list[list[int]], EvalResult, SearchReport]:
    """Delete sweep + edge shifts over the pre-flag body."""
    hard = (flag_frame - 2) if flag_frame else len(frames)
    hard = max(1, min(hard, len(frames)))
    best, result, rep1 = systematic_delete_sweep(
        frames,
        evaluator,
        window=(0, hard),
        stride=delete_stride,
        verbose=verbose,
    )
    best, result, rep2 = edge_shift_search(
        best,
        evaluator,
        window=(1, min(hard, len(best))),
        verbose=verbose,
    )
    combined = SearchReport(
        improvements=rep1.improvements + rep2.improvements,
        baseline_clear=rep1.baseline_clear,
        best_clear=rep2.best_clear or rep1.best_clear,
        moves=rep1.moves + rep2.moves,
        elapsed_s=rep1.elapsed_s + rep2.elapsed_s,
    )
    return best, result, combined


# NES-9 slots (stable-retro; L+R at 6/7 — never sanitize).
NES_B, NES_A, NES_LEFT, NES_RIGHT = 0, 8, 6, 7
ORACLE_CONTROL_FRAME = 13121
ORACLE_FIRST_OBSTACLE_FRAME = 13235
ORACLE_FIRST_DIVERGENCE_OFFSET = 101  # movie 13222
ORACLE_FIRST_OBSTACLE_OFFSET = 114
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


def _clone(body: list[list[int]]) -> list[list[int]]:
    return [list(fr) for fr in body]


def _fill(
    body: list[list[int]], slot: int, lo: int, hi: int, val: int
) -> list[list[int]]:
    out = _clone(body)
    for i in range(max(0, lo), min(hi, len(out))):
        out[i][slot] = val
    return out


def _bit(body: list[list[int]], index: int, slot: int, val: int) -> list[list[int]]:
    return _fill(body, slot, index, index + 1, val)


def _btn_fmt(frame: list[int]) -> str:
    names = ("B", "Sel", "St", "U", "D", "L", "R", "A")
    return "".join(n for n, i in zip(names, (0, 2, 3, 4, 5, 6, 7, 8)) if frame[i]) or "."


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


def apply_a_release(
    body: list[list[int]], release_from: int, release_to: int = 120
) -> list[list[int]]:
    """Zero A on [release_from, release_to). Preserves L+R."""
    return _fill(body, NES_A, release_from, release_to, 0)


def mut_a_release_tail(
    body: list[list[int]], release_from: int, end: int = 140
) -> list[list[int]]:
    return apply_a_release(body, release_from, release_to=end)


def mut_a_clear_single(body: list[list[int]], index: int) -> list[list[int]]:
    return _bit(body, index, NES_A, 0)


def mut_a_dual_edge(
    body: list[list[int]],
    release_from: int,
    rehold_at: int,
    rehold_len: int,
) -> list[list[int]]:
    """Clear A on [release_from, rehold_at); optional rehold, then clear to 140."""
    end_clear = rehold_at if rehold_len > 0 else min(140, len(body))
    out = _fill(body, NES_A, release_from, end_clear, 0)
    if rehold_len > 0:
        out = _fill(out, NES_A, rehold_at, rehold_at + rehold_len, 1)
        out = _fill(out, NES_A, rehold_at + rehold_len, 140, 0)
    return out


def mut_r_drop_single(body: list[list[int]], index: int) -> list[list[int]] | None:
    """Clear RIGHT only if R and not L (never touch L+R frames)."""
    if not (0 <= index < len(body) and body[index][NES_RIGHT] and not body[index][NES_LEFT]):
        return None
    return _bit(body, index, NES_RIGHT, 0)


def count_lr(body: list[list[int]]) -> int:
    return sum(1 for fr in body if fr[NES_LEFT] and fr[NES_RIGHT])


def lr_broken(base: list[list[int]], mut: list[list[int]]) -> bool:
    """True if L+R count drops or any L+R frame loses a side."""
    if count_lr(mut) < count_lr(base):
        return True
    return any(
        b[NES_LEFT] and b[NES_RIGHT] and not (m[NES_LEFT] and m[NES_RIGHT])
        for b, m in zip(base, mut)
    )


def body_window_sig(body: list[list[int]], lo: int = 90, hi: int = 140) -> tuple:
    return tuple(
        (fr[NES_B], fr[NES_LEFT], fr[NES_RIGHT], fr[NES_A])
        for fr in body[lo:hi]
    )


def _abs99(v: Any) -> int:
    return -abs(v if v is not None else 99)


def rank_v3(row: dict[str, Any], oracle_114: dict[str, Any]) -> tuple:
    """Exact 114 pose first; max_x last (diagnostic only)."""
    g = row.get("gate_progress") or {}
    s = row.get("s114") or {}
    return (
        int(bool(row.get("exact_114"))),
        int(bool(row.get("yys_exact_114"))),
        int(bool(g.get("first_obstacle_xy"))),
        _abs99(row.get("dy114")),
        _abs99(row.get("dys114")),
        _abs99(row.get("dx114")),
        int(s.get("timer") == oracle_114.get("timer")),
        int(s.get("timer_mod21") == oracle_114.get("timer_mod21")),
        int(bool(row.get("ys101_match"))),
        int(bool(g.get("x900"))),
        int(row.get("x_at_x900_offset") or 0),
        int(bool(g.get("x1600"))),
        int(bool(g.get("flag_or_leave"))),
        int(bool(g.get("control_8_4"))),
        int(not row.get("lr_broken")),
        int(row.get("max_x") or 0),
        -(row.get("death") or 10**9),
    )


def should_prune_p1(row: dict[str, Any], *, dy_max: int = 10) -> str | None:
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
    """Bounded jump-3 mutations (deduped). Window ~90–140."""
    out: list[tuple[str, list[list[int]], list[dict[str, Any]]]] = []
    seen: set[tuple] = set()

    def add(name: str, body: list[list[int]] | None, ops: list[dict[str, Any]]) -> None:
        if body is None:
            return
        sig = body_window_sig(body)
        if sig in seen:
            return
        seen.add(sig)
        out.append((name, body, ops))

    add("baseline_fm2", base_body, [{"op": "baseline"}])
    for r in range(95, 121):
        add(f"a_release_tail_{r}", mut_a_release_tail(base_body, r), [{"op": "a_release_tail", "r": r}])
    for i in range(95, 126):
        if i < len(base_body) and base_body[i][NES_A]:
            add(f"a_clear_single_{i}", mut_a_clear_single(base_body, i), [{"op": "a_clear_single", "i": i}])
    for i in range(110, 140):
        if i < len(base_body) and not base_body[i][NES_A]:
            add(f"a_set_single_{i}", _bit(base_body, i, NES_A, 1), [{"op": "a_set_single", "i": i}])
    for length in (2, 3, 4):
        for s in range(98, 116):
            add(
                f"a_window_off_{s}_L{length}",
                _fill(base_body, NES_A, s, s + length, 0),
                [{"op": "a_window_off", "s": s, "L": length}],
            )
    for r in range(99, 109):
        for p in range(r + 1, r + 5):
            for h in (0, 1, 2, 3):
                add(
                    f"a_dual_edge_r{r}_p{p}_h{h}",
                    mut_a_dual_edge(base_body, r, p, h),
                    [{"op": "a_dual_edge", "r": r, "p": p, "h": h}],
                )
    for onset in (95, 96, 97):
        for first_off in range(max(onset + 1, 99), 106):
            held = _clone(base_body)
            for i in range(94, min(130, len(held))):
                held[i][NES_A] = int(onset <= i < first_off)
            add(
                f"a_onset_{onset}_off_{first_off}",
                held,
                [{"op": "a_onset_shift", "onset": onset, "first_off": first_off}],
            )
    if include_b_r:
        for i in range(90, 93):
            if i < len(base_body) and base_body[i][NES_B]:
                add(f"b_clear_{i}", _bit(base_body, i, NES_B, 0), [{"op": "b_clear_single", "i": i}])
        add("b_set_93", _bit(base_body, 93, NES_B, 1), [{"op": "b_set_single", "i": 93}])
        add("r_drop_96", mut_r_drop_single(base_body, 96), [{"op": "r_drop_single", "i": 96}])
        for r in (100, 101, 102, 103):
            rel = mut_a_release_tail(base_body, r)
            for bi in range(90, 93):
                if bi < len(base_body) and base_body[bi][NES_B]:
                    add(
                        f"a_rel_{r}_b_clear_{bi}",
                        _bit(rel, bi, NES_B, 0),
                        [{"op": "a_release_tail", "r": r}, {"op": "b_clear_single", "i": bi}],
                    )
            add(
                f"a_rel_{r}_b_set_93",
                _bit(rel, 93, NES_B, 1),
                [{"op": "a_release_tail", "r": r}, {"op": "b_set_single", "i": 93}],
            )
            add(
                f"a_rel_{r}_r_drop_96",
                mut_r_drop_single(rel, 96),
                [{"op": "a_release_tail", "r": r}, {"op": "r_drop_single", "i": 96}],
            )
    return out


def dense_compare_to_oracle(
    dense: dict[int, dict[str, Any]],
    oracle_trace: dict[int, dict[str, Any]],
    body: list[list[int]],
    *,
    control_frame: int = ORACLE_CONTROL_FRAME,
    until: int = 120,
) -> list[DenseRow]:
    rows: list[DenseRow] = []
    for off in range(until + 1):
        o, f = oracle_trace.get(control_frame + off) or {}, dense.get(off) or {}
        if not o and not f:
            continue
        y_div = bool(o and f) and (
            o.get("player_y") != f.get("player_y") or o.get("y_speed") != f.get("y_speed")
        )
        rows.append(
            DenseRow(
                body_offset=off,
                movie_frame=control_frame + off,
                oracle=_pose(o) if o else {},
                fceumm=_pose(f) if f else {},
                buttons=_btn_fmt(body[off - 1]) if 0 < off <= len(body) else ".",
                y_div=y_div,
                any_div=bool(o and f and _diff_pose(o, f)),
            )
        )
    return rows


def gate_progress(gates: dict[str, Any], *, xy_tol: int = 1) -> dict[str, bool]:
    """Ordered success flags. max_x alone is never a pass."""
    early = gates.get("early_8_3_after_first_obstacle") or {}
    f, o = early.get("fceumm") or {}, early.get("oracle") or {}
    fy, oy = f.get("player_y"), o.get("player_y")
    xy_ok = (
        f.get("player_x") == o.get("player_x")
        and isinstance(fy, int)
        and isinstance(oy, int)
        and abs(fy - oy) <= xy_tol
    )
    leave = gates.get("leave_8_3_to_8_4") or {}
    lx = (leave.get("fceumm") or {}).get("player_x")

    def near(name: str, target: int, tol: int = 8) -> bool:
        fx = ((gates.get(name) or {}).get("fceumm") or {}).get("player_x")
        return isinstance(fx, int) and abs(fx - target) <= tol

    return {
        "first_obstacle_exact": bool(early.get("match")),
        "first_obstacle_xy": bool(xy_ok),
        "x900": near("mid_8_3_x900", 900),
        "x1600": near("mid_8_3_x1600", 1600),
        "flag_or_leave": bool(leave.get("match"))
        or (isinstance(lx, int) and abs(lx - 3554) <= 20),
        "control_8_4": bool((gates.get("control_8_4") or {}).get("match")),
    }
