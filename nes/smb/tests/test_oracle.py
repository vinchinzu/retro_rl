"""Focused tests for FCEUX oracle extract + fceumm compare helpers (no live FCEUX)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smb.paths import REPO_ROOT
from smb.tas.oracle.compare_fceumm_chain import (
    find_first_divergence,
    oracle_body_offsets,
)
from smb.tas.oracle.extract_fceux_checkpoints import (
    ORACLE_DIR,
    ORACLE_EVIDENCE_DIR,
    build_checkpoint_table,
    load_jsonl,
    write_run_config,
)
from smb.tas.search import (
    ORACLE_CONTROL_FRAME,
    ORACLE_FIRST_DIVERGENCE_OFFSET,
    ORACLE_FIRST_OBSTACLE_FRAME,
    ORACLE_FIRST_OBSTACLE_OFFSET,
    apply_a_release,
    clone_fm2_body,
    count_lr,
    dense_compare_to_oracle,
    enumerate_local_v3,
    gate_progress,
    lr_broken,
    mut_a_clear_single,
    mut_a_dual_edge,
    mut_a_release_tail,
    mut_r_drop_single,
    rank_v3,
    should_prune_p1,
)


def test_write_run_config_uses_absolute_out_paths(tmp_path: Path) -> None:
    out_trace = tmp_path / "sub" / "trace.jsonl"
    out_named = tmp_path / "sub" / "named.jsonl"
    cfg_path = write_run_config(
        out_trace=out_trace,
        out_named=out_named,
        start_frame=10,
        end_frame=20,
        dense_from=10,
        dense_to=20,
    )
    text = cfg_path.read_text(encoding="utf-8")
    assert 'out_trace = "/' in text
    assert 'out_named = "/' in text
    assert str(out_trace.resolve()) in text
    assert str(out_named.resolve()) in text
    assert "start_frame = 10" in text
    assert "end_frame = 20" in text
    # Config lives next to dump_ram_trace.lua so script_dir() finds it.
    assert cfg_path.parent == ORACLE_DIR
    assert (ORACLE_DIR / "dump_ram_trace.lua").is_file()
    # Restore production-window config so unit tests do not leave a smoke window.
    write_run_config(
        out_trace=ORACLE_EVIDENCE_DIR / "fceux_ram_trace.jsonl",
        out_named=ORACLE_EVIDENCE_DIR / "fceux_named_checkpoints.jsonl",
    )


def test_build_checkpoint_table_skips_summary() -> None:
    rows = [
        {"name": "control_8_3", "movie_frame": 100, "player_x": 40, "world": 7, "level": 2},
        {"name": "_summary", "end_frame": 200},
        {"name": "mid_8_3_x900", "movie_frame": 400, "player_x": 900},
    ]
    table = build_checkpoint_table(rows)
    assert [r["name"] for r in table] == ["control_8_3", "mid_8_3_x900"]
    assert table[0]["movie_frame"] == 100
    assert table[0]["snapshot"]["player_x"] == 40


def test_oracle_body_offsets_and_first_divergence() -> None:
    oracle = {
        "control_8_3": {
            "movie_frame": 1000,
            "world": 7,
            "level": 2,
            "oper_mode": 1,
            "player_state": 7,
            "player_x": 40,
            "player_y": 176,
            "x_speed": 0,
            "y_speed": 0,
            "grounded": True,
            "timer": 301,
            "timer_mod21": 7,
            "lives": 2,
            "screen_x": 0,
            "enemies": [],
        },
        "early_8_3_after_first_obstacle": {
            "movie_frame": 1114,
            "world": 7,
            "level": 2,
            "oper_mode": 1,
            "player_state": 8,
            "player_x": 280,
            "player_y": 135,
            "x_speed": 40,
            "y_speed": -1,
            "grounded": False,
            "timer": 296,
            "timer_mod21": 2,
            "lives": 2,
            "screen_x": 168,
            "enemies": [{"type": 0}],
        },
        "mid_8_3_x900": {
            "movie_frame": 1362,
            "world": 7,
            "level": 2,
            "oper_mode": 1,
            "player_state": 8,
            "player_x": 900,
            "player_y": 120,
            "x_speed": 40,
            "y_speed": 0,
            "grounded": True,
            "timer": 280,
            "timer_mod21": 7,
            "lives": 2,
            "screen_x": 400,
            "enemies": [],
        },
    }
    offs = oracle_body_offsets(oracle)
    assert offs["control_8_3"] == 0
    assert offs["early_8_3_after_first_obstacle"] == 114
    assert offs["mid_8_3_x900"] == 362

    # Match at control; diverge at early (player_y) — mid-level, not entry/death only.
    fceumm = {
        0: {
            **{k: oracle["control_8_3"][k] for k in oracle["control_8_3"] if k != "enemies"},
            "enemies": [],
        },
        114: {
            "world": 7,
            "level": 2,
            "oper_mode": 1,
            "player_state": 8,
            "player_x": 280,
            "player_y": 109,  # diverge
            "x_speed": 40,
            "y_speed": -3,
            "grounded": False,
            "timer": 296,
            "timer_mod21": 2,
            "lives": 2,
            "screen_x": 168,
            "enemies": [{"type": 0}],
        },
        362: {
            "world": 7,
            "level": 2,
            "oper_mode": 1,
            "player_state": 8,
            "player_x": 400,  # further diverge
            "player_y": 120,
            "x_speed": 20,
            "y_speed": 0,
            "grounded": True,
            "timer": 280,
            "timer_mod21": 7,
            "lives": 2,
            "screen_x": 200,
            "enemies": [],
        },
    }
    divs = find_first_divergence(oracle, fceumm, offs, x_tol=2)
    meaningful = [
        d
        for d in divs
        if d.field not in ("x_frac", "y_frac", "frame_counter", "screen_x", "__missing_sample__")
    ]
    assert meaningful, "expected a mid-level field mismatch"
    first = meaningful[0]
    assert first.name == "early_8_3_after_first_obstacle"
    assert first.field == "player_y"
    assert first.body_offset == 114
    assert first.oracle_value == 135
    assert first.fceumm_value == 109


def test_load_jsonl_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "t.jsonl"
    p.write_text('{"a":1}\n\n{"b":2}\n', encoding="utf-8")
    rows = load_jsonl(p)
    assert rows == [{"a": 1}, {"b": 2}]
    assert load_jsonl(tmp_path / "missing.jsonl") == []


@pytest.mark.skipif(
    not (ORACLE_EVIDENCE_DIR / "fceux_checkpoints.json").is_file(),
    reason="oracle extract artifacts not present",
)
def test_oracle_evidence_has_mid_level_checkpoints() -> None:
    """Validate real extract artifacts when present (post rr-sw9v dump)."""
    path = ORACLE_EVIDENCE_DIR / "fceux_checkpoints.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    names = set(data.get("checkpoint_names") or [])
    required = {
        "control_8_3",
        "early_8_3_after_first_obstacle",
        "mid_8_3_x900",
        "mid_8_3_x1600",
        "flagpole_grab_8_3",
        "leave_8_3_to_8_4",
        "control_8_4",
    }
    assert required <= names, f"missing mid-level landmarks: {required - names}"
    assert (ORACLE_EVIDENCE_DIR / "fceux_ram_trace.jsonl").stat().st_size > 0
    assert (ORACLE_EVIDENCE_DIR / "fceux_named_checkpoints.jsonl").stat().st_size > 0
    # Paths in evidence should stay under game recordings (not repo root clutter).
    assert ORACLE_EVIDENCE_DIR.is_relative_to(REPO_ROOT / "nes" / "smb" / "recordings")


@pytest.mark.skipif(
    not (ORACLE_EVIDENCE_DIR / "compare_evidence.json").is_file(),
    reason="compare_evidence.json not present",
)
def test_compare_evidence_reports_mid_level_divergence() -> None:
    data = json.loads(
        (ORACLE_EVIDENCE_DIR / "compare_evidence.json").read_text(encoding="utf-8")
    )
    assert data.get("success") is True
    first = data.get("first_meaningful_divergence")
    assert first is not None
    # Must not be entry-only: either a mid landmark name or mid_level_trace.
    name = first.get("name") or ""
    assert name not in ("", "control_8_2")
    # body_offset should be present for body-relative compare
    assert first.get("body_offset") is not None
    offsets = data.get("body_offsets") or {}
    assert "mid_8_3_x900" in offsets
    assert "early_8_3_after_first_obstacle" in offsets
    # Control entry matched (phase at handoff is aligned; body drifts after).
    assert data.get("oracle_control_movie_frame") == 13121
    # Entry must be clean — do not re-litigate 8-2→8-3 transition.
    assert data.get("entry_diffs") == []
    # First meaningful mid-body break is early_8_3 y/vy (not entry-only).
    assert first.get("name") == "early_8_3_after_first_obstacle"
    assert first.get("body_offset") == ORACLE_FIRST_OBSTACLE_OFFSET
    assert first.get("field") in ("player_y", "y_speed")
    primary = data.get("primary_body") or {}
    # Raw FM2 dies early; max_x alone is not a port success.
    assert primary.get("reached_8_4_control") is False
    assert primary.get("leave") is None


def test_apply_a_release_preserves_lr_and_zeros_a() -> None:
    body = [
        [0, 0, 0, 0, 0, 0, 1, 1, 1],  # L+R+A
        [0, 0, 0, 0, 0, 0, 0, 1, 1],  # R+A
        [1, 0, 0, 0, 0, 0, 0, 1, 0],  # B+R
    ]
    out = apply_a_release(body, release_from=0, release_to=2)
    assert out[0][6] == 1 and out[0][7] == 1  # L+R kept
    assert out[0][8] == 0  # A cleared
    assert out[1][8] == 0
    assert out[2][8] == 0  # outside window untouched only if release_to excludes — 2 is exclusive so index 2 kept
    assert count_lr(out) == 1
    # Original unmodified
    assert body[0][8] == 1


def test_dense_compare_finds_first_y_div_at_101() -> None:
    """Unit: dense oracle vs fceumm series flags y/vy break at offset 101."""
    control = ORACLE_CONTROL_FRAME
    oracle_trace = {}
    fceumm_dense = {}
    body = [[0] * 9 for _ in range(120)]
    for off in range(0, 115):
        mf = control + off
        # match through 100
        y = 152 if off >= 100 else 176
        ys = -5 if off >= 97 else 0
        o_ys = -3 if off >= 101 else ys
        o_y = 152 if off == 101 else (135 if off == 114 else y)
        f_y = 152 if off == 101 else (109 if off == 114 else y)
        f_ys = -5 if off >= 101 else ys
        oracle_trace[mf] = {
            "player_x": 248 if off >= 101 else 40,
            "player_y": o_y if off >= 101 else y,
            "y_speed": o_ys,
            "x_speed": 40 if off >= 97 else 0,
            "timer": 296 if off >= 97 else 301,
            "timer_mod21": 2,
            "grounded": off < 97,
        }
        fceumm_dense[off] = {
            "player_x": 248 if off >= 101 else 40,
            "player_y": f_y if off >= 101 else y,
            "y_speed": f_ys,
            "x_speed": 40 if off >= 97 else 0,
            "timer": 296 if off >= 97 else 301,
            "timer_mod21": 2,
            "grounded": off < 97,
        }
    rows = dense_compare_to_oracle(fceumm_dense, oracle_trace, body, until=114)
    first_y = next(r for r in rows if r.body_offset > 0 and r.y_div)
    assert first_y.body_offset == ORACLE_FIRST_DIVERGENCE_OFFSET
    assert first_y.movie_frame == control + ORACLE_FIRST_DIVERGENCE_OFFSET
    r114 = next(r for r in rows if r.body_offset == ORACLE_FIRST_OBSTACLE_OFFSET)
    assert r114.y_div is True
    assert r114.oracle["player_y"] == 135
    assert r114.fceumm["player_y"] == 109


def test_gate_progress_requires_ordered_landmarks() -> None:
    """max_x-only progress must not count as x900 / leave success."""
    gates = {
        "early_8_3_after_first_obstacle": {
            "match": False,
            "fceumm": {
                "player_x": 280,
                "player_y": 136,
                "y_speed": 2,
                "x_speed": 40,
                "timer": 296,
                "timer_mod21": 2,
                "grounded": False,
            },
            "oracle": {
                "player_x": 280,
                "player_y": 135,
                "y_speed": -1,
                "x_speed": 40,
                "timer": 296,
                "timer_mod21": 2,
                "grounded": False,
            },
        },
        "mid_8_3_x900": {
            "match": False,
            "fceumm": {"player_x": 528, "player_y": 145},
            "oracle": {"player_x": 900, "player_y": 61},
        },
        "mid_8_3_x1600": {
            "match": False,
            "fceumm": {"player_x": 528},
            "oracle": {"player_x": 1600},
        },
        "leave_8_3_to_8_4": {
            "match": False,
            "fceumm": {"player_x": 526},
            "oracle": {"player_x": 3554},
        },
        "control_8_4": {"match": False, "fceumm": {}, "oracle": {}},
    }
    prog = gate_progress(gates, xy_tol=1)
    assert prog["first_obstacle_xy"] is True
    assert prog["first_obstacle_exact"] is False
    assert prog["x900"] is False
    assert prog["x1600"] is False
    assert prog["control_8_4"] is False


def test_clone_fm2_body_length_and_control_pin() -> None:
    # synthetic frames: index i has marker in slot 0
    frames = [[i % 2, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(ORACLE_CONTROL_FRAME + 50)]
    body = clone_fm2_body(frames, n=20)
    assert len(body) == 20
    assert body[0][0] == frames[ORACLE_CONTROL_FRAME][0]
    assert body[5][0] == frames[ORACLE_CONTROL_FRAME + 5][0]


@pytest.mark.skipif(
    not (ORACLE_EVIDENCE_DIR / "early83_jump_repair_evidence.json").is_file(),
    reason="early83 jump repair evidence not present",
)
def test_early83_repair_evidence_honest_gates() -> None:
    """Documented repair improves |dy| but does not claim full port / x900."""
    data = json.loads(
        (ORACLE_EVIDENCE_DIR / "early83_jump_repair_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    assert data.get("schema") == "smb.oracle_early83_jump_repair.v1"
    diag = data.get("diagnosis") or {}
    assert diag.get("entry_diffs") == []
    assert diag.get("first_y_vy_divergence_body_offset") == ORACLE_FIRST_DIVERGENCE_OFFSET
    best = data.get("best_variant")
    assert best
    row = (data.get("variants") or {})[best]
    # Measured improvement: |dy114| < baseline 26
    assert abs(row.get("dy114") or 99) < 26
    gates = row.get("gate_order_pass") or {}
    # Must not claim x900 / leave / 8-4 without evidence
    assert gates.get("x900") is False
    assert gates.get("x1600") is False
    assert gates.get("leave") is False
    assert gates.get("control_84") is False
    # Candidate path is under oracle evidence dir (not shared models/)
    cand = data.get("candidate") or ""
    assert "oracle_happylee_8_3" in cand
    assert "natural_82" not in cand


@pytest.mark.skipif(
    not (
        ORACLE_EVIDENCE_DIR / "smb_8_3_oracle_early_jump_repair_candidate.json"
    ).is_file(),
    reason="early jump repair candidate not present",
)
def test_early_jump_repair_candidate_preserves_lr() -> None:
    from smb.policy import expand_nes9_rle, load_nes9_rle_seed

    path = ORACLE_EVIDENCE_DIR / "smb_8_3_oracle_early_jump_repair_candidate.json"
    data = load_nes9_rle_seed(path)
    assert data.get("route_id") == "smb_8_3_oracle_early_jump_repair_candidate"
    meta = data.get("oracle_meta") or {}
    assert meta.get("preserve_lr") is True
    assert meta.get("no_natural_82_splice") is True
    frames = expand_nes9_rle(data)
    assert count_lr(frames) >= 1 or any(fr[6] and fr[7] for fr in frames)
    # Distinct artifact: lives under recordings, not models/
    assert path.is_relative_to(ORACLE_EVIDENCE_DIR)


def test_mut_a_release_tail_zeros_a_preserves_lr() -> None:
    body = [
        [0, 0, 0, 0, 0, 0, 1, 1, 1],  # L+R+A
        [0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1],
    ]
    out = mut_a_release_tail(body, release_from=1, end=3)
    assert out[0][8] == 1 and out[0][6] == 1 and out[0][7] == 1
    assert out[1][8] == 0 and out[1][7] == 1
    assert out[2][8] == 0
    assert count_lr(out) == count_lr(body)
    assert not lr_broken(body, out)
    assert body[1][8] == 1  # base unmodified


def test_mut_a_clear_single_and_dual_edge() -> None:
    body = [[0, 0, 0, 0, 0, 0, 0, 0, 1] for _ in range(10)]
    single = mut_a_clear_single(body, 3)
    assert single[3][8] == 0 and single[2][8] == 1 and single[4][8] == 1
    dual = mut_a_dual_edge(body, release_from=2, rehold_at=4, rehold_len=2)
    # [2,4) cleared, [4,6) rehold A, then clear to 140 cap within body
    assert dual[2][8] == 0 and dual[3][8] == 0
    assert dual[4][8] == 1 and dual[5][8] == 1
    assert dual[6][8] == 0
    assert body[2][8] == 1  # unmodified


def test_mut_r_drop_skips_lr_frames() -> None:
    body = [
        [0, 0, 0, 0, 0, 0, 1, 1, 0],  # L+R — must not drop
        [0, 0, 0, 0, 0, 0, 0, 1, 0],  # R only
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    assert mut_r_drop_single(body, 0) is None
    out = mut_r_drop_single(body, 1)
    assert out is not None and out[1][7] == 0 and out[1][6] == 0
    # synthetic lr break
    broken = [list(fr) for fr in body]
    broken[0][7] = 0
    assert lr_broken(body, broken) is True


def test_rank_v3_exact_beats_high_max_x() -> None:
    o114 = {
        "player_x": 280,
        "player_y": 135,
        "y_speed": -1,
        "x_speed": 40,
        "timer": 296,
        "timer_mod21": 2,
        "grounded": False,
    }
    exact = {
        "exact_114": True,
        "yys_exact_114": True,
        "gate_progress": {
            "first_obstacle_xy": True,
            "x900": False,
            "x1600": False,
            "flag_or_leave": False,
            "control_8_4": False,
        },
        "dy114": 0,
        "dys114": 0,
        "dx114": 0,
        "s114": dict(o114),
        "ys101_match": False,
        "lr_broken": False,
        "max_x": 300,
        "death": None,
    }
    far = {
        "exact_114": False,
        "yys_exact_114": False,
        "gate_progress": {
            "first_obstacle_xy": False,
            "x900": True,
            "x1600": False,
            "flag_or_leave": False,
            "control_8_4": False,
        },
        "dy114": -26,
        "dys114": -2,
        "dx114": 0,
        "s114": {"player_x": 280, "player_y": 109, "y_speed": -3, "timer": 296, "timer_mod21": 2},
        "ys101_match": False,
        "lr_broken": False,
        "max_x": 2000,
        "death": None,
    }
    assert rank_v3(exact, o114) > rank_v3(far, o114)
    xy_only = {
        **far,
        "gate_progress": {**far["gate_progress"], "x900": False, "first_obstacle_xy": True},
        "dy114": 1,
        "dys114": 3,
        "max_x": 532,
        "s114": {
            "player_x": 280,
            "player_y": 136,
            "y_speed": 2,
            "timer": 296,
            "timer_mod21": 2,
        },
    }
    yys = {
        **xy_only,
        "yys_exact_114": True,
        "dy114": 0,
        "dys114": 0,
        "s114": dict(o114),
        "max_x": 400,
    }
    assert rank_v3(yys, o114) > rank_v3(xy_only, o114)
    # same pose quality → x900 beats non-x900 regardless of max_x
    a = {
        **xy_only,
        "gate_progress": {**xy_only["gate_progress"], "x900": True},
        "max_x": 100,
    }
    b = {
        **xy_only,
        "gate_progress": {**xy_only["gate_progress"], "x900": False},
        "max_x": 9000,
    }
    assert rank_v3(a, o114) > rank_v3(b, o114)


def test_prune_rules_predicates() -> None:
    assert should_prune_p1({"lr_broken": True, "s114": {"player_y": 136}, "dy114": 1}) == "lr_broken"
    assert (
        should_prune_p1({"lr_broken": False, "death": 50, "s114": {}, "dy114": 0})
        == "death_before_114"
    )
    assert should_prune_p1({"lr_broken": False, "death": None, "s114": {}}) == "missing_s114"
    assert (
        should_prune_p1(
            {
                "lr_broken": False,
                "death": None,
                "s114": {"player_y": 100},
                "dy114": 20,
            }
        )
        == "dy114_gt_10"
    )
    assert (
        should_prune_p1(
            {
                "lr_broken": False,
                "death": None,
                "s114": {"player_y": 136},
                "dy114": 1,
            }
        )
        is None
    )


def test_enumerate_local_v3_bounded_and_deduped() -> None:
    body = [[0] * 9 for _ in range(200)]
    # synthetic jump-3 A hold 96-115 + R+B run-up
    for i in range(90, 93):
        body[i][0] = 1
        body[i][7] = 1
    body[96][7] = 1
    body[96][8] = 1
    for i in range(97, 116):
        body[i][8] = 1
    body[1][6] = 1
    body[1][7] = 1  # L+R early
    cands = enumerate_local_v3(body, include_b_r=True)
    assert 50 < len(cands) <= 320
    names = [n for n, _, _ in cands]
    assert names[0] == "baseline_fm2"
    assert any(n.startswith("a_release_tail_") for n in names)
    assert any(n.startswith("a_dual_edge_") for n in names)
    # L+R preserved on all
    for _, mut, _ in cands:
        assert not lr_broken(body, mut)


def test_v3_evidence_schema_constants() -> None:
    assert ORACLE_FIRST_OBSTACLE_OFFSET == 114
    assert ORACLE_FIRST_DIVERGENCE_OFFSET == 101
    v3_name = "early83_local_search_v3_evidence.json"
    cand_v3 = "smb_8_3_oracle_early_jump_repair_candidate_v3.json"
    assert v3_name != "early83_jump_repair_evidence.json"
    assert cand_v3 != "smb_8_3_oracle_early_jump_repair_candidate.json"


@pytest.mark.skipif(
    not (ORACLE_EVIDENCE_DIR / "early83_local_search_v3_evidence.json").is_file(),
    reason="v3 local-search evidence not present",
)
def test_early83_v3_evidence_honest_if_present() -> None:
    data = json.loads(
        (ORACLE_EVIDENCE_DIR / "early83_local_search_v3_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    assert data.get("schema") == "smb.oracle_early83_local_search.v3"
    assert data.get("full_port") is False
    best = data.get("best") or {}
    exact = bool(data.get("exact_114_found"))
    assert exact == bool(best.get("exact_114"))
    if not exact:
        g = best.get("gate_progress") or {}
        # may still have soft xy; must not claim leave/8-4 without exact
        assert g.get("flag_or_leave") is not True or exact
        assert g.get("control_8_4") is not True or exact
    residual = data.get("residual") or {}
    assert residual.get("schema") == "smb.oracle_early83_local_search_residual.v3"
    cand = data.get("candidate") or ""
    if cand:
        assert "oracle_happylee_8_3" in cand
        assert "candidate_v3" in cand
        assert "natural_82" not in cand
