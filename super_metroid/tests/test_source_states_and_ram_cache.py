"""Unit tests for selective-RAM helpers, source catalog, graph path API."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from super_metroid.ram import (
    StateCache,
    parse_counts,
    parse_env_state,
    parse_state,
    probe_pin,
    reset_parse_counts,
)
from super_metroid.source_states import (
    get_source,
    match_source_by_path,
    suggest_source_path,
    suggest_sources_for_room,
    validate_fingerprint,
)


def _nav_state(**kwargs):
    base = parse_state(np.zeros(0x2000, dtype=np.uint8))
    return replace(base, **kwargs)


def test_probe_pin_fields() -> None:
    st = _nav_state(room_id=0xA6E2, pose=2, samus_x=100, samus_y=150, door_transition=0)
    pin = probe_pin(st)
    assert pin["room"] == "0xA6E2"
    assert pin["pose"] == 2
    assert pin["x"] == 100
    assert pin["y"] == 150
    assert pin["door_transition"] == 0


def test_parse_counts_nav_vs_full() -> None:
    class _Env:
        def get_ram(self):
            return np.zeros(0x2000, dtype=np.uint8)

        class data:  # noqa: N801
            class memory:
                blocks = {0x7E0000: np.zeros(0x20000, dtype=np.uint8)}

    # read_bank7e_wram needs env.data.memory.blocks[SNES_WRAM_BANK]
    from super_metroid.ram import SNES_WRAM_BANK

    class Env:
        def get_ram(self):
            return np.zeros(0x2000, dtype=np.uint8)

        def __init__(self) -> None:
            self.data = type(
                "D",
                (),
                {
                    "memory": type(
                        "M",
                        (),
                        {"blocks": {SNES_WRAM_BANK: np.zeros(0x20000, dtype=np.uint8)}},
                    )()
                },
            )()

    env = Env()
    reset_parse_counts()
    parse_env_state(env, mode="nav")
    parse_env_state(env, mode="nav")
    parse_env_state(env, mode="full")
    counts = parse_counts()
    assert counts["nav"] == 2
    assert counts["full"] == 1


def test_state_cache_local_parse_stats() -> None:
    from super_metroid.ram import SNES_WRAM_BANK

    class Env:
        def get_ram(self):
            return np.zeros(0x2000, dtype=np.uint8)

        def __init__(self) -> None:
            self.data = type(
                "D",
                (),
                {
                    "memory": type(
                        "M",
                        (),
                        {"blocks": {SNES_WRAM_BANK: np.zeros(0x20000, dtype=np.uint8)}},
                    )()
                },
            )()

    env = Env()
    cache = StateCache(env, mode="nav")
    cache.get(frame=0)
    cache.get(frame=0)  # hit
    cache.get(frame=1)  # miss / nav parse
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 2
    assert stats["nav_parses"] == 2
    assert stats["full_parses"] == 0
    cache.reset_stats()
    assert cache.stats()["hits"] == 0
    assert cache.stats()["nav_parses"] == 0


def test_path_summary_unifies_pure_gate_and_path_verification() -> None:
    from super_metroid.progression import START_TO_SPEED_GRAPH

    caps = frozenset(
        {
            "morph_ball",
            "bombs",
            "missiles",
            "super_missiles",
            "hi_jump",
            "varia_suit",
        }
    )
    summary = START_TO_SPEED_GRAPH.path_summary(
        0xB167, 0xAD1B, caps, min_verification="controller_dev"
    )
    assert summary["reachable"] is True
    assert summary["pure_gated"] is False
    # From Frog tip: reverse to Business then Cathedral (Speedway is post-Speed).
    assert summary["blocking_edge_id"] == "frog_save_to_business"
    assert summary["blocking"]["edgeId"] == "frog_save_to_business"

    # Continuous-prefer suggest matches legacy suggest_next_hops ranking.
    next_hops = START_TO_SPEED_GRAPH.suggest_next_hops(0xB167, capabilities=caps)
    pure = START_TO_SPEED_GRAPH.suggest_pure_work(0xB167, capabilities=caps)
    unified = START_TO_SPEED_GRAPH.suggest_edges(
        0xB167, capabilities=caps, prefer="pure_work",
        exclude_verifications=frozenset({"continuous"}),
    )
    assert pure == unified
    assert next_hops[0].edge_id == pure[0].edge_id


def test_state_cache_hits_same_frame() -> None:
    from super_metroid.ram import SNES_WRAM_BANK

    class Env:
        def get_ram(self):
            return np.zeros(0x2000, dtype=np.uint8)

        def __init__(self) -> None:
            self.data = type(
                "D",
                (),
                {
                    "memory": type(
                        "M",
                        (),
                        {"blocks": {SNES_WRAM_BANK: np.zeros(0x20000, dtype=np.uint8)}},
                    )()
                },
            )()

    env = Env()
    cache = StateCache(env, mode="nav")
    a = cache.get(frame=10)
    b = cache.get(frame=10)
    assert a is b
    assert cache.hits == 1
    assert cache.misses == 1
    cache.get(frame=11)
    assert cache.misses == 2
    stats = cache.stats()
    assert stats["mode"] == "nav"


def test_source_catalog_varia_and_suggest() -> None:
    row = get_source("post_varia_collected")
    assert row.room_id == 0xA6E2
    assert "varia" in row.use_for
    ranked = suggest_sources_for_room(0xA6E2, segment_hint="varia-to-kraid")
    assert ranked
    assert ranked[0].source_id == "post_varia_collected"
    path = suggest_source_path(0xA59F, segment_hint="kraid-to-eye-return")
    assert path is not None
    assert "post_varia_to_kraid" in path.as_posix() or path.name.endswith(".state")


def test_full_continuous_kihunter_fingerprint_validates_launch_band() -> None:
    source = get_source("post_varia_continuous_to_kihunter")
    good = _nav_state(room_id=0xA4DA, pose=165, samus_x=461, samus_y=395)
    assert validate_fingerprint(good, source=source).ok
    wrong_pose = _nav_state(room_id=0xA4DA, pose=137, samus_x=461, samus_y=395)
    assert validate_fingerprint(wrong_pose, source=source).ok is False


def test_validate_fingerprint_room_mismatch() -> None:
    st = _nav_state(room_id=0xA59F, pose=1, samus_x=50, samus_y=50)
    check = validate_fingerprint(st, expected_room=0xA6E2)
    assert check.ok is False
    assert any("room" in f for f in check.failures)
    ok = validate_fingerprint(st, expected_room=0xA59F)
    assert ok.ok is True


def test_match_source_by_path_suffix() -> None:
    path = Path("scratch/post_varia_collected.state")
    row = match_source_by_path(path)
    assert row is not None
    assert row.source_id == "post_varia_collected"


def test_suggest_pure_work_and_pure_gate_after_frog_continuous_tip() -> None:
    from super_metroid.progression import START_TO_SPEED_GRAPH

    caps = frozenset(
        {
            "morph_ball",
            "bombs",
            "missiles",
            "super_missiles",
            "hi_jump",
            "varia_suit",
        }
    )
    pure = START_TO_SPEED_GRAPH.suggest_pure_work(0xB167, capabilities=caps)
    # Frog tip pure outs: reverse to Business (Cathedral repath) and optional Speedway.
    assert {edge.edge_id for edge in pure} >= {
        "frog_save_to_business",
        "frog_save_to_speedway",
    }

    gate = START_TO_SPEED_GRAPH.pure_gate(0xB167, 0xAD1B, caps)
    assert gate["reachable"] is True
    assert gate["pure_gated"] is False
    assert gate["blocking"] is not None
    assert gate["blocking"]["edgeId"] == "frog_save_to_business"
    assert gate["blocking"]["verification"] == "unverified"

    # Continuous-only gate from Kraid entry (edge continuous) should clear short hop.
    cont = START_TO_SPEED_GRAPH.pure_gate(
        0xA59F, 0xA6E2, caps, min_verification="continuous"
    )
    assert cont["reachable"] is True
    assert cont["pure_gated"] is True
    assert cont["blocking"] is None


def test_frog_save_successor_source_is_cataloged() -> None:
    source = get_source("post_frog_continuous")
    assert source.room_id == 0xB167
    assert source.continuous_like is True
    assert source.y_min == 130
    assert source.y_max == 145
    assert source.poses == frozenset({1})
    assert match_source_by_path(Path("scratch/post_frog_continuous.state")) is source
    ranked = suggest_sources_for_room(0xB167, segment_hint="frog-save-to-speedway")
    assert ranked[0] is source


def test_scaffold_tip_dry_run_exits_zero() -> None:
    import subprocess
    import sys

    cmd = [
        sys.executable,
        str(Path("super_metroid/scripts/scaffold_tip.py")),
        "--segment",
        "test_scaffold_hop",
        "--from-room",
        "0xA7DE",
        "--to-room",
        "0xB167",
        "--module",
        "k4_norfair",
        "--card-id",
        "SM-TEST-SCAFFOLD",
        "--dry-run",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "play_test_scaffold_hop" in proc.stdout
    assert "Tip-extension checklist" in proc.stdout
