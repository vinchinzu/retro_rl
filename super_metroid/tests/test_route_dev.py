"""Unit tests for full-route hop tables and route helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from super_metroid.dev.route_dev import (
    DEFAULT_HYBRID_START_LEG,
    FULL_LEG_ORDER,
    LATE_LEG_ORDER,
    NULL_DOOR_SUBSTITUTES,
    leg_key,
    load_full_hops,
    load_late_hops,
    resolve_hop_door,
    summarize_full_graph_legs,
)
from super_metroid.video import concat_videos


def test_late_hop_table_covers_all_late_legs() -> None:
    hops = load_late_hops()
    for source, target in LATE_LEG_ORDER:
        key = leg_key(source, target)
        assert key in hops, key
        chain = hops[key]
        assert chain, key
        assert all(h.get("door") for h in chain), key


def test_late_legs_chain_rooms() -> None:
    hops = load_late_hops()
    for source, target in LATE_LEG_ORDER:
        chain = hops[leg_key(source, target)]
        for prev, cur in zip(chain, chain[1:]):
            assert prev["to"] == cur["from"], (source, target, prev, cur)


def test_phantoon_to_ridley_and_ridley_to_mb_hop_counts() -> None:
    hops = load_late_hops()
    p2g = hops["phantoon__gravity_suit"]
    d2r = hops["draygon__ridley"]
    r2s = hops["ridley__statues"]
    t2m = hops["tourian_elevator__mother_brain"]
    assert len(p2g) == 6
    assert len(d2r) == 28
    assert len(r2s) == 30
    assert len(t2m) == 11
    assert d2r[-1]["to"] == "0xB32E"
    assert t2m[-1]["to"] == "0xDD58"


def test_completion_summary_marks_late_legs() -> None:
    rows = summarize_full_graph_legs()
    late_targets = {t for _, t in LATE_LEG_ORDER}
    covered = [r for r in rows if r["target"] in late_targets and r["inLateHopTable"]]
    assert len(covered) >= 8


def test_full_hop_table_covers_all_full_legs() -> None:
    hops = load_full_hops()
    assert len(FULL_LEG_ORDER) == 22
    total = 0
    for source, target in FULL_LEG_ORDER:
        key = leg_key(source, target)
        assert key in hops, key
        chain = hops[key]
        assert chain, key
        total += len(chain)
    assert total == 199


def test_full_legs_chain_rooms() -> None:
    hops = load_full_hops()
    for source, target in FULL_LEG_ORDER:
        chain = hops[leg_key(source, target)]
        for prev, cur in zip(chain, chain[1:]):
            assert prev["to"] == cur["from"], (source, target, prev, cur)


def test_full_route_exactly_one_null_door() -> None:
    hops = load_full_hops()
    nulls: list[dict] = []
    for source, target in FULL_LEG_ORDER:
        for hop in hops[leg_key(source, target)]:
            if hop.get("door") is None:
                nulls.append({**hop, "leg": leg_key(source, target)})
    assert len(nulls) == 1
    hop = nulls[0]
    assert hop["from"] == "0xDF45"
    assert hop["to"] == "0x91F8"
    assert hop["leg"] == "ceres_ridley__landing_site"


def test_null_door_substitute_documented() -> None:
    assert NULL_DOOR_SUBSTITUTES[(0xDF45, 0x91F8)] == 0x896A
    door, is_null, sub = resolve_hop_door(
        {"from": "0xDF45", "to": "0x91F8", "door": None}
    )
    assert is_null is True
    assert door == 0x896A
    assert sub == "0x896A"
    door2, is_null2, sub2 = resolve_hop_door(
        {"from": "0xDEDE", "to": "0x96BA", "door": "0xAB34"}
    )
    assert is_null2 is False
    assert door2 == 0xAB34
    assert sub2 is None


def test_null_door_without_substitute_raises() -> None:
    with pytest.raises(ValueError, match="no substitute"):
        resolve_hop_door({"from": "0x0001", "to": "0x0002", "door": None})


def test_completion_summary_marks_full_legs() -> None:
    rows = summarize_full_graph_legs()
    assert len(rows) == 22
    full_covered = [r for r in rows if r["inFullHopTable"]]
    assert len(full_covered) == 22
    null_row = next(
        r for r in rows if r["source"] == "ceres_ridley" and r["target"] == "landing_site"
    )
    assert null_row["nullDoors"] == 1
    late_only_ok = [r for r in rows if r["inLateHopTable"]]
    assert len(late_only_ok) == 9
    for r in rows:
        assert "inLateHopTable" in r
        assert "inFullHopTable" in r
        assert "nullDoors" in r


def test_full_legs_inter_leg_continuity() -> None:
    hops = load_full_hops()
    for (s0, t0), (s1, t1) in zip(FULL_LEG_ORDER, FULL_LEG_ORDER[1:]):
        last = hops[leg_key(s0, t0)][-1]["to"]
        first = hops[leg_key(s1, t1)][0]["from"]
        assert last == first, ((s0, t0), (s1, t1), last, first)


def test_late_legs_are_suffix_of_full() -> None:
    assert LATE_LEG_ORDER == FULL_LEG_ORDER[-len(LATE_LEG_ORDER) :]
    full = load_full_hops()
    late = load_late_hops()
    for source, target in LATE_LEG_ORDER:
        key = leg_key(source, target)
        assert full[key] == late[key]


def test_hybrid_start_leg_is_post_supers() -> None:
    assert DEFAULT_HYBRID_START_LEG == "spore_spawn_supers"
    sources = [s for s, _ in FULL_LEG_ORDER]
    assert DEFAULT_HYBRID_START_LEG in sources
    # Suffix after Super collect still reaches Landing Site finish.
    started = False
    targets: list[str] = []
    for source, target in FULL_LEG_ORDER:
        if source == DEFAULT_HYBRID_START_LEG:
            started = True
        if started:
            targets.append(target)
    assert started
    assert targets[-1] == "landing_site_finish"
    assert "early_power_bombs" in targets
    assert "mother_brain" in targets


def test_concat_videos_stream_copy(tmp_path: Path) -> None:
    """Tiny synthetic clips: concat_videos must produce a readable output."""
    import subprocess

    def _tiny(path: Path, color: str) -> None:
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:s=64x64:d=0.1:r=60",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            check=True,
            capture_output=True,
        )

    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    out = tmp_path / "out.mp4"
    _tiny(a, "red")
    _tiny(b, "blue")
    report = concat_videos([a, b], out)
    assert out.is_file()
    assert out.stat().st_size > 0
    assert report["path"] == str(out)
    assert len(report["parts"]) == 2


def test_concat_videos_missing_part(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        concat_videos([tmp_path / "missing.mp4"], tmp_path / "out.mp4")
