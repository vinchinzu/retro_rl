"""Practice-hack repertoire: human pins + bot policy/route-edge/recovery."""

from __future__ import annotations

import pytest

from super_metroid import practice_repertoire as pr
from super_metroid.paths import (
    PRACTICE_REPERTOIRE_PATH,
    SHARED_PRACTICE_ROM,
    SHARED_ROM,
    VANILLA_ROM_SHA1,
)
from super_metroid.scripts.setup_practice_rom import apply_ips


def test_catalog_on_disk() -> None:
    assert PRACTICE_REPERTOIRE_PATH.is_file()
    cat = pr.load_catalog()
    assert cat["product_category"] == "kpdr25"
    assert len(cat["categories"]) >= 20
    assert len(cat["sessions"]) >= 3000


def test_kpdr25_has_core_sessions() -> None:
    ids = {s.id for s in pr.sessions(category="kpdr25")}
    for need in (
        "kpdr25/crateria/morph",
        "kpdr25/crateria/bomb_torizo",
        "kpdr25/brinstar/big_pink",
        "kpdr25/upper_norfair/bat_cave",
        "kpdr25/wrecked_ship/phantoon",
        "kpdr25/tourian/mother_brain_2",
    ):
        assert need in ids


def test_morph_fingerprint() -> None:
    s = pr.get_session("kpdr25/crateria/morph")
    assert s.room_id == 0x9E9F
    assert s.name == "Morph"
    assert s.canonical_state_path.name == "morph.state"
    assert "practice_repertoire" in str(s.canonical_state_path)
    assert not hasattr(type(s), "roles")


def test_product_map_resolves() -> None:
    rows = pr.mapped_sessions()
    assert len(rows) >= 10
    morph = pr.get_session("kpdr25/crateria/morph")
    m = morph.product_map()
    assert m is not None
    assert m["start_preset"] == "morph"
    assert morph.living_state_path() is not None


def test_route_order_and_neighbors() -> None:
    route = pr.route_sessions("kpdr25")
    assert len(route) >= 100
    assert route[0].route_index == 0
    morph = pr.get_session("kpdr25/crateria/morph")
    prev_s, next_s = pr.neighbors(morph.id)
    assert next_s is not None
    # Morph is mid-Crateria; both neighbors should exist.
    assert prev_s is not None
    assert prev_s.category == "kpdr25"
    assert next_s.category == "kpdr25"


def test_route_edge_and_hop_key() -> None:
    edge = pr.route_edge("kpdr25/crateria/morph")
    assert edge is not None
    assert edge.from_session == "kpdr25/crateria/morph"
    assert edge.to_session
    assert edge.hop_key.startswith("0x9E9F:")
    assert not hasattr(edge, "roles")
    board = pr.product_route_edges("kpdr25")
    assert len(board) == len(pr.route_sessions("kpdr25")) - 1
    # Back-compat aliases still resolve.
    assert pr.stitch_seam is pr.route_edge
    assert pr.StitchSeam is pr.RouteEdge
    assert pr.product_stitch_board is pr.product_route_edges


def test_recovery_by_room() -> None:
    # Pre-Morph inventory → Morph preset (items 0). Post-Morph (0x0004) may
    # land construction_zone; product_spine morph still preferred when close.
    pre = pr.recover_session(0x9E9F, 0x0000)
    assert pre is not None
    assert pre.session_id == "kpdr25/crateria/morph"
    assert pre.hop_key is not None
    assert pre.grade in pr.GRADES

    post = pr.recover_session(0x9E9F, 0x0004)
    assert post is not None
    assert post.room_id == 0x9E9F
    # Living product pin wins over unmapped same-room presets when inv close.
    assert post.session_id == "kpdr25/crateria/morph"
    assert post.grade == "product_spine"


def test_recovery_hint_for_state_duck() -> None:
    class _S:
        room_id = 0x9E9F
        collected_items = 0x0004

    hint = pr.recovery_hint_for_state(_S())
    assert hint is not None
    assert hint.session_id == "kpdr25/crateria/morph"


def test_policy_board_card() -> None:
    card = pr.policy_board_card("kpdr25/crateria/morph")
    assert card.session_id.endswith("morph")
    assert card.hop_key is not None
    assert card.grade in pr.GRADES
    assert "optimize_room_policy" in card.tune_command
    assert not hasattr(card, "roles") or "roles" not in card.to_dict()


def test_session_work_card() -> None:
    card = pr.session_work_card("kpdr25/crateria/morph")
    assert card["hop_key"]
    assert card["route_edge"] is not None
    assert card["policy_board"]["grade"] in pr.GRADES
    assert "roles" not in card


def test_gap_report_shape() -> None:
    report = pr.gap_report("kpdr25")
    assert report["session_count"] == len(pr.sessions(category="kpdr25"))
    assert report["mapped_count"] >= 10
    assert report["route_edges"] >= 100
    assert "by_grade" in report
    assert "roles" not in report
    assert report["vanilla_rom_ready"] == SHARED_ROM.is_file()


def test_categories_cli_ids() -> None:
    ids = [c["id"] for c in pr.categories()]
    assert ids[0] == "kpdr20"
    assert "kpdr25" in ids
    assert ids.index("kpdr25") == 4  # mainmenu.asm order


def test_apply_ips_identity_expand() -> None:
    patch = b"PATCH" + bytes([0x00, 0x00, 0x10, 0x00, 0x01, 0xAB]) + b"EOF"
    out = apply_ips(b"\x00" * 16, patch, min_size=32)
    assert len(out) >= 32
    assert out[0x10] == 0xAB


@pytest.mark.skipif(not SHARED_ROM.is_file(), reason="vanilla ROM not present")
def test_vanilla_sha1_when_present() -> None:
    import hashlib

    h = hashlib.sha1(SHARED_ROM.read_bytes()).hexdigest()
    assert h == VANILLA_ROM_SHA1


@pytest.mark.skipif(
    not SHARED_PRACTICE_ROM.is_file(),
    reason="practice ROM not built (run setup_practice_rom.py)",
)
def test_practice_rom_size() -> None:
    assert SHARED_PRACTICE_ROM.stat().st_size == 4_194_304
