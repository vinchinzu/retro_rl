"""Generalist corpus status (no emulator)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import super_metroid.generalist.corpus as corpus_module
from super_metroid.generalist.corpus import corpus_status, load_rows
from super_metroid.generalist.goals import Goal, leave_spec_for
from super_metroid.paths import VANILLA_ROM_SHA1
from super_metroid.practice_repertoire.catalog import PRODUCT_CATEGORY, RepertoireSession


def test_corpus_status_shape() -> None:
    report = corpus_status()
    assert report["category"] == PRODUCT_CATEGORY
    assert report["sessions"] >= 100
    assert report["captured"] >= 0
    assert report["missing"] == report["sessions"] - report["captured"]
    assert report["practice_only"] is True
    assert report["trainable"] >= 0
    assert report["crateria_trainable"] >= 0


def test_load_rows_crateria_excludes_ceres_and_has_goal() -> None:
    rows = load_rows(area="crateria", exclude_ceres=True, dedupe=False)
    if not rows:
        return
    assert all("ceres" not in row.session_id for row in rows)
    ship = next((row for row in rows if row.session_id.endswith("/ship")), None)
    if ship is None:
        return
    assert ship.goal_session_id.endswith("/parlor")
    assert ship.room_id == 0x91F8
    goal = ship.goal()
    assert goal.resolved is True
    same = load_rows(area="crateria", exclude_ceres=True, dedupe=True, same_room=True)
    if same:
        assert all(row.room_id == row.goal_room_id for row in same)
        assert any(row.session_id.endswith("/ship") for row in same)


def test_vanilla_sha1_constant_not_used_as_practice() -> None:
    assert VANILLA_ROM_SHA1 == "da957f0d63d14cb441d215462904c4fa8519c613"


def test_load_rows_preserves_next_session_morph_pose_for_join(
    tmp_path: Path, monkeypatch: Any
) -> None:
    current = RepertoireSession(
        id="pre-morph",
        category=PRODUCT_CATEGORY,
        area="brinstar",
        slug="pre-morph",
        name="Pre Morph",
        room_id=0x9E9F,
        x=128,
        y=160,
        pose=1,
        items=0,
    )
    next_session = RepertoireSession(
        id="morph-door",
        category=PRODUCT_CATEGORY,
        area="brinstar",
        slug="morph-door",
        name="Morph Door",
        room_id=0x9E9F,
        x=64,
        y=176,
        pose=29,
    )
    (tmp_path / "pre-morph.state").touch()
    monkeypatch.setattr(corpus_module, "route_sessions", lambda _category: [current])
    monkeypatch.setattr(
        corpus_module,
        "neighbors",
        lambda *_args, **_kwargs: (None, next_session),
    )
    monkeypatch.setattr(
        corpus_module,
        "goal_from_session",
        lambda *_args, **_kwargs: Goal(
            session_id=next_session.id,
            room_id=int(next_session.room_id or 0),
            x=int(next_session.x or 0),
            y=int(next_session.y or 0),
            pose=next_session.pose,
            start_room_id=current.room_id,
        ),
    )

    rows = load_rows(root=tmp_path, dedupe=False)

    assert len(rows) == 1
    row = rows[0]
    goal = row.goal()
    assert row.goal_pose == 29
    assert goal.pose == 29
    assert leave_spec_for(goal).pose_class == "morph"
