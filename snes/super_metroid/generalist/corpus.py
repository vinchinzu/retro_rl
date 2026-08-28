"""Entry-state inventory over captured practice-repertoire gs=8 pins."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any

from super_metroid.generalist.goals import Goal, goal_from_session
from super_metroid.hop_glance import pose_class
from super_metroid.paths import (
    PRACTICE_CONTRACTOR_STATE_DIR,
    SHARED_PRACTICE_ROM,
    VANILLA_ROM_SHA1,
)
from super_metroid.practice_repertoire.catalog import (
    PRODUCT_CATEGORY,
    RepertoireSession,
    load_catalog,
    neighbors,
    route_sessions,
)


@dataclass(frozen=True)
class CorpusRow:
    """One practice-ROM training pin + Goal (next repertoire session)."""

    session_id: str
    state_path: str
    area: str
    room_id: int
    x: int
    y: int
    pose: int | None
    items: int | None
    goal_session_id: str
    goal_room_id: int
    goal_x: int
    goal_y: int
    prev_room_id: int | None = None
    goal_pose: int | None = None

    def goal(self) -> Goal:
        return Goal(
            session_id=self.goal_session_id,
            room_id=self.goal_room_id,
            x=self.goal_x,
            y=self.goal_y,
            pose=self.goal_pose,
            start_room_id=self.room_id,
        )


def captured_states(
    *,
    category: str | None = PRODUCT_CATEGORY,
    root: Path = PRACTICE_CONTRACTOR_STATE_DIR,
) -> list[Path]:
    """Existing canonical `.state` files for catalog sessions."""

    rows = load_catalog()["sessions"]
    out: list[Path] = []
    for rec in rows:
        if category and rec.get("category") != category:
            continue
        path = root / f"{rec['id']}.state"
        if path.is_file():
            out.append(path)
    return out


def _row_for(session: RepertoireSession, root: Path) -> CorpusRow | None:
    path = root / f"{session.id}.state"
    if not path.is_file():
        return None
    if session.room_id is None or session.x is None or session.y is None:
        return None
    prev_s, next_s = neighbors(session.id, category=session.category)
    if next_s is None or next_s.room_id is None:
        return None
    goal = goal_from_session(session.id, next_session=next_s)
    return CorpusRow(
        session_id=session.id,
        state_path=str(path),
        area=session.area,
        room_id=int(session.room_id),
        x=int(session.x),
        y=int(session.y),
        pose=session.pose,
        items=session.items,
        goal_session_id=goal.session_id,
        goal_room_id=goal.room_id,
        goal_x=goal.x,
        goal_y=goal.y,
        goal_pose=goal.pose,
        prev_room_id=prev_s.room_id if prev_s is not None else None,
    )


def load_rows(
    *,
    category: str = PRODUCT_CATEGORY,
    area: str | None = None,
    root: Path = PRACTICE_CONTRACTOR_STATE_DIR,
    dedupe: bool = True,
    exclude_ceres: bool = False,
    session_ids: Sequence[str] | None = None,
    same_room: bool | None = None,
) -> list[CorpusRow]:
    """kpdr25 (or other category) rows whose practice `.state` exists."""

    wanted = set(session_ids) if session_ids else None
    out: list[CorpusRow] = []
    seen: set[tuple[Any, ...]] = set()
    for session in route_sessions(category):
        if wanted is not None and session.id not in wanted:
            continue
        if area and session.area != area:
            continue
        if exclude_ceres and session.area == "crateria" and "ceres" in session.slug:
            continue
        row = _row_for(session, root)
        if row is None:
            continue
        if same_room is True and row.room_id != row.goal_room_id:
            continue
        if same_room is False and row.room_id == row.goal_room_id:
            continue
        if dedupe:
            key = (
                row.room_id,
                row.items,
                pose_class(int(row.pose or 0)),
                row.prev_room_id,
            )
            if key in seen:
                continue
            seen.add(key)
        out.append(row)
    return out


def practice_rom_sha1(path: Path = SHARED_PRACTICE_ROM) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def assert_practice_rom(path: Path = SHARED_PRACTICE_ROM) -> str:
    """Fail closed if the contractor env would load vanilla product ROM."""

    if not path.is_file():
        raise FileNotFoundError(f"practice ROM missing: {path}")
    digest = practice_rom_sha1(path)
    if digest == VANILLA_ROM_SHA1:
        raise ValueError("generalist must not load the vanilla product ROM")
    return digest


def corpus_status(
    *,
    category: str | None = PRODUCT_CATEGORY,
    root: Path = PRACTICE_CONTRACTOR_STATE_DIR,
    area: str | None = None,
) -> dict[str, Any]:
    catalog = load_catalog()
    sessions = [
        rec
        for rec in catalog["sessions"]
        if (not category or rec.get("category") == category)
        and (not area or rec.get("area") == area)
    ]
    present = captured_states(category=category, root=root)
    if area:
        present = [path for path in present if f"/{area}/" in path.as_posix()]
    rows = load_rows(
        category=category or PRODUCT_CATEGORY,
        area=area,
        root=root,
        dedupe=False,
        exclude_ceres=False,
    )
    crateria = load_rows(
        category=category or PRODUCT_CATEGORY,
        area="crateria",
        root=root,
        dedupe=False,
        exclude_ceres=True,
    )
    return {
        "category": category,
        "area": area,
        "sessions": len(sessions),
        "captured": len(present),
        "missing": len(sessions) - len(present),
        "trainable": len(rows),
        "crateria_trainable": len(crateria),
        "root": str(root),
        "practice_only": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--category", default=PRODUCT_CATEGORY)
    parser.add_argument("--area", default=None)
    args = parser.parse_args(argv)
    report = corpus_status(category=args.category, area=args.area)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CorpusRow",
    "assert_practice_rom",
    "captured_states",
    "corpus_status",
    "load_rows",
    "practice_rom_sha1",
]
