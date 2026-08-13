"""Policy board, graduation, gap report, and unified work cards."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from super_metroid.paths import SHARED_PRACTICE_ROM, SHARED_ROM
from super_metroid.practice_repertoire.catalog import (
    GRADES,
    PRODUCT_CATEGORY,
    REACTIVE_POLICY_DIR,
    RepertoireSession,
    get_session,
    neighbors,
    route_sessions,
)
from super_metroid.practice_repertoire.spine import (
    PRODUCT_SESSION_MAP,
    hop_key_for_session,
    recover_session,
    route_edge,
)


@dataclass
class PolicyBoardCard:
    """Workspace for room-by-room policy tune → graduate."""

    session_id: str
    room_id: int | None
    hop_key: str | None
    grade: str
    entry_state: str | None
    plan_path: str
    policy_glob: str
    demo_stem: str
    start_preset: str | None = None
    next_session_id: str | None = None
    prev_session_id: str | None = None
    existing_policies: list[str] = field(default_factory=list)
    tune_command: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _list_room_policies(room_id: int | None) -> list[Path]:
    if room_id is None or not REACTIVE_POLICY_DIR.is_dir():
        return []
    needle = f"room_{room_id:04x}_"
    return sorted(
        p
        for p in REACTIVE_POLICY_DIR.glob("*.json")
        if needle in p.name.lower() or f"0x{room_id:04x}" in p.name.lower()
    )


def _policy_statuses(paths: list[Path]) -> list[str]:
    statuses: list[str] = []
    for p in paths:
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        st = raw.get("status")
        if st:
            statuses.append(str(st))
    return statuses


def graduation_status(session: RepertoireSession) -> str:
    """Infer how far this session has graduated toward product use."""
    living = session.living_state_path()
    if living is not None and living.is_file() and session.id in PRODUCT_SESSION_MAP:
        return "product_spine"
    policies = _list_room_policies(session.room_id)
    statuses = _policy_statuses(policies)
    if any(s == "verified_live_anchor" for s in statuses):
        return "verified_live_anchor"
    if policies or session.policy_plan_path().is_file():
        return "candidate"
    demo_json = Path(str(session.canonical_demo_stem) + ".json")
    if (
        session.canonical_state_path.is_file()
        or demo_json.is_file()
        or (living is not None and living.is_file())
    ):
        return "draft"
    return "none"


def policy_board_card(session_id: str) -> PolicyBoardCard:
    """Full workspace card for tuning + graduating a room policy."""
    s = get_session(session_id)
    prev_s, next_s = neighbors(session_id)
    hop = hop_key_for_session(s, next_session=next_s, prev_session=prev_s)
    entry = s.resolve_state_path()
    policies = _list_room_policies(s.room_id)
    start = None
    m = s.product_map()
    if m:
        start = m.get("start_preset")
    room_arg = s.room_hex or "0x0000"
    from_arg = f"0x{prev_s.room_id:04X}" if prev_s and prev_s.room_id else "start"
    exit_arg = f"0x{next_s.room_id:04X}" if next_s and next_s.room_id else "leave"
    body_hint = f"tasks/*_hops/*{s.slug}*.json"
    tune = (
        "uv run python snes/super_metroid/scripts/tools/optimize_room_policy.py "
        f"--body {body_hint} --room {room_arg} --from-room {from_arg} "
        f"--exit-room {exit_arg} --variant base --takeover-sweep"
    )
    return PolicyBoardCard(
        session_id=s.id,
        room_id=s.room_id,
        hop_key=hop,
        grade=graduation_status(s),
        entry_state=str(entry) if entry else None,
        plan_path=str(s.policy_plan_path()),
        policy_glob=str(REACTIVE_POLICY_DIR / s.policy_json_glob()),
        demo_stem=str(s.canonical_demo_stem),
        start_preset=start,
        next_session_id=next_s.id if next_s else None,
        prev_session_id=prev_s.id if prev_s else None,
        existing_policies=[str(p) for p in policies],
        tune_command=tune,
    )


def policy_board(category: str = PRODUCT_CATEGORY) -> list[PolicyBoardCard]:
    return [policy_board_card(s.id) for s in route_sessions(category)]


def iter_product_sessions() -> Iterator[RepertoireSession]:
    yield from route_sessions(PRODUCT_CATEGORY)


def mapped_sessions() -> list[tuple[RepertoireSession, dict[str, str]]]:
    rows: list[tuple[RepertoireSession, dict[str, str]]] = []
    for sid, meta in PRODUCT_SESSION_MAP.items():
        try:
            rows.append((get_session(sid), meta))
        except KeyError:
            continue
    return rows


def gap_report(category: str = PRODUCT_CATEGORY) -> dict[str, Any]:
    """Coverage for pins, route edges, policies, and graduation."""
    all_s = route_sessions(category)
    mapped_ids: set[str] = set()
    for sid in PRODUCT_SESSION_MAP:
        try:
            s = get_session(sid)
        except KeyError:
            continue
        if s.category == category:
            mapped_ids.add(sid)

    by_grade: dict[str, int] = {g: 0 for g in GRADES}
    missing_state: list[str] = []
    present_state: list[str] = []
    edge_count = 0
    for s in all_s:
        g = graduation_status(s)
        by_grade[g] = by_grade.get(g, 0) + 1
        living = s.living_state_path()
        if (living and living.is_file()) or s.canonical_state_path.is_file():
            present_state.append(s.id)
        elif s.id in mapped_ids:
            missing_state.append(s.id)
        if route_edge(s.id) is not None:
            edge_count += 1

    return {
        "category": category,
        "grades": GRADES,
        "session_count": len(all_s),
        "mapped_count": len(mapped_ids),
        "unmapped_count": len(all_s) - len(mapped_ids),
        "route_edges": edge_count,
        "by_grade": by_grade,
        "mapped_missing_binary": missing_state,
        "mapped_with_binary": present_state,
        "practice_rom_ready": SHARED_PRACTICE_ROM.is_file(),
        "vanilla_rom_ready": SHARED_ROM.is_file(),
        "reactive_policy_dir": str(REACTIVE_POLICY_DIR),
    }


def session_work_card(session_id: str) -> dict[str, Any]:
    """Unified card: human + policy + route-edge + recovery view of one session."""
    s = get_session(session_id)
    prev_s, next_s = neighbors(session_id)
    edge = route_edge(session_id)
    card = policy_board_card(session_id)
    m = s.product_map()
    return {
        "session": s.fingerprint(),
        "grade": graduation_status(s),
        "product_map": m,
        "living_state": str(s.living_state_path()) if s.living_state_path() else None,
        "canonical_state": str(s.canonical_state_path),
        "canonical_demo": str(s.canonical_demo_stem),
        "resolve_state": str(s.resolve_state_path()) if s.resolve_state_path() else None,
        "prev_session": prev_s.id if prev_s else None,
        "next_session": next_s.id if next_s else None,
        "hop_key": hop_key_for_session(s, next_session=next_s, prev_session=prev_s),
        "route_edge": edge.to_dict() if edge else None,
        "policy_board": card.to_dict(),
        "recovery": recover_session(
            s.room_id or 0, s.items, category=s.category
        ).to_dict()
        if s.room_id
        else None,
    }
