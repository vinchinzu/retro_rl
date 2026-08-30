"""Read-only task cards and assembly tables from one route manifest.

``generate_cards`` does not mutate the manifest or write bank.json.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from super_metroid.splice.errors import SchemaError
from super_metroid.splice.schema import (
    FORBIDDEN_HOT_FILES,
    INTERVENTION_PROFILES,
    NON_CLAIMS,
    CompletionReport,
    EntryContract,
    JoinPredicate,
    RouteEdge,
    RouteManifest,
    TaskCard,
    candidate_kind,
    rel_path,
)


def artifact_dir(task_id: str) -> str:
    safe = str(task_id).replace(":", "_").replace("/", "_").replace("\\", "_")
    return f"snes/super_metroid/recordings/splice/{safe}/"


def residual_path(task_id: str) -> str:
    safe = str(task_id).replace(":", "_").replace("/", "_").replace("\\", "_")
    return f"snes/super_metroid/docs/tasks/{safe}-residual.md"


def assembly_table(manifest: RouteManifest) -> tuple[dict[str, Any], ...]:
    """Planner assembly rows in manifest edge order (not a hand-copied hop list)."""
    return tuple(
        {
            "order": i,
            "task_id": edge.task_id,
            "hop_key": edge.hop_key,
            "room": f"0x{edge.room_id:04X}",
            "predecessor_task_id": edge.predecessor_task_id,
            "successor_task_id": edge.successor_task_id,
            "selected": edge.selected_map(),
            "allowed_kinds": list(edge.allowed_kinds),
            "invalid_room": edge.invalid_room,
        }
        for i, edge in enumerate(manifest.edges)
    )


def _as_manifest(manifest: RouteManifest | Mapping[str, Any]) -> RouteManifest:
    if isinstance(manifest, RouteManifest):
        return manifest
    payload = json.loads(json.dumps(dict(manifest)))
    return RouteManifest.from_dict(payload)


def _adapter(edge: RouteEdge, profile: str) -> tuple[str, str]:
    selected = edge.selected_map().get(profile, "")
    if selected:
        return candidate_kind(selected), selected
    return edge.allowed_kinds[0], ""


def _card_for(
    edge: RouteEdge,
    *,
    successor: RouteEdge | None,
    others: tuple[str, ...],
    profile: str,
    revision: int,
) -> TaskCard:
    kind, selected_id = _adapter(edge, profile)
    art = artifact_dir(edge.task_id)
    leftover = f"{art}leftover.state"
    owned = [edge.owner_package, art]
    if edge.tape_path:
        owned.append(edge.tape_path)
    forbidden = list(FORBIDDEN_HOT_FILES)
    forbidden.extend(residual_path(tid) for tid in others)
    next_entry: EntryContract | None = None if successor is None else successor.entry
    commands = (
        f"checkbox: make {edge.task_id} sync_green or leave exact residual",
        f"profile={profile} kind={kind} selected={selected_id or 'unselected'}",
        f"timeout_frames={edge.max_frames}",
        f"emit candidate under {art}",
    )
    paths = tuple(rel_path(item) or item for item in owned if item)
    return TaskCard(
        task_id=edge.task_id,
        hop_key=edge.hop_key,
        revision=int(revision),
        checkbox="sync_green",
        exact_residual=residual_path(edge.task_id),
        entry_state_path=edge.entry.state_path,
        entry_state_digest=edge.entry.state_digest,
        tape_digest=edge.tape_digest,
        segment=edge.segment,
        hop_index=edge.hop_index,
        frame_start=edge.frame_start,
        frame_end=edge.frame_end,
        source_notes=edge.source_notes,
        entry_fingerprint=edge.entry.fingerprint,
        join=JoinPredicate(leave=edge.successor_leave, next_entry=next_entry),
        adapter_kind=kind,
        intervention_profile=profile,
        timeout_frames=edge.max_frames,
        commands=commands,
        owned_paths=paths,
        candidate_artifact_dir=art,
        forbidden_files=tuple(dict.fromkeys(forbidden)),
        non_claims=NON_CLAIMS,
        completion=CompletionReport(
            leftover_state_path=leftover,
            screenshot_path=f"{art}red.png",
            trace_path=f"{art}trace.json",
            next_boot_on_red=leftover,
        ),
        next_task_id=edge.successor_task_id,
        invalid_room=edge.invalid_room,
    )


def generate_cards(
    manifest: RouteManifest | Mapping[str, Any],
    *,
    profile: str = "scaffold",
    revision: int = 1,
) -> tuple[TaskCard, ...]:
    """Immutable cards, one per route edge, in manifest order."""
    if not str(profile).strip() or profile not in INTERVENTION_PROFILES:
        raise SchemaError(
            f"unknown intervention profile {profile!r}",
            code="schema.profile",
            details={"profile": profile},
        )
    if int(revision) < 1:
        raise SchemaError("revision must be >= 1", code="schema.revision")
    route = _as_manifest(manifest)
    ids = tuple(e.task_id for e in route.edges)
    cards: list[TaskCard] = []
    for i, edge in enumerate(route.edges):
        succ = route.edges[i + 1] if i + 1 < len(route.edges) else None
        others = tuple(tid for tid in ids if tid != edge.task_id)
        cards.append(
            _card_for(edge, successor=succ, others=others, profile=profile, revision=revision)
        )
    return tuple(cards)


def format_card(card: TaskCard) -> str:
    leave = card.join.leave
    nxt = card.join.next_entry
    lines = [
        f"task_id: {card.task_id}",
        f"hop_key: {card.hop_key}",
        f"checkbox: make {card.task_id} {card.checkbox} or leave exact residual {card.exact_residual}",
        f"entry_path: {card.entry_state_path}",
        f"entry_digest: {card.entry_state_digest}",
        f"tape_digest: {card.tape_digest}",
        (
            f"join: {leave.hop} room=0x{leave.room:04X} "
            f"x={list(leave.x)} y={list(leave.y)} digest={leave.digest}"
        ),
        f"next_task: {card.next_task_id}",
        f"next_entry_digest: {None if nxt is None else nxt.state_digest}",
        f"adapter_kind: {card.adapter_kind}",
        f"profile: {card.intervention_profile}",
        f"timeout_frames: {card.timeout_frames}",
        f"owned: {', '.join(card.owned_paths)}",
        f"candidate_dir: {card.candidate_artifact_dir}",
        f"invalid_room: {card.invalid_room}",
        "forbidden:",
        *[f"  - {p}" for p in card.forbidden_files],
        "non_claims:",
        *[f"  - {c}" for c in card.non_claims],
        f"replay_green: {card.completion.replay_green}",
        f"sync_green: {card.completion.sync_green}",
        f"next_boot_on_red: {card.completion.next_boot_on_red}",
    ]
    return "\n".join(lines)


def format_cards(cards: tuple[TaskCard, ...] | list[TaskCard]) -> str:
    if not cards:
        return "no hops inventoried"
    return "\n---\n".join(format_card(c) for c in cards)
