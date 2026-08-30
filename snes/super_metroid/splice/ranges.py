"""Attic→Gravity Scaffold range over ``tips.play_hops``.

Planning/verification only. Loads s23 Attic+Bowling tape candidates and
placeholder edges for West Ocean / Pancakes / Homing Geemer / Gravity.
Never boots unless assemble is given a session; never loads a room state
mid-run; never claims Survival / STATUS / living Tip. Main Shaft stays serial.
Scaffold HP clamp is recorded for the Attic gray door only — not enabled
globally and not applied here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from super_metroid.assist import attic_ordinary_enemy_allowlist
from super_metroid.hop_id import make_hop_key
from super_metroid.splice.assemble import (
    Assembly,
    HopFactory,
    PlayHops,
    assemble as assemble_route,
)
from super_metroid.splice.errors import AssembleError
from super_metroid.splice.manifest import dest_leave_spec
from super_metroid.splice.schema import (
    NON_CLAIMS,
    CandidateArtifact,
    EntryContract,
    EntryFingerprint,
    LeaveSpecRef,
    RouteEdge,
    RouteManifest,
)
from super_metroid.splice.select import CandidateOffer, Selection, select
from super_metroid.splice.tapes import (
    ATTIC_ROOM,
    ATTIC_TASK_ID,
    BOWLING_ROOM,
    BOWLING_TASK_ID,
    GRAVITY_ROOM,
    MAIN_SHAFT_ROOM,
    OWNER_PACKAGE,
    SEGMENT,
    WEST_OCEAN_ROOM,
    TapeCandidate,
    load_s23_tape_candidates,
)

ROUTE_ID = "attic_to_gravity"
PROFILE = "scaffold"
PLACEHOLDER_KIND_ID = "controller:placeholder"
WEST_OCEAN_TASK_ID = "west_ocean"
PANCAKES_TASK_ID = "pancakes"
HOMING_GEEMER_TASK_ID = "homing_geemer"
GRAVITY_TASK_ID = "gravity"
PANCAKES_ROOM = 0x9461
HOMING_GEEMER_ROOM = 0x968F
GRAVITY_GOAL = "gravity_collect"
PLACEHOLDER_MAX_FRAMES = 10_000
HP_CLAMP_TASKS = (ATTIC_TASK_ID,)
TAPE_TASKS = (ATTIC_TASK_ID, BOWLING_TASK_ID)
PLACEHOLDER_TASKS = (
    WEST_OCEAN_TASK_ID,
    PANCAKES_TASK_ID,
    HOMING_GEEMER_TASK_ID,
    GRAVITY_TASK_ID,
)
TASK_ORDER = (
    ATTIC_TASK_ID,
    WEST_OCEAN_TASK_ID,
    PANCAKES_TASK_ID,
    HOMING_GEEMER_TASK_ID,
    BOWLING_TASK_ID,
    GRAVITY_TASK_ID,
)
RANGE_NON_CLAIMS: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *NON_CLAIMS,
            "route-ready before Main Shaft reaches Attic",
            "living Tip / DEFAULT_CONTINUOUS_TIP promotion",
            "Survival/Finish from Scaffold range evidence",
            "global Scaffold HP clamp",
        )
    )
)
_SOURCE_NOTES = (
    "Scaffold range; development-only — not Survival/STATUS/living Tip",
    "Main Shaft / rr-kw8t remains serial",
    "May replay-green from archived s23 anchors; not route-ready",
)

AssembleFn = Callable[..., Assembly]


@dataclass(frozen=True)
class RangePlan:
    """Attic→Gravity Scaffold selection. Not a promotion and not a bank write."""

    route_id: str
    profile: str
    manifest: RouteManifest
    selection: Selection
    offers: tuple[CandidateOffer, ...]
    tape_tasks: tuple[str, ...]
    placeholder_tasks: tuple[str, ...]
    hp_clamp_tasks: tuple[str, ...]
    hp_clamp_allowlist: tuple[Any, ...]
    hp_clamp_global: bool = False
    non_claims: tuple[str, ...] = RANGE_NON_CLAIMS
    source_notes: tuple[str, ...] = _SOURCE_NOTES

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(edge.task_id for edge in self.manifest.edges)

    def hp_clamp_allowed(self, task_id: str) -> bool:
        return (not self.hp_clamp_global) and task_id in self.hp_clamp_tasks

    def to_dict(self) -> dict[str, Any]:
        allowlist = []
        for entry in self.hp_clamp_allowlist:
            room_id = int(getattr(entry, "room_id"))
            allowlist.append(
                {
                    "room_id": room_id,
                    "room": f"0x{room_id:04X}",
                    "enemy_id": int(getattr(entry, "enemy_id")),
                }
            )
        return {
            "route_id": self.route_id,
            "profile": self.profile,
            "route_ready": False,
            "living_tip": False,
            "survival_claim": False,
            "hp_clamp_global": self.hp_clamp_global,
            "hp_clamp_tasks": list(self.hp_clamp_tasks),
            "hp_clamp_allowlist": allowlist,
            "tape_tasks": list(self.tape_tasks),
            "placeholder_tasks": list(self.placeholder_tasks),
            "task_ids": list(self.task_ids),
            "selected": dict(self.selection.selected),
            "non_claims": list(self.non_claims),
            "source_notes": list(self.source_notes),
            "manifest": self.manifest.to_dict(),
            "selection": self.selection.to_dict(),
        }


def format_range(plan: RangePlan) -> str:
    tasks = ", ".join(plan.task_ids)
    placeholders = ", ".join(plan.placeholder_tasks) or "none"
    clamp = ", ".join(plan.hp_clamp_tasks) or "none"
    lines = [
        f"route {plan.route_id} profile={plan.profile} (development-only)",
        f"tasks: {tasks}",
        f"placeholders: {placeholders}",
        f"hp_clamp: {clamp} (not global)",
        "non-claims: " + "; ".join(plan.non_claims),
    ]
    return "\n".join(lines)


def _require_scaffold(profile: str) -> str:
    prof = str(profile).strip()
    if prof != PROFILE:
        raise AssembleError(
            "Attic→Gravity range is scaffold-only (development); not Survival/Tip",
            code="assemble.profile",
            details={"profile": profile, "route_id": ROUTE_ID},
        )
    return PROFILE


def _attic_clamp_allowlist() -> tuple[Any, ...]:
    # Attic gray door only. Drop any non-Attic rows rather than enabling globally.
    return tuple(
        entry
        for entry in attic_ordinary_enemy_allowlist()
        if int(entry.room_id) == int(ATTIC_ROOM)
    )


def _retarget(edge: RouteEdge, *, order: int, extra_notes: Sequence[str] = ()) -> dict[str, Any]:
    payload = edge.to_dict()
    payload["integration_order"] = int(order)
    payload.pop("predecessor_task_id", None)
    payload.pop("successor_task_id", None)
    notes = [str(n) for n in (payload.get("source_notes") or ())]
    for note in extra_notes:
        if note not in notes:
            notes.append(str(note))
    payload["source_notes"] = notes
    return payload


def _placeholder_edge(
    *,
    task_id: str,
    room_id: int,
    predecessor_room_id: int | None,
    next_room_id: int | None,
    items: int | None,
    order: int,
    notes: Sequence[str],
    goal: str | None = None,
) -> dict[str, Any]:
    hop_key = make_hop_key(
        int(room_id),
        from_room_id=predecessor_room_id,
        to_room_id=next_room_id,
        items=items,
        goal=goal,
    )
    leave_room = int(next_room_id) if next_room_id is not None else int(room_id)
    leave = dest_leave_spec(hop=hop_key, room_id=leave_room)
    max_frames = PLACEHOLDER_MAX_FRAMES
    notes_t = tuple(dict.fromkeys((*_SOURCE_NOTES, *notes)))
    entry = EntryContract(
        fingerprint=EntryFingerprint(
            room_id=int(room_id),
            items=items,
            prior_room_id=predecessor_room_id,
        )
    )
    return {
        "task_id": task_id,
        "hop_key": hop_key,
        "room_id": int(room_id),
        "predecessor_room_id": predecessor_room_id,
        "next_room_id": next_room_id,
        "goal": goal,
        "required_items": items,
        "entry": entry.to_dict(),
        "successor_leave": LeaveSpecRef.from_leave_spec(leave).to_dict(),
        "allowed_kinds": ["controller", "tape"],
        "selected": {PROFILE: PLACEHOLDER_KIND_ID},
        "owner_package": OWNER_PACKAGE,
        "integration_order": int(order),
        "max_frames": max_frames,
        "max_no_progress": max(1, min(600, max_frames)),
        "segment": SEGMENT,
        "hop_index": int(order),
        "source_notes": list(notes_t),
    }


def _tape_offer(candidate: TapeCandidate) -> CandidateOffer:
    artifact = candidate.artifact
    if not isinstance(artifact, CandidateArtifact):
        artifact = CandidateArtifact.from_dict(artifact.to_dict())
    return CandidateOffer(artifact=artifact, profile=PROFILE)


def attic_to_gravity_range(
    segment_dir: Path | str | None = None,
    *,
    profile: str = PROFILE,
) -> RangePlan:
    """RouteManifest + scaffold selection over s23 Attic/Bowling plus placeholders."""
    _require_scaffold(profile)
    attic, bowling = load_s23_tape_candidates(segment_dir)
    if attic.room_id == MAIN_SHAFT_ROOM or bowling.room_id == MAIN_SHAFT_ROOM:
        raise AssembleError(
            "Attic→Gravity range excludes Main Shaft hops",
            code="assemble.selected",
            details={"route_id": ROUTE_ID},
        )
    items = attic.edge.required_items
    edges = [
        _retarget(
            attic.edge,
            order=0,
            extra_notes=(
                "Attic 0xCA52 → West Ocean 0x93FE (kill-all gray door)",
                "Scaffold HP clamp allowed for Attic gray door only",
            ),
        ),
        _placeholder_edge(
            task_id=WEST_OCEAN_TASK_ID,
            room_id=WEST_OCEAN_ROOM,
            predecessor_room_id=ATTIC_ROOM,
            next_room_id=PANCAKES_ROOM,
            items=items,
            order=1,
            notes=("West Ocean → Pancakes and Wavers 0x9461; placeholder (no tape adapter)",),
        ),
        _placeholder_edge(
            task_id=PANCAKES_TASK_ID,
            room_id=PANCAKES_ROOM,
            predecessor_room_id=WEST_OCEAN_ROOM,
            next_room_id=HOMING_GEEMER_ROOM,
            items=items,
            order=2,
            notes=("Pancakes and Wavers → Homing Geemer 0x968F; placeholder (no tape adapter)",),
        ),
        _placeholder_edge(
            task_id=HOMING_GEEMER_TASK_ID,
            room_id=HOMING_GEEMER_ROOM,
            predecessor_room_id=PANCAKES_ROOM,
            next_room_id=BOWLING_ROOM,
            items=items,
            order=3,
            notes=("Homing Geemer → Bowling 0xC98E; placeholder (no tape adapter)",),
        ),
        _retarget(
            bowling.edge,
            order=4,
            extra_notes=(
                "Bowling 0xC98E internal split; one external natural-entry→Gravity contract",
            ),
        ),
        _placeholder_edge(
            task_id=GRAVITY_TASK_ID,
            room_id=GRAVITY_ROOM,
            predecessor_room_id=BOWLING_ROOM,
            next_room_id=None,
            items=items,
            order=5,
            notes=("Gravity entry, natural PLM collect, settled post-collect leave; placeholder",),
            goal=GRAVITY_GOAL,
        ),
    ]
    manifest = RouteManifest.from_dict(
        {"route_id": ROUTE_ID, "variant": "kpdr", "edges": edges}
    )
    rooms = tuple(edge.room_id for edge in manifest.edges)
    if MAIN_SHAFT_ROOM in rooms:
        raise AssembleError(
            "Attic→Gravity range excludes Main Shaft hops",
            code="assemble.selected",
            details={"route_id": ROUTE_ID, "rooms": [f"0x{r:04X}" for r in rooms]},
        )
    offers = (_tape_offer(attic), _tape_offer(bowling))
    selection = select(manifest, offers, profile=PROFILE)
    return RangePlan(
        route_id=ROUTE_ID,
        profile=PROFILE,
        manifest=manifest,
        selection=selection,
        offers=offers,
        tape_tasks=TAPE_TASKS,
        placeholder_tasks=PLACEHOLDER_TASKS,
        hp_clamp_tasks=HP_CLAMP_TASKS,
        hp_clamp_allowlist=_attic_clamp_allowlist(),
        hp_clamp_global=False,
        non_claims=RANGE_NON_CLAIMS,
        source_notes=_SOURCE_NOTES,
    )


gravity_range = attic_to_gravity_range


def assemble_attic_to_gravity(
    segment_dir: Path | str | None = None,
    *,
    profile: str = PROFILE,
    assemble: AssembleFn | None = None,
    play_hops: PlayHops | None = None,
    session: Any | None = None,
    session_factory: Callable[[], Any] | None = None,
    hop_factory: HopFactory | None = None,
    plan: RangePlan | None = None,
) -> Assembly:
    """Drive the Attic→Gravity Scaffold range through ``splice.assemble``."""
    _require_scaffold(profile)
    rng = plan if plan is not None else attic_to_gravity_range(segment_dir, profile=profile)
    if rng.profile != PROFILE or rng.hp_clamp_global:
        raise AssembleError(
            "Attic→Gravity range is scaffold-only and must not enable HP clamp globally",
            code="assemble.profile",
            details={
                "profile": rng.profile,
                "hp_clamp_global": rng.hp_clamp_global,
                "route_id": rng.route_id,
            },
        )
    runner = assemble if assemble is not None else assemble_route
    return runner(
        rng.route_id,
        rng.selection,
        manifest=rng.manifest,
        profile=PROFILE,
        candidates=rng.offers,
        play_hops=play_hops,
        session=session,
        session_factory=session_factory,
        hop_factory=hop_factory,
    )
