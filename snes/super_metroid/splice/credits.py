"""Scaffold credits planning and live finish-splice benchmarks over ``tips.play_hops``.

The Scaffold path stitches the ten item-seam lanes plus the
Attic→Gravity range into one RouteManifest and drives ``splice.assemble``
under profile=scaffold. Never boots unless a session factory is injected;
never loads a room state mid-run; never writes bank.json; never claims
Survival credits, STATUS, or Finish. Development-only splice milestone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from super_metroid.hop_id import make_hop_key
from super_metroid.splice.assemble import (
    Assembly,
    HopFactory,
    PlayHops,
    assemble as assemble_route,
)
from super_metroid.splice.cards import assembly_table
from super_metroid.splice.errors import AssembleError
from super_metroid.splice.lanes import ITEM_SEAM_LANES, Lane, inventory_from_manifest
from super_metroid.splice.manifest import dest_leave_spec
from super_metroid.splice.ranges import (
    PLACEHOLDER_KIND_ID,
    PLACEHOLDER_MAX_FRAMES,
    RangePlan,
    attic_to_gravity_range,
)
from super_metroid.splice.schema import (
    NON_CLAIMS,
    CandidateArtifact,
    EntryContract,
    EntryFingerprint,
    LeaveSpecRef,
    MemoryWrite,
    RouteEdge,
    RouteManifest,
)
from super_metroid.splice.select import CandidateOffer, Selection, select
from super_metroid.splice.tapes import GRAVITY_ROOM, MAIN_SHAFT_ROOM, OWNER_PACKAGE

ROUTE_ID = "scaffold_credits"
PROFILE = "scaffold"
CREDITS_GOAL = "credits"
CREDITS_TASK_ID = "ridley_credits"
LANDING_ROOM = 0x91F8
# Start rooms of lanes 2–10 (Gravity→Grapple … Ridley→credits).
TAIL_ROOMS: tuple[int, ...] = (
    0xAC2B,  # Grapple
    0xCFC9,  # Main Street
    0xD9AA,  # Space Jump
    0xD2AA,  # Plasma
    0xB283,  # Golden Torizo
    0xB6C1,  # Screw Attack
    0xB62B,  # Metal Pirates
    0xB32E,  # Ridley
    LANDING_ROOM,  # ship / credits
)
_SOURCE_NOTES = (
    "Scaffold credits chain; development-only — not Survival/STATUS/Finish",
    "Zero mid-run state load; power-on session semantics",
    "Main Shaft / rr-kw8t remains serial",
    "Not a live credits clear and not a Survival credits claim",
)
CREDITS_NON_CLAIMS: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *NON_CLAIMS,
            "Survival credits from Scaffold chain evidence",
            "Finish / STATUS from Scaffold credits",
            "living Tip / DEFAULT_CONTINUOUS_TIP promotion",
            "route-ready before Main Shaft reaches Attic",
        )
    )
)

AssembleFn = Callable[..., Assembly]


@dataclass(frozen=True)
class LedgerRow:
    """One hop's recorded interventions. Empty writes still belong in the ledger."""

    task_id: str
    candidate_id: str
    lane_id: str | None
    writes: tuple[MemoryWrite, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "candidate_id": self.candidate_id,
            "lane_id": self.lane_id,
            "writes": [w.to_dict() for w in self.writes],
        }


@dataclass(frozen=True)
class CreditsPlan:
    """Ten-lane Scaffold credits selection. Not a promotion and not a bank write."""

    route_id: str
    profile: str
    manifest: RouteManifest
    selection: Selection
    offers: tuple[CandidateOffer, ...]
    lanes: tuple[Lane, ...]
    range_plan: RangePlan
    tape_tasks: tuple[str, ...]
    placeholder_tasks: tuple[str, ...]
    hp_clamp_tasks: tuple[str, ...]
    hp_clamp_allowlist: tuple[Any, ...]
    hp_clamp_global: bool = False
    non_claims: tuple[str, ...] = CREDITS_NON_CLAIMS
    source_notes: tuple[str, ...] = _SOURCE_NOTES

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(edge.task_id for edge in self.manifest.edges)

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "profile": self.profile,
            "development_only": True,
            "route_ready": False,
            "living_tip": False,
            "survival_claim": False,
            "finish_claim": False,
            "status_claim": False,
            "zero_state_load": True,
            "hp_clamp_global": self.hp_clamp_global,
            "hp_clamp_tasks": list(self.hp_clamp_tasks),
            "tape_tasks": list(self.tape_tasks),
            "placeholder_tasks": list(self.placeholder_tasks),
            "task_ids": list(self.task_ids),
            "lane_ids": [lane.lane_id for lane in self.lanes],
            "lanes": [lane.to_dict() for lane in self.lanes],
            "selected": dict(self.selection.selected),
            "non_claims": list(self.non_claims),
            "source_notes": list(self.source_notes),
            "manifest": self.manifest.to_dict(),
            "selection": self.selection.to_dict(),
        }


@dataclass(frozen=True)
class CreditsReport:
    """Assembly report with intervention ledger and room split table.

    Development-only. Never Survival credits, STATUS, or Finish.
    """

    assembly: Assembly
    plan: CreditsPlan
    intervention_ledger: tuple[LedgerRow, ...]
    room_splits: tuple[dict[str, Any], ...]
    non_claims: tuple[str, ...] = CREDITS_NON_CLAIMS
    source_notes: tuple[str, ...] = _SOURCE_NOTES

    @property
    def route_id(self) -> str:
        return self.assembly.route_id

    @property
    def profile(self) -> str:
        return self.assembly.profile

    @property
    def hop_ids(self) -> tuple[str, ...]:
        return self.assembly.hop_ids

    @property
    def lanes(self) -> tuple[Lane, ...]:
        return self.plan.lanes

    @property
    def session(self) -> Any:
        return self.assembly.session

    def to_dict(self) -> dict[str, Any]:
        payload = self.plan.to_dict()
        payload.update(self.assembly.to_dict())
        payload["development_only"] = True
        payload["survival_claim"] = False
        payload["finish_claim"] = False
        payload["status_claim"] = False
        payload["living_tip"] = False
        payload["route_ready"] = False
        payload["zero_state_load"] = True
        payload["intervention_ledger"] = [row.to_dict() for row in self.intervention_ledger]
        payload["room_splits"] = [dict(row) for row in self.room_splits]
        payload["non_claims"] = list(self.non_claims)
        payload["source_notes"] = list(self.source_notes)
        return payload


def format_credits(plan: CreditsPlan) -> str:
    tasks = ", ".join(plan.task_ids)
    lanes = ", ".join(lane.lane_id for lane in plan.lanes)
    lines = [
        f"route {plan.route_id} profile={plan.profile} (development-only)",
        f"lanes ({len(plan.lanes)}): {lanes}",
        f"tasks: {tasks}",
        f"goal: {CREDITS_GOAL}",
        "zero mid-run state load; not Survival/STATUS/Finish",
        "non-claims: " + "; ".join(plan.non_claims),
    ]
    return "\n".join(lines)


def _require_scaffold(profile: str) -> str:
    prof = str(profile).strip()
    if prof != PROFILE:
        raise AssembleError(
            "Scaffold credits chain is development-only; not Survival/STATUS/Finish",
            code="assemble.profile",
            details={"profile": profile, "route_id": ROUTE_ID},
        )
    return PROFILE


def _retarget_next(edge: RouteEdge, *, next_room_id: int, order: int) -> dict[str, Any]:
    hop_key = make_hop_key(
        int(edge.room_id),
        from_room_id=edge.predecessor_room_id,
        to_room_id=int(next_room_id),
        items=edge.required_items,
        goal=edge.goal,
    )
    payload = edge.to_dict()
    payload.pop("predecessor_task_id", None)
    payload.pop("successor_task_id", None)
    payload["hop_key"] = hop_key
    payload["next_room_id"] = int(next_room_id)
    payload["integration_order"] = int(order)
    payload["hop_index"] = int(order)
    payload["successor_leave"] = LeaveSpecRef.from_leave_spec(
        dest_leave_spec(hop=hop_key, room_id=int(next_room_id))
    ).to_dict()
    notes = [str(n) for n in (payload.get("source_notes") or ())]
    for note in _SOURCE_NOTES:
        if note not in notes:
            notes.append(note)
    payload["source_notes"] = notes
    return payload


def _edge_payload(edge: RouteEdge, *, order: int) -> dict[str, Any]:
    payload = edge.to_dict()
    payload.pop("predecessor_task_id", None)
    payload.pop("successor_task_id", None)
    payload["integration_order"] = int(order)
    payload["hop_index"] = int(order)
    notes = [str(n) for n in (payload.get("source_notes") or ())]
    for note in _SOURCE_NOTES:
        if note not in notes:
            notes.append(note)
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
    segment: str,
    owner_package: str,
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
    notes_t = tuple(dict.fromkeys((*_SOURCE_NOTES, *notes)))
    entry = EntryContract(
        fingerprint=EntryFingerprint(
            room_id=int(room_id),
            items=items,
            prior_room_id=predecessor_room_id,
        )
    )
    max_frames = PLACEHOLDER_MAX_FRAMES
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
        "owner_package": owner_package or OWNER_PACKAGE,
        "integration_order": int(order),
        "max_frames": max_frames,
        "max_no_progress": max(1, min(600, max_frames)),
        "segment": segment,
        "hop_index": int(order),
        "source_notes": list(notes_t),
    }


def _gravity_edges(rng: RangePlan, *, next_room_id: int) -> list[dict[str, Any]]:
    edges = list(rng.manifest.edges)
    if not edges:
        raise AssembleError(
            "Attic→Gravity range produced no edges",
            code="assemble.selected",
            details={"route_id": ROUTE_ID},
        )
    out: list[dict[str, Any]] = []
    last = len(edges) - 1
    for i, edge in enumerate(edges):
        if i == last:
            out.append(_retarget_next(edge, next_room_id=next_room_id, order=i))
        else:
            out.append(_edge_payload(edge, order=i))
    return out


def _tail_edges(*, items: int | None, start_order: int) -> list[dict[str, Any]]:
    specs = ITEM_SEAM_LANES[1:]
    if len(specs) != len(TAIL_ROOMS):
        raise AssembleError(
            "credits tail does not match item-seam lanes",
            code="assemble.selected",
            details={"lanes": len(specs), "rooms": len(TAIL_ROOMS)},
        )
    rows: list[dict[str, Any]] = []
    pred = GRAVITY_ROOM
    for i, (spec, room) in enumerate(zip(specs, TAIL_ROOMS)):
        nxt = TAIL_ROOMS[i + 1] if i + 1 < len(TAIL_ROOMS) else None
        segment = spec.segments[0] if spec.segments else "s33"
        goal = CREDITS_GOAL if spec.lane_id == CREDITS_TASK_ID else None
        extra = (
            f"{spec.name}; placeholder (no tape adapter)",
            *spec.labels,
        )
        rows.append(
            _placeholder_edge(
                task_id=spec.lane_id,
                room_id=int(room),
                predecessor_room_id=pred,
                next_room_id=nxt,
                items=items,
                order=start_order + i,
                segment=segment,
                owner_package=spec.owner_package,
                notes=extra,
                goal=goal,
            )
        )
        pred = int(room)
    return rows


def _lane_for_task(lanes: Sequence[Lane], task_id: str) -> str | None:
    for lane in lanes:
        if task_id in lane.task_ids:
            return lane.lane_id
    return None


def _writes_of(offer: CandidateOffer | None) -> tuple[MemoryWrite, ...]:
    if offer is None:
        return ()
    artifact = offer.artifact
    if not isinstance(artifact, CandidateArtifact):
        return ()
    return tuple(artifact.memory_writes)


def _ledger(plan: CreditsPlan, assembly: Assembly) -> tuple[LedgerRow, ...]:
    selected = dict(assembly.selected) or dict(plan.selection.selected)
    rows: list[LedgerRow] = []
    for edge in plan.manifest.edges:
        cid = selected.get(edge.task_id, "")
        offer = plan.selection.offer_for(edge.task_id, cid, profile=PROFILE)
        if offer is None:
            for extra in plan.offers:
                if extra.task_id == edge.task_id and extra.candidate_id == cid:
                    offer = extra
                    break
        rows.append(
            LedgerRow(
                task_id=edge.task_id,
                candidate_id=cid,
                lane_id=_lane_for_task(plan.lanes, edge.task_id),
                writes=_writes_of(offer),
            )
        )
    return tuple(rows)


def _live_split(split: Any) -> dict[str, Any]:
    if hasattr(split, "to_dict"):
        payload = split.to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    if isinstance(split, Mapping):
        return dict(split)
    return {
        "split_id": getattr(split, "split_id", None),
        "repr": str(split),
    }


def _room_splits(
    plan: CreditsPlan,
    live: Sequence[Any] = (),
) -> tuple[dict[str, Any], ...]:
    table = []
    for row in assembly_table(plan.manifest):
        item = dict(row)
        item["lane_id"] = _lane_for_task(plan.lanes, str(row["task_id"]))
        item["goal"] = None
        for edge in plan.manifest.edges:
            if edge.task_id == row["task_id"]:
                item["goal"] = edge.goal
                item["room_id"] = int(edge.room_id)
                break
        table.append(item)
    for i, split in enumerate(live):
        if i < len(table):
            table[i] = dict(table[i])
            table[i]["split"] = _live_split(split)
    return tuple(table)


def credits_chain(
    segment_dir: Path | str | None = None,
    *,
    profile: str = PROFILE,
    range_plan: RangePlan | None = None,
) -> CreditsPlan:
    """RouteManifest spanning ten item-seam lanes plus Attic→Gravity."""
    _require_scaffold(profile)
    rng = range_plan if range_plan is not None else attic_to_gravity_range(
        segment_dir, profile=PROFILE
    )
    if rng.profile != PROFILE or rng.hp_clamp_global:
        raise AssembleError(
            "Scaffold credits chain is scaffold-only and must not enable HP clamp globally",
            code="assemble.profile",
            details={
                "profile": rng.profile,
                "hp_clamp_global": rng.hp_clamp_global,
                "route_id": ROUTE_ID,
            },
        )
    rooms = tuple(edge.room_id for edge in rng.manifest.edges)
    if MAIN_SHAFT_ROOM in rooms:
        raise AssembleError(
            "Scaffold credits chain excludes Main Shaft hops",
            code="assemble.selected",
            details={"route_id": ROUTE_ID},
        )
    first_tail = TAIL_ROOMS[0]
    edges = _gravity_edges(rng, next_room_id=first_tail)
    items = rng.manifest.edges[-1].required_items if rng.manifest.edges else None
    edges.extend(_tail_edges(items=items, start_order=len(edges)))
    manifest = RouteManifest.from_dict(
        {"route_id": ROUTE_ID, "variant": "kpdr", "edges": edges}
    )
    chained_rooms = tuple(edge.room_id for edge in manifest.edges)
    if MAIN_SHAFT_ROOM in chained_rooms:
        raise AssembleError(
            "Scaffold credits chain excludes Main Shaft hops",
            code="assemble.selected",
            details={"route_id": ROUTE_ID, "rooms": [f"0x{r:04X}" for r in chained_rooms]},
        )
    if not manifest.edges or manifest.edges[-1].goal != CREDITS_GOAL:
        raise AssembleError(
            "credits chain must end at the credits goal",
            code="assemble.selected",
            details={"route_id": ROUTE_ID},
        )
    lanes = inventory_from_manifest(manifest)
    if len(lanes) != len(ITEM_SEAM_LANES):
        raise AssembleError(
            "credits chain must span ten item-seam lanes",
            code="assemble.selected",
            details={"lanes": [lane.lane_id for lane in lanes]},
        )
    offers = rng.offers
    selection = select(manifest, offers, profile=PROFILE)
    placeholders = tuple(
        edge.task_id
        for edge in manifest.edges
        if edge.selected_map().get(PROFILE) == PLACEHOLDER_KIND_ID
        and edge.task_id not in rng.tape_tasks
    )
    return CreditsPlan(
        route_id=ROUTE_ID,
        profile=PROFILE,
        manifest=manifest,
        selection=selection,
        offers=offers,
        lanes=lanes,
        range_plan=rng,
        tape_tasks=rng.tape_tasks,
        placeholder_tasks=placeholders,
        hp_clamp_tasks=rng.hp_clamp_tasks,
        hp_clamp_allowlist=rng.hp_clamp_allowlist,
        hp_clamp_global=False,
        non_claims=CREDITS_NON_CLAIMS,
        source_notes=_SOURCE_NOTES,
    )


def assemble_credits(
    segment_dir: Path | str | None = None,
    *,
    profile: str = PROFILE,
    assemble: AssembleFn | None = None,
    play_hops: PlayHops | None = None,
    session: Any | None = None,
    session_factory: Callable[[], Any] | None = None,
    hop_factory: HopFactory | None = None,
    plan: CreditsPlan | None = None,
) -> CreditsReport:
    """Drive the Scaffold credits chain through ``splice.assemble``."""
    _require_scaffold(profile)
    chain = plan if plan is not None else credits_chain(segment_dir, profile=profile)
    if chain.profile != PROFILE or chain.hp_clamp_global:
        raise AssembleError(
            "Scaffold credits chain is scaffold-only and must not enable HP clamp globally",
            code="assemble.profile",
            details={
                "profile": chain.profile,
                "hp_clamp_global": chain.hp_clamp_global,
                "route_id": chain.route_id,
            },
        )
    split_buf: list[Any] = []
    runner = assemble if assemble is not None else assemble_route
    assembly = runner(
        chain.route_id,
        chain.selection,
        manifest=chain.manifest,
        profile=PROFILE,
        candidates=chain.offers,
        play_hops=play_hops,
        session=session,
        session_factory=session_factory,
        hop_factory=hop_factory,
        splits=split_buf,
    )
    if not isinstance(assembly, Assembly):
        raise AssembleError(
            "assemble must return an Assembly",
            code="assemble.play",
            details={"route_id": chain.route_id},
        )
    return CreditsReport(
        assembly=assembly,
        plan=chain,
        intervention_ledger=_ledger(chain, assembly),
        room_splits=_room_splits(chain, split_buf),
        non_claims=chain.non_claims,
        source_notes=chain.source_notes,
    )


def _play_finish_splice(
    start: bytes, delay: int, *, recorded: tuple[Any, list[dict[str, Any]]] | None = None,
) -> tuple[dict[str, Any], Any]:
    """Run the production finish controllers or their recorded buttons from one pin."""
    from contextlib import ExitStack

    import numpy as np

    from retro_harness.actions import idle_action
    from super_metroid.combat.escape import (
        ESCAPE_ROW_CONTROLLERS,
        finish_controller_for,
        play_credits_tail,
    )
    from super_metroid.dev.common import make_dev_env
    from super_metroid.ram import GameplayPhase, credits_complete, parse_env_state
    from super_metroid.routes.tips import play_hops
    from super_metroid.splice.assemble import AssemblyHop, _chain_session, _guard_loads, _guard_writes

    actions, rows, splits = [], [], []
    replay, expected_rows = recorded if recorded is not None else (None, [])
    loads, writes = [0], [0]
    with ExitStack() as cleanup:
        env = make_dev_env()
        cleanup.callback(env.close)
        env.reset()
        core = env.em
        core.set_state(start)

        class CoreProxy:
            def __getattr__(self, name: str) -> Any:
                return getattr(core, name)

        env.em = CoreProxy()  # Allow the guards to shadow the native methods.
        cleanup.callback(setattr, env, "em", core)
        session = _chain_session(env)
        cleanup.callback(_guard_writes(env, writes))
        cleanup.callback(_guard_loads(session, loads))
        step = session.step

        def checked_step(action: Any, reason: str) -> Any:
            state = step(action, reason)
            if state.dead or state.health <= 0:
                raise RuntimeError(f"finish splice died in room 0x{state.room_id:04X} at frame {session.frame}")
            if replay is None:
                actions.append(action.copy())
            return state

        session.step = checked_step
        idle = idle_action(dtype=np.int8)
        for frame in range(delay):
            session.step(idle if replay is None else replay[frame], "finish_entry_delay")

        for edge in ESCAPE_ROW_CONTROLLERS:
            source, destination, goal_phase = edge
            origin, target = int(source, 16), int(destination or source, 16)
            if session.state.room_id != origin:
                continue
            begin = session.frame
            play = finish_controller_for(session.state, *edge)
            if play is None:
                raise RuntimeError(f"finish controller unavailable in {source}")
            if replay is not None:
                if len(rows) >= len(expected_rows) or expected_rows[len(rows)]["room"] != source:
                    raise RuntimeError(f"finish replay diverged at {source} on frame {begin}")
                end = begin + expected_rows[len(rows)]["frames"]

                def play_recorded(live: Any) -> None:
                    live.raw_actions(replay[begin:end], "finish_replay")

                play = play_recorded
            play_hops(session, splits, [AssemblyHop(source, play, origin, target, source, "finish_splice")])
            state = session.state
            phase = GameplayPhase(goal_phase) if goal_phase else GameplayPhase.ORDINARY_GAMEPLAY
            if state.room_id != target or state.phase is not phase:
                raise RuntimeError(f"finish splice missed {source} to {target:04X} at frame {session.frame}")
            rows.append({
                "room": source, "to_room": f"0x{target:04X}", "frames": session.frame - begin,
                "energy": state.health, "x": state.samus_x, "y": state.samus_y,
                # The timer stores packed decimal digits.
                "escape_clock": (
                    f"{state.escape_timer_minutes:X}:{state.escape_timer_seconds:02X}."
                    f"{state.escape_timer_frames:02X}"
                ),
            })

        if not rows or session.state.phase is not GameplayPhase.ENDING_OR_CREDITS:
            raise RuntimeError("finish splice did not launch the ship")
        ending_entry = session.frame
        if replay is None:
            tail = play_credits_tail(env)
            tape = np.concatenate((np.asarray(actions, dtype=np.int8), tail))
        else:
            session.raw_actions(replay[ending_entry:], "finish_credits_replay")
            tape = replay
        if not credits_complete(env) or loads[0] or writes[0]:
            raise RuntimeError("finish splice failed its final screen or intervention check")
        return {
            "rows": rows,
            "ending_entry_frame": ending_entry,
            "credits_tail_frames": len(tape) - ending_entry,
            "credits_complete": True,
            "total_frames": len(tape),
            "final_state": parse_env_state(env, frame=len(tape)).to_dict(),
            "state_loads_after_startup": loads[0], "ram_writes": writes[0], "deaths": 0,
        }, tape


def benchmark_finish_splice(
    pin: Path, *, delay: int = 0, tape_out: Path | None = None, compare: Path | None = None,
) -> dict[str, Any]:
    """Time a live pin-to-credits splice and verify it in a fresh buttons-only boot."""
    import hashlib
    import json

    import numpy as np

    from retro_harness.env import read_state_bytes
    from super_metroid.human_tape.rta_clock import fmt_time
    from super_metroid.paths import INTEGRATION_DIR, VANILLA_ROM_SHA1

    if delay < 0:
        raise ValueError("entry delay must be nonnegative")
    rom = (INTEGRATION_DIR / "rom.sfc").read_bytes()
    if hashlib.sha1(rom).hexdigest() != VANILLA_ROM_SHA1:
        raise ValueError("finish benchmark requires the vanilla ROM")
    start = read_state_bytes(pin)
    case = {
        "rom_sha256": hashlib.sha256(rom).hexdigest(),
        "pin_sha256": hashlib.sha256(start).hexdigest(),
        "entry_delay_frames": delay,
    }
    baseline = None
    if compare is not None:
        baseline = json.loads(compare.read_text())
        if (
            not isinstance(baseline, dict)
            or baseline.get("kind") != "finish_splice"
            or baseline.get("schema_version") != 1
            or baseline.get("case") != case
            or baseline.get("replay_verified") is not True
        ):
            raise ValueError("compare needs a verified finish splice from the same ROM, pin and delay")
    live, tape = _play_finish_splice(start, delay)
    replay, _ = _play_finish_splice(start, delay, recorded=(tape, live["rows"]))
    if live != replay:
        raise RuntimeError("buttons-only finish replay did not match the live splits and endpoint")
    report = {
        "kind": "finish_splice", "schema_version": 1, "case": case, "pin": str(pin),
        "replay_verified": True, "tape_sha256": hashlib.sha256(tape.tobytes()).hexdigest(),
        "ending_entry_time_60fps": fmt_time(live["ending_entry_frame"]),
        "final_screen_time_60fps": fmt_time(live["total_frames"]),
        **live,
    }
    if baseline is not None:
        prior_rows = {row["room"]: row for row in baseline["rows"]}
        for row in report["rows"]:
            row["frames_saved"] = prior_rows[row["room"]]["frames"] - row["frames"]
        report["frames_saved_to_ending"] = baseline["ending_entry_frame"] - live["ending_entry_frame"]
        report["frames_saved_to_final_screen"] = baseline["total_frames"] - live["total_frames"]
    if tape_out is not None:
        with tape_out.open("xb") as output:
            np.savez_compressed(output, actions=tape)
    return report
