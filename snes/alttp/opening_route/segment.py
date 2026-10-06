"""Segment / natural-entry contract for ALTTP opening-route hops.

Parallel in spirit to ``super_metroid.routes.segment``:

- entry requirement (room/screen + position + inventory)
- exit / success predicate
- ``play(env)`` → evidence (frames, final snapshot, phase report)

Existing scripts (``castle_to_sword``, ``secret_entrance_clear``) adapt to
this surface. Continuous composition should walk verified Segments; a hop
is route-ready only when it succeeds from the real predecessor continuous
state (no privileged warps in published evidence).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from alttp.opening_route.anchors import opening_anchors
from alttp.ram import AlttpSnapshot, snapshot_to_diag
from alttp.route_report import RoutePhaseResult, SegmentResult
from alttp.startup import BootEnv


@runtime_checkable
class Segment(Protocol):
    """Uniform play unit for continuous opening hops."""

    @property
    def id(self) -> str: ...

    def play(self, env: BootEnv, **kwargs: Any) -> SegmentResult: ...


@dataclass(frozen=True)
class EntryRequirement:
    """What must be true before a segment may start (natural-entry gate)."""

    description: str
    room_base_id: int | None = None
    screen_id: int | None = None
    require_indoors: bool | None = None
    require_fighter_sword: bool = False
    require_lamp: bool = False
    require_zelda_follower: bool = False
    require_control: bool = True
    # Optional multi-truth anchor ids that all must match
    anchor_ids: tuple[str, ...] = ()
    # Graph node this segment expects as predecessor
    graph_node_id: str = ""

    def matches(self, snapshot: AlttpSnapshot) -> bool:
        if self.require_control and not snapshot.has_control:
            return False
        if self.room_base_id is not None:
            if not snapshot.indoors or snapshot.room_base_id != self.room_base_id:
                return False
        if self.screen_id is not None and not snapshot.indoors:
            if snapshot.screen_id != self.screen_id:
                return False
        if self.require_indoors is True and not snapshot.indoors:
            return False
        if self.require_indoors is False and snapshot.indoors:
            return False
        if self.require_fighter_sword and not snapshot.has_fighter_sword:
            return False
        if self.require_lamp and not snapshot.has_lamp:
            return False
        if self.require_zelda_follower and not snapshot.has_zelda_follower:
            return False
        if self.anchor_ids:
            by_id = {a.anchor_id: a for a in opening_anchors()}
            for aid in self.anchor_ids:
                anchor = by_id.get(aid)
                if anchor is None or not anchor.matches(snapshot):
                    return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "roomBaseId": self.room_base_id,
            "screenId": self.screen_id,
            "requireIndoors": self.require_indoors,
            "requireFighterSword": self.require_fighter_sword,
            "requireLamp": self.require_lamp,
            "requireZeldaFollower": self.require_zelda_follower,
            "requireControl": self.require_control,
            "anchorIds": list(self.anchor_ids),
            "graphNodeId": self.graph_node_id,
        }


@dataclass(frozen=True)
class ExitPredicate:
    """Success condition for a finished segment."""

    description: str
    acceptance_keys: tuple[str, ...] = ()
    # If set, all listed acceptance keys from evaluate_acceptance must be True
    require_all: bool = True
    graph_node_id: str = ""
    verification: str = "planned"  # planned | isolated | natural_entry | continuous

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "acceptanceKeys": list(self.acceptance_keys),
            "requireAll": self.require_all,
            "graphNodeId": self.graph_node_id,
            "verification": self.verification,
        }


@dataclass
class SegmentEvidence:
    """Uniform evidence wrapper around :class:`SegmentResult`."""

    segment_id: str
    ok: bool
    frames: int
    snapshot: AlttpSnapshot
    source: str
    phase: str
    acceptance: dict[str, bool] = field(default_factory=dict)
    diagnostics: dict[str, bool] = field(default_factory=dict)
    blocker: str = ""
    notes: list[str] = field(default_factory=list)
    # Preserve the controller-level phases rather than reducing a composed
    # natural-chain report to one opaque segment row.
    phases: list[RoutePhaseResult] = field(default_factory=list)
    development_only: bool = True
    matched_anchors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "segmentId": self.segment_id,
            "ok": self.ok,
            "frames": self.frames,
            "source": self.source,
            "phase": self.phase,
            "cleanChain": self.source == "natural_boot" and self.ok,
            "developmentOnly": self.development_only,
            "acceptance": dict(self.acceptance),
            "blocker": self.blocker,
            "notes": list(self.notes),
            "phases": [
                {
                    "phase": phase.phase,
                    "ok": phase.ok,
                    "frames": phase.frames,
                    "detail": phase.detail,
                    "diag": phase.diag or snapshot_to_diag(phase.snapshot),
                }
                for phase in self.phases
            ],
            "matchedAnchors": list(self.matched_anchors),
            "final": snapshot_to_diag(self.snapshot),
        }
        if self.diagnostics:
            out["diagnostics"] = dict(self.diagnostics)
        return out

    @classmethod
    def from_segment_result(
        cls,
        segment_id: str,
        result: SegmentResult,
        *,
        matched_anchors: Sequence[str] | None = None,
    ) -> SegmentEvidence:
        return cls(
            segment_id=segment_id,
            ok=result.ok,
            frames=result.frames,
            snapshot=result.snapshot,
            source=result.source,
            phase=result.phase,
            acceptance=dict(result.acceptance),
            diagnostics=dict(result.diagnostics),
            blocker=result.blocker,
            notes=list(result.notes),
            phases=list(result.phases),
            development_only=result.source != "natural_boot",
            matched_anchors=list(matched_anchors or ()),
        )


PlayFn = Callable[..., SegmentResult]


@dataclass(frozen=True)
class ScriptSegment:
    """Adapter: existing route script callable → Segment contract."""

    segment_id: str
    play_fn: PlayFn
    entry: EntryRequirement
    exit: ExitPredicate
    label: str = ""
    graph_edge_id: str = ""

    @property
    def id(self) -> str:
        return self.segment_id

    def play(self, env: BootEnv, **kwargs: Any) -> SegmentResult:
        return self.play_fn(env, **kwargs)

    def play_checked(
        self,
        env: BootEnv,
        *,
        enforce_entry: bool = True,
        **kwargs: Any,
    ) -> SegmentEvidence:
        """Play and wrap evidence.

        When ``enforce_entry`` is True (default), snapshot the env and call
        :meth:`EntryRequirement.matches` first. On mismatch, return
        ``phase=\"entry_rejected\"`` with ``ok=False`` **without** calling
        ``play_fn``. Pass ``enforce_entry=False`` only for intentional tests
        that skip the natural-entry gate.
        """
        from alttp.opening_route.anchors import match_anchors
        from alttp.startup import snapshot_env

        if enforce_entry:
            pre = snapshot_env(env)
            if not self.entry.matches(pre):
                matched = [a.anchor_id for a in match_anchors(pre)]
                source = str(kwargs.get("source", "unknown"))
                return SegmentEvidence(
                    segment_id=self.segment_id,
                    ok=False,
                    frames=0,
                    snapshot=pre,
                    source=source,
                    phase="entry_rejected",
                    acceptance={},
                    blocker=(
                        f"entry requirement not met for {self.segment_id}: "
                        f"{self.entry.description}"
                    ),
                    notes=["play_fn not called (entry gate)"],
                    development_only=source != "natural_boot",
                    matched_anchors=matched,
                )

        result = self.play(env, **kwargs)
        snap = result.snapshot
        matched = [a.anchor_id for a in match_anchors(snap)]
        return SegmentEvidence.from_segment_result(
            self.segment_id, result, matched_anchors=matched
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "segmentId": self.segment_id,
            "label": self.label or self.segment_id,
            "graphEdgeId": self.graph_edge_id,
            "entry": self.entry.to_dict(),
            "exit": self.exit.to_dict(),
        }


def visit_sahasrahla_hut(env: BootEnv, *, source: str = "state_load_dev") -> SegmentResult:
    """Mapped hut visit from Eastern grounds; no pendant or bomb grants."""
    from dataclasses import asdict

    from alttp import primitives
    from alttp.opening_route.room_engine import move_path_combat_aware, run_room_edge
    from alttp.overworld import _dismiss_text
    from alttp.ram import read_inventory, read_u16, wram_index
    from alttp.room_map import load_room_map
    from alttp.route_report import phase_from_primitive
    from alttp.startup import snapshot_env

    phases: list[RoutePhaseResult] = []
    indoor = load_room_map("room_05")
    before = read_inventory(env.get_ram())
    marker_before = int(env.get_ram()[wram_index(0xF3C7)])

    def record(label: str, result: Any) -> None:
        phases.append(phase_from_primitive(label, result))
        if not result.ok:
            raise RuntimeError(result.reason)

    def walk(label: str) -> None:
        point = indoor.point(label)
        if point is None:
            raise RuntimeError(f"missing hut point {label}")
        record(label, primitives.move_to(
            env, primitives.Waypoint(point.x, point.y, tolerance=0, room=5, label=label),
            step_size=1,
        ))

    def edge(map_id: str, door: str) -> None:
        result = run_room_edge(env, map_id, door, clear=False, source=source)
        phases.extend(result.phases)
        if not result.ok:
            raise RuntimeError(result.blocker)

    blocker = ""
    try:
        if before.pendants or before.boots or before.bombs or marker_before != 1:
            raise RuntimeError("hut visit requires the unvisited no-pendant, no-bomb predecessor")
        edge("screen_1e_sahasrahla", "hut_door_to_0x05")
        point = indoor.point("elder_front")
        if point is None:
            raise RuntimeError("missing elder approach")
        record("elder_front", move_path_combat_aware(
            env, [primitives.Waypoint(point.x, point.y, tolerance=1, room=5)],
            room=5, policy=indoor.clear_policy, walker=True,
        ))
        record("talk_to_elder", primitives.run_script(env, ((('UP',), 1), (('A',), 1), ((), 6))))
        if not snapshot_env(env).is_text_mode or read_u16(env.get_ram(), 0x1CF0) != 0x32:
            raise RuntimeError("elder's no-pendant dialogue did not open")
        closed, frames = _dismiss_text(env, 3000, max_pages=20)
        record("elder_dialogue", primitives.PrimitiveResult(closed, "advance messages 0x32/0x33", frames, snapshot_env(env)))
        if int(env.get_ram()[wram_index(0xF3C7)]) != 3:
            raise RuntimeError("elder did not mark the pendant locations")
        for label in ("west_south", "west_north", "pot_left_aisle", "pot_left_front", "pot_right_front"):
            walk(label)
            if not label.endswith("_front"):
                continue
            record(label + "_face", primitives.run_script(env, ((('UP',), 1),)))
            record(label + "_lift", primitives.lift(env))
            phases[-1].diag.update(sprites=[asdict(s) for s in primitives.active_sprites(env)])
            record(label + "_throw", primitives.throw(env))
            record(label + "_settle", primitives.run_script(env, (((), 60),)))
            phases[-1].diag.update(inventory=asdict(read_inventory(env.get_ram())))
        edge("room_05", "south_to_0x1e")
    except RuntimeError as exc:
        blocker = str(exc)
    snap = snapshot_env(env)
    after = read_inventory(env.get_ram())
    marker = int(env.get_ram()[wram_index(0xF3C7)])
    acceptance = {
        "hut_completed": not blocker and not snap.indoors and snap.has_control
        and snap.screen_id == 0x1E and marker == 3 and snap.follower == 0,
    }
    return SegmentResult(
        ok=acceptance["hut_completed"], phase="hut_completed" if not blocker else "hut_blocked",
        frames=sum(p.frames for p in phases), snapshot=snap, phases=phases,
        source=source, acceptance=acceptance, blocker=blocker,
        notes=[f"Map indicator {marker_before}->{marker}; inventory before={asdict(before)} after={asdict(after)}",
               "Two accessible pots inspected; left pot can restore a heart. Rear 50 rupees/3 bombs/50 rupees chests remain closed behind type0x28 bombable wall; no bombs or boots.",
               "Development predecessor evidence; no Clean tip promotion."],
    )


def approach_eastern_palace(env: BootEnv, *, source: str = "state_load_dev") -> SegmentResult:
    """Follow live outdoor collision paths from the completed hut to the plaza."""
    from dataclasses import asdict

    from alttp.overworld import read_overworld_tiles, standing_screen, walk_to_screen
    from alttp.ram import read_inventory, wram_index
    from alttp.room_map import load_room_map
    from alttp.startup import snapshot_env

    start = snapshot_env(env)
    before = read_inventory(env.get_ram())
    story = tuple(int(env.get_ram()[wram_index(a)]) for a in (0xF3C5, 0xF3C6, 0xF3C7))
    exit_door = load_room_map("room_05").door("south_to_0x1e")
    outdoor = load_room_map("screen_1e_eastern")
    point = outdoor.point("eastern_door")
    if exit_door is None or exit_door.landing_xy is None or point is None:
        raise ValueError("missing mapped hut exit or Eastern approach")
    entry_ok = (
        not start.indoors and not start.dark_world and start.has_control
        and start.screen_id == outdoor.screen_id and start.follower == 0
        and start.has_fighter_sword and start.has_lamp and start.hp > 0
        and story == (2, 0x15, 3) and before.max_hp == 32
        and max(abs(v - target) for v, target in
                zip((start.link_x, start.link_y), exit_door.landing_xy)) <= 4
    )
    if not entry_ok:
        return SegmentResult(
            ok=False, phase="entry_rejected", frames=0, snapshot=start, source=source,
            blocker="Eastern approach requires the completed hut's mapped outdoor exit",
        )
    tiles = read_overworld_tiles(env.get_ram())
    target = standing_screen(tiles, point.x, point.y)
    walked = walk_to_screen(
        env, target, target_xy=(point.x, point.y), tolerance=4,
        max_frames=6000, guard=True,
    )
    snap = snapshot_env(env)
    after = read_inventory(env.get_ram())
    after_story = tuple(int(env.get_ram()[wram_index(a)]) for a in (0xF3C5, 0xF3C6, 0xF3C7))
    after_fields = asdict(after)
    acceptance = {
        "mapped_approach": bool(walked["ok"] and snap.has_control and not snap.indoors
                                and not snap.dark_world and snap.screen_id == outdoor.screen_id
                                and snap.hp > 0 and snap.follower == 0
                                and max(abs(snap.link_x - point.x), abs(snap.link_y - point.y)) <= 4),
        "inventory_preserved": all(value == after_fields[name]
                                   for name, value in asdict(before).items() if name != "hp"),
        "story_preserved": after_story == story,
    }
    acceptance["eastern_approach"] = all(acceptance.values())
    ok = acceptance["eastern_approach"]
    return SegmentResult(
        ok=ok, phase="eastern_approach", frames=int(walked["frames"]), snapshot=snap,
        source=source, acceptance=acceptance,
        blocker="" if ok else f"Eastern approach failed: {walked['reason']}; {acceptance}",
        phases=[RoutePhaseResult(
            phase="hedge_to_eastern_door", ok=ok, frames=int(walked["frames"]), snapshot=snap,
            detail=str(walked["line"]), diag={**snapshot_to_diag(snap), "walk": walked},
        )],
        notes=[f"HP {before.hp}->{after.hp}; damage events={walked['hits']}. Loaded screen and standing quadrant are distinct.",
               "Mapped target with live hedge, boulder and stair collisions; guarded button path. Development evidence; no Clean tip promotion."],
    )


def collect_sanctuary_heart(env: BootEnv, *, source: str = "state_load_dev") -> SegmentResult:
    """Collect the Sanctuary Heart Container chest in room 0x12."""
    import numpy as np
    from alttp import primitives
    from alttp.ram import read_inventory
    from alttp.room_map import load_room_map
    from alttp.route_report import phase_from_primitive
    from alttp.startup import snapshot_env

    phases: list[RoutePhaseResult] = []
    indoor = load_room_map("room_12")

    def record(label: str, result: Any) -> None:
        phases.append(phase_from_primitive(label, result))
        if not result.ok:
            raise RuntimeError(result.reason)

    blocker = ""
    try:
        snap = snapshot_env(env)
        if not snap.indoors or snap.room_base_id != 18:
            raise RuntimeError(f"expected indoor room 0x12, got {snap.room_base_id}")
        inv_before = read_inventory(np.asarray(env.get_ram(), dtype=np.uint8))
        if inv_before.max_hp >= 32:
            raise RuntimeError("heart container already collected")

        for pt_name in ("heart_aisle_south", "heart_west_south", "heart_south_lip", "heart_open_front"):
            pt = indoor.point(pt_name)
            if pt is None:
                raise RuntimeError(f"missing point {pt_name}")
            record(pt_name, primitives.move_to(
                env, primitives.Waypoint(pt.x, pt.y, tolerance=0, room=18, label=pt_name),
                step_size=1,
            ))

        record("face_north", primitives.run_script(env, ((('UP',), 1),)))
        record("open_chest", primitives.run_script(env, ((('A',), 2),)))

        record("heart_receipt", primitives.run_script(
            env, tuple(action for i in range(27)
                       for action in (((), 3), (("B" if i % 2 == 0 else "A",), 2))),
        ))

        settle = primitives.settle_control(env, max_frames=240)
        record("heart_dialogue_settle", settle)
    except RuntimeError as exc:
        blocker = str(exc)

    snap = snapshot_env(env)
    inv = read_inventory(np.asarray(env.get_ram(), dtype=np.uint8))
    acceptance = {
        "heart_collected": not blocker and snap.indoors and snap.room_base_id == 18
        and snap.has_control and inv.max_hp == 32 and snap.hp == 32,
    }
    return SegmentResult(
        ok=acceptance["heart_collected"],
        phase="heart_collected" if not blocker else "heart_blocked",
        frames=sum(p.frames for p in phases),
        snapshot=snap,
        phases=phases,
        source=source,
        acceptance=acceptance,
        blocker=blocker,
        notes=[
            "Sanctuary Heart Container chest (room 0x12) collected using validated map waypoints.",
            f"Max HP: {inv_before.max_hp if 'inv_before' in locals() else 'unknown'} -> {inv.max_hp}; HP refilled to {snap.hp}.",
        ],
    )


def exit_sanctuary_to_outdoors(env: BootEnv, *, source: str = "state_load_dev") -> SegmentResult:
    """Exit Sanctuary through the mapped south door to screen 0x13."""
    from alttp import primitives
    from alttp.ram import SANCTUARY_SCREEN
    from alttp.room_map import load_room_map
    from alttp.route_report import phase_from_primitive
    from alttp.startup import snapshot_env

    phases: list[RoutePhaseResult] = []
    indoor = load_room_map("room_12")
    door = indoor.door("sanctuary_door_to_0x13")
    if door is None:
        raise RuntimeError("missing door sanctuary_door_to_0x13")

    def record(label: str, result: Any) -> None:
        phases.append(phase_from_primitive(label, result))
        if not result.ok:
            raise RuntimeError(result.reason)

    blocker = ""
    try:
        snap = snapshot_env(env)
        if not snap.indoors or snap.room_base_id != 18:
            raise RuntimeError(f"expected indoor room 0x12, got {snap.room_base_id}")

        for pt_name in door.path:
            pt = indoor.point(pt_name)
            if pt is None:
                raise RuntimeError(f"missing door path point {pt_name}")
            tol = door.tolerance_for(pt_name)
            record(pt_name, primitives.move_to(
                env, primitives.Waypoint(pt.x, pt.y, tolerance=tol, room=18, label=pt_name),
                step_size=1,
            ))

        record("sanctuary_door_push", primitives.run_script(
            env, (((door.direction,), door.push_frames),),
        ))
        settle = primitives.settle_control(env, max_frames=480)
        record("sanctuary_exit_settle", settle)
    except RuntimeError as exc:
        blocker = str(exc)

    snap = snapshot_env(env)
    acceptance = {
        "sanctuary_exited": not blocker and not snap.indoors and snap.screen_id == SANCTUARY_SCREEN
        and snap.has_control and snap.hp > 0,
    }
    return SegmentResult(
        ok=acceptance["sanctuary_exited"],
        phase="sanctuary_exited" if not blocker else "sanctuary_exit_blocked",
        frames=sum(p.frames for p in phases),
        snapshot=snap,
        phases=phases,
        source=source,
        acceptance=acceptance,
        blocker=blocker,
        notes=[
            f"Sanctuary south door exit lands outdoors screen 0x{SANCTUARY_SCREEN:02X} at ({snap.link_x},{snap.link_y}).",
        ],
    )


def walk_to_kakariko(env: BootEnv, *, source: str = "state_load_dev") -> SegmentResult:
    """Compose westward tile walks from the collected-heart Sanctuary leave."""
    from alttp.overworld import walk_to_screen
    from alttp.ram import (
        KAKARIKO_SCREEN, NORTH_FIELD_SCREEN, SANCTUARY_SCREEN, SANCTUARY_WEST_SCREEN,
        SRAM_PROGRESS, SRAM_PROGRESS_FLAGS, wram_index,
    )
    from alttp.startup import snapshot_env

    phases: list[RoutePhaseResult] = []
    trace: list[int] = []
    snap = snapshot_env(env)
    ram = env.get_ram()
    ready = (
        not snap.indoors and not snap.dark_world and snap.screen_id == SANCTUARY_SCREEN
        and snap.has_control and snap.max_hp == 32 and snap.hp > 0
        and snap.follower == 0 and int(ram[wram_index(SRAM_PROGRESS)]) == 2
        and int(ram[wram_index(SRAM_PROGRESS_FLAGS)]) & 4
    )
    blocker = "" if ready else "requires the completed four-heart Sanctuary exit"
    if ready:
        for target in (SANCTUARY_WEST_SCREEN, NORTH_FIELD_SCREEN, KAKARIKO_SCREEN):
            walked = walk_to_screen(env, target, max_frames=5000, guard=True)
            snap = snapshot_env(env)
            for screen in walked["trace"]:
                if not trace or trace[-1] != screen:
                    trace.append(screen)
            phases.append(RoutePhaseResult(
                phase=f"walk_to_screen_{target:02x}", ok=bool(walked["ok"]),
                frames=int(walked["frames"]), snapshot=snap,
                detail=str(walked["reason"]),
                diag={"trace": walked["trace"], "hits": walked["hits"]},
            ))
            if not walked["ok"]:
                blocker = f"leg to 0x{target:02X} failed: {walked['reason']}"
                break
    # Kakariko is a big loaded area: the northern entry may stand in 0x19
    # while $8A remains 0x18. The tile walker proves the target quadrant.
    ok = bool(not blocker and not snap.indoors and snap.screen_id == KAKARIKO_SCREEN
              and snap.has_control and snap.hp > 0)
    return SegmentResult(
        ok=ok, phase="in_kakariko" if ok else "kakariko_walk_failed",
        frames=sum(p.frames for p in phases), snapshot=snap, phases=phases,
        source=source, acceptance={"in_kakariko": ok}, blocker=blocker,
        notes=["Overworld trace: " + " -> ".join(f"0x{s:02X}" for s in trace)],
    )


def _build_registry() -> dict[str, ScriptSegment]:
    """Lazy registry so heavy route modules import only when needed.

    Live segments must be able to succeed under their exit contract.
    The rescue suffix starts at the real B1 west predecessor; the measured
    castle prefix remains registered separately.
    """
    from alttp.opening_route import (
        castle_dungeon,
        castle_to_sword,
        escort_to_sanctuary,
        main_hall_to_zelda,
        pocket_to_main_hall,
        secret_entrance_clear,
    )
    from alttp.ram import (
        HYRULE_CASTLE_B1_EAST_ROOM,
        HYRULE_CASTLE_B1_PIT_ROOM,
        HYRULE_CASTLE_B1_WEST_ROOM,
        HYRULE_CASTLE_MAIN_HALL_ROOM,
        HYRULE_CASTLE_NORTH_CONNECTOR_ROOM,
        HYRULE_CASTLE_NW_ROOM,
        HYRULE_CASTLE_SCREEN,
        SANCTUARY_ROOM,
        SANCTUARY_SCREEN,
        SECRET_PASSAGE_ROOM,
    )

    return {
        "eastern_approach": ScriptSegment(
            segment_id="eastern_approach", play_fn=approach_eastern_palace,
            entry=EntryRequirement(
                description="Completed Sahasrahla hut at its mapped outdoor exit",
                screen_id=0x1E, require_indoors=False, require_fighter_sword=True,
                require_lamp=True, graph_node_id="sahasrahla_hut",
            ),
            exit=ExitPredicate(
                description="Controllable at the mapped Eastern entrance approach",
                acceptance_keys=("eastern_approach",), graph_node_id="eastern_approach",
                verification="planned",
            ),
            label="Sahasrahla hut exit → Eastern Palace entrance plaza",
        ),
        "sahasrahla_hut": ScriptSegment(
            segment_id="sahasrahla_hut", play_fn=visit_sahasrahla_hut,
            entry=EntryRequirement(
                description="Eastern grounds with sword, lamp, control and no follower",
                screen_id=0x1E, require_indoors=False, require_fighter_sword=True,
                require_lamp=True, graph_node_id="eastern_grounds",
            ),
            exit=ExitPredicate(
                description="Elder marked pendants, accessible pots inspected, hut exited",
                acceptance_keys=("hut_completed",), graph_node_id="sahasrahla_hut",
                verification="planned",
            ),
            label="Eastern grounds → Sahasrahla dialogue and hut exit",
        ),
        "castle_to_sword": ScriptSegment(
            segment_id="castle_to_sword",
            play_fn=castle_to_sword.run_from_castle_grounds,
            entry=EntryRequirement(
                description="Controllable on Hyrule Castle grounds (screen 0x1B)",
                screen_id=HYRULE_CASTLE_SCREEN,
                require_indoors=False,
                graph_node_id="castle_grounds",
                anchor_ids=("HyruleCastle_GroundsSpawn_Controllable",),
            ),
            exit=ExitPredicate(
                description="Fighter sword equip RAM ≥ 1 in secret entrance",
                acceptance_keys=("fighter_sword_ram", "secret_passage"),
                graph_node_id="room_55_sword",
                verification="continuous",
            ),
            label="Castle grounds → uncle fighter sword",
            graph_edge_id="grounds_to_hole+hole_to_sword",
        ),
        "sword_to_secret_entrance_clear": ScriptSegment(
            segment_id="sword_to_secret_entrance_clear",
            play_fn=secret_entrance_clear.run_from_sword,
            entry=EntryRequirement(
                description="Fighter sword in secret entrance room 0x55",
                room_base_id=SECRET_PASSAGE_ROOM,
                require_indoors=True,
                require_fighter_sword=True,
                graph_node_id="room_55_sword",
            ),
            exit=ExitPredicate(
                description="Left secret entrance outdoors (courtyard pocket)",
                acceptance_keys=("left_secret_entrance", "fighter_sword_ram"),
                graph_node_id="courtyard_secret_pocket",
                verification="continuous",
            ),
            label="Post-sword → south chamber → stairs outdoor clear",
            graph_edge_id="sword_to_south_chamber+south_stairs_to_courtyard_pocket",
        ),
        "pocket_to_main_hall": ScriptSegment(
            segment_id="pocket_to_main_hall",
            play_fn=pocket_to_main_hall.run_from_pocket,
            entry=EntryRequirement(
                description=(
                    "Outdoors on castle screen 0x1B with fighter sword "
                    "(courtyard secret pocket or open court)"
                ),
                screen_id=HYRULE_CASTLE_SCREEN,
                require_indoors=False,
                require_fighter_sword=True,
                graph_node_id="courtyard_secret_pocket",
            ),
            exit=ExitPredicate(
                description="Indoors main hall room 0x61",
                acceptance_keys=("main_hall", "fighter_sword_ram"),
                graph_node_id="room_61",
                verification="continuous",
            ),
            label="Courtyard pocket → main castle door → room 0x61",
            graph_edge_id="pocket_to_main_hall",
        ),
        "castle_dungeon_prefix": ScriptSegment(
            segment_id="castle_dungeon_prefix",
            play_fn=castle_dungeon.run_from_main_hall,
            entry=EntryRequirement(
                description="Indoors main hall room 0x61 with fighter sword + control",
                room_base_id=HYRULE_CASTLE_MAIN_HALL_ROOM,
                require_indoors=True,
                require_fighter_sword=True,
                require_control=True,
                graph_node_id="room_61",
                anchor_ids=("HyruleCastle_MainHall",),
            ),
            exit=ExitPredicate(
                description="Continuous clean prefix reaches room 0x50",
                acceptance_keys=("northwest_0x50",),
                graph_node_id="room_50",
                verification="continuous",
            ),
            label="Main hall → west 0x60 → NW chamber 0x50",
            graph_edge_id="main_hall_west_to_0x60+room_60_north_to_0x50",
        ),
        "room_50_east_to_0x01": ScriptSegment(
            segment_id="room_50_east_to_0x01",
            play_fn=castle_dungeon.run_east_to_0x01,
            entry=EntryRequirement(
                description="Indoors NW chamber 0x50 with fighter sword + control",
                room_base_id=HYRULE_CASTLE_NW_ROOM,
                require_indoors=True,
                require_fighter_sword=True,
                require_control=True,
                graph_node_id="room_50",
                anchor_ids=("HyruleCastle_NW_0x50",),
            ),
            exit=ExitPredicate(
                description="East door lands north connector 0x01",
                acceptance_keys=("at_door_dest",),
                graph_node_id="room_01",
                verification="natural_entry",
            ),
            label="NW chamber 0x50 east → room 0x01",
            graph_edge_id="room_50_east_to_0x01",
        ),
        "room_01_down_to_0x72": ScriptSegment(
            segment_id="room_01_down_to_0x72",
            play_fn=castle_dungeon.run_down_to_0x72,
            entry=EntryRequirement(
                description="Indoors north connector 0x01 with fighter sword + control",
                room_base_id=HYRULE_CASTLE_NORTH_CONNECTOR_ROOM,
                require_indoors=True,
                require_fighter_sword=True,
                require_control=True,
                graph_node_id="room_01",
            ),
            exit=ExitPredicate(
                description="North-wall well UP lands B1 pit 0x72",
                acceptance_keys=("at_door_dest",),
                graph_node_id="room_72",
                verification="natural_entry",
            ),
            label="Room 0x01 well UP → room 0x72",
            graph_edge_id="room_01_down_to_0x72",
        ),
        "room_72_south_to_0x82": ScriptSegment(
            segment_id="room_72_south_to_0x82",
            play_fn=castle_dungeon.run_south_to_0x82,
            entry=EntryRequirement(
                description="Indoors B1 pit 0x72 with fighter sword + control",
                room_base_id=HYRULE_CASTLE_B1_PIT_ROOM,
                require_indoors=True,
                require_fighter_sword=True,
                require_control=True,
                graph_node_id="room_72",
                anchor_ids=("HyruleCastle_B1_0x72",),
            ),
            exit=ExitPredicate(
                description="South door from the north ledge lands B1 east 0x82",
                acceptance_keys=("at_door_dest",),
                graph_node_id="room_82",
                verification="natural_entry",
            ),
            label="Room 0x72 key door and south → room 0x82",
            graph_edge_id="room_72_south_to_0x82",
        ),
        "room_82_west_to_0x81": ScriptSegment(
            segment_id="room_82_west_to_0x81",
            play_fn=castle_dungeon.run_west_to_0x81,
            entry=EntryRequirement(
                description="Indoors B1 east 0x82 with fighter sword + control",
                room_base_id=HYRULE_CASTLE_B1_EAST_ROOM,
                require_indoors=True,
                require_fighter_sword=True,
                require_control=True,
                graph_node_id="room_82",
                anchor_ids=("HyruleCastle_B1East_0x82",),
            ),
            exit=ExitPredicate(
                description="West bridge door lands B1 west 0x81",
                acceptance_keys=("at_door_dest",),
                graph_node_id="room_81",
                verification="natural_entry",
            ),
            label="Room 0x82 west → room 0x81",
            graph_edge_id="room_82_west_to_0x81",
        ),
        "room_81_to_zelda": ScriptSegment(
            segment_id="room_81_to_zelda",
            play_fn=main_hall_to_zelda.run_from_b1_west,
            entry=EntryRequirement(
                description="Controllable B1 west 0x81 with fighter sword",
                room_base_id=HYRULE_CASTLE_B1_WEST_ROOM,
                require_indoors=True, require_fighter_sword=True,
                require_control=True, graph_node_id="room_81",
            ),
            exit=ExitPredicate(
                description="Zelda follows after B3 cell dialogue",
                acceptance_keys=("zelda_follower", "in_zelda_cell"),
                graph_node_id="room_80", verification="isolated",
            ),
            label="B1 west → small key → B3 big key → Zelda rescue",
        ),
        "escort_to_sanctuary": ScriptSegment(
            segment_id="escort_to_sanctuary",
            play_fn=escort_to_sanctuary.run_from_escort,
            entry=EntryRequirement(
                description="Zelda follower + lamp + control (escort start)",
                require_zelda_follower=True,
                require_lamp=True,
                require_control=True,
                graph_node_id="room_80",
            ),
            exit=ExitPredicate(
                description="Sanctuary dialogue complete, Zelda safely delivered",
                acceptance_keys=("in_sanctuary", "sanctuary_completed"),
                graph_node_id="sanctuary",
                verification="planned",
            ),
            label="Zelda escort → Sanctuary (planned)",
            # Planned multi-edge leg (mantle → sewers → sanctuary); no single edge yet.
            graph_edge_id="",
        ),
        "sanctuary_heart": ScriptSegment(
            segment_id="sanctuary_heart",
            play_fn=collect_sanctuary_heart,
            entry=EntryRequirement(
                description="Sanctuary 0x12 post-escort delivery point",
                room_base_id=SANCTUARY_ROOM,
                require_indoors=True,
                require_fighter_sword=True,
                require_lamp=True,
                require_control=True,
                graph_node_id="sanctuary",
            ),
            exit=ExitPredicate(
                description="Sanctuary Heart Container collected (4 hearts / 32 max HP)",
                acceptance_keys=("heart_collected",),
                graph_node_id="sanctuary_heart",
                verification="natural_entry",
            ),
            label="Sanctuary → collect Heart Container chest",
            graph_edge_id="sanctuary_to_heart",
        ),
        "sanctuary_exit": ScriptSegment(
            segment_id="sanctuary_exit",
            play_fn=exit_sanctuary_to_outdoors,
            entry=EntryRequirement(
                description="Sanctuary 0x12 with Heart Container collected",
                room_base_id=SANCTUARY_ROOM,
                require_indoors=True,
                require_fighter_sword=True,
                require_lamp=True,
                require_control=True,
                graph_node_id="sanctuary_heart",
            ),
            exit=ExitPredicate(
                description="Outdoors on screen 0x13 at Sanctuary door",
                acceptance_keys=("sanctuary_exited",),
                graph_node_id="sanctuary_exit",
                verification="natural_entry",
            ),
            label="Sanctuary south door → outdoors screen 0x13",
            graph_edge_id="sanctuary_heart_to_exit",
        ),
        "sanctuary_to_kakariko": ScriptSegment(
            segment_id="sanctuary_to_kakariko",
            play_fn=walk_to_kakariko,
            entry=EntryRequirement(
                description="Sanctuary exterior screen 0x13 with 4 hearts",
                screen_id=SANCTUARY_SCREEN,
                require_indoors=False,
                require_fighter_sword=True,
                require_lamp=True,
                require_control=True,
                graph_node_id="sanctuary_exit",
            ),
            exit=ExitPredicate(
                description="Kakariko Village screen 0x18/0x19 outdoors",
                acceptance_keys=("in_kakariko",),
                graph_node_id="kakariko",
                verification="natural_entry",
            ),
            label="Sanctuary exterior → walk to Kakariko Village (screens 0x12, 0x11, 0x18)",
            graph_edge_id="sanctuary_exit_to_kakariko",
        ),
    }


_REGISTRY: dict[str, ScriptSegment] | None = None


def segment_registry() -> Mapping[str, ScriptSegment]:
    """Return the opening-route Segment registry (cached)."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def get_segment(segment_id: str) -> ScriptSegment:
    reg = segment_registry()
    if segment_id not in reg:
        known = ", ".join(sorted(reg))
        raise KeyError(f"unknown segment {segment_id!r}; known: {known}")
    return reg[segment_id]


def list_segments() -> list[str]:
    return sorted(segment_registry())
