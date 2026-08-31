"""Fail-closed one-frame policies for unobserved Level 8 interior stages.

Hypothesis rooms cannot press a direction on the cumulative spine.  Blue Gohma
requires naturally owned Bow + wooden arrows and never pokes L6's one-time
arrow grant or L6 room 0x1C.  Four-head Gleeok waits for a live object type;
0x45 is not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_idle_action
from zelda_i.dungeon.ids import GOHMA_BLUE_OBJECT_TYPE, GOHMA_OBJECT_TYPE
from zelda_i.level8.dungeon import (
    BLUE_GOHMA_ARROWS_REQUIRED,
    ENTRY_TO_MAGIC_KEY_SPEC,
    GLEEOK_FOUR_HEAD_OBJECT_TYPE,
    MAGIC_KEY_TO_SHARD_SPEC,
    UNOBSERVED_LEVEL8_TOPOLOGY,
    Level8ChapterSpec,
    Level8Topology,
)
from zelda_i.ram import ZeldaSnapshot

# L6 red Gohma is 0x33; L8 source is blue 0x34.  Red is accepted only as a
# live-type observation, never as an L6 room check.
_GOHMA_TYPES = frozenset({GOHMA_BLUE_OBJECT_TYPE, GOHMA_OBJECT_TYPE})


@dataclass
class UnverifiedLevel8PathController:
    """Stop immediately when a chapter has no live one-frame policy."""

    stage_id: str
    missing_evidence: str
    spec: Level8ChapterSpec | None = None
    max_frames: int = 1
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    writes: int = 0

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "spec_id": self.stage_id,
            "chapter_id": None if self.spec is None else self.spec.chapter_id,
            "evidence": "hypothesis",
            "route_eligible": False,
            "missing_evidence": self.missing_evidence,
            "writes": self.writes,
            "notes": list(self.notes),
        }

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.failed = True
        note = (
            f"blocked_unverified:{self.stage_id}:"
            f"L{snap.level}:0x{snap.screen:02x}:m{snap.mode}:"
            f"xy={snap.link_x},{snap.link_y}"
        )
        if not self.notes:
            self.notes.append(note)
        return FrameAction(nes_idle_action(), "blocked_unverified")


def unverified_path_controller(
    stage_id: str, missing_evidence: str, *, spec: Level8ChapterSpec | None = None
) -> UnverifiedLevel8PathController:
    return UnverifiedLevel8PathController(stage_id, missing_evidence, spec=spec)


@dataclass
class Level8BlueGohmaController:
    """Fail closed until topology is live and Bow/arrows arrived naturally."""

    topology: Level8Topology = UNOBSERVED_LEVEL8_TOPOLOGY
    max_frames: int = 1
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    writes: int = 0
    poked_arrows: bool = False
    l6_room_check: bool = False

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "spec_id": "level8_blue_gohma",
            "arrows_required": BLUE_GOHMA_ARROWS_REQUIRED,
            "gohma_types": tuple(sorted(_GOHMA_TYPES)),
            "poked_arrows": self.poked_arrows,
            "l6_room_check": self.l6_room_check,
            "writes": self.writes,
            "route_eligible": False,
            "notes": list(self.notes),
        }

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.failed = True
        if snap.bow < 1 or snap.arrows < 1:
            reason = "l8_gohma_requires_natural_bow_arrows"
        elif not self.topology.route_eligible:
            reason = "l8_gohma_topology_unobserved"
        else:
            reason = "l8_gohma_room_unobserved"
        if not self.notes:
            self.notes.append(reason)
        return FrameAction(nes_idle_action(), reason)


@dataclass
class Level8FourHeadGleeokController:
    """Fail closed until a live L8 Gleeok body type is observed."""

    topology: Level8Topology = UNOBSERVED_LEVEL8_TOPOLOGY
    observed_body_type: int | None = GLEEOK_FOUR_HEAD_OBJECT_TYPE
    max_frames: int = 1
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    writes: int = 0

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "spec_id": "level8_four_head_gleeok",
            "observed_body_type": self.observed_body_type,
            "assumed_0x45": False,
            "writes": self.writes,
            "route_eligible": False,
            "notes": list(self.notes),
        }

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.failed = True
        if self.observed_body_type == 0x45:
            reason = "l8_gleeok_refuses_assumed_0x45"
        elif self.observed_body_type is None:
            reason = "l8_gleeok_object_type_unobserved"
        elif not self.topology.route_eligible or self.topology.boss_room is None:
            reason = "l8_gleeok_topology_unobserved"
        else:
            reason = "l8_gleeok_room_unobserved"
        if not self.notes:
            self.notes.append(reason)
        _ = snap
        return FrameAction(nes_idle_action(), reason)


def make_north_manhandla_controller() -> UnverifiedLevel8PathController:
    return unverified_path_controller(
        "level8_north_manhandla_bomb",
        "live entry room, north Manhandla census, and bomb-north stand",
        spec=ENTRY_TO_MAGIC_KEY_SPEC,
    )


def make_darknut_key_controller() -> UnverifiedLevel8PathController:
    return unverified_path_controller(
        "level8_darknut_key_up",
        "live darknut key rooms and KEY-UP shutter without invented room IDs",
        spec=ENTRY_TO_MAGIC_KEY_SPEC,
    )


def make_blue_gohma_controller(
    *, topology: Level8Topology = UNOBSERVED_LEVEL8_TOPOLOGY
) -> Level8BlueGohmaController:
    return Level8BlueGohmaController(topology=topology)


def make_magic_key_stairs_controller() -> UnverifiedLevel8PathController:
    return unverified_path_controller(
        "level8_magic_key_stairs",
        "live Magical Key cellar and natural ADDR_MAGIC_KEY 0-to-1",
        spec=ENTRY_TO_MAGIC_KEY_SPEC,
    )


def make_gleeok_passage_controller() -> UnverifiedLevel8PathController:
    return unverified_path_controller(
        "level8_return_passage",
        "live return from Magical Key through the west passage",
        spec=MAGIC_KEY_TO_SHARD_SPEC,
    )


def make_four_head_gleeok_controller(
    *, topology: Level8Topology = UNOBSERVED_LEVEL8_TOPOLOGY
) -> Level8FourHeadGleeokController:
    return Level8FourHeadGleeokController(topology=topology)


def make_shard_leave_controller() -> UnverifiedLevel8PathController:
    return unverified_path_controller(
        "level8_heart_shard_leave",
        "live heart container, shard 0x80, and settled post-fanfare OW leave",
        spec=MAGIC_KEY_TO_SHARD_SPEC,
    )


__all__ = [
    "Level8BlueGohmaController",
    "Level8FourHeadGleeokController",
    "UnverifiedLevel8PathController",
    "make_blue_gohma_controller",
    "make_darknut_key_controller",
    "make_four_head_gleeok_controller",
    "make_gleeok_passage_controller",
    "make_magic_key_stairs_controller",
    "make_north_manhandla_controller",
    "make_shard_leave_controller",
    "unverified_path_controller",
]
