"""Sanctuary-path room/save-state work queue for ALttP.

Catalogs ``.state`` files under ``custom_integrations/Zelda3-Snes/`` into a
ranked queue toward Sanctuary. Smaller and ALTTP-specific than Super Metroid's
room-problem board: filename heuristics + curated status from STATUS/docs.

Units are **save-state practice checkpoints**, not a full room-graph topology.

Continuous tip (STATUS): NW chamber **0x50** after ``castle_dungeon_prefix``.
Primary next work: discover the physical exit after 0x50, then B1 → Zelda
cell → follower → escort → Sanctuary.
Internal 0x55 key/shutter path is **alternate practice**, not the primary route.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from alttp.paths import (
    INTEGRATION_DIR,
    ROOM_WORK_QUEUE_JSON,
    ROOM_WORK_QUEUE_MD,
)

CATALOG_ID = "alttp_sanctuary_work_queue"
SCHEMA_VERSION = 1

DEFAULT_QUEUE_JSON = ROOM_WORK_QUEUE_JSON
DEFAULT_QUEUE_MD = ROOM_WORK_QUEUE_MD

# Status ladder (docs-facing).
STATUS_VALUES = frozenset(
    {
        "unstarted",
        "probe_state",
        "segment_scripted",
        "natural_chain",
        "blocker",
    }
)

# Queue tiers: critical-path friction vs deferred side states.
TIER_VALUES = frozenset({"easy", "standard", "blocker", "later"})

_TIER_RANK = {"blocker": 0, "easy": 1, "standard": 2, "later": 3}

# Path phase for Sanctuary ordering (lower = earlier on the route / work sooner).
# Continuous tip is room 0x50; key/shutter and room 0x55 exit are alternate.
_PHASE = {
    "frontier": 0,  # continuous tip: room 0x50 NW chamber
    "zelda": 1,  # Zelda cell / follower approach
    "b1_path": 2,  # B1 legs on the Zelda approach
    "b1": 3,  # general B1 practice
    "main": 4,  # completed prefix / east-wing exploration
    "key_shutter": 5,  # demoted alternate internal-key path
    "post_sword": 6,  # secret-entrance clear done; not continuous tip
    "room_55": 6,  # alternate / historical; not primary blocker
    "escort": 7,  # after follower
    "room": 8,
    "b2": 9,
    "b3": 10,
    "opening": 11,  # already largely done; low new-work priority
    "unknown": 12,
}

_ROOM_RE = re.compile(r"^Castle(?:Room|_|)([0-9A-Fa-f]{1,2})$")
_ROOM_SUFFIX_RE = re.compile(r"^CastleRoom([0-9A-Fa-f]{1,2})")


@dataclass(frozen=True)
class WorkItem:
    """One save-state checkpoint on the opening / Sanctuary path."""

    state_name: str
    group: str
    tier: str
    goal: str
    status: str
    notes: str = ""
    predecessor: str | None = None
    acceptance_ram: dict[str, Any] = field(default_factory=dict)
    phase: int = 9
    # Stable sort key components filled by ranker.
    rank: int = 0
    rank_score: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def list_state_names(integration_dir: Path | None = None) -> list[str]:
    """Return sorted basenames (no ``.state``) for all save states on disk."""
    root = integration_dir or INTEGRATION_DIR
    if not root.is_dir():
        return []
    names = sorted(p.stem for p in root.glob("*.state") if p.is_file())
    return names


# ---------------------------------------------------------------------------
# Heuristic classification
# ---------------------------------------------------------------------------


def _parse_room_id(state_name: str) -> int | None:
    if state_name == "Castle_55":
        return 0x55
    m = _ROOM_SUFFIX_RE.match(state_name) or _ROOM_RE.match(state_name)
    if not m:
        return None
    try:
        return int(m.group(1), 16) if len(m.group(1)) <= 2 else int(m.group(1), 10)
    except ValueError:
        try:
            return int(m.group(1), 10)
        except ValueError:
            return None


def classify_group(state_name: str) -> str:
    """Map a state filename to a coarse segment group."""
    n = state_name
    if n in {
        "YazeSlot000",
        "LinksHouseWake",
        "FirstAction",
        "HyruleCastleGrounds",
    }:
        return "opening"
    if n.startswith("FighterSword"):
        return "post_sword"
    if n in {"Castle_55"} or n.startswith("Castle_55"):
        return "room_55"
    if n.startswith("CastleMantle"):
        return "escort"
    if n.startswith("CastleZelda") or "Zelda" in n and n.startswith("Castle"):
        if "Mantle" in n:
            return "escort"
        return "zelda"
    if n.startswith("CastleB3"):
        return "b3"
    if n.startswith("CastleB2"):
        return "b2"
    if n.startswith("CastleB1"):
        # Key/shutter states are alternate internal-key practice, not tip blockers.
        low = n.lower()
        if "key" in low or "shutter" in low:
            return "key_shutter"
        return "b1"
    if n.startswith("CastleMain"):
        if "Zelda" in n:
            return "zelda"
        return "main"
    if n.startswith("CastleRoom") or n.startswith("Castle_"):
        rid = _parse_room_id(n)
        if rid == 0x55:
            return "room_55"
        if rid == 0x50:
            return "frontier"
        return "room"
    return "unknown"


def _default_goal(state_name: str, group: str) -> str:
    if group == "opening":
        if state_name == "YazeSlot000":
            return "boot_title"
        if state_name == "LinksHouseWake":
            return "exit_links_house"
        if state_name == "HyruleCastleGrounds":
            return "reach_secret_hole"
        return "opening_progress"
    if group == "post_sword":
        return "secret_entrance_clear"
    if group == "room_55":
        return "exit_0x55"
    if group == "key_shutter":
        if "Key" in state_name:
            return "obtain_key"
        return "open_shutter"
    if group == "zelda":
        if "Follower" in state_name:
            return "zelda_follower"
        return "reach_zelda_cell"
    if group == "escort":
        return "sanctuary"
    if group == "b3":
        return "ball_and_chain"
    if group == "b2":
        return "traverse_b2"
    if group == "b1":
        return "traverse_b1"
    if group == "frontier":
        return "discover_after_0x50"
    if group == "main":
        return "main_hall_to_zelda"
    rid = _parse_room_id(state_name)
    if rid is not None:
        return f"clear_room_0x{rid:02X}"
    return "probe"


def _default_tier(group: str, status: str) -> str:
    if status == "blocker":
        return "blocker"
    if group in {"opening"}:
        return "easy"
    if group in {"frontier", "main", "zelda"}:
        return "standard"
    if group in {"key_shutter", "post_sword", "room_55"}:
        # Alternate / done segments — not primary continuous-tip blockers.
        return "later" if group == "key_shutter" else "standard"
    if group in {"escort", "b3"}:
        return "later"
    if group in {"b1", "b2", "room"}:
        return "standard"
    return "standard"


def _default_status(group: str) -> str:
    """Conservative default when no curated fact is known."""
    if group == "opening":
        return "probe_state"
    if group in {"post_sword", "room_55"}:
        return "probe_state"
    return "unstarted"


def _default_acceptance(group: str, state_name: str) -> dict[str, Any]:
    if group == "opening":
        if state_name == "HyruleCastleGrounds":
            return {"on_castle_grounds": True}
        return {"has_control": True}
    if group in {"post_sword", "room_55"}:
        return {
            "has_fighter_sword": True,
            "in_secret_passage": True,
        }
    if group == "key_shutter":
        return {"has_fighter_sword": True, "dungeon_key_count": ">=1"}
    if group == "main":
        return {"indoors_room": "0x61", "has_control": True}
    if group == "frontier":
        return {"indoors_room": "0x50", "has_control": True}
    if group == "zelda":
        return {"has_zelda_follower": True}
    if group == "escort":
        return {"has_zelda_follower": True, "in_sanctuary": True}
    if group == "b3":
        return {"has_control": True}
    return {}


def _default_predecessor(state_name: str, group: str) -> str | None:
    if state_name == "LinksHouseWake":
        return "YazeSlot000"
    if state_name == "FirstAction":
        return "LinksHouseWake"
    if state_name == "HyruleCastleGrounds":
        return "LinksHouseWake"
    if state_name == "FighterSword":
        return "HyruleCastleGrounds"
    if state_name == "FighterSwordLamp":
        return "FighterSword"
    if state_name == "Castle_55":
        return "HyruleCastleGrounds"
    if group == "main":
        return "FighterSword"
    if group == "frontier":
        return "CastleMain"
    if (
        group in {"room_55", "key_shutter", "post_sword"}
        and state_name != "FighterSword"
    ):
        return "FighterSword"
    if group == "b1":
        return "CastleMain"
    if group == "zelda":
        return "CastleMain"
    if group == "escort":
        return "CastleZeldaFollower"
    if group in {"b2", "b3"}:
        return "CastleB1South"
    return None


# ---------------------------------------------------------------------------
# Curated facts from docs/STATUS (override heuristics)
# ---------------------------------------------------------------------------


def curated_overrides() -> dict[str, dict[str, Any]]:
    """Known statuses/notes from STATUS.md and verified segments.

    Keys are ``state_name``. Only fields that should override heuristics.
    Continuous tip is NW chamber room 0x50; primary work is the next physical
    exit, then B1 → Zelda → escort.
    """
    return {
        "YazeSlot000": {
            "status": "natural_chain",
            "tier": "easy",
            "goal": "boot_title",
            "notes": "Boot / title entry; natural chain to castle grounds exists.",
            "acceptance_ram": {"module": "title_or_file"},
        },
        "LinksHouseWake": {
            "status": "segment_scripted",
            "tier": "easy",
            "goal": "exit_links_house",
            "notes": "Wake / exit house on proven button script.",
            "predecessor": "YazeSlot000",
        },
        "FirstAction": {
            "status": "probe_state",
            "tier": "easy",
            "goal": "opening_progress",
            "notes": "Dev first-action checkpoint.",
            "predecessor": "LinksHouseWake",
        },
        "HyruleCastleGrounds": {
            "status": "natural_chain",
            "tier": "easy",
            "goal": "reach_secret_hole",
            "notes": (
                "Boot path natural progress to screen 0x1B; approach near_secret_hole "
                "~(2430,1704). Predecessor for castle_to_sword."
            ),
            "predecessor": "LinksHouseWake",
            "acceptance_ram": {"on_castle_grounds": True, "overworld_screen": "0x1B"},
        },
        "FighterSword": {
            "status": "segment_scripted",
            "tier": "standard",
            "goal": "secret_entrance_clear",
            "notes": (
                "Dev checkpoint after uncle/fighter sword (castle_to_sword). "
                "Secret-entrance clear (stairs → outdoor pocket → courtyard) is "
                "continuous via outdoor path; not the continuous tip. Tip is NW "
                "chamber room 0x50 after castle_dungeon_prefix."
            ),
            "predecessor": "HyruleCastleGrounds",
            "acceptance_ram": {
                "has_fighter_sword": True,
                "in_secret_passage": True,
            },
        },
        "FighterSwordLamp": {
            "status": "probe_state",
            "tier": "standard",
            "goal": "secret_entrance_clear",
            "notes": (
                "Sword + lamp checkpoint; secret entrance clear is continuous. "
                "Not the continuous tip (NW chamber 0x50)."
            ),
            "predecessor": "FighterSword",
            "acceptance_ram": {
                "has_fighter_sword": True,
                "has_lamp": True,
            },
        },
        "Castle_55": {
            "status": "segment_scripted",
            "tier": "standard",
            "goal": "exit_0x55",
            "notes": (
                "Secret passage room 0x55. Secret-entrance clear is continuous via "
                "outdoor path (stairs → pocket → courtyard → main door → 0x61 → 0x50). "
                "Internal key/shutter exit is alternate practice only — not a "
                "primary continuous-tip blocker."
            ),
            "predecessor": "HyruleCastleGrounds",
            "acceptance_ram": {
                "in_secret_passage": True,
                "has_fighter_sword": True,
            },
        },
        "CastleMain": {
            "status": "segment_scripted",
            "tier": "standard",
            "goal": "castle_dungeon_prefix",
            "notes": (
                "Development checkpoint for the now-continuous first-dungeon prefix "
                "0x61 → 0x60 → 0x50. The continuous tip is room 0x50."
            ),
            "predecessor": "FighterSword",
            "acceptance_ram": {"indoors_room": "0x61", "has_control": True},
        },
        "CastleMainEast": {
            "status": "probe_state",
            "tier": "standard",
            "goal": "east_wing_exploration",
            "notes": "East-wing exploration from main hall; not the continuous tip.",
            "predecessor": "CastleMain",
            "acceptance_ram": {"indoors_room": "0x61", "has_control": True},
        },
        "CastleMainZeldaReady": {
            "status": "probe_state",
            "tier": "standard",
            "goal": "reach_zelda_cell",
            "notes": (
                "State-local Zelda-ready probe from main hall; not evidence of the "
                "physical route after continuous-tip room 0x50."
            ),
            "predecessor": "CastleMain",
            "phase": 1,
        },
        "CastleMainZeldaBoomerang": {
            "status": "probe_state",
            "tier": "standard",
            "goal": "reach_zelda_cell",
            "notes": (
                "Main-hall Zelda path with boomerang loadout; state-local probe, "
                "not continuous-tip evidence."
            ),
            "predecessor": "CastleMain",
            "phase": 1,
        },
        "CastleRoom50": {
            "status": "probe_state",
            "tier": "blocker",
            "goal": "discover_after_0x50",
            "notes": (
                "Continuous tip / natural-entry frontier. Isolate the next physical "
                "exit from room 0x50 before asserting a Zelda route."
            ),
            "predecessor": "CastleMain",
            "acceptance_ram": {"indoors_room": "0x50", "has_control": True},
        },
        "CastleB1Key": {
            "status": "probe_state",
            "tier": "standard",
            "goal": "obtain_key",
            "notes": (
                "Alternate internal_key practice from 0x55 / B1 — not the primary "
                "continuous-tip blocker. Primary route now reaches room 0x50 first."
            ),
            "predecessor": "FighterSword",
            "acceptance_ram": {"dungeon_key_count": ">=1"},
        },
        "CastleB1SecondKey": {
            "status": "probe_state",
            "tier": "later",
            "goal": "obtain_key",
            "notes": "Second key probe; alternate internal_key ladder, not tip.",
            "predecessor": "CastleB1Key",
        },
        "CastleB1Shutter": {
            "status": "probe_state",
            "tier": "standard",
            "goal": "open_shutter",
            "notes": (
                "Shutter door on alternate internal key/shutter path — demoted vs "
                "main hall → Zelda."
            ),
            "predecessor": "CastleB1Key",
        },
        "CastleB1ShutterGuard": {
            "status": "probe_state",
            "tier": "later",
            "goal": "open_shutter",
            "notes": "Shutter room with guard; alternate practice, not tip blocker.",
            "predecessor": "CastleB1Shutter",
        },
        "CastleB1ShutterRoom": {
            "status": "probe_state",
            "tier": "later",
            "goal": "open_shutter",
            "notes": "Shutter room probe; alternate internal_key path.",
            "predecessor": "CastleB1Shutter",
        },
        "CastleZeldaFollower": {
            "status": "probe_state",
            "tier": "standard",
            "goal": "zelda_follower",
            "notes": (
                "Expected $F3CC==1 (Zelda tagalong). Not verified on natural path. "
                "State-local follower checkpoint; natural predecessor after room 0x50 "
                "is not verified."
            ),
            "predecessor": "CastleMain",
            "acceptance_ram": {"has_zelda_follower": True, "follower": 1},
        },
        "CastleMantleZelda": {
            "status": "unstarted",
            "tier": "later",
            "goal": "sanctuary",
            "notes": (
                "Escort / mantle checkpoint toward Sanctuary (room 0x12 / OW 0x13). "
                "Defer until follower==1 verified from main-hall Zelda path."
            ),
            "predecessor": "CastleZeldaFollower",
            "acceptance_ram": {
                "has_zelda_follower": True,
                "in_sanctuary": True,
            },
        },
        "CastleB3": {
            "status": "unstarted",
            "tier": "later",
            "goal": "ball_and_chain",
            "notes": "Ball-and-chain area; not on immediate main-hall → Zelda path.",
        },
        "CastleB3BallApproach": {
            "status": "unstarted",
            "tier": "later",
            "goal": "ball_and_chain",
            "notes": (
                "Ball-and-chain approach; Sanctuary escort is higher path priority "
                "only after Zelda follower."
            ),
        },
        "CastleB3BossOneHitBoomerang": {
            "status": "unstarted",
            "tier": "later",
            "goal": "ball_and_chain",
            "notes": "Boss one-hit probe; deferred vs main hall → Zelda.",
        },
    }


def build_item(
    state_name: str,
    *,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> WorkItem:
    """Build one work item from heuristics + optional curated overrides."""
    group = classify_group(state_name)
    status = _default_status(group)
    goal = _default_goal(state_name, group)
    notes = ""
    predecessor = _default_predecessor(state_name, group)
    acceptance = _default_acceptance(group, state_name)
    # Mark "Cleared"/"Done" filenames as probe_state (saved progress, not scripted).
    if any(tag in state_name for tag in ("Cleared", "Done", "Full", "Landing")):
        if status == "unstarted":
            status = "probe_state"
            notes = notes or "Named progress save; treat as probe until scripted."

    ov = (overrides or curated_overrides()).get(state_name) or {}
    if "group" in ov:
        group = str(ov["group"])
    if "status" in ov:
        status = str(ov["status"])
    if "goal" in ov:
        goal = str(ov["goal"])
    if "notes" in ov:
        notes = str(ov["notes"])
    if "predecessor" in ov:
        predecessor = ov["predecessor"]  # may be None
    if "acceptance_ram" in ov:
        acceptance = dict(ov["acceptance_ram"])  # type: ignore[arg-type]

    tier = str(ov["tier"]) if "tier" in ov else _default_tier(group, status)
    if tier not in TIER_VALUES:
        tier = "standard"
    if status not in STATUS_VALUES:
        status = "unstarted"

    phase = int(ov.get("phase", _PHASE.get(group, 9)))

    return WorkItem(
        state_name=state_name,
        group=group,
        tier=tier,
        goal=goal,
        status=status,
        notes=notes,
        predecessor=predecessor if predecessor else None,
        acceptance_ram=acceptance,
        phase=phase,
    )


def rank_score(item: WorkItem) -> int:
    """Lower score = higher on the Sanctuary work queue.

    Policy (continuous tip = NW chamber room 0x50):
    - Prefer the physical frontier, then Zelda goals (discover_after_0x50,
      reach_zelda_cell, zelda_follower) over exit_0x55 / obtain_key /
      open_shutter.
    - Demote key/shutter; boost the physical frontier and Zelda goals.
    - Escort / B3 / already-natural opening later.
    - Within a phase, unfinished work before natural_chain milestones.
    """
    status_boost = {
        "blocker": 0,
        "probe_state": 10,
        "unstarted": 20,
        "segment_scripted": 40,
        "natural_chain": 60,
    }
    # Opening natural_chain is catalog context, not next work.
    phase = item.phase
    if item.group == "opening" and item.status in {"natural_chain", "segment_scripted"}:
        phase = 12
    # Done post_sword / room_55 segments stay in their phase (not tip).
    if item.group in {"post_sword", "room_55"} and item.status == "segment_scripted":
        phase = max(phase, _PHASE.get(item.group, 5))

    score = phase * 1000
    score += _TIER_RANK.get(item.tier, 2) * 100
    score += status_boost.get(item.status, 25)
    # Prefer the continuous-tip physical frontier and Zelda goals.
    if item.goal in {
        "discover_after_0x50",
        "reach_zelda_cell",
        "zelda_follower",
    }:
        score -= 10
    # Demote alternate internal key/shutter and historical 0x55 exit goals.
    if item.goal in {"exit_0x55", "obtain_key", "open_shutter"}:
        score += 15
    if item.goal == "secret_entrance_clear":
        score += 10
    if item.goal == "sanctuary":
        score += 50
    # Stable tie-break by name (applied in rank_items sort, not here).
    return score


def rank_items(items: Sequence[WorkItem]) -> list[WorkItem]:
    """Return new WorkItems with rank and rank_score filled, sorted ascending."""
    decorated: list[tuple[int, str, WorkItem]] = []
    for item in items:
        score = rank_score(item)
        decorated.append((score, item.state_name, item))
    decorated.sort(key=lambda t: (t[0], t[1]))
    ranked: list[WorkItem] = []
    for index, (score, _name, item) in enumerate(decorated, start=1):
        ranked.append(
            WorkItem(
                state_name=item.state_name,
                group=item.group,
                tier=item.tier,
                goal=item.goal,
                status=item.status,
                notes=item.notes,
                predecessor=item.predecessor,
                acceptance_ram=dict(item.acceptance_ram),
                phase=item.phase,
                rank=index,
                rank_score=score,
            )
        )
    return ranked


def build_catalog(
    *,
    integration_dir: Path | None = None,
    state_names: Sequence[str] | None = None,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[WorkItem]:
    """Enumerate states and return ranked work items."""
    names = (
        list(state_names)
        if state_names is not None
        else list_state_names(integration_dir)
    )
    ov = overrides if overrides is not None else curated_overrides()
    items = [build_item(name, overrides=ov) for name in names]
    return rank_items(items)


def build_work_queue(
    *,
    integration_dir: Path | None = None,
    state_names: Sequence[str] | None = None,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build machine-readable queue payload."""
    items = build_catalog(
        integration_dir=integration_dir,
        state_names=state_names,
        overrides=overrides,
    )
    by_status = Counter(i.status for i in items)
    by_group = Counter(i.group for i in items)
    by_tier = Counter(i.tier for i in items)
    by_goal = Counter(i.goal for i in items)

    focus = [
        i
        for i in items
        if i.status in {"blocker", "probe_state", "unstarted"}
        and i.group in {"frontier", "zelda", "b1"}
    ][:12]

    return {
        "schemaVersion": SCHEMA_VERSION,
        "catalogId": CATALOG_ID,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "unitNote": (
            "Units are Zelda3-Snes save states on the boot → fighter sword → "
            "secret-entrance clear → courtyard pocket → main hall → NW chamber "
            "room 0x50 → Zelda → Sanctuary path. Continuous tip is **room 0x50**; "
            "next work is the physical exit after 0x50, then B1 → Zelda cell → "
            "follower → escort. Internal 0x55 "
            "key/shutter is alternate practice only. Sanctuary not claimed."
        ),
        "source": {
            "integrationDir": str(integration_dir or INTEGRATION_DIR),
            "stateCount": len(items),
        },
        "summary": {
            "stateCount": len(items),
            "byStatus": dict(sorted(by_status.items())),
            "byGroup": dict(sorted(by_group.items())),
            "byTier": dict(sorted(by_tier.items())),
            "byGoal": dict(sorted(by_goal.items())),
            "workFocusCount": len(focus),
            "sanctuaryClaimed": False,
            "verifiedMilestones": [
                "title_to_castle_grounds",
                "castle_to_fighter_sword",
                "secret_entrance_clear",
                "pocket_to_main_hall_0x61",
                "castle_dungeon_prefix_0x50",
            ],
        },
        "workFocus": [i.to_dict() for i in focus],
        "items": [i.to_dict() for i in items],
    }


def work_queue_to_markdown(payload: Mapping[str, Any]) -> str:
    """Render a short human-readable queue doc."""
    summary = payload.get("summary") or {}
    items = list(payload.get("items") or [])
    focus = list(payload.get("workFocus") or [])
    generated = str(payload.get("generatedAt", ""))

    lines: list[str] = [
        "# ALTTP — Room / Save-State Work Queue",
        "",
        "Sanctuary-path practice queue for `Zelda3-Snes` save states.",
        "Continuous tip is **NW chamber room 0x50** (after `castle_dungeon_prefix`).",
        "Ranked for next work: **physical exit after 0x50 → B1 → Zelda cell → follower → escort**.",
        "Internal 0x55 key/shutter path is **alternate practice**, not primary.",
        "",
        f"Generated: `{generated}`",
        f"Catalog: `{payload.get('catalogId')}` schema {payload.get('schemaVersion')}",
        f"States: **{summary.get('stateCount', len(items))}**",
        f"Sanctuary claimed: **{summary.get('sanctuaryClaimed', False)}**",
        "",
        "## Regenerate",
        "",
        "```bash",
        "uv run python alttp/scripts/export_work_queue.py",
        "uv run python alttp/scripts/export_work_queue.py --json",
        "```",
        "",
        "Artifacts: `docs/routes/ROOM_WORK_QUEUE.md` · "
        "`recordings/room_work_queue.json`.",
        "",
        "## Work focus (next toward Sanctuary)",
        "",
    ]

    if focus:
        lines.append("| Rank | State | Group | Goal | Status | Tier | Notes |")
        lines.append("|-----:|-------|-------|------|--------|------|-------|")
        for row in focus:
            notes = str(row.get("notes") or "").replace("|", "\\|")
            if len(notes) > 80:
                notes = notes[:77] + "..."
            lines.append(
                f"| {row.get('rank')} | `{row.get('state_name')}` | "
                f"{row.get('group')} | {row.get('goal')} | {row.get('status')} | "
                f"{row.get('tier')} | {notes} |"
            )
    else:
        lines.append("_No open continuous-tip items._")

    lines.extend(
        [
            "",
            "## Status summary",
            "",
            "```",
            f"byStatus: {summary.get('byStatus')}",
            f"byGroup:  {summary.get('byGroup')}",
            f"byTier:   {summary.get('byTier')}",
            "```",
            "",
            "Verified milestones (docs): "
            + ", ".join(summary.get("verifiedMilestones") or []),
            "",
            "## Full ranked table",
            "",
            "| Rank | State | Group | Goal | Status | Tier | Predecessor |",
            "|-----:|-------|-------|------|--------|------|-------------|",
        ]
    )

    for row in items:
        pred = row.get("predecessor") or ""
        if pred:
            pred = f"`{pred}`"
        lines.append(
            f"| {row.get('rank')} | `{row.get('state_name')}` | "
            f"{row.get('group')} | {row.get('goal')} | {row.get('status')} | "
            f"{row.get('tier')} | {pred} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Continuous tip is **NW chamber room 0x50** after "
            "`castle_dungeon_prefix` (courtyard pocket → main door → 0x60 → 0x50).",
            "- Secret-entrance clear (stairs → outdoor pocket) is already "
            "continuous; do **not** treat `Castle_55` internal exit as the top "
            "blocker.",
            "- Primary next work: physical exit after 0x50 / B1 → Zelda cell → follower → "
            "escort → Sanctuary.",
            "- Internal 0x55 key/shutter path is **alternate practice** only.",
            "- `FighterSword` is a **dev checkpoint** after uncle sword; natural "
            "sword claim needs `--natural` on `castle_to_sword`.",
            "- Acceptance for rescue: `has_zelda_follower` (`$F3CC == 1`).",
            "- Sanctuary: room `0x12` / OW screen `0x13` — not claimed.",
            "",
            str(payload.get("unitNote") or ""),
            "",
        ]
    )
    return "\n".join(lines)


def export_work_queue(
    *,
    json_output: Path | None = DEFAULT_QUEUE_JSON,
    md_output: Path | None = DEFAULT_QUEUE_MD,
    integration_dir: Path | None = None,
) -> dict[str, Any]:
    """Build queue and optionally write JSON + markdown artifacts."""
    payload = build_work_queue(integration_dir=integration_dir)
    if json_output is not None:
        json_output = Path(json_output)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
    if md_output is not None:
        md_output = Path(md_output)
        md_output.parent.mkdir(parents=True, exist_ok=True)
        md_output.write_text(work_queue_to_markdown(payload), encoding="utf-8")
    return payload


def top_items(
    payload: Mapping[str, Any] | None = None,
    *,
    n: int = 15,
    integration_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Return the first *n* ranked items (convenience for CLI)."""
    data = (
        payload
        if payload is not None
        else build_work_queue(integration_dir=integration_dir)
    )
    return list(data.get("items") or [])[:n]


__all__ = [
    "CATALOG_ID",
    "DEFAULT_QUEUE_JSON",
    "DEFAULT_QUEUE_MD",
    "SCHEMA_VERSION",
    "WorkItem",
    "build_catalog",
    "build_item",
    "build_work_queue",
    "classify_group",
    "curated_overrides",
    "export_work_queue",
    "list_state_names",
    "rank_items",
    "rank_score",
    "top_items",
    "work_queue_to_markdown",
]
