"""Sanctuary-path save-state work queue for ALttP.

Discover ``.state`` files, merge curated facts from ``work_queue_data.yaml``,
rank by path tags + policy, export JSON/markdown. Python stays thin; prose
and overrides live in data.

Tip: NW chamber **0x50** (east→0x01 natural_entry). Next: B1 stairs after
0x01 chain → Zelda → Sanctuary. Key/shutter is alternate practice only.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from functools import lru_cache
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from alttp.opening_route.escape_graph import PATH_INTERNAL_KEY, PATH_PRIMARY
from alttp.paths import INTEGRATION_DIR, ROOM_WORK_QUEUE_JSON, ROOM_WORK_QUEUE_MD

CATALOG_ID = "alttp_sanctuary_work_queue"
SCHEMA_VERSION = 1
DEFAULT_QUEUE_JSON = ROOM_WORK_QUEUE_JSON
DEFAULT_QUEUE_MD = ROOM_WORK_QUEUE_MD
DATA_PATH = Path(__file__).with_name("work_queue_data.yaml")

STATUS_VALUES = frozenset(
    {"unstarted", "probe_state", "segment_scripted", "natural_chain", "blocker"}
)
TIER_VALUES = frozenset({"easy", "standard", "blocker", "later"})
_ROOM_RE = re.compile(r"^Castle(?:Room|_|)([0-9A-Fa-f]{1,2})$")
_ROOM_SUFFIX_RE = re.compile(r"^CastleRoom([0-9A-Fa-f]{1,2})")
_PROGRESS_TAGS = ("Cleared", "Done", "Full", "Landing")
_OPENING = frozenset(
    {"YazeSlot000", "LinksHouseWake", "FirstAction", "HyruleCastleGrounds"}
)


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
    path_tag: str = PATH_PRIMARY
    rank: int = 0
    rank_score: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@lru_cache(maxsize=1)
def load_queue_data(path: str | None = None) -> dict[str, Any]:
    """Load ``work_queue_data.yaml`` (catalog prose, policy, overrides)."""
    p = Path(path) if path else DATA_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"work queue data must be a mapping: {p}")
    return raw


def curated_overrides(
    data: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Curated per-state facts from data (status/notes/goals/…)."""
    src = data if data is not None else load_queue_data()
    raw = src.get("overrides") or {}
    return {str(k): dict(v) for k, v in raw.items() if isinstance(v, Mapping)}


def list_state_names(integration_dir: Path | None = None) -> list[str]:
    """Return sorted basenames (no ``.state``) for all save states on disk."""
    root = integration_dir or INTEGRATION_DIR
    if not root.is_dir():
        return []
    return sorted(p.stem for p in root.glob("*.state") if p.is_file())


def _parse_room_id(state_name: str) -> int | None:
    if state_name == "Castle_55":
        return 0x55
    m = _ROOM_SUFFIX_RE.match(state_name) or _ROOM_RE.match(state_name)
    if not m:
        return None
    token = m.group(1)
    try:
        return int(token, 16) if len(token) <= 2 else int(token, 10)
    except ValueError:
        try:
            return int(token, 10)
        except ValueError:
            return None


def classify_group(state_name: str) -> str:
    """Map a state filename to a coarse segment group."""
    n = state_name
    if n in _OPENING:
        return "opening"
    if n.startswith("FighterSword"):
        return "post_sword"
    if n == "Castle_55" or n.startswith("Castle_55"):
        return "room_55"
    if n.startswith("CastleMantle"):
        return "escort"
    if n.startswith("CastleZelda") or ("Zelda" in n and n.startswith("Castle")):
        return "escort" if "Mantle" in n else "zelda"
    if n.startswith("CastleB3"):
        return "b3"
    if n.startswith("CastleB2"):
        return "b2"
    if n.startswith("CastleB1"):
        low = n.lower()
        return "key_shutter" if ("key" in low or "shutter" in low) else "b1"
    if n.startswith("CastleMain"):
        return "zelda" if "Zelda" in n else "main"
    if n.startswith("CastleRoom") or n.startswith("Castle_"):
        rid = _parse_room_id(n)
        if rid == 0x55:
            return "room_55"
        return "frontier" if rid == 0x50 else "room"
    return "unknown"


def _refine_goal(state_name: str, group: str, goal: str) -> str:
    if group == "key_shutter" and "Key" in state_name:
        return "obtain_key"
    if group == "zelda" and "Follower" in state_name:
        return "zelda_follower"
    if group == "room":
        rid = _parse_room_id(state_name)
        if rid is not None:
            return f"clear_room_0x{rid:02X}"
    return goal


def build_item(
    state_name: str,
    *,
    data: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> WorkItem:
    """Build one work item: group defaults + state defaults + curated overrides."""
    src = data if data is not None else load_queue_data()
    group = classify_group(state_name)
    gdef = dict((src.get("groupDefaults") or {}).get(group) or {})
    sdef = dict((src.get("stateDefaults") or {}).get(state_name) or {})
    ov_map = overrides if overrides is not None else curated_overrides(src)
    ov = dict(ov_map.get(state_name) or {})
    merged: dict[str, Any] = {**gdef, **sdef, **ov}
    if "group" in merged:
        group = str(merged["group"])

    status = str(merged.get("status") or "unstarted")
    goal = str(merged.get("goal") or "probe")
    if "goal" not in ov:
        goal = _refine_goal(state_name, group, goal)
    notes = str(merged.get("notes") or "")
    predecessor = merged.get("predecessor") or None
    acceptance = dict(merged.get("acceptance_ram") or {})

    if any(t in state_name for t in _PROGRESS_TAGS) and status == "unstarted":
        status = "probe_state"
        notes = notes or "Named progress save; treat as probe until scripted."

    tier = str(merged.get("tier") or "standard")
    if tier not in TIER_VALUES:
        tier = "standard"
    if status not in STATUS_VALUES:
        status = "unstarted"

    phases = src.get("phases") or {}
    phase = int(merged.get("phase", phases.get(group, 9)))
    path_tag = str((src.get("pathTags") or {}).get(group, PATH_PRIMARY))

    return WorkItem(
        state_name=state_name,
        group=group,
        tier=tier,
        goal=goal,
        status=status,
        notes=notes,
        predecessor=predecessor,
        acceptance_ram=acceptance,
        phase=phase,
        path_tag=path_tag,
    )


def rank_score(
    item: WorkItem, *, data: Mapping[str, Any] | None = None
) -> int:
    """Lower score = higher on the Sanctuary work queue."""
    policy = (data if data is not None else load_queue_data()).get("rank") or {}
    phase = item.phase
    if item.group == "opening" and item.status in {
        "natural_chain",
        "segment_scripted",
    }:
        phase = int(policy.get("openingDonePhase", 12))
    done_min = policy.get("doneGroupsMinPhase") or {}
    if item.group in done_min and item.status == "segment_scripted":
        phase = max(phase, int(done_min[item.group]))

    score = phase * 1000
    score += int((policy.get("tierRank") or {}).get(item.tier, 2)) * 100
    score += int((policy.get("statusBoost") or {}).get(item.status, 25))
    if item.goal in set(policy.get("goalPrefer") or ()):
        score += int(policy.get("goalPreferDelta", -10))
    score += int((policy.get("goalDemote") or {}).get(item.goal, 0))
    if item.path_tag == PATH_INTERNAL_KEY:
        score += int(policy.get("internalKeyDemote", 15))
    return score


def rank_items(
    items: Sequence[WorkItem], *, data: Mapping[str, Any] | None = None
) -> list[WorkItem]:
    """Return WorkItems with rank / rank_score filled, sorted ascending."""
    src = data if data is not None else load_queue_data()
    decorated = sorted(
        ((rank_score(i, data=src), i.state_name, i) for i in items),
        key=lambda t: (t[0], t[1]),
    )
    return [
        replace(item, rank=idx, rank_score=score)
        for idx, (score, _n, item) in enumerate(decorated, start=1)
    ]


def build_catalog(
    *,
    integration_dir: Path | None = None,
    state_names: Sequence[str] | None = None,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
    data: Mapping[str, Any] | None = None,
) -> list[WorkItem]:
    """Enumerate states and return ranked work items."""
    src = data if data is not None else load_queue_data()
    names = (
        list(state_names)
        if state_names is not None
        else list_state_names(integration_dir)
    )
    ov = overrides if overrides is not None else curated_overrides(src)
    return rank_items(
        [build_item(n, data=src, overrides=ov) for n in names], data=src
    )


def build_work_queue(
    *,
    integration_dir: Path | None = None,
    state_names: Sequence[str] | None = None,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
    data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build machine-readable queue payload."""
    src = data if data is not None else load_queue_data()
    items = build_catalog(
        integration_dir=integration_dir,
        state_names=state_names,
        overrides=overrides,
        data=src,
    )
    focus_groups = set(src.get("workFocusGroups") or {"frontier", "zelda", "b1"})
    focus = [
        i
        for i in items
        if i.status in {"blocker", "probe_state", "unstarted"}
        and i.group in focus_groups
    ][:12]
    return {
        "schemaVersion": int(src.get("schemaVersion") or SCHEMA_VERSION),
        "catalogId": str(src.get("catalogId") or CATALOG_ID),
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "unitNote": str(src.get("unitNote") or ""),
        "source": {
            "integrationDir": str(integration_dir or INTEGRATION_DIR),
            "dataPath": str(DATA_PATH),
            "stateCount": len(items),
        },
        "summary": {
            "stateCount": len(items),
            "byStatus": dict(sorted(Counter(i.status for i in items).items())),
            "byGroup": dict(sorted(Counter(i.group for i in items).items())),
            "byTier": dict(sorted(Counter(i.tier for i in items).items())),
            "byGoal": dict(sorted(Counter(i.goal for i in items).items())),
            "workFocusCount": len(focus),
            "sanctuaryClaimed": False,
            "verifiedMilestones": list(src.get("verifiedMilestones") or []),
        },
        "workFocus": [i.to_dict() for i in focus],
        "items": [i.to_dict() for i in items],
    }


def _clip_notes(text: str, limit: int = 80) -> str:
    notes = text.replace("|", "\\|")
    return notes if len(notes) <= limit else notes[: limit - 3] + "..."


def work_queue_to_markdown(payload: Mapping[str, Any]) -> str:
    """Render human-readable queue doc from payload + data prose."""
    src = load_queue_data()
    summary = payload.get("summary") or {}
    items = list(payload.get("items") or [])
    focus = list(payload.get("workFocus") or [])
    lines = [
        "# ALTTP — Room / Save-State Work Queue",
        "",
        *list(src.get("mdIntro") or []),
        "",
        f"Generated: `{payload.get('generatedAt', '')}`",
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
        "Source data: `opening_route/work_queue_data.yaml` "
        "(curated status/notes; Python discovers + ranks).",
        "",
        "## Work focus (next toward Sanctuary)",
        "",
    ]
    if focus:
        lines += [
            "| Rank | State | Group | Goal | Status | Tier | Notes |",
            "|-----:|-------|-------|------|--------|------|-------|",
        ]
        for row in focus:
            lines.append(
                f"| {row.get('rank')} | `{row.get('state_name')}` | "
                f"{row.get('group')} | {row.get('goal')} | {row.get('status')} | "
                f"{row.get('tier')} | {_clip_notes(str(row.get('notes') or ''))} |"
            )
    else:
        lines.append("_No open continuous-tip items._")
    lines += [
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
    for row in items:
        pred = f"`{row['predecessor']}`" if row.get("predecessor") else ""
        lines.append(
            f"| {row.get('rank')} | `{row.get('state_name')}` | "
            f"{row.get('group')} | {row.get('goal')} | {row.get('status')} | "
            f"{row.get('tier')} | {pred} |"
        )
    lines += ["", "## Notes", ""]
    lines += [f"- {n}" for n in (src.get("mdNotes") or [])]
    lines += ["", str(payload.get("unitNote") or ""), ""]
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
        out = Path(json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
    if md_output is not None:
        out = Path(md_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(work_queue_to_markdown(payload), encoding="utf-8")
    return payload


def top_items(
    payload: Mapping[str, Any] | None = None,
    *,
    n: int = 15,
    integration_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Return the first *n* ranked items (convenience for CLI)."""
    data = payload if payload is not None else build_work_queue(
        integration_dir=integration_dir
    )
    return list(data.get("items") or [])[:n]


__all__ = [
    "CATALOG_ID",
    "DATA_PATH",
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
    "load_queue_data",
    "rank_items",
    "rank_score",
    "top_items",
    "work_queue_to_markdown",
]
