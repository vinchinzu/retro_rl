"""Practice-hack repertoire catalog: sessions, route order, neighbors."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from super_metroid.paths import (
    GAME_DIR,
    INTEGRATION_DIR,
    PRACTICE_REPERTOIRE_DEMO_DIR,
    PRACTICE_REPERTOIRE_PATH,
    PRACTICE_REPERTOIRE_STATE_DIR,
)

# Product spine: practice-hack KPDR Early Ice (menu index 4 in mainmenu.asm).
PRODUCT_CATEGORY = "kpdr25"
PRODUCT_ROUTE_ID = "kpdr"

REACTIVE_POLICY_DIR = GAME_DIR / "policies" / "reactive_rooms"
REACTIVE_PLAN_DIR = REACTIVE_POLICY_DIR / "plans"

# Graduation ladder (low → high). Aligns with reactive policy statuses where
# possible; product_spine is harness-route promotion beyond a single room skill.
GRADES = (
    "none",
    "draft",  # named pin or hop body only
    "candidate",  # policy JSON written, not dual-green
    "verified_live_anchor",  # reactive policy status / dual-green hop
    "product_spine",  # living full_start / continuous tip pin
)


def _safe_id(session_id: str) -> str:
    return session_id.replace("/", "__")


def _parse_hex_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    s = str(value).strip()
    if not s:
        return None
    return int(s, 0)


@dataclass(frozen=True)
class RepertoireSession:
    """One practice-hack preset leaf (menu entry + WRAM fingerprint)."""

    id: str
    category: str
    area: str
    slug: str
    name: str
    room_id: int | None = None
    items: int | None = None
    beams: int | None = None
    x: int | None = None
    y: int | None = None
    pose: int | None = None
    menu_label: str = ""
    data_label: str = ""
    route_index: int = -1

    @property
    def room_hex(self) -> str | None:
        return f"0x{self.room_id:04X}" if self.room_id is not None else None

    @property
    def canonical_state_path(self) -> Path:
        return PRACTICE_REPERTOIRE_STATE_DIR / f"{self.id}.state"

    @property
    def canonical_demo_stem(self) -> Path:
        return PRACTICE_REPERTOIRE_DEMO_DIR / self.id

    def product_map(self) -> dict[str, str] | None:
        # Lazy import avoids catalog ↔ spine cycle (PRODUCT_SESSION_MAP lives in spine).
        from super_metroid.practice_repertoire.spine import PRODUCT_SESSION_MAP

        return PRODUCT_SESSION_MAP.get(self.id)

    def living_state_path(self) -> Path | None:
        """Resolved harness pin if product map names a state under integration."""
        m = self.product_map()
        if not m or "state" not in m:
            return None
        return INTEGRATION_DIR / m["state"]

    def resolve_state_path(self) -> Path | None:
        """Best available boot pin: living map → canonical repertoire state."""
        living = self.living_state_path()
        if living is not None and living.is_file():
            return living
        if self.canonical_state_path.is_file():
            return self.canonical_state_path
        if living is not None:
            return living  # expected path even if missing
        return self.canonical_state_path

    def policy_plan_path(self) -> Path:
        return REACTIVE_PLAN_DIR / f"{_safe_id(self.id)}.json"

    def policy_json_glob(self) -> str:
        """Glob under reactive_rooms for room-local policies (room hex)."""
        if self.room_hex is None:
            return "*.json"
        return f"room_{self.room_hex[2:].lower()}_*.json"

    def fingerprint(self) -> dict[str, Any]:
        out: dict[str, Any] = {"session_id": self.id, "name": self.name}
        if self.room_id is not None:
            out["room_id"] = self.room_id
            out["room_hex"] = self.room_hex
        if self.items is not None:
            out["items"] = self.items
            out["items_hex"] = f"0x{self.items:04X}"
        if self.beams is not None:
            out["beams"] = self.beams
            out["beams_hex"] = f"0x{self.beams:04X}"
        for k in ("x", "y", "pose"):
            v = getattr(self, k)
            if v is not None:
                out[k] = v
        return out

    @classmethod
    def from_record(cls, rec: dict[str, Any], *, route_index: int = -1) -> RepertoireSession:
        return cls(
            id=str(rec["id"]),
            category=str(rec["category"]),
            area=str(rec["area"]),
            slug=str(rec["slug"]),
            name=str(rec["name"]),
            room_id=rec.get("room_id"),
            items=rec.get("items"),
            beams=rec.get("beams"),
            x=rec.get("x"),
            y=rec.get("y"),
            pose=rec.get("pose"),
            menu_label=str(rec.get("menu_label") or ""),
            data_label=str(rec.get("data_label") or ""),
            route_index=route_index,
        )


@lru_cache(maxsize=1)
def load_catalog(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else PRACTICE_REPERTOIRE_PATH
    if not p.is_file():
        raise FileNotFoundError(
            f"practice repertoire catalog missing: {p}\n"
            "Run: uv run python snes/super_metroid/scripts/export/practice_repertoire.py"
        )
    return json.loads(p.read_text(encoding="utf-8"))


def categories() -> list[dict[str, Any]]:
    return list(load_catalog()["categories"])


def sessions(
    *,
    category: str | None = None,
    area: str | None = None,
) -> list[RepertoireSession]:
    out: list[RepertoireSession] = []
    for rec in load_catalog()["sessions"]:
        if category and rec["category"] != category:
            continue
        if area and rec["area"] != area:
            continue
        out.append(RepertoireSession.from_record(rec))
    return out


def route_sessions(category: str = PRODUCT_CATEGORY) -> list[RepertoireSession]:
    """Ordered product route (practice-hack menu order = route-edge order)."""
    rows = sessions(category=category)
    return [replace(s, route_index=i) for i, s in enumerate(rows)]


@lru_cache(maxsize=8)
def _route_index(category: str) -> tuple[str, ...]:
    return tuple(s.id for s in route_sessions(category))


def get_session(session_id: str) -> RepertoireSession:
    for s in sessions():
        if s.id != session_id:
            continue
        try:
            idx = _route_index(s.category).index(s.id)
        except ValueError:
            return s
        return replace(s, route_index=idx)
    raise KeyError(f"unknown repertoire session {session_id!r}")


def neighbors(
    session_id: str,
    *,
    category: str | None = None,
) -> tuple[RepertoireSession | None, RepertoireSession | None]:
    """Previous and next sessions on the category route (route edges)."""
    s = get_session(session_id)
    cat = category or s.category
    route = route_sessions(cat)
    ids = [r.id for r in route]
    try:
        i = ids.index(session_id)
    except ValueError:
        return None, None
    prev_s = route[i - 1] if i > 0 else None
    next_s = route[i + 1] if i + 1 < len(route) else None
    return prev_s, next_s
