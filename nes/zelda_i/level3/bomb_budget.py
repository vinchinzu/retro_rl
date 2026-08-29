"""L3 bomb budget from isolated Raft / Manhandla tapes — library only.

Not a spine close for ``rr-4d53.3.2``. Isolated ``--poke-bombs 16`` on
``run_level3_to_boss`` is recon (Level3Raft pin starts empty). Continuous
spine must carry ``.3.0`` 0x7c bombs or farm 0x5b Darknut drops. Never
write ``max_bombs``; do not add poke-16 to Survival.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from zelda_i.dungeon.ids import MANHANDLA_OBJECT_TYPE
from zelda_i.level3.dungeon import (
    ROOM_L3_BOMB_SHORTCUT,
    ROOM_L3_BOSS,
    ROOM_L3_COMPASS,
    ROOM_L3_DARKNUTS,
    ROOM_L3_ENTRY,
    ROOM_L3_WEST_DARKNUTS,
)
from zelda_i.level3.geometry import BOMB_STAND_59_RIGHT, BOMB_STAND_5B_RIGHT
from zelda_i.paths import RECORDINGS_DIR

Evidence = Literal["verified", "assumed"]

# --- Verified bomb-RIGHT walls (Raft → boss). One successful placement each. ---
# 0x5c walk-RIGHT@y141 bomb-R fallback is optional and is NOT in this spend.
L3_BOMB_R_59: tuple[int, int, tuple[int, int]] = (
    ROOM_L3_WEST_DARKNUTS,
    ROOM_L3_COMPASS,
    BOMB_STAND_59_RIGHT,
)
L3_BOMB_R_5B: tuple[int, int, tuple[int, int]] = (
    ROOM_L3_DARKNUTS,
    ROOM_L3_BOMB_SHORTCUT,
    BOMB_STAND_5B_RIGHT,
)
L3_BOMB_R_WALLS: tuple[tuple[int, int, tuple[int, int]], ...] = (
    L3_BOMB_R_59,
    L3_BOMB_R_5B,
)
L3_BOMB_WALL_SPEND: int = len(L3_BOMB_R_WALLS)  # 2; verified geometry

# Manhandla: live 5 heads type 0x3c HP64 on 0x4d. Bombs preferred.
# Per-head placement count is assumed — not a TAS-perfect spend.
MANHANDLA_HEADS_LIVE: int = 5
MANHANDLA_HEADS_EVIDENCE: Evidence = "verified"
MANHANDLA_BOMB_SPEND_ASSUMED: int = MANHANDLA_HEADS_LIVE
MANHANDLA_BOMB_SPEND_EVIDENCE: Evidence = "assumed"

L3_BOMB_BUDGET_VERIFIED: int = L3_BOMB_WALL_SPEND
L3_BOMB_BUDGET_ASSUMED: int = MANHANDLA_BOMB_SPEND_ASSUMED
L3_BOMB_BUDGET: int = L3_BOMB_BUDGET_VERIFIED + L3_BOMB_BUDGET_ASSUMED  # 7

# 0x5b Darknuts "bombs drop on clear" (LEVEL3_ROUTE interior). Natural farm
# for rr-4d53.3.2; isolated Raft did not pick any up.
L3_BOMB_FARM_ROOM: int = ROOM_L3_DARKNUTS

# Isolated / spine measurements (named; JSON parse is optional confirmation).
ISOLATED_RAFT_RECORDING: str = "level3_raft_assisted.json"
ISOLATED_TO_BOSS_RECORDING: str = "level3_to_boss_assisted.json"
SPINE_ENTRANCE_RECORDING: str = "l3_entrance_bombtopup.json"

ISOLATED_RAFT_BOMBS: int = 0  # Level3Raft pin starts empty
ISOLATED_TO_BOSS_POKE_BOMBS: int = 16  # recon; not a route claim
ISOLATED_POKE16_CLOSES_SPINE: bool = False

# Live spine predecessor rr-4d53.3.0 (Survival count top-up until rr-doua).
SPINE_0X7C_BOMBS: int = 8
SPINE_0X7C_KEYS: int = 4
SPINE_0X7C_ROOM: int = ROOM_L3_ENTRY
SPINE_L2_ENTRY_BOMBS: int = 0

# West raft path (0x5b LEFT → Compass → 0x59 → 0x69 → Raft) spends 0 bombs,
# so dest 0x5b carry equals Raft carry unless 0x5b Darknuts drop bombs.
L3_BOMBS_IN_AT_DEST_5B: int = L3_BOMB_BUDGET
L3_BOMBS_IN_AT_RAFT: int = L3_BOMB_BUDGET


@dataclass(frozen=True)
class BombSpendItem:
    """One planned placement (wall or boss heads)."""

    label: str
    room: int
    dest: int | None
    stand: tuple[int, int] | None
    count: int
    evidence: Evidence
    note: str


@dataclass(frozen=True)
class PlannedBombSpend:
    """Planned Raft→Manhandla bomb spend. Library accounting, not a spine close."""

    walls: tuple[BombSpendItem, ...]
    manhandla: BombSpendItem
    wall_bombs: int
    manhandla_bombs: int
    verified: int
    assumed: int
    total: int
    bombs_in_at_dest_5b: int
    bombs_in_at_raft: int
    isolated_raft_bombs: int
    isolated_poke16_closes_spine: bool

    def report(self) -> dict[str, Any]:
        return {
            "wall_bombs": self.wall_bombs,
            "manhandla_bombs": self.manhandla_bombs,
            "manhandla_evidence": self.manhandla.evidence,
            "verified": self.verified,
            "assumed": self.assumed,
            "total": self.total,
            "bombs_in_at_dest_5b": self.bombs_in_at_dest_5b,
            "bombs_in_at_raft": self.bombs_in_at_raft,
            "isolated_raft_bombs": self.isolated_raft_bombs,
            "isolated_poke16_closes_spine": False,
            "walls": [
                {
                    "label": w.label,
                    "room": w.room,
                    "dest": w.dest,
                    "stand": list(w.stand) if w.stand is not None else None,
                    "count": w.count,
                    "evidence": w.evidence,
                }
                for w in self.walls
            ],
        }


def planned_bomb_spend() -> PlannedBombSpend:
    """Bomb-R 0x59 + bomb-R 0x5b (verified) + Manhandla heads estimate (assumed)."""
    walls = (
        BombSpendItem(
            label="bomb_r_0x59",
            room=L3_BOMB_R_59[0],
            dest=L3_BOMB_R_59[1],
            stand=L3_BOMB_R_59[2],
            count=1,
            evidence="verified",
            note="walk-RIGHT sealed post-Raft; BOMB_RIGHT @(192,141) reopens 0x5a",
        ),
        BombSpendItem(
            label="bomb_r_0x5b",
            room=L3_BOMB_R_5B[0],
            dest=L3_BOMB_R_5B[1],
            stand=L3_BOMB_R_5B[2],
            count=1,
            evidence="verified",
            note="BOMB_RIGHT @(192,141) → 0x5c boss shortcut",
        ),
    )
    manhandla = BombSpendItem(
        label="manhandla_heads",
        room=ROOM_L3_BOSS,
        dest=None,
        stand=None,
        count=MANHANDLA_BOMB_SPEND_ASSUMED,
        evidence=MANHANDLA_BOMB_SPEND_EVIDENCE,
        note=(
            f"type 0x{MANHANDLA_OBJECT_TYPE:02x}, {MANHANDLA_HEADS_LIVE} heads "
            "live-verified; bombs preferred — 1 bomb/head is assumed, not TAS"
        ),
    )
    return PlannedBombSpend(
        walls=walls,
        manhandla=manhandla,
        wall_bombs=L3_BOMB_WALL_SPEND,
        manhandla_bombs=MANHANDLA_BOMB_SPEND_ASSUMED,
        verified=L3_BOMB_BUDGET_VERIFIED,
        assumed=L3_BOMB_BUDGET_ASSUMED,
        total=L3_BOMB_BUDGET,
        bombs_in_at_dest_5b=L3_BOMBS_IN_AT_DEST_5B,
        bombs_in_at_raft=L3_BOMBS_IN_AT_RAFT,
        isolated_raft_bombs=ISOLATED_RAFT_BOMBS,
        isolated_poke16_closes_spine=ISOLATED_POKE16_CLOSES_SPINE,
    )


def bomb_budget() -> int:
    """Planned spend: verified walls + assumed Manhandla-head estimate."""
    return planned_bomb_spend().total


def isolated_raft_requires_poke16(bombs_at_raft: int | None = ISOLATED_RAFT_BOMBS) -> bool:
    """Empty Level3Raft pin cannot cover even the verified bomb-R walls."""
    if bombs_at_raft is None:
        return True
    return int(bombs_at_raft) < L3_BOMB_WALL_SPEND


def bombs_from_snapshot(snap: object) -> int | None:
    """Inventory bombs, or None when the field is absent (not zero)."""
    if snap is None:
        return None
    if isinstance(snap, Mapping):
        if "bombs" not in snap:
            return None
        val = snap["bombs"]
    elif hasattr(snap, "bombs"):
        val = getattr(snap, "bombs")
    else:
        return None
    if val is None:
        return None
    return int(val)


def recording_path(name: str) -> Path:
    return RECORDINGS_DIR / name


def load_isolated_report(name: str) -> dict[str, Any] | None:
    """JSON report if present; None when ``recordings/`` is gitignored/missing."""
    path = recording_path(name)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def raft_stop_bombs(report: Mapping[str, Any] | None) -> int | None:
    """Final bombs on an isolated Raft runner report (last trial)."""
    if not isinstance(report, Mapping):
        return None
    reports = report.get("reports")
    if isinstance(reports, list) and reports:
        last = reports[-1]
        if isinstance(last, Mapping):
            got = bombs_from_snapshot(last.get("final"))
            if got is not None:
                return got
    final = report.get("final")
    if isinstance(final, Mapping):
        return bombs_from_snapshot(final)
    return bombs_from_snapshot(report)


def report_used_poke16(report: Mapping[str, Any] | None) -> bool:
    """True if an isolated to-boss / spine report admits ``--poke-bombs 16``."""
    if not isinstance(report, Mapping):
        return False
    pb = report.get("poke_bombs")
    if pb is True or (isinstance(pb, int) and pb == ISOLATED_TO_BOSS_POKE_BOMBS):
        return True
    runner = str(report.get("runner") or "")
    if f"--poke-bombs {ISOLATED_TO_BOSS_POKE_BOMBS}" in runner:
        return True
    assist = report.get("inventory_assist")
    if isinstance(assist, Mapping):
        apb = assist.get("poke_bombs")
        if apb is True or (isinstance(apb, int) and apb == ISOLATED_TO_BOSS_POKE_BOMBS):
            return True
    return False


__all__ = [
    "BOMB_STAND_59_RIGHT",
    "BOMB_STAND_5B_RIGHT",
    "ISOLATED_POKE16_CLOSES_SPINE",
    "ISOLATED_RAFT_BOMBS",
    "ISOLATED_RAFT_RECORDING",
    "ISOLATED_TO_BOSS_POKE_BOMBS",
    "ISOLATED_TO_BOSS_RECORDING",
    "L3_BOMB_BUDGET",
    "L3_BOMB_BUDGET_ASSUMED",
    "L3_BOMB_BUDGET_VERIFIED",
    "L3_BOMB_FARM_ROOM",
    "L3_BOMB_R_59",
    "L3_BOMB_R_5B",
    "L3_BOMB_R_WALLS",
    "L3_BOMB_WALL_SPEND",
    "L3_BOMBS_IN_AT_DEST_5B",
    "L3_BOMBS_IN_AT_RAFT",
    "MANHANDLA_BOMB_SPEND_ASSUMED",
    "MANHANDLA_BOMB_SPEND_EVIDENCE",
    "MANHANDLA_HEADS_EVIDENCE",
    "MANHANDLA_HEADS_LIVE",
    "MANHANDLA_OBJECT_TYPE",
    "SPINE_0X7C_BOMBS",
    "SPINE_0X7C_KEYS",
    "SPINE_0X7C_ROOM",
    "SPINE_ENTRANCE_RECORDING",
    "SPINE_L2_ENTRY_BOMBS",
    "BombSpendItem",
    "PlannedBombSpend",
    "bomb_budget",
    "bombs_from_snapshot",
    "isolated_raft_requires_poke16",
    "load_isolated_report",
    "planned_bomb_spend",
    "raft_stop_bombs",
    "recording_path",
    "report_used_poke16",
]
