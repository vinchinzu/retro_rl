"""Contract-guarded survival assist for Zelda I first-pass routing.

See ``docs/ASSIST_CONTRACT.md``. Zelda I segment scripts default to Survival
(``--infinite-life``). Opt out with ``--no-infinite-life``. Clean M5 stays
``run_level1_complete`` without the flag.

**Strategy:** infinite life unblocks pathfinding and puzzle geometry first.
Damage is observed and aggregated so Clean combat harden can target hot
rooms later — do not prioritize sword polish over route completion.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from zelda_i.ram import (
    ADDR_HEALTH,
    PLAY_MODE,
    ZeldaSnapshot,
    health_byte_for_containers,
    read_snapshot,
)

# RoomItemId for a dungeon/boss heart container (see dungeon_ids.ROOM_ITEM_NAMES).
HEART_CONTAINER_ITEM = 0x1A
UNDERWORLD_PASSAGE_MODE = 9

# Cap stored event samples (reports stay small; totals remain unbounded).
_MAX_DAMAGE_SAMPLES = 64


class _RetroData(Protocol):
    def set_value(self, key: str, value: int) -> None: ...


@dataclass
class ResourceCounter:
    restored: int = 0
    writes: int = 0
    first_active_frame: int | None = None


@dataclass
class DamageEvent:
    """One observed filled-heart loss before assist refill."""

    frame: int
    amount: int
    level: int
    screen: int
    link_x: int
    link_y: int

    def location_key(self) -> str:
        return f"L{int(self.level)}:0x{int(self.screen):02x}"

    def to_dict(self) -> dict[str, object]:
        return {
            "frame": self.frame,
            "amount": self.amount,
            "level": self.level,
            "screen": self.screen,
            "screen_hex": f"0x{int(self.screen):02x}",
            "location": self.location_key(),
            "link_x": self.link_x,
            "link_y": self.link_y,
        }


@dataclass
class AssistTelemetry:
    health: ResourceCounter = field(default_factory=ResourceCounter)
    suspended_phase_frames: Counter[str] = field(default_factory=Counter)
    maximum_single_frame_damage: int = 0
    # Cumulative filled-heart units lost (observed before refill). Primary
    # signal for later Clean combat harden prioritization.
    total_damage: int = 0
    damage_events: int = 0
    damage_by_location: Counter[str] = field(default_factory=Counter)
    damage_samples: list[DamageEvent] = field(default_factory=list)
    deaths: int = 0
    progression_writes: int = 0
    capacity_writes: int = 0
    # Corrective clamps when RAM showed more containers than earned.
    container_clamps: int = 0
    accepted_containers: int | None = None

    def to_dict(self) -> dict[str, object]:
        # Top locations by total damage (hottest rooms first).
        by_loc = dict(
            sorted(
                self.damage_by_location.items(),
                key=lambda kv: (-kv[1], kv[0]),
            )
        )
        return {
            "health": asdict(self.health),
            "suspended_phase_frames": dict(self.suspended_phase_frames),
            "maximum_single_frame_damage": self.maximum_single_frame_damage,
            "total_damage": self.total_damage,
            "damage_events": self.damage_events,
            "damage_by_location": by_loc,
            "damage_samples": [e.to_dict() for e in self.damage_samples],
            "deaths": self.deaths,
            "progression_writes": self.progression_writes,
            "capacity_writes": self.capacity_writes,
            "container_clamps": self.container_clamps,
            "accepted_containers": self.accepted_containers,
        }


def assist_phase_name(snap: ZeldaSnapshot) -> str:
    """Coarse phase for assist guards (mirrors SM GameplayPhase idea)."""
    if snap.mode == 17:
        return "death"
    if snap.mode == 18:
        return "triforce_fanfare"
    if snap.mode in (0, 1, 2, 3, 4):
        return "menu_or_boot"
    if snap.transitioning:
        return "transition"
    if snap.mode in (PLAY_MODE, UNDERWORLD_PASSAGE_MODE) or snap.in_cave:
        return "ordinary_gameplay"
    return f"mode_{snap.mode}"


def location_key(snap: ZeldaSnapshot) -> str:
    """Stable location id for damage heatmaps: ``L{level}:0x{screen}``."""
    return f"L{int(snap.level)}:0x{int(snap.screen):02x}"


class UnlimitedHealthAssist:
    """Refill filled hearts to the natural container max; never grant containers.

    Writes only ``health`` (``ADDR_HEALTH`` / data.json key) under the contract.
    Tracks observed damage (total, per-location heatmap) so later Clean passes
    know which rooms hurt most without blocking first-pass geometry work.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.telemetry = AssistTelemetry()
        self._prev_filled: int | None = None
        self._prev_phase: str | None = None
        self._accepted_containers: int | None = None

    def report(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "class": "survival",
            "kind": "unlimited_health",
            **self.telemetry.to_dict(),
        }

    def _record_damage(self, snap: ZeldaSnapshot, amount: int, *, frame: int) -> None:
        if amount <= 0:
            return
        tel = self.telemetry
        tel.total_damage += amount
        tel.damage_events += 1
        tel.maximum_single_frame_damage = max(
            tel.maximum_single_frame_damage,
            amount,
        )
        loc = location_key(snap)
        tel.damage_by_location[loc] += amount
        if len(tel.damage_samples) < _MAX_DAMAGE_SAMPLES:
            tel.damage_samples.append(
                DamageEvent(
                    frame=frame,
                    amount=amount,
                    level=int(snap.level),
                    screen=int(snap.screen),
                    link_x=int(snap.link_x),
                    link_y=int(snap.link_y),
                )
            )

    def apply_snapshot(
        self,
        data: _RetroData,
        snap: ZeldaSnapshot,
        *,
        frame: int = 0,
    ) -> ZeldaSnapshot | None:
        """Apply assist from a snapshot. Returns None if no write happened."""
        if not self.enabled:
            return None

        prev_phase = self._prev_phase
        phase = assist_phase_name(snap)
        if phase == "death" and prev_phase != "death":
            self.telemetry.deaths += 1
        self._prev_phase = phase

        if phase != "ordinary_gameplay":
            self.telemetry.suspended_phase_frames[phase] += 1
            self._prev_filled = None
            return None

        observed = int(snap.heart_containers)
        if self._accepted_containers is None:
            self._accepted_containers = max(1, observed)
        accepted = self._accepted_containers
        left_fanfare = prev_phase == "triforce_fanfare"
        legal_plus_one = observed == accepted + 1 and (
            left_fanfare or int(snap.room_item_id) == HEART_CONTAINER_ITEM
        )
        if legal_plus_one:
            accepted = observed
            self._accepted_containers = accepted
        elif left_fanfare and observed > accepted:
            # Triforce grants one container. A glitched high nibble after
            # fanfare (tape: 5 → 7) is not a second container.
            accepted += 1
            self._accepted_containers = accepted
        elif observed > accepted:
            self.telemetry.container_clamps += 1

        self.telemetry.accepted_containers = accepted

        filled = snap.filled_hearts
        if (
            self._prev_filled is not None
            and observed == accepted
            and observed == ((int(snap.health) >> 4) & 0x0F) + 1
        ):
            damage = max(0, self._prev_filled - filled)
            self._record_damage(snap, damage, frame=frame)

        target = health_byte_for_containers(accepted)
        partial = int(getattr(snap, "heart_partial", 0xFF)) & 0xFF
        if accepted <= 0:
            self._prev_filled = target & 0x0F
            return None
        if snap.health == target and partial == 0xFF:
            self._prev_filled = target & 0x0F
            return None

        counter = self.telemetry.health
        if counter.first_active_frame is None:
            counter.first_active_frame = frame
        restored = max(0, (target & 0x0F) - (int(snap.health) & 0x0F))
        data.set_value("health", target)
        data.set_value("heart_partial", 0xFF)
        counter.restored += restored
        counter.writes += 1
        self._prev_filled = target & 0x0F
        return None

    def apply_env(self, env: Any, *, frame: int = 0) -> None:
        """Read RAM from ``env``, apply, leave env mutated when writing."""
        if not self.enabled:
            return
        snap = read_snapshot(env.get_ram())
        self.apply_snapshot(env.data, snap, frame=frame)


def write_health_u8(env: Any, value: int) -> None:
    """Low-level health write (tests / diagnostics). Prefer the assist class."""
    env.data.set_value("health", int(value) & 0xFF)


__all__ = [
    "AssistTelemetry",
    "DamageEvent",
    "HEART_CONTAINER_ITEM",
    "ResourceCounter",
    "UnlimitedHealthAssist",
    "assist_phase_name",
    "location_key",
    "write_health_u8",
    "ADDR_HEALTH",
]
