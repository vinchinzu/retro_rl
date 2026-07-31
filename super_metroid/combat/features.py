"""Structured combat features: positions, HP, and known hitbox geometry.

Hitbox dimensions come from sm-json-data ``enemies/bosses/main.json`` (width/
height in pixels around the enemy center). Samus uses a conservative standing
AABB until pose-specific boxes are wired from the reverse-engineered source.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from super_metroid.ram import SuperMetroidState

# Approximate Samus standing hurtbox (center-based; refined later per pose).
SAMUS_STAND_W = 14
SAMUS_STAND_H = 32


@dataclass(frozen=True)
class AxisAlignedBox:
    """Axis-aligned box in room pixel coordinates (center + half extents)."""

    cx: float
    cy: float
    half_w: float
    half_h: float

    @property
    def left(self) -> float:
        return self.cx - self.half_w

    @property
    def right(self) -> float:
        return self.cx + self.half_w

    @property
    def top(self) -> float:
        return self.cy - self.half_h

    @property
    def bottom(self) -> float:
        return self.cy + self.half_h

    def overlaps(self, other: AxisAlignedBox) -> bool:
        return (
            self.left < other.right
            and self.right > other.left
            and self.top < other.bottom
            and self.bottom > other.top
        )

    def separation(self, other: AxisAlignedBox) -> tuple[float, float]:
        """Signed center deltas (other − self)."""
        return other.cx - self.cx, other.cy - self.cy


@dataclass(frozen=True)
class BossCatalogEntry:
    """Static boss facts used for features and reward shaping."""

    name: str
    room_id: int
    max_hp: int
    width: int
    height: int
    contact_damage: int
    primary_weapon: str  # "missiles" | "supers" | "beam" | ...


def bomb_torizo_catalog() -> BossCatalogEntry:
    """Bomb Torizo facts from sm-json-data bosses catalog."""
    return BossCatalogEntry(
        name="Bomb Torizo",
        room_id=0x9804,
        max_hp=800,
        width=73,
        height=90,
        contact_damage=8,
        primary_weapon="missiles",
    )


def kraid_catalog() -> BossCatalogEntry:
    """Kraid body facts (sm-json-data + live entry probes).

    Multi-slot fight: body is enemy0 at HP 1000; nails/spikes use other
    slots. Hitbox dims are approximate for contact checks — the strategy
    lane-keeps rather than relying on AABB aim.
    """
    return BossCatalogEntry(
        name="Kraid",
        room_id=0xA59F,
        max_hp=1000,
        width=80,
        height=120,
        contact_damage=20,
        primary_weapon="supers",
    )


@dataclass(frozen=True)
class CombatFeatures:
    """One-frame full-knowledge combat summary (no pixels)."""

    room_id: int
    samus_x: int
    samus_y: int
    samus_vx: int
    samus_vy: int
    samus_pose: int
    samus_health: int
    samus_max_health: int
    missiles: int
    max_missiles: int
    selected_item: int
    enemy_x: int
    enemy_y: int
    enemy_hp: int
    enemy_max_hp: int
    enemy_spritemap: int
    dx: int
    dy: int
    distance: float
    contact_overlap: bool
    enemy_active: bool
    enemy_defeated: bool
    boss_name: str

    def to_dict(self) -> dict[str, object]:
        return {
            "room_id": self.room_id,
            "room_id_hex": f"0x{self.room_id:04X}",
            "samus_x": self.samus_x,
            "samus_y": self.samus_y,
            "samus_pose": self.samus_pose,
            "samus_health": self.samus_health,
            "missiles": self.missiles,
            "selected_item": self.selected_item,
            "enemy_x": self.enemy_x,
            "enemy_y": self.enemy_y,
            "enemy_hp": self.enemy_hp,
            "enemy_max_hp": self.enemy_max_hp,
            "enemy_spritemap": self.enemy_spritemap,
            "enemy_spritemap_hex": f"0x{self.enemy_spritemap:04X}",
            "dx": self.dx,
            "dy": self.dy,
            "distance": self.distance,
            "contact_overlap": self.contact_overlap,
            "enemy_active": self.enemy_active,
            "enemy_defeated": self.enemy_defeated,
            "boss_name": self.boss_name,
        }


def _box(cx: int, cy: int, width: int, height: int) -> AxisAlignedBox:
    return AxisAlignedBox(
        cx=float(cx),
        cy=float(cy),
        half_w=width / 2.0,
        half_h=height / 2.0,
    )


# Bomb Torizo pre-combat spritemaps (not hittable combat AI yet).
# 0x804F = room-load / chozo spawn; 0x87D0 = idle statue before touch.
BOMB_TORIZO_INACTIVE_SPRITEMAPS = frozenset({0x87D0, 0x804F})


def features_from_state(
    state: SuperMetroidState,
    catalog: BossCatalogEntry,
    *,
    inactive_spritemaps: frozenset[int] = BOMB_TORIZO_INACTIVE_SPRITEMAPS,
) -> CombatFeatures:
    """Build combat features from typed RAM state + boss catalog."""
    samus = _box(state.samus_x, state.samus_y, SAMUS_STAND_W, SAMUS_STAND_H)
    enemy = _box(state.enemy0_x, state.enemy0_y, catalog.width, catalog.height)
    dx = state.enemy0_x - state.samus_x
    dy = state.enemy0_y - state.samus_y
    defeated = state.enemy0_hp == 0 and state.room_id == catalog.room_id
    # Active once the boss leaves idle/spawn spritemaps with a real boss HP
    # bar. Room-entry frames often still carry the previous room's enemy0 slot
    # (low HP + random spritemap + many enemy slots) — reject those.
    active = (
        state.room_id == catalog.room_id
        and 0 < state.enemy0_hp <= catalog.max_hp
        and state.enemy0_spritemap not in inactive_spritemaps
        and state.enemy0_spritemap != 0
        and state.num_enemies <= 4
    )
    return CombatFeatures(
        room_id=state.room_id,
        samus_x=state.samus_x,
        samus_y=state.samus_y,
        samus_vx=state.velocity_x,
        samus_vy=state.velocity_y,
        samus_pose=state.pose,
        samus_health=state.health,
        samus_max_health=state.max_health,
        missiles=state.missiles,
        max_missiles=state.max_missiles,
        selected_item=state.selected_item,
        enemy_x=state.enemy0_x,
        enemy_y=state.enemy0_y,
        enemy_hp=state.enemy0_hp,
        enemy_max_hp=catalog.max_hp,
        enemy_spritemap=state.enemy0_spritemap,
        dx=dx,
        dy=dy,
        distance=float(np.hypot(dx, dy)),
        contact_overlap=samus.overlaps(enemy),
        enemy_active=active,
        enemy_defeated=defeated,
        boss_name=catalog.name,
    )


# Fixed layout length for structured-state RL (keep in sync with feature_vector).
FEATURE_DIM = 14


def feature_vector(features: CombatFeatures) -> np.ndarray:
    """Normalized float32 vector for structured-state RL (fixed layout).

    Layout (FEATURE_DIM = 14):
      0  samus_x / 512
      1  samus_y / 512
      2  samus_vx / 16
      3  samus_vy / 16
      4  enemy_x / 512
      5  enemy_y / 512
      6  enemy_hp / max_hp
      7  samus_health / max_health
      8  missiles / max(max_missiles, 1)
      9  dx / 256
      10 dy / 256
      11 distance / 256
      12 contact_overlap
      13 selected_missiles (1 if selected_item == 1)
    """
    max_hp = max(features.enemy_max_hp, 1)
    max_health = max(features.samus_max_health, 1)
    max_missiles = max(features.max_missiles, 1)
    vec = np.asarray(
        [
            features.samus_x / 512.0,
            features.samus_y / 512.0,
            features.samus_vx / 16.0,
            features.samus_vy / 16.0,
            features.enemy_x / 512.0,
            features.enemy_y / 512.0,
            features.enemy_hp / max_hp,
            features.samus_health / max_health,
            features.missiles / max_missiles,
            features.dx / 256.0,
            features.dy / 256.0,
            features.distance / 256.0,
            1.0 if features.contact_overlap else 0.0,
            1.0 if features.selected_item == 1 else 0.0,
        ],
        dtype=np.float32,
    )
    assert vec.shape == (FEATURE_DIM,)
    return vec
