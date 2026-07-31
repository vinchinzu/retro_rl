"""Structured combat features: positions, HP, and known hitbox geometry.

Hitbox dimensions come from sm-json-data ``enemies/bosses/main.json`` (width/
height in pixels around the enemy center) unless live probes refine them.
Samus uses a conservative standing AABB until pose-specific boxes are wired
from the reverse-engineered source.

Full boss registry + pipeline: ``docs/BOSS_PIPELINE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
class BossPhaseSpec:
    """One phase of a multi-phase boss (Phantoon, Mother Brain, Ridley, …)."""

    phase_id: str
    max_hp: int
    width: int
    height: int
    notes: str = ""


@dataclass(frozen=True)
class BossCatalogEntry:
    """Static boss facts used for features, strategies, and reward shaping.

    Core fields (name, room_id, max_hp, width, height, contact_damage,
    primary_weapon) are required. Extended fields support the full boss
    pipeline registry without breaking existing call sites.
    """

    name: str
    room_id: int
    max_hp: int
    width: int
    height: int
    contact_damage: int
    primary_weapon: str  # "missiles" | "supers" | "beam" | "acid_push" | ...
    boss_id: str = ""
    secondary_weapon: str = ""
    # Defeat: boss_bits[area_index] & bit_mask, and/or event flag id.
    boss_area_index: int | None = None
    boss_bit_mask: int = 0
    defeat_event_id: int | None = None
    inactive_spritemaps: frozenset[int] = field(default_factory=frozenset)
    max_enemy_slots: int = 8
    phases: tuple[BossPhaseSpec, ...] = ()
    closeout: str = ""
    recommended_loadout: str = ""
    # Lower = earlier on KPDR continuous spine (see BOSS_PIPELINE.md).
    kpdr_priority: int = 99
    # continuous | wired | deferred | optional | side
    continuous_status: str = "deferred"
    sm_json_id: int | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.boss_id:
            # Derive a stable id from the display name when omitted.
            object.__setattr__(
                self,
                "boss_id",
                self.name.lower().replace(" ", "_").replace("'", ""),
            )


# ---------------------------------------------------------------------------
# Catalog constructors (existing call sites + full registry)
# ---------------------------------------------------------------------------

# Bomb Torizo pre-combat spritemaps (not hittable combat AI yet).
# 0x804F = room-load / chozo spawn; 0x87D0 = idle statue before touch.
BOMB_TORIZO_INACTIVE_SPRITEMAPS = frozenset({0x87D0, 0x804F})


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
        boss_id="bomb_torizo",
        boss_area_index=0,  # Crateria $7E:D828
        boss_bit_mask=0x04,
        inactive_spritemaps=BOMB_TORIZO_INACTIVE_SPRITEMAPS,
        max_enemy_slots=4,
        closeout="Bombs PLM + Flyway exit (hash-pinned continuous replay)",
        recommended_loadout="Missiles; Morph + Bombs target",
        kpdr_priority=0,
        continuous_status="continuous",
        sm_json_id=201,
    )


def spore_spawn_catalog() -> BossCatalogEntry:
    """Spore Spawn (continuous controller in routes/spore_spawn_controller)."""
    return BossCatalogEntry(
        name="Spore Spawn",
        room_id=0x9DC7,
        max_hp=960,
        width=73,
        height=90,
        contact_damage=12,
        primary_weapon="missiles",
        boss_id="spore_spawn",
        boss_area_index=1,  # Brinstar $7E:D829
        boss_bit_mask=0x02,
        max_enemy_slots=4,
        closeout="Super Missile room collect",
        recommended_loadout="Missiles / charged beam",
        kpdr_priority=0,
        continuous_status="continuous",
        sm_json_id=202,
    )


def kraid_catalog() -> BossCatalogEntry:
    """Kraid body facts (sm-json-data + live entry probes).

    Multi-slot fight: body is enemy0 at HP 1000; nails/spikes use other
    slots. Hitbox dims are approximate for contact checks — the strategy
    lane-keeps rather than relying on AABB aim. sm-json-data dims are
    192×303; feature box stays tighter for contact-overlap signal.
    """
    return BossCatalogEntry(
        name="Kraid",
        room_id=0xA59F,
        max_hp=1000,
        width=80,
        height=120,
        contact_damage=20,
        primary_weapon="supers",
        boss_id="kraid",
        secondary_weapon="missiles",
        boss_area_index=1,  # Brinstar $7E:D829 bit 0
        boss_bit_mask=0x01,
        max_enemy_slots=8,
        closeout="Rear blue door → Varia Suit Room PLM",
        recommended_loadout="Super Missiles (capacity ≥ 5)",
        kpdr_priority=1,
        continuous_status="wired",
        sm_json_id=203,
        notes="sm-json-data hitbox 192×303; feature box 80×120 for lane contact",
    )


def phantoon_catalog() -> BossCatalogEntry:
    """Phantoon (Wrecked Ship). Multi-phase open/close eye windows."""
    return BossCatalogEntry(
        name="Phantoon",
        room_id=0xCD13,
        max_hp=2500,
        width=140,
        height=224,
        contact_damage=40,
        primary_weapon="missiles",
        boss_id="phantoon",
        secondary_weapon="supers",
        boss_area_index=3,  # Wrecked Ship $7E:D82B bit 0
        boss_bit_mask=0x01,
        max_enemy_slots=8,
        phases=(
            BossPhaseSpec("round", 2500, 140, 224, "Eye open vulnerable windows"),
        ),
        closeout="WS basement exit + power restore path",
        recommended_loadout="Missiles + Supers; Gravity not required for fight",
        kpdr_priority=2,
        continuous_status="deferred",
        sm_json_id=205,
    )


def botwoon_catalog() -> BossCatalogEntry:
    """Botwoon (Maridia worm). sm-json dims are 0×0 — head box is approximate."""
    return BossCatalogEntry(
        name="Botwoon",
        room_id=0xD95E,
        max_hp=1500,
        width=32,
        height=32,
        contact_damage=128,
        primary_weapon="supers",
        boss_id="botwoon",
        secondary_weapon="missiles",
        boss_area_index=4,  # Maridia $7E:D82C bit 1
        boss_bit_mask=0x02,
        max_enemy_slots=8,
        closeout="Botwoon hall exit toward Draygon path",
        recommended_loadout="Supers / Missiles; Gravity preferred",
        kpdr_priority=3,
        continuous_status="deferred",
        sm_json_id=206,
        notes="Hitbox approximate (sm-json w/h = 0); refine with live probes",
    )


def draygon_catalog() -> BossCatalogEntry:
    """Draygon + Space Jump closeout."""
    return BossCatalogEntry(
        name="Draygon",
        room_id=0xDA60,
        max_hp=6000,
        width=262,
        height=284,
        contact_damage=160,
        primary_weapon="supers",
        boss_id="draygon",
        secondary_weapon="missiles",
        boss_area_index=4,  # Maridia $7E:D82C bit 0
        boss_bit_mask=0x01,
        max_enemy_slots=8,
        phases=(
            BossPhaseSpec("body", 6000, 262, 284, "Turrets + grab phases"),
        ),
        closeout="Space Jump collect (Colosseum / Big Pink return per route)",
        recommended_loadout="Gravity + Supers; X-Ray optional",
        kpdr_priority=4,
        continuous_status="deferred",
        sm_json_id=207,
    )


def crocomire_catalog() -> BossCatalogEntry:
    """Crocomire is an acid-push fight — not pure HP damage (sm-json hp=0)."""
    return BossCatalogEntry(
        name="Crocomire",
        room_id=0xA98D,
        max_hp=0,
        width=145,
        height=118,
        contact_damage=40,
        primary_weapon="acid_push",
        boss_id="crocomire",
        secondary_weapon="power_bombs",
        boss_area_index=2,  # Norfair $7E:D82A bit 1
        boss_bit_mask=0x02,
        max_enemy_slots=4,
        closeout="Post-Croc shaft / Power Bomb / Speed side path",
        recommended_loadout="Varia + Missiles/Supers to push; heat management",
        kpdr_priority=5,
        continuous_status="side",
        sm_json_id=204,
        notes="Defeat by pushing into acid wall; HP field unused for win",
    )


def ridley_catalog() -> BossCatalogEntry:
    """Lower Norfair Ridley."""
    return BossCatalogEntry(
        name="Ridley",
        room_id=0xB32E,
        max_hp=18000,
        width=79,
        height=89,
        contact_damage=160,
        primary_weapon="supers",
        boss_id="ridley",
        secondary_weapon="missiles",
        boss_area_index=2,  # Norfair $7E:D82A bit 0
        boss_bit_mask=0x01,
        max_enemy_slots=4,
        phases=(
            BossPhaseSpec("flight", 18000, 79, 89, "Hover + swoop + fireballs"),
        ),
        closeout="LN exit / escape path toward Statues",
        recommended_loadout="Varia/Gravity + Supers; high ammo",
        kpdr_priority=6,
        continuous_status="deferred",
        sm_json_id=209,
    )


def golden_torizo_catalog() -> BossCatalogEntry:
    """Golden Torizo (optional / speed side; multi-phase practice)."""
    return BossCatalogEntry(
        name="Golden Torizo",
        room_id=0xB283,
        max_hp=13500,
        width=73,
        height=90,
        contact_damage=160,
        primary_weapon="supers",
        boss_id="golden_torizo",
        secondary_weapon="charge_beam",
        boss_area_index=None,  # Not a major boss bit for statues
        boss_bit_mask=0,
        max_enemy_slots=4,
        phases=(
            BossPhaseSpec("main", 13500, 73, 90, "Super-absorb + egg spam"),
        ),
        closeout="Screw Attack path (route-dependent)",
        recommended_loadout="Charge beam / Supers; Screw optional",
        kpdr_priority=7,
        continuous_status="optional",
        sm_json_id=208,
    )


def mother_brain_catalog() -> BossCatalogEntry:
    """Mother Brain multi-phase (Zebetites → brain → rainbow)."""
    return BossCatalogEntry(
        name="Mother Brain",
        room_id=0xDD58,
        max_hp=3000,  # Phase 1 glass/brain
        width=112,
        height=112,
        contact_damage=120,
        primary_weapon="missiles",
        boss_id="mother_brain",
        secondary_weapon="supers",
        boss_area_index=5,  # Tourian $7E:D82D
        boss_bit_mask=0x02,
        defeat_event_id=0x0E,  # Escape start
        max_enemy_slots=8,
        phases=(
            BossPhaseSpec("mb1", 3000, 112, 112, "Glass + turrets / Zebetites"),
            BossPhaseSpec("mb2", 18000, 138, 171, "Body + rainbow setup"),
            BossPhaseSpec("mb3", 36000, 138, 171, "Rainbow beam / hyper"),
        ),
        closeout="Escape timer + ship / credits",
        recommended_loadout="Full loadout; hyper beam after MB2",
        kpdr_priority=8,
        continuous_status="deferred",
        sm_json_id=210,
        notes="Primary max_hp is MB1; phases list full HP progression",
    )


def _build_boss_catalog() -> dict[str, BossCatalogEntry]:
    entries = (
        bomb_torizo_catalog(),
        spore_spawn_catalog(),
        kraid_catalog(),
        phantoon_catalog(),
        botwoon_catalog(),
        draygon_catalog(),
        crocomire_catalog(),
        ridley_catalog(),
        golden_torizo_catalog(),
        mother_brain_catalog(),
    )
    return {e.boss_id: e for e in entries}


BOSS_CATALOG: dict[str, BossCatalogEntry] = _build_boss_catalog()

# Spine order for documentation / iteration (excludes continuous-done early).
BOSS_SPINE_ORDER: tuple[str, ...] = (
    "kraid",
    "phantoon",
    "botwoon",
    "draygon",
    "crocomire",
    "ridley",
    "golden_torizo",
    "mother_brain",
)


def get_boss_catalog(boss_id: str) -> BossCatalogEntry:
    """Return catalog entry by ``boss_id`` (e.g. ``\"kraid\"``)."""
    try:
        return BOSS_CATALOG[boss_id]
    except KeyError as exc:
        known = ", ".join(sorted(BOSS_CATALOG))
        raise KeyError(f"unknown boss_id {boss_id!r}; known: {known}") from exc


def list_boss_catalog(
    *,
    continuous_status: str | None = None,
) -> list[BossCatalogEntry]:
    """All catalog entries, optionally filtered, sorted by KPDR priority."""
    entries = list(BOSS_CATALOG.values())
    if continuous_status is not None:
        entries = [e for e in entries if e.continuous_status == continuous_status]
    return sorted(entries, key=lambda e: (e.kpdr_priority, e.boss_id))


def boss_defeated_in_state(state: SuperMetroidState, catalog: BossCatalogEntry) -> bool:
    """True when the catalog defeat condition holds on parsed state.

    Uses ``state.boss_bits`` (low WRAM path) and optional event flag.
    For full bank-$7E peeks during a fight, strategies may also read via
    ``read_bank7e_wram`` (see Kraid).
    """
    if catalog.boss_area_index is not None and catalog.boss_bit_mask:
        bits = state.boss_bits
        if catalog.boss_area_index < len(bits):
            if bits[catalog.boss_area_index] & catalog.boss_bit_mask:
                return True
    if catalog.defeat_event_id is not None:
        event_id = catalog.defeat_event_id
        byte_index = event_id >> 3
        bit = 1 << (event_id & 7)
        flags = state.event_flags
        if byte_index < len(flags) and (flags[byte_index] & bit):
            return True
    return False


def validate_live_enemy(
    state: SuperMetroidState,
    catalog: BossCatalogEntry,
    *,
    require_active: bool = False,
    require_full_hp: bool = False,
) -> list[str]:
    """Return human-readable mismatches between live enemy0 and catalog.

    Empty list means the live snapshot is consistent enough for strategy /
    capture work. Room-load garbage and wrong rooms produce failures.
    """
    failures: list[str] = []
    if state.room_id != catalog.room_id:
        failures.append(
            f"room 0x{state.room_id:04X} != catalog 0x{catalog.room_id:04X}"
        )
    if catalog.max_hp > 0:
        if state.enemy0_hp < 0 or state.enemy0_hp > catalog.max_hp:
            # Allow 0 after defeat.
            if state.enemy0_hp != 0:
                failures.append(
                    f"enemy0_hp {state.enemy0_hp} outside 0..{catalog.max_hp}"
                )
        if require_full_hp and state.enemy0_hp != catalog.max_hp:
            failures.append(
                f"enemy0_hp {state.enemy0_hp} != full {catalog.max_hp}"
            )
    if state.num_enemies > catalog.max_enemy_slots:
        failures.append(
            f"num_enemies {state.num_enemies} > max_enemy_slots "
            f"{catalog.max_enemy_slots} (room-load garbage?)"
        )
    if state.enemy0_spritemap == 0:
        failures.append("enemy0_spritemap is 0")
    if (
        catalog.inactive_spritemaps
        and state.enemy0_spritemap in catalog.inactive_spritemaps
        and require_active
    ):
        failures.append(
            f"spritemap 0x{state.enemy0_spritemap:04X} still inactive"
        )
    if require_active:
        feat = features_from_state(state, catalog)
        if not feat.enemy_active:
            failures.append("features report enemy_active=False")
    return failures


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


def features_from_state(
    state: SuperMetroidState,
    catalog: BossCatalogEntry,
    *,
    inactive_spritemaps: frozenset[int] | None = None,
) -> CombatFeatures:
    """Build combat features from typed RAM state + boss catalog.

    ``inactive_spritemaps`` overrides the catalog set when provided (legacy
    Bomb Torizo call sites pass the module constant explicitly).
    """
    inactive = (
        inactive_spritemaps
        if inactive_spritemaps is not None
        else catalog.inactive_spritemaps
    )
    samus = _box(state.samus_x, state.samus_y, SAMUS_STAND_W, SAMUS_STAND_H)
    enemy = _box(state.enemy0_x, state.enemy0_y, catalog.width, catalog.height)
    dx = state.enemy0_x - state.samus_x
    dy = state.enemy0_y - state.samus_y
    defeated = (
        boss_defeated_in_state(state, catalog)
        or (state.enemy0_hp == 0 and state.room_id == catalog.room_id)
    )
    # Active once the boss leaves idle/spawn spritemaps with a real boss HP
    # bar. Room-entry frames often still carry the previous room's enemy0 slot
    # (low HP + random spritemap + many enemy slots) — reject those.
    # Crocomire (max_hp==0) treats any non-zero spritemap in-room as active.
    if catalog.max_hp <= 0:
        hp_ok = state.room_id == catalog.room_id and state.enemy0_spritemap != 0
    else:
        hp_ok = 0 < state.enemy0_hp <= catalog.max_hp
    active = (
        state.room_id == catalog.room_id
        and hp_ok
        and state.enemy0_spritemap not in inactive
        and state.enemy0_spritemap != 0
        and state.num_enemies <= catalog.max_enemy_slots
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
