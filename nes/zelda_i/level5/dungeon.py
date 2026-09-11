"""Level 5 (Lizard) dungeon room specs and stop predicates.

Isolated pure for early L5 rooms. Imports combat infrastructure from
``dungeon`` only — do not edit ``dungeon.py`` from L5 agents.

Live recon:
    Entry 0x76 -> 0x66 (Gibdos/key) -> 0x77 (Pols Voice/key).
    West transit routes north then east to central aisle (x=96..144).
    See LEVEL5_ROUTE.md and tasks/rr-npv.2-residual.md for details.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.combat import direction_to_facing
from zelda_i.dungeon.engine import (
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonRoomSpec,
    GenericDungeonRoomController,
    KEESE_OBJECT_TYPE,
    RewardKind,
    RewardSpec,
    dungeon_room_cleared,
    inventory_reward_success,
    register_room_spec,
)
from zelda_i.dungeon import ids as _ids
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot, read_snapshot

LEVEL_5 = 5
ROOM_L5_ENTRY = 0x76
ROOM_L5_GIBDO_66 = 0x66
# East residual of cleared 0x66 (live probe 2026-08-06); Bubble dead-end.
ROOM_L5_EAST_67 = 0x67
# East of entry 0x76 — key door to Pols Voice + replacement small key.
ROOM_L5_POLS_77 = 0x77
# West of 0x66 when west door forced (PARTIAL natural).
ROOM_L5_WEST_65 = 0x65
# North of 0x65 when doors forced (PARTIAL / dark-room chain).
ROOM_L5_NORTH_55 = 0x55
# North of 0x66 (live from Level5EastKey: free UP).
ROOM_L5_NORTH_56 = 0x56
# North of cleared 0x37 (live from Level5Cleared37: free UP).
ROOM_L5_NORTH_27 = 0x27
# West of cleared 0x27 (live from Level5Cleared27: west key).
ROOM_L5_WEST_26 = 0x26
# West of cleared 0x26 (live from Level5Cleared26: free WEST y=141).
ROOM_L5_WEST_25 = 0x25
# West of cleared 0x25 (live from Level5Cleared25: west key). Digdogger 0x38 — door only.
ROOM_L5_WEST_24 = 0x24
# East Zols on the TF approach; do NOT clear them — the secret foes_item
# drops statue 0x5f at (128,128) and seals the north channel at y≈125.
ROOM_L5_EAST_ZOLS = 0x57
# North of 0x57 (ROM N=open).
ROOM_L5_NORTH_GIBDOS = 0x47

# --- Whistle / cellar leg (0x65 west bomb → 0x64 stairs → 0x07 → 0x06/0x05/0x04) ---
ROOM_L5_BLUE_64 = 0x64
ROOM_L5_CELLAR_07 = 0x07
ROOM_L5_PASSAGE_06 = 0x06
ROOM_L5_WHISTLE_05 = 0x05
ROOM_L5_WHISTLE_ITEM = 0x04
BOMB_WEST_STAND = (40, 141)
# Cleared 0x66 west is a ROM bomb wall → 0x65. River locks x-move at y=141;
# south-band y=189 then the west column reaches the bricks.
BOMB_WEST_66_STAND = (32, 141)
# Live bomb-east 0x65 → 0x66 (diamond y=109 then east; stand at east wall).
BOMB_EAST_STAND = (224, 141)

# Type 0x30 — Gibdo-correlated (HP=112 at spawn; TYPE_AND_HP liveness).
GIBDO_OBJECT_TYPE = _ids.GIBDO_OBJECT_TYPE
# Type 0x16 — Pols Voice (HP=160; sword works with backstep; key 0x19).
POLS_VOICE_OBJECT_TYPE = _ids.POLS_VOICE_OBJECT_TYPE
# Type 0x40 — Bubble (HP=240; sword does not reduce HP; invincible residual).
BUBBLE_OBJECT_TYPE = _ids.BUBBLE_OBJECT_TYPE
# Type 0x4e — non-combat residual on 0x67 (hp0; trap/fire-correlated).
ROOM_67_TRAP_TYPE = 0x4E
# Type 0x13 — Zol (same id as L3 west-key room); seen on 0x55.
ZOL_OBJECT_TYPE = _ids.ZOL_OBJECT_TYPE

ROOM_ITEM_SMALL_KEY = 0x19
# Live room-item id 0x03 = no inventory reward (same as L4 ROOM_ITEM_NONE).
ROOM_ITEM_NONE = 0x03

# After clear of 0x66, ``cur_opened_doors`` becomes 0x08 and east opens → 0x67.
# North/west still blocked from this room without further items/geometry.
ROOM_66_EAST_DOOR_BIT = 0x08
# 0x67 settles with left doorway open back to 0x66.
ROOM_67_WEST_DOOR_BIT = 0x02
# 0x65 settles with east doorway open back to 0x66 (when entered west).
ROOM_65_EAST_DOOR_BIT = 0x01

# Stalfos-style room sweep; engage tighter for multi-hit Gibdos.
_ROOM_66_PATROL: tuple[tuple[int, int], ...] = (
    (64, 117),
    (112, 117),
    (160, 117),
    (192, 117),
    (192, 149),
    (160, 149),
    (112, 149),
    (64, 149),
    (64, 181),
    (112, 181),
    (160, 181),
    (192, 181),
)

_ROOM_77_PATROL: tuple[tuple[int, int], ...] = (
    (48, 109),
    (120, 109),
    (200, 109),
    (200, 141),
    (200, 173),
    (120, 173),
    (48, 173),
    (48, 141),
    (120, 141),
)

_ROOM_67_PATROL: tuple[tuple[int, int], ...] = (
    (64, 117),
    (120, 117),
    (176, 117),
    (176, 141),
    (120, 141),
    (64, 141),
)

# From entry 0x76 south mouth ~(120,205) walk north into 0x66.
# Also valid when already in 0x66 at south spawn (L5_Room_66).
ROOM_66_SPEC = DungeonRoomSpec(
    spec_id="level5_room66_gibdos",
    source_room=ROOM_L5_ENTRY,
    room_id=ROOM_L5_GIBDO_66,
    entry=DoorRoute(
        "UP",
        ((120, 205), (120, 93)),
    ),
    enemy_types=(GIBDO_OBJECT_TYPE,),
    expected_enemy_count=3,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_ROOM_66_PATROL,
        engage_distance=56,
        engage_attack_period=6,
        engage_attack_hold=3,
        patrol_attack_period=10,
        patrol_attack_hold=3,
        # River locks cardinal patrol. TF suffix leftover (79,165) 1 Gibdo
        # north of the water; same occupancy as ROOM_66_SPINE_SPEC.
        occupancy_patrol=True,
        occupancy_bounds=(16, 216, 77, 205),
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    required_open_doors=ROOM_66_EAST_DOOR_BIT,
    exit_routes=(
        DoorRoute("DOWN", ((120, 205),)),
        DoorRoute("RIGHT", ((120, 141), (208, 141))),
    ),
    max_frames=12000,
    level=LEVEL_5,
)

# East residual of cleared 0x66 — Bubbles only; no clear pure (invincible).
# Graph node: enter RIGHT from 0x66, exit LEFT only.
ROOM_67_SPEC = DungeonRoomSpec(
    spec_id="level5_room67_bubbles",
    source_room=ROOM_L5_GIBDO_66,
    room_id=ROOM_L5_EAST_67,
    entry=DoorRoute(
        "RIGHT",
        ((120, 141), (208, 141)),
    ),
    enemy_types=(BUBBLE_OBJECT_TYPE,),
    expected_enemy_count=2,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_ROOM_67_PATROL,
        engage_distance=32,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    # Left doorway open on settle (back to 0x66); R/U/D solid.
    required_open_doors=0,
    exit_routes=(DoorRoute("LEFT", ((120, 141), (32, 141))),),
    max_frames=4000,
    level=LEVEL_5,
)

# East of entry: 5× Pols Voice 0x16 + fixed small key 0x19.
# A key from room 0x66 is required. Spec is pure once room-ready on 0x77.
ROOM_77_SPEC = DungeonRoomSpec(
    spec_id="level5_room77_pols_voice",
    source_room=ROOM_L5_ENTRY,
    room_id=ROOM_L5_POLS_77,
    entry=DoorRoute(
        "RIGHT",
        # Approach geometry lives in level5_path.EAST_DOOR_* .
        (
            (120, 157),
            (200, 157),
            (200, 141),
        ),
    ),
    enemy_types=(POLS_VOICE_OBJECT_TYPE,),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_ROOM_77_PATROL,
        engage_distance=72,
        engage_attack_period=5,
        engage_attack_hold=3,
        patrol_attack_period=6,
        patrol_attack_hold=3,
    ),
    reward=RewardSpec(
        kind=RewardKind.FIXED_INVENTORY,
        inventory_field="keys",
        target=(120, 141),
        waypoints=(
            (120, 141),
            (96, 117),
            (144, 165),
            (96, 141),
            (144, 141),
            (120, 157),
            (120, 125),
        ),
    ),
    room_item_id=ROOM_ITEM_SMALL_KEY,
    exit_routes=(DoorRoute("LEFT", ((120, 141), (32, 141))),),
    max_frames=18000,
    level=LEVEL_5,
)

# Natural from cleared 0x55 DOWN. 5× Gibdo — same combat as ROOM_66_SPEC.
# Prior 14000f timeout left 2/5 (HP 16/32); tanky HP=112 needs extra frames.
ROOM_65_SPEC = DungeonRoomSpec(
    spec_id="level5_room65_gibdos",
    source_room=ROOM_L5_NORTH_55,
    room_id=ROOM_L5_WEST_65,
    entry=DoorRoute(
        "DOWN",
        ((120, 141), (120, 205)),
    ),
    enemy_types=(GIBDO_OBJECT_TYPE,),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_ROOM_66_PATROL,
        engage_distance=56,
        engage_attack_period=6,
        engage_attack_hold=3,
        patrol_attack_period=10,
        patrol_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    exit_routes=(
        DoorRoute("UP", ((120, 141), (120, 93))),
        DoorRoute("DOWN", ((120, 141), (120, 205))),
        DoorRoute("LEFT", ((120, 141), (32, 141))),
        DoorRoute("RIGHT", ((120, 141), (208, 141))),
    ),
    max_frames=28000,
    level=LEVEL_5,
)

# Natural from cleared 0x37 UP. Mixed 2 Pols + 2 Gibdo + 2 Keese + key 0x19.
# Keese HP=0 while alive — type_only under TYPE_AND_HP.
ROOM_27_SPEC = DungeonRoomSpec(
    spec_id="level5_room27_mixed",
    source_room=0x37,
    room_id=ROOM_L5_NORTH_27,
    entry=DoorRoute(
        "UP",
        ((120, 141), (120, 93)),
    ),
    enemy_types=(POLS_VOICE_OBJECT_TYPE, GIBDO_OBJECT_TYPE, KEESE_OBJECT_TYPE),
    expected_enemy_count=6,
    alive_rule=AliveRule.TYPE_AND_HP,
    type_only_enemy_types=(KEESE_OBJECT_TYPE,),
    combat=CombatTuning(
        patrol=_ROOM_77_PATROL,
        engage_distance=72,
        engage_attack_period=5,
        engage_attack_hold=3,
        patrol_attack_period=6,
        patrol_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
    room_item_id=ROOM_ITEM_SMALL_KEY,
    exit_routes=(
        DoorRoute("DOWN", ((120, 141), (120, 205))),
        DoorRoute("LEFT", ((120, 141), (32, 141))),
    ),
    max_frames=28000,
    level=LEVEL_5,
)

# Natural from cleared 0x27 WEST. 5× Gibdo — same combat as ROOM_66_SPEC.
ROOM_26_SPEC = DungeonRoomSpec(
    spec_id="level5_room26_gibdos",
    source_room=ROOM_L5_NORTH_27,
    room_id=ROOM_L5_WEST_26,
    entry=DoorRoute(
        "LEFT",
        ((224, 141), (208, 141)),
    ),
    enemy_types=(GIBDO_OBJECT_TYPE,),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_ROOM_66_PATROL,
        engage_distance=56,
        engage_attack_period=6,
        engage_attack_hold=3,
        patrol_attack_period=10,
        patrol_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    room_item_id=ROOM_ITEM_SMALL_KEY,
    exit_routes=(
        DoorRoute("LEFT", ((120, 141), (32, 141))),
        DoorRoute("RIGHT", ((120, 141), (208, 141))),
    ),
    max_frames=28000,
    level=LEVEL_5,
)

# Natural from cleared 0x26 WEST. 5× Pols Voice — same combat as ROOM_77_SPEC.
ROOM_25_SPEC = DungeonRoomSpec(
    spec_id="level5_room25_pols_voice",
    source_room=ROOM_L5_WEST_26,
    room_id=ROOM_L5_WEST_25,
    entry=DoorRoute(
        "LEFT",
        ((224, 141), (208, 141)),
    ),
    enemy_types=(POLS_VOICE_OBJECT_TYPE,),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_ROOM_77_PATROL,
        engage_distance=72,
        engage_attack_period=5,
        engage_attack_hold=3,
        patrol_attack_period=6,
        patrol_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    room_item_id=ROOM_ITEM_NONE,
    exit_routes=(
        DoorRoute("RIGHT", ((120, 141), (208, 141))),
        DoorRoute("LEFT", ((120, 141), (32, 141))),
    ),
    max_frames=28000,
    level=LEVEL_5,
)

register_room_spec(ROOM_66_SPEC)
register_room_spec(ROOM_67_SPEC)
register_room_spec(ROOM_77_SPEC)
register_room_spec(ROOM_65_SPEC)
register_room_spec(ROOM_27_SPEC)
register_room_spec(ROOM_26_SPEC)
register_room_spec(ROOM_25_SPEC)


def level5_room_66_cleared(ram: np.ndarray) -> bool:
    """Isolated pure: 0x66 3× Gibdo dead, RoomAllDead≥20, east door bit 0x08."""
    return dungeon_room_cleared(ram, ROOM_66_SPEC)


def level5_in_room_66(ram: np.ndarray) -> bool:
    """Play mode inside L5 room 0x66 (pre- or post-clear)."""
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL_5
        and snap.screen == ROOM_L5_GIBDO_66
        and snap.mode == PLAY_MODE
    )


def level5_in_room_67(ram: np.ndarray) -> bool:
    """Play mode inside L5 residual room 0x67 (east of cleared 0x66)."""
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL_5
        and snap.screen == ROOM_L5_EAST_67
        and snap.mode == PLAY_MODE
    )


def level5_room_67_arrived(ram: np.ndarray) -> bool:
    """Graph stop: room-ready 0x67 with west door bit.

    Bubbles (type 0x40) spawn after a short settle; do not require them for the
    graph stop (sword-immune residual — arrival only, not clear).
    """
    snap = read_snapshot(ram)
    return (
        level5_in_room_67(ram)
        and (snap.cur_opened_doors & ROOM_67_WEST_DOOR_BIT) == ROOM_67_WEST_DOOR_BIT
    )


def level5_in_room_77(ram: np.ndarray) -> bool:
    """Play mode inside L5 Pols Voice room 0x77."""
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL_5
        and snap.screen == ROOM_L5_POLS_77
        and snap.mode == PLAY_MODE
    )


def level5_room_77_key_success(ram: np.ndarray) -> bool:
    """Isolated pure: 0x77 keys≥1 and no live Pols Voice (type 0x16).

    FIXED_INVENTORY stop: inventory + liveness only (RoomAllDead may lag).
    """
    return inventory_reward_success(ram, ROOM_77_SPEC, min_value=1)


def level5_in_room_65(ram: np.ndarray) -> bool:
    """Play mode inside L5 west room 0x65 (PARTIAL natural entry)."""
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL_5
        and snap.screen == ROOM_L5_WEST_65
        and snap.mode == PLAY_MODE
    )



def level5_in_room_56(ram: np.ndarray) -> bool:
    """Play mode inside L5 room 0x56 (north of 0x66)."""
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL_5
        and snap.screen == ROOM_L5_NORTH_56
        and snap.mode == PLAY_MODE
    )


def level5_room_56_arrived(ram: np.ndarray) -> bool:
    """Graph stop: room-ready 0x56 after free UP from 0x66."""
    return level5_in_room_56(ram)

def level5_room_65_arrived(ram: np.ndarray) -> bool:
    """Graph stop: room-ready 0x65 after the west key door from 0x66."""
    return level5_in_room_65(ram)


def level5_room_65_cleared(ram: np.ndarray) -> bool:
    """Isolated pure: 0x65 5× Gibdo dead, RoomAllDead settle (ROOM_66 rule)."""
    return dungeon_room_cleared(ram, ROOM_65_SPEC)


def level5_in_room_27(ram: np.ndarray) -> bool:
    """Play mode inside L5 room 0x27 (north of 0x37)."""
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL_5
        and snap.screen == ROOM_L5_NORTH_27
        and snap.mode == PLAY_MODE
    )


def level5_room_27_cleared(ram: np.ndarray) -> bool:
    """Isolated pure: 0x27 2 Pols + 2 Gibdo + 2 Keese dead."""
    return dungeon_room_cleared(ram, ROOM_27_SPEC)


def level5_in_room_26(ram: np.ndarray) -> bool:
    """Play mode inside L5 room 0x26 (west of 0x27)."""
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL_5
        and snap.screen == ROOM_L5_WEST_26
        and snap.mode == PLAY_MODE
    )


def level5_room_26_cleared(ram: np.ndarray) -> bool:
    """Isolated pure: 0x26 5× Gibdo dead (ROOM_66 rule)."""
    return dungeon_room_cleared(ram, ROOM_26_SPEC)


def level5_in_room_25(ram: np.ndarray) -> bool:
    """Play mode inside L5 room 0x25 (west of 0x26)."""
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL_5
        and snap.screen == ROOM_L5_WEST_25
        and snap.mode == PLAY_MODE
    )


def level5_room_25_cleared(ram: np.ndarray) -> bool:
    """Isolated pure: 0x25 5× Pols Voice dead (ROOM_77 rule)."""
    return dungeon_room_cleared(ram, ROOM_25_SPEC)


def level5_in_room_24(ram: np.ndarray) -> bool:
    """Play mode inside L5 room 0x24 (west of 0x25; Digdogger — arrival only)."""
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL_5
        and snap.screen == ROOM_L5_WEST_24
        and snap.mode == PLAY_MODE
    )


@dataclass
class Level5PolsVoiceController(GenericDungeonRoomController):
    """Pols Voice clear + tactical spacing/backstep and focus-fire.

    Pols Voice: HP=160 (10 wooden sword hits), hops in arcs, no knockback.
    Link transits from west door north then east into central aisle (x=96..144).
    Tactical evasion side-steps leaping Pols Voices perpendicularly and backsteps
    from grounded threats. Strikes only grounded enemies and retreats to maintain
    safe distance. Avoids false occupancy miss walling and outer block clusters.
    """

    last_progress_frame: int = 0
    prev_live_count: int = -1
    backstep_frames: int = 0
    backstep_dir: str = "LEFT"
    blocked_cells: set[tuple[int, int]] = field(default_factory=set)
    misses: int = 0
    stuck_frames: int = 0
    last_pos: tuple[int, int] | None = None
    last_dir: str | None = None
    attack_cooldown_frames: int = 0
    entered_central: bool = False
    enemy_prev_pos: dict[int, tuple[int, int]] = field(default_factory=dict)

    def _ahead_pos(
        self, x: int, y: int, direction: str, step: int = 4
    ) -> tuple[int, int]:
        if direction == "LEFT":
            return x - step, y
        if direction == "RIGHT":
            return x + step, y
        if direction == "UP":
            return x, y - step
        return x, y + step

    def _is_solid(self, x: int, y: int) -> bool:
        if (x, y) in self.blocked_cells:
            return True
        if x < 44 or x > 204 or y < 93 or y > 189:
            return True
        if y > 185 and x < 185:
            return True
        if 56 <= x <= 97 and 109 < y <= 164:
            return True
        if 145 <= x <= 184 and y >= 109:
            return True
        if x < 88 and y > 145:
            return True
        return False

    def _can_move(self, x: int, y: int, direction: str, step: int = 4) -> bool:
        if direction == "LEFT":
            return not self._is_solid(x - step, y)
        if direction == "RIGHT":
            return not self._is_solid(x + step, y)
        if direction == "UP":
            return not self._is_solid(x, y - step)
        if direction == "DOWN":
            return not self._is_solid(x, y + step)
        return False

    def _combat(
        self, snap: ZeldaSnapshot, live: tuple[ZeldaObject, ...]
    ) -> FrameAction:
        self.combat_frames += 1
        if self.attack_cooldown_frames > 0:
            self.attack_cooldown_frames -= 1

        n_live = len(live)
        if self.prev_live_count < 0:
            self.prev_live_count = n_live
            self.last_progress_frame = self.frames
        elif n_live < self.prev_live_count:
            self.prev_live_count = n_live
            self.last_progress_frame = self.frames
            self.notes.append(f"kill_to_{n_live}_f{self.frames}")

        if not live:
            self.last_dir = None
            return FrameAction(nes_idle_action(), "combat_all_dead")

        lx, ly = int(snap.link_x), int(snap.link_y)
        if 96 <= lx <= 144 and 112 <= ly <= 168:
            self.entered_central = True

        def _cheb(x1: int, y1: int, x2: int, y2: int) -> int:
            return max(abs(x1 - x2), abs(y1 - y2))

        def _manh(x1: int, y1: int, x2: int, y2: int) -> int:
            return abs(x1 - x2) + abs(y1 - y2)

        # Track enemy velocities and projected positions
        projected = []
        for e in live:
            old_x, old_y = self.enemy_prev_pos.get(e.slot, (e.x, e.y))
            vx = e.x - old_x
            vy = e.y - old_y
            self.enemy_prev_pos[e.slot] = (e.x, e.y)
            proj_x = e.x + (vx * 4 if e.state == 1 else 0)
            proj_y = e.y + (vy * 4 if e.state == 1 else 0)
            projected.append((e, proj_x, proj_y))

        # Occupancy miss tracking: only when NOT in attack cooldown and NOT backstepping
        if self.last_pos == (lx, ly):
            if (
                self.last_dir is not None
                and self.attack_cooldown_frames == 0
                and self.backstep_frames == 0
            ):
                self.stuck_frames += 1
                if self.stuck_frames >= 4:
                    cell_ahead = self._ahead_pos(lx, ly, self.last_dir)
                    if not any(
                        _cheb(cell_ahead[0], cell_ahead[1], e.x, e.y) <= 20
                        for e in live
                    ):
                        self.blocked_cells.add(cell_ahead)
                        self.misses += 1
                        self.notes.append(
                            f"occupancy_miss_{self.last_dir}_{cell_ahead}"
                        )
                    self.stuck_frames = 0
        else:
            self.stuck_frames = 0
            self.last_pos = (lx, ly)

        if lx < 44:
            self.last_dir = "RIGHT"
            return FrameAction(nes_action("RIGHT"), "enter_room")

        closest_enemy = min(live, key=lambda o: _cheb(lx, ly, o.x, o.y))
        min_cheb = _cheb(lx, ly, closest_enemy.x, closest_enemy.y)

        # Uninterruptible backstep / evasion continuation
        if self.backstep_frames > 0:
            self.backstep_frames -= 1
            if self._can_move(lx, ly, self.backstep_dir):
                self.last_dir = self.backstep_dir
                return FrameAction(
                    nes_action(self.backstep_dir),
                    f"backstep_{self.backstep_dir}",
                )
            for alt in ("UP", "DOWN", "LEFT", "RIGHT"):
                if self._can_move(lx, ly, alt):
                    self.backstep_dir = alt
                    self.last_dir = alt
                    return FrameAction(
                        nes_action(alt),
                        f"backstep_alt_{alt}",
                    )
            self.backstep_frames = 0

        # Helper: find safest move away from all enemies and walls
        def safest_move(dirs: tuple[str, ...]) -> str | None:
            valid = [d for d in dirs if self._can_move(lx, ly, d)]
            if not valid:
                return None

            def safety_score(d: str) -> float:
                nx, ny = self._ahead_pos(lx, ly, d, step=6)
                min_cheb_d = min(
                    min(_cheb(nx, ny, e.x, e.y), _cheb(nx, ny, px, py))
                    for e, px, py in projected
                )
                min_eucl_sq = min(
                    min(
                        (nx - e.x) ** 2 + (ny - e.y) ** 2,
                        (nx - px) ** 2 + (ny - py) ** 2,
                    )
                    for e, px, py in projected
                )
                center_dist = abs(nx - 120) + abs(ny - 141)
                center_bonus = max(0, 40 - center_dist)
                aisle_bonus = 20 if (96 <= nx <= 144 and 115 <= ny <= 165) else 0
                wall_penalty = -30 if (ny <= 97 or ny >= 181 or nx <= 48 or nx >= 200) else 0
                danger_penalty = -100 if min_cheb_d <= 14 else 0
                arena_penalty = -50 if (self.entered_central and (nx < 96 or nx > 144 or ny < 112 or ny > 165)) else 0
                return (
                    min_cheb_d * 20 + (min_eucl_sq ** 0.5) + center_bonus
                    + aisle_bonus + wall_penalty + danger_penalty + arena_penalty
                )

            return max(valid, key=safety_score)

        # Emergency contact evasion: Pols Voice within 14 px
        if min_cheb <= 14:
            c = safest_move(("LEFT", "RIGHT", "UP", "DOWN"))
            if c:
                self.backstep_dir = c
                self.backstep_frames = 6
                self.last_dir = c
                return FrameAction(nes_action(c), f"evade_contact_{c}")
            dx = closest_enemy.x - lx
            dy = closest_enemy.y - ly
            c_dir = ("RIGHT" if dx > 0 else "LEFT") if abs(dx) >= abs(dy) else ("DOWN" if dy > 0 else "UP")
            self.last_dir = None
            self.attack_cooldown_frames = 12
            if snap.facing != direction_to_facing(c_dir):
                return FrameAction(nes_action(c_dir, "A"), f"corner_strike_{c_dir}")
            return FrameAction(
                nes_action("A") if self.combat_frames % 4 < 3 else nes_idle_action(),
                "corner_swing",
            )

        # West door exit transit: route north then east into central aisle
        if lx < 88:
            if ly > 109 and self._can_move(lx, ly, "UP"):
                self.last_dir = "UP"
                return FrameAction(nes_action("UP"), "route_north_from_west_door")
            elif self._can_move(lx, ly, "RIGHT"):
                self.last_dir = "RIGHT"
                return FrameAction(nes_action("RIGHT"), "traverse_east_to_aisle")

        if lx < 112 and ly <= 109 and self._can_move(lx, ly, "RIGHT"):
            self.last_dir = "RIGHT"
            return FrameAction(nes_action("RIGHT"), "reach_central_aisle")

        # Arena re-entry if knocked out of central arena
        if self.entered_central:
            for chk, d in ((ly > 165, "UP"), (ly < 112, "DOWN"), (lx < 96, "RIGHT"), (lx > 144, "LEFT")):
                if chk and self._can_move(lx, ly, d):
                    self.last_dir = d
                    return FrameAction(nes_action(d), f"arena_return_{d}")

        # Strike Timing against Grounded Enemy
        c_dx = closest_enemy.x - lx
        c_dy = closest_enemy.y - ly
        col_aligned = abs(c_dx) <= 12 and 12 <= abs(c_dy) <= 24
        row_aligned = abs(c_dy) <= 12 and 12 <= abs(c_dx) <= 24

        if (col_aligned or row_aligned) and closest_enemy.state == 0:
            if col_aligned:
                dir_to = "DOWN" if c_dy > 0 else "UP"
            else:
                dir_to = "RIGHT" if c_dx > 0 else "LEFT"
            self.last_dir = None
            best_retreat = safest_move(("UP", "DOWN", "LEFT", "RIGHT"))
            if best_retreat:
                self.backstep_dir = best_retreat
                self.backstep_frames = 6
            self.attack_cooldown_frames = 12
            if snap.facing != direction_to_facing(dir_to):
                return FrameAction(
                    nes_action(dir_to, "A"), f"turn_strike_{dir_to}"
                )
            return FrameAction(nes_action("A"), "strike_grounded_retreat")

        # Leaping Threat Evasion: Pols Voice leaping towards Link (state == 1, dist <= 36)
        leaping_threats = [
            e for e in live if e.state == 1 and _cheb(lx, ly, e.x, e.y) <= 36
        ]
        if leaping_threats:
            threat = min(leaping_threats, key=lambda o: _cheb(lx, ly, o.x, o.y))
            tdx = threat.x - lx
            tdy = threat.y - ly
            if abs(tdx) >= abs(tdy):
                p_dirs = ("DOWN", "UP") if ly < 141 else ("UP", "DOWN")
            else:
                p_dirs = ("RIGHT", "LEFT") if lx < 120 else ("LEFT", "RIGHT")
            c = safest_move(p_dirs)
            if not c:
                c = safest_move(("UP", "DOWN", "LEFT", "RIGHT"))
            if c:
                self.backstep_dir = c
                self.backstep_frames = 5
                self.last_dir = c
                return FrameAction(nes_action(c), f"evade_leap_{c}")

        if 96 <= lx <= 144 and ly < 125:
            if self._can_move(lx, ly, "DOWN"):
                self.last_dir = "DOWN"
                return FrameAction(nes_action("DOWN"), "center_down")

        # Target selection: prioritize grounded and wounded enemies near central aisle
        def score(o: ZeldaObject) -> int:
            d = _manh(lx, ly, o.x, o.y)
            hp_cost = (o.hp // 16) * 15
            state_cost = 30 if o.state == 1 else 0
            pocket_cost = 40 if (o.x < 88 and o.y > 109) else 0
            return d + hp_cost + state_cost + pocket_cost

        tgt = min(live, key=score)
        dx = tgt.x - lx
        dy = tgt.y - ly

        steps = []
        if abs(dx) >= abs(dy):
            if abs(dy) > 4:
                steps.append("DOWN" if dy > 0 else "UP")
            steps.append("RIGHT" if dx > 0 else "LEFT")
            steps.append("DOWN" if dy > 0 else "UP")
            steps.append("LEFT" if dx > 0 else "RIGHT")
        else:
            if abs(dx) > 4:
                steps.append("RIGHT" if dx > 0 else "LEFT")
            steps.append("DOWN" if dy > 0 else "UP")
            steps.append("RIGHT" if dx > 0 else "LEFT")
            steps.append("UP" if dy > 0 else "DOWN")

        for step_d in steps:
            nx, ny = self._ahead_pos(lx, ly, step_d)
            if (
                min(_cheb(nx, ny, e.x, e.y) for e in live) >= 14
                and self._can_move(lx, ly, step_d)
            ):
                self.last_dir = step_d
                return FrameAction(nes_action(step_d), f"move_{step_d}")

        c = safest_move(("UP", "DOWN", "LEFT", "RIGHT"))
        if c:
            self.last_dir = c
            return FrameAction(nes_action(c), f"safe_step_{c}")

        self.last_dir = None
        return FrameAction(nes_idle_action(), "stand_no_path")

    def _collect_reward(self, snap: ZeldaSnapshot) -> FrameAction:
        self.entered_central = False
        lx, ly = int(snap.link_x), int(snap.link_y)
        if (lx >= 185 or lx < 88) and ly > 109:
            if self._can_move(lx, ly, "UP"):
                return FrameAction(nes_action("UP"), "collect_route_north")
        if lx < 88 and self._can_move(lx, ly, "RIGHT"):
            return FrameAction(nes_action("RIGHT"), "collect_route_east")
        return super()._collect_reward(snap)


def make_pols_voice_controller() -> Level5PolsVoiceController:
    """Factory for room-77 Pols Voice + key controller."""
    return Level5PolsVoiceController(spec=ROOM_77_SPEC)


@dataclass
class Level5East67Controller:
    """Walk east from cleared 0x66 into residual 0x67; stop on arrival.

    No combat clear — Bubbles are sword-immune. Success = room-ready 0x67.
    Standalone (not GenericDungeonRoomController) so we skip clear logic.
    """

    max_frames: int = 4000
    settle_frames: int = 45
    frames: int = 0
    settle_left: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)

    def report(self) -> dict:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": ROOM_67_SPEC.spec_id,
        }

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        from retro_harness.nes import nes_idle_action

        self.frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            self.failed = True
            return FrameAction(nes_idle_action(), "timeout")

        if snap.mode == 17:
            self.failed = True
            self.notes.append("link_death")
            return FrameAction(nes_idle_action(), "link_death")

        if (
            snap.level == LEVEL_5
            and snap.screen == ROOM_L5_EAST_67
            and snap.mode == PLAY_MODE
        ):
            if self.settle_left <= 0 and "settling_67" not in self.notes:
                self.settle_left = self.settle_frames
                self.notes.append("settling_67")
            if self.settle_left > 0:
                self.settle_left -= 1
                if self.settle_left > 0:
                    return FrameAction(nes_idle_action(), "settle_67")
            self.success = True
            self.notes.append("arrived_67")
            return FrameAction(nes_idle_action(), "arrived_67")

        if snap.transitioning or snap.mode != PLAY_MODE:
            return FrameAction(nes_action("RIGHT"), "scroll")

        # Leave south-ish positions then push RIGHT at y≈141.
        if snap.link_y > 170:
            return FrameAction(nes_action("UP"), "leave_south")
        if abs(snap.link_y - 141) > 4:
            btn = "UP" if snap.link_y > 141 else "DOWN"
            return FrameAction(nes_action(btn), "align_east_y")
        return FrameAction(nes_action("RIGHT"), "enter_67")


def make_east_67_controller() -> Level5East67Controller:
    return Level5East67Controller()


__all__ = [
    "LEVEL_5",
    "ROOM_L5_ENTRY",
    "ROOM_L5_GIBDO_66",
    "ROOM_L5_EAST_67",
    "ROOM_L5_POLS_77",
    "ROOM_L5_WEST_65",
    "ROOM_L5_NORTH_55",
    "ROOM_L5_NORTH_56",
    "ROOM_L5_NORTH_27",
    "ROOM_L5_WEST_26",
    "ROOM_L5_WEST_25",
    "ROOM_L5_WEST_24",
    "GIBDO_OBJECT_TYPE",
    "POLS_VOICE_OBJECT_TYPE",
    "BUBBLE_OBJECT_TYPE",
    "ROOM_67_TRAP_TYPE",
    "ZOL_OBJECT_TYPE",
    "ROOM_ITEM_SMALL_KEY",
    "ROOM_ITEM_NONE",
    "ROOM_66_EAST_DOOR_BIT",
    "ROOM_67_WEST_DOOR_BIT",
    "ROOM_65_EAST_DOOR_BIT",
    "ROOM_66_SPEC",
    "ROOM_67_SPEC",
    "ROOM_77_SPEC",
    "ROOM_65_SPEC",
    "ROOM_27_SPEC",
    "ROOM_26_SPEC",
    "ROOM_25_SPEC",
    "level5_room_66_cleared",
    "level5_in_room_66",
    "level5_in_room_67",
    "level5_room_67_arrived",
    "level5_in_room_77",
    "level5_room_77_key_success",
    "level5_in_room_65",
    "level5_in_room_56",
    "level5_room_56_arrived",
    "level5_room_65_arrived",
    "level5_room_65_cleared",
    "level5_in_room_27",
    "level5_room_27_cleared",
    "level5_in_room_26",
    "level5_room_26_cleared",
    "level5_in_room_25",
    "level5_room_25_cleared",
    "level5_in_room_24",
    "Level5PolsVoiceController",
    "Level5East67Controller",
    "make_pols_voice_controller",
    "make_east_67_controller",
]
