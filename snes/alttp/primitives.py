"""Controller-only navigation, combat, pickup, and interaction primitives.

These helpers deliberately operate on observed RAM and controller input.  They
do not write progression state, warp rooms, or assume that every active sprite
slot belongs to the part of a large room currently visible on screen.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from retro_harness.adventure.nav import Waypoint as SharedWaypoint
from retro_harness.adventure.nav import direction_to_waypoint, reached_waypoint
from alttp.dungeon.tilemap import read_room_tiles
from alttp.ram import ABILITY_DASH, MODULE, PEG_STATE, AlttpSnapshot, read_inventory, wram_index
from alttp.sprite_ids import vulnerability
from alttp.startup import action_for, no_action, snapshot_env, step_frames

Buttons: TypeAlias = tuple[str, ...]
ButtonScript: TypeAlias = Sequence[tuple[Buttons, int]]
SnapshotPredicate: TypeAlias = Callable[[AlttpSnapshot], bool]

SPRITE_Y_LOW = 0x0D00
SPRITE_X_LOW = 0x0D10
SPRITE_Y_HIGH = 0x0D20
SPRITE_X_HIGH = 0x0D30
SPRITE_AI = 0x0D80
SPRITE_STATE = 0x0DD0
SPRITE_DIRECTION = 0x0DE0
SPRITE_TYPE = 0x0E20
SPRITE_HP = 0x0E50
# Hit reaction. Counts down while bit 7 is set. Sword contact on CastleB1Guard
# type 0x41 moved this; stun table $0B58 stayed 0 (no hookshot on that pin).
SPRITE_HIT_TIMER = 0x0EF0
# Sprite layer. 0 matches Link on $EE == 0; the other-floor soldier was 1.
SPRITE_FLOOR = 0x0F20
SPRITE_SLOTS = 16

# Ten ancilla slots. Type 0 is empty. Soldier arrows are sprite 0x1B, not here.
ANCILLA_SLOTS = 10
ANCILLA_Y_LOW = 0x0BFA
ANCILLA_X_LOW = 0x0C04
ANCILLA_Y_HIGH = 0x0C0E
ANCILLA_X_HIGH = 0x0C18
ANCILLA_TYPE = 0x0C4A
ANCILLA_TIMER = 0x0C68
ANCILLA_FLOOR = 0x0C7C

# Link's shots and bombs (Zelda_3_RAM.log $0C4A). Visual ancilla are omitted.
LINK_PROJECTILE_ANCILLA = frozenset(
    {
        0x01,  # somaria blast
        0x02,  # fire rod shot
        0x04,  # beam hit
        0x05,  # boomerang
        0x07,  # bomb
        0x09,  # arrow
        0x0A,  # halted arrow
        0x0B,  # ice rod shot
        0x0C,  # sword beam
        0x11,  # ice shot spread
        0x13,  # ice shot sparkle
        0x1F,  # hookshot
        0x3A,  # super bomb explosion
    }
)

SPRITE_BLUE_SOLDIER = 0x41
SPRITE_GREEN_SOLDIER = 0x42
SPRITE_KNIGHT = 0x4B
SPRITE_BALL_AND_CHAIN = 0x6A
SPRITE_ZELDA = 0x76
SPRITE_HEART = 0xD8
SPRITE_CRYSTAL_SWITCH = 0x1E
SPRITE_SMALL_KEY = 0xE4
SPRITE_BIG_KEY = 0xE5
SPRITE_MANTLE = 0xEE

CASTLE_HOSTILE_TYPES = frozenset(
    {
        SPRITE_BLUE_SOLDIER,
        SPRITE_GREEN_SOLDIER,
        SPRITE_KNIGHT,
        SPRITE_BALL_AND_CHAIN,
    }
)

DIRECTION_BUTTONS = frozenset({"UP", "DOWN", "LEFT", "RIGHT"})


@dataclass(frozen=True)
class SpriteSnapshot:
    """One active sprite slot with full 16-bit room coordinates."""

    slot: int
    sprite_type: int
    state: int
    hp: int
    x: int
    y: int
    ai_state: int
    direction: int
    floor: int
    hit_timer: int

    def distance_to(self, snapshot: AlttpSnapshot) -> int:
        return abs(self.x - snapshot.link_x) + abs(self.y - snapshot.link_y)


@dataclass(frozen=True)
class AncillaSnapshot:
    """One active Link projectile or bomb in the ancilla table."""

    slot: int
    ancilla_type: int
    x: int
    y: int
    floor: int
    timer: int

    def distance_to(self, snapshot: AlttpSnapshot) -> int:
        return abs(self.x - snapshot.link_x) + abs(self.y - snapshot.link_y)


@dataclass(frozen=True)
class PrimitiveResult:
    """Observed outcome from a bounded controller primitive."""

    ok: bool
    reason: str
    frames: int
    snapshot: AlttpSnapshot


@dataclass(frozen=True)
class Waypoint:
    """An absolute in-room navigation target (ALTTP room-aware).

    Mirrors :class:`retro_harness.adventure.nav.Waypoint` with an optional room id
    for multi-screen castle chambers.
    """

    x: int
    y: int
    tolerance: int = 5
    room: int | None = None
    label: str = ""

    def as_shared(self) -> SharedWaypoint:
        return SharedWaypoint(
            x=self.x,
            y=self.y,
            tolerance=self.tolerance,
            label=self.label,
            room=self.room,
        )

    def reached(self, x: int, y: int) -> bool:
        return reached_waypoint(x, y, self.as_shared())

    def direction_from(self, x: int, y: int) -> str | None:
        return direction_to_waypoint(x, y, self.as_shared())


@dataclass(frozen=True)
class CombatResult:
    """Combat outcome, including the actual slots defeated."""

    ok: bool
    reason: str
    frames: int
    snapshot: AlttpSnapshot
    defeated_slots: tuple[int, ...] = ()


def active_sprites(env: object) -> tuple[SpriteSnapshot, ...]:
    """Read active sprites using the correct 16-bit X/Y coordinates.

    Large indoor rooms can keep sprites for off-screen subrooms active.  The
    high bytes are therefore required; comparing only low bytes aliases those
    sprites onto the current screen and makes combat chase false duplicates.
    """
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    sprites: list[SpriteSnapshot] = []
    for slot in range(SPRITE_SLOTS):
        state = int(ram[SPRITE_STATE + slot])
        if state == 0:
            continue
        sprites.append(
            SpriteSnapshot(
                slot=slot,
                sprite_type=int(ram[SPRITE_TYPE + slot]),
                state=state,
                hp=int(ram[SPRITE_HP + slot]),
                x=(
                    int(ram[SPRITE_X_LOW + slot])
                    | (int(ram[SPRITE_X_HIGH + slot]) << 8)
                ),
                y=(
                    int(ram[SPRITE_Y_LOW + slot])
                    | (int(ram[SPRITE_Y_HIGH + slot]) << 8)
                ),
                ai_state=int(ram[SPRITE_AI + slot]),
                direction=int(ram[SPRITE_DIRECTION + slot]),
                floor=int(ram[SPRITE_FLOOR + slot]),
                hit_timer=int(ram[SPRITE_HIT_TIMER + slot]),
            )
        )
    return tuple(sprites)


def active_ancillae(env: object) -> tuple[AncillaSnapshot, ...]:
    """Link's projectiles and bombs. Enemy shots are sprites, not ancilla.

    Confirmed empty on CastleB1Guard (sword only; no bow, bomb, or boomerang).
    A soldier arrow is sprite type ``0x1B`` and will not show up here.
    """
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    found: list[AncillaSnapshot] = []
    for slot in range(ANCILLA_SLOTS):
        kind = int(ram[ANCILLA_TYPE + slot])
        if kind not in LINK_PROJECTILE_ANCILLA:
            continue
        found.append(
            AncillaSnapshot(
                slot=slot,
                ancilla_type=kind,
                x=(
                    int(ram[ANCILLA_X_LOW + slot])
                    | (int(ram[ANCILLA_X_HIGH + slot]) << 8)
                ),
                y=(
                    int(ram[ANCILLA_Y_LOW + slot])
                    | (int(ram[ANCILLA_Y_HIGH + slot]) << 8)
                ),
                floor=int(ram[ANCILLA_FLOOR + slot]),
                timer=int(ram[ANCILLA_TIMER + slot]),
            )
        )
    return tuple(found)


def sprites_of_type(
    env: object,
    sprite_types: Iterable[int],
    *,
    max_distance: int | None = None,
) -> tuple[SpriteSnapshot, ...]:
    """Return active sprites of the requested types, optionally screen-local."""
    wanted = frozenset(int(value) for value in sprite_types)
    snapshot = snapshot_env(env)
    found = tuple(
        sprite for sprite in active_sprites(env) if sprite.sprite_type in wanted
    )
    if max_distance is None:
        return found
    return tuple(
        sprite
        for sprite in found
        if abs(sprite.x - snapshot.link_x) <= max_distance
        and abs(sprite.y - snapshot.link_y) <= max_distance
    )


def run_script(
    env: object,
    script: ButtonScript,
    *,
    stop_when: SnapshotPredicate | None = None,
) -> PrimitiveResult:
    """Run a bounded multi-button script with optional RAM acceptance."""
    frames = 0
    for buttons, hold in script:
        action = (
            no_action()
            if not buttons or buttons == ("NONE",)
            else action_for(*buttons)
        )
        for _ in range(max(0, hold)):
            env.step(action)  # type: ignore[attr-defined]
            frames += 1
            snapshot = snapshot_env(env)
            if stop_when is not None and stop_when(snapshot):
                return PrimitiveResult(True, "acceptance reached", frames, snapshot)
            if snapshot.game_mode == 0x12:
                return PrimitiveResult(False, "Link died", frames, snapshot)
    snapshot = snapshot_env(env)
    return PrimitiveResult(
        stop_when is None or stop_when(snapshot),
        "script complete" if stop_when is None else "acceptance not reached",
        frames,
        snapshot,
    )


def settle_control(env: object, *, max_frames: int = 240) -> PrimitiveResult:
    """Wait for ordinary gameplay control, advancing item/text sequences.

    ``game_mode == 0x07`` alone is not enough: item pickups spend roughly one
    hundred frames in the hold-up pose before opening their message.  Route
    movement must not start while ``link_action == HoldUpItem``.
    """
    frames = 0
    while frames < max_frames:
        snapshot = snapshot_env(env)
        if (
            snapshot.has_control
            and not snapshot.is_text_mode
            and not snapshot.is_hold_up_item
        ):
            return PrimitiveResult(True, "control ready", frames, snapshot)
        if snapshot.game_mode == 0x12:
            return PrimitiveResult(False, "Link died", frames, snapshot)
        if snapshot.is_text_mode:
            button = "A" if (frames // 4) % 2 == 0 else "B"
            step_frames(env, action_for(button), 2)
            step_frames(env, no_action(), 2)
            frames += 4
        else:
            step_frames(env, no_action(), 4)
            frames += 4
    return PrimitiveResult(False, "control timeout", frames, snapshot_env(env))


def move_to(
    env: object,
    waypoint: Waypoint,
    *,
    max_frames: int = 600,
    step_size: int = 3,
    stuck_cycles: int = 28,
) -> PrimitiveResult:
    """Move toward an absolute waypoint with feedback and bounded failure.

    A waypoint is intentionally not a path planner.  Route code supplies
    intermediate points around walls, pits, statues, and ledges.
    """
    frames = 0
    unchanged = 0
    previous_xy: tuple[int, int] | None = None
    while frames < max_frames:
        snapshot = snapshot_env(env)
        if waypoint.room is not None and snapshot.room_base_id != waypoint.room:
            return PrimitiveResult(
                False,
                (
                    f"left room 0x{waypoint.room:02X} before "
                    f"{waypoint.label or 'waypoint'}"
                ),
                frames,
                snapshot,
            )
        if waypoint.reached(snapshot.link_x, snapshot.link_y):
            return PrimitiveResult(True, waypoint.label or "waypoint reached", frames, snapshot)
        if snapshot.game_mode == 0x12:
            return PrimitiveResult(False, "Link died", frames, snapshot)

        # Multi-button diagonal keeps simultaneous axes (direction_from is
        # single-cardinal only and would slow diagonals).
        dx = waypoint.x - snapshot.link_x
        dy = waypoint.y - snapshot.link_y
        buttons: list[str] = []
        if abs(dx) > waypoint.tolerance:
            buttons.append("RIGHT" if dx > 0 else "LEFT")
        if abs(dy) > waypoint.tolerance:
            buttons.append("DOWN" if dy > 0 else "UP")
        step_frames(env, action_for(*buttons), step_size)
        frames += step_size

        xy = (snapshot.link_x, snapshot.link_y)
        if xy == previous_xy:
            unchanged += 1
        else:
            unchanged = 0
            previous_xy = xy
        if unchanged >= stuck_cycles:
            return PrimitiveResult(
                False,
                f"stuck before {waypoint.label or 'waypoint'}",
                frames,
                snapshot_env(env),
            )
    return PrimitiveResult(
        False,
        f"timeout before {waypoint.label or 'waypoint'}",
        frames,
        snapshot_env(env),
    )


def move_path(
    env: object,
    waypoints: Sequence[Waypoint],
    *,
    max_frames_per_waypoint: int = 600,
) -> PrimitiveResult:
    """Follow explicit waypoints, stopping on the first failed segment."""
    frames = 0
    for waypoint in waypoints:
        result = move_to(
            env,
            waypoint,
            max_frames=max_frames_per_waypoint,
        )
        frames += result.frames
        if not result.ok:
            return PrimitiveResult(False, result.reason, frames, result.snapshot)
    return PrimitiveResult(True, "path complete", frames, snapshot_env(env))


def move_until(
    env: object,
    buttons: Buttons,
    predicate: SnapshotPredicate,
    *,
    max_frames: int = 600,
    step_size: int = 2,
) -> PrimitiveResult:
    """Hold movement until a RAM predicate succeeds or the bound expires."""
    frames = 0
    action = action_for(*buttons)
    while frames < max_frames:
        snapshot = snapshot_env(env)
        if predicate(snapshot):
            return PrimitiveResult(True, "acceptance reached", frames, snapshot)
        if snapshot.game_mode == 0x12:
            return PrimitiveResult(False, "Link died", frames, snapshot)
        step_frames(env, action, step_size)
        frames += step_size
    snapshot = snapshot_env(env)
    return PrimitiveResult(
        predicate(snapshot),
        "acceptance reached" if predicate(snapshot) else "movement timeout",
        frames,
        snapshot,
    )


def spin_attack(env: object, *, charge_frames: int = 100) -> PrimitiveResult:
    """Charge and release the fighter-sword spin attack."""
    return run_script(
        env,
        (
            (("B",), charge_frames),
            (("NONE",), 24),
        ),
    )


def hit_switch(
    env: object,
    target: tuple[int, int] | None = None,
    *,
    room: int | None = None,
) -> PrimitiveResult:
    """Sword-hit one crystal switch. Ok when ``$C172`` changes in play.

    Measured hit, no RAM write of the peg byte: walk to the crystal, up to
    six B taps, then one spin (charge 80) if the byte is unchanged.
    HeraEntry crystal (3840, 3968) and room 0x87 crystal (3648, 4432) both
    toggle it. Raised pegs are attribute 0x67 and stay walls.
    """
    frames = 0
    if not _in_play(env):
        return PrimitiveResult(False, "not in play", 0, snapshot_env(env))
    snap = snapshot_env(env)
    expected = snap.room_base_id if room is None else int(room)
    before = _ram_u8(env, PEG_STATE)
    if target is None:
        crystals = sprites_of_type(env, (SPRITE_CRYSTAL_SWITCH,))
        if not crystals:
            return PrimitiveResult(False, "no crystal", 0, snap)
        crystal = min(crystals, key=lambda row: row.distance_to(snap))
        target = (crystal.x, crystal.y)
    moved = move_to(
        env,
        Waypoint(int(target[0]), int(target[1]), tolerance=12, room=expected, label="crystal"),
        max_frames=700,
    )
    frames += int(moved.frames)
    snap = snapshot_env(env)
    if snap.room_base_id != expected:
        return PrimitiveResult(False, f"left room 0x{expected:02X}", frames, snap)
    dx = int(target[0]) - snap.link_x
    dy = int(target[1]) - snap.link_y
    if dx or dy:
        step_frames(env, action_for(_face_toward(dx, dy)), 2)
        frames += 2
    for _ in range(6):
        step_frames(env, action_for("B"), 8)
        step_frames(env, no_action(), 8)
        frames += 16
        if _ram_u8(env, PEG_STATE) != before:
            break
    if _ram_u8(env, PEG_STATE) == before:
        spun = spin_attack(env, charge_frames=80)
        frames += int(spun.frames)
        settled = settle_control(env, max_frames=120)
        frames += int(settled.frames)
        step_frames(env, no_action(), 30)
        frames += 30
    after = _ram_u8(env, PEG_STATE)
    snap = snapshot_env(env)
    if after != before and snap.room_base_id == expected:
        return PrimitiveResult(
            True, f"peg ${before:02X}->${after:02X} at {target}", frames, snap
        )
    return PrimitiveResult(
        False, f"peg stayed ${before:02X} at {target}", frames, snap
    )


def _face_toward(dx: int, dy: int) -> str:
    if abs(dx) >= abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    return "DOWN" if dy > 0 else "UP"


def fight_nearby(
    env: object,
    *,
    sprite_types: Iterable[int] = CASTLE_HOSTILE_TYPES,
    room: int | None = None,
    max_distance: int = 150,
    attack_distance: int = 60,
    stop_when: Callable[[object], bool] | None = None,
    max_cycles: int = 500,
) -> CombatResult:
    """Fight the nearest on-screen target with short movement and sword taps.

    ``stop_when`` should encode progression when a fight produces an item or
    opens a door.  That lets callers stop as soon as the key appears rather
    than chasing an off-screen guard in another subroom.
    """
    wanted = frozenset(int(value) for value in sprite_types)
    start = snapshot_env(env)
    expected_room = start.room_base_id if room is None else room
    frames = 0
    initial_slots = {
        sprite.slot
        for sprite in sprites_of_type(env, wanted, max_distance=max_distance)
    }
    for _ in range(max_cycles):
        snapshot = snapshot_env(env)
        if stop_when is not None and stop_when(env):
            defeated = initial_slots - {
                sprite.slot
                for sprite in sprites_of_type(
                    env, wanted, max_distance=max_distance
                )
            }
            return CombatResult(
                True,
                "combat acceptance reached",
                frames,
                snapshot,
                tuple(sorted(defeated)),
            )
        if snapshot.room_base_id != expected_room:
            return CombatResult(
                False,
                f"left combat room 0x{expected_room:02X}",
                frames,
                snapshot,
            )
        if snapshot.game_mode == 0x12:
            return CombatResult(False, "Link died", frames, snapshot)

        targets = sprites_of_type(env, wanted, max_distance=max_distance)
        if not targets:
            defeated = initial_slots - {
                sprite.slot
                for sprite in sprites_of_type(
                    env, wanted, max_distance=max_distance
                )
            }
            return CombatResult(
                True,
                "no nearby targets",
                frames,
                snapshot,
                tuple(sorted(defeated)),
            )

        target = min(targets, key=lambda sprite: sprite.distance_to(snapshot))
        dx = target.x - snapshot.link_x
        dy = target.y - snapshot.link_y
        if abs(dx) <= attack_distance and abs(dy) <= attack_distance:
            step_frames(env, action_for(_face_toward(dx, dy)), 2)
            step_frames(env, action_for("B"), 4)
            step_frames(env, no_action(), 5)
            frames += 11
            continue

        buttons: list[str] = []
        if abs(dx) > attack_distance - 8:
            buttons.append("RIGHT" if dx > 0 else "LEFT")
        if abs(dy) > attack_distance - 8:
            buttons.append("DOWN" if dy > 0 else "UP")
        step_frames(env, action_for(*buttons), 3)
        frames += 3

    return CombatResult(
        False,
        "combat cycle limit",
        frames,
        snapshot_env(env),
        (),
    )


def collect_nearby(
    env: object,
    sprite_types: Iterable[int],
    *,
    max_distance: int = 180,
    max_frames: int = 480,
) -> PrimitiveResult:
    """Walk directly onto the nearest matching pickup."""
    frames = 0
    wanted = frozenset(int(value) for value in sprite_types)
    while frames < max_frames:
        snapshot = snapshot_env(env)
        items = sprites_of_type(env, wanted, max_distance=max_distance)
        if not items:
            return PrimitiveResult(True, "pickup collected", frames, snapshot)
        target = min(items, key=lambda sprite: sprite.distance_to(snapshot))
        dx = target.x - snapshot.link_x
        dy = target.y - snapshot.link_y
        buttons: list[str] = []
        if abs(dx) > 3:
            buttons.append("RIGHT" if dx > 0 else "LEFT")
        if abs(dy) > 3:
            buttons.append("DOWN" if dy > 0 else "UP")
        step_frames(
            env,
            no_action() if not buttons else action_for(*buttons),
            3,
        )
        frames += 3
        if snapshot.game_mode == 0x12:
            return PrimitiveResult(False, "Link died", frames, snapshot)
    return PrimitiveResult(False, "pickup timeout", frames, snapshot_env(env))


def interact_until(
    env: object,
    predicate: SnapshotPredicate,
    *,
    max_cycles: int = 80,
) -> PrimitiveResult:
    """Alternate A/B interactions and text advancement until accepted."""
    frames = 0
    for cycle in range(max_cycles):
        snapshot = snapshot_env(env)
        if predicate(snapshot):
            return PrimitiveResult(True, "interaction accepted", frames, snapshot)
        if snapshot.game_mode == 0x12:
            return PrimitiveResult(False, "Link died", frames, snapshot)
        button = "A" if cycle % 2 == 0 else "B"
        step_frames(env, action_for(button), 2)
        step_frames(env, no_action(), 3)
        frames += 5
    snapshot = snapshot_env(env)
    return PrimitiveResult(
        predicate(snapshot),
        "interaction accepted" if predicate(snapshot) else "interaction timeout",
        frames,
        snapshot,
    )


# hud_cur_item. The menu writes this when the cursor moves; it is not poked.
# FighterSwordLamp already reads 0x0B here, and $0303 (current_item_y) is 0x09.
HUD_CUR_ITEM = 0x0202
# overworld_map_state while the inventory cursor accepts the pad.
MENU_STATE = 0x0200
MENU_MODULE = 0x0E
MENU_CURSOR = 4
MENU_BOTTLE = 10
# Bottle possession is the index at $F34F, not one of the four contents bytes.
BOTTLE_INDEX = 0xF34F
# One owned item per step. Twenty Y slots plus the empty slot.
_CURSOR_STEPS = 21
# Scroll-down plus the cursor state. Measured open was 33 frames.
_MENU_OPEN_LIMIT = 80
_MENU_CLOSE_LIMIT = 80

# Hud slot, then the inventory field that must be non-zero to land on it.
# Slot 1 is bow at $F340; lamp is slot 0x0B. The original grid is 5 wide.
_Y_ITEMS: dict[str, tuple[int, str]] = {
    "bow": (0x01, "bow"),
    "boomerang": (0x02, "boomerang"),
    "hookshot": (0x03, "hookshot"),
    "bombs": (0x04, "bombs"),
    "bomb": (0x04, "bombs"),
    "powder": (0x05, "powder"),
    "mushroom": (0x05, "powder"),
    "fire_rod": (0x06, "fire_rod"),
    "ice_rod": (0x07, "ice_rod"),
    "bombos": (0x08, "bombos"),
    "ether": (0x09, "ether"),
    "quake": (0x0A, "quake"),
    "lamp": (0x0B, "lamp"),
    "hammer": (0x0C, "hammer"),
    "flute": (0x0D, "flute"),
    "shovel": (0x0D, "flute"),
    "net": (0x0E, "net"),
    "book": (0x0F, "book"),
    "bottle": (0x10, "bottle"),
    "somaria": (0x11, "somaria"),
    "byrna": (0x12, "byrna"),
    "cape": (0x13, "cape"),
    "mirror": (0x14, "mirror"),
}


def _ram_u8(env: object, addr: int) -> int:
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    if addr >= 0x2000:
        addr = wram_index(addr)
    return int(ram[addr])


def _owns_y_item(env: object, field: str) -> bool:
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    if field == "bottle":
        if _ram_u8(env, BOTTLE_INDEX) != 0:
            return True
        inv = read_inventory(ram)
        return any(int(getattr(inv, f"bottle_{i}")) != 0 for i in range(1, 5))
    return int(getattr(read_inventory(ram), field)) != 0


def _in_play(env: object) -> bool:
    snap = snapshot_env(env)
    return (
        snap.has_control
        and not snap.is_text_mode
        and not snap.is_hold_up_item
        and snap.game_mode != 0x12
    )


def _cursor_ready(env: object) -> bool:
    return (
        _ram_u8(env, MODULE) == MENU_MODULE
        and _ram_u8(env, MENU_STATE) in (MENU_CURSOR, MENU_BOTTLE)
    )


def _hold(env: object, button: str, count: int, frames: list[int]) -> None:
    action = no_action() if button == "NONE" else action_for(button)
    step_frames(env, action, count)
    frames[0] += count


def _until(env: object, ready: Callable[[], bool], limit: int, frames: list[int]) -> bool:
    for _ in range(limit):
        if ready():
            return True
        if _ram_u8(env, MODULE) == 0x12:
            return False
        _hold(env, "NONE", 1, frames)
    return ready()


def _open_menu(env: object, frames: list[int]) -> int | None:
    """Press START and return the frames until the cursor accepts input."""
    started = frames[0]
    if _cursor_ready(env):
        return 0
    if not _in_play(env):
        return None
    # START is an edge. A held START would also be the close press.
    _hold(env, "NONE", 1, frames)
    _hold(env, "START", 1, frames)
    _hold(env, "NONE", 1, frames)
    if not _until(env, lambda: _cursor_ready(env), _MENU_OPEN_LIMIT, frames):
        return None
    return frames[0] - started


def _close_menu(env: object, frames: list[int]) -> int | None:
    """Press START and return the frames until gameplay control is back."""
    started = frames[0]
    if not _cursor_ready(env):
        return None
    _hold(env, "NONE", 1, frames)
    _hold(env, "START", 1, frames)
    _hold(env, "NONE", 1, frames)
    if not _until(env, lambda: _in_play(env), _MENU_CLOSE_LIMIT, frames):
        return None
    return frames[0] - started


def _nudge_cursor(env: object, frames: list[int]) -> bool:
    """One RIGHT edge. True only when ``hud_cur_item`` changes on that press."""
    before = _ram_u8(env, HUD_CUR_ITEM)
    _hold(env, "RIGHT", 1, frames)
    changed = _ram_u8(env, HUD_CUR_ITEM) != before
    _hold(env, "NONE", 1, frames)
    if not _cursor_ready(env):
        # Landing on the bottle opens its submenu; wait until the pad works.
        _until(env, lambda: _cursor_ready(env), 40, frames)
    return changed


def _move_to_item(env: object, want: int, frames: list[int]) -> str:
    """Cycle RIGHT until ``want`` is selected. Empty string means it is."""
    start = _ram_u8(env, HUD_CUR_ITEM)
    changed = False
    for _ in range(_CURSOR_STEPS):
        if _ram_u8(env, HUD_CUR_ITEM) == want and (changed or start == want):
            # Already there: one press must move the byte or prove it cannot.
            if changed or start != want:
                return ""
            if not _nudge_cursor(env, frames):
                if _ram_u8(env, HUD_CUR_ITEM) == want and _cursor_ready(env):
                    return ""
                return "cursor did not change"
            changed = True
            continue
        if not _cursor_ready(env):
            return "menu lost the cursor"
        if not _nudge_cursor(env, frames):
            # The frame map state becomes 4 ignores one edge.
            if not _nudge_cursor(env, frames):
                return "cursor did not change"
        changed = True
        if _ram_u8(env, HUD_CUR_ITEM) == want:
            return ""
    if _ram_u8(env, HUD_CUR_ITEM) == want and changed:
        return ""
    return "cursor missed the item"


def select_y_item(env: object, name: str) -> PrimitiveResult:
    """Open the inventory, move the cursor onto ``name``, and close it.

    Confirmation is ``$0202`` changing onto that hud slot, or staying there
    when it is the only owned Y item (a RIGHT press then does not move).
    The selector never writes the byte.
    """
    frames = [0]
    key = name.strip().lower().replace(" ", "_")
    spec = _Y_ITEMS.get(key)
    if spec is None:
        return PrimitiveResult(False, f"unknown y item {name}", 0, snapshot_env(env))
    want, field = spec
    if not _owns_y_item(env, field):
        return PrimitiveResult(False, f"{key} is not owned", 0, snapshot_env(env))
    before = _ram_u8(env, HUD_CUR_ITEM)
    opened = _open_menu(env, frames)
    if opened is None:
        return PrimitiveResult(False, "menu did not open", frames[0], snapshot_env(env))
    reason = _move_to_item(env, want, frames)
    if reason:
        return PrimitiveResult(False, reason, frames[0], snapshot_env(env))
    if _ram_u8(env, HUD_CUR_ITEM) != want:
        return PrimitiveResult(
            False, "selected byte is not the item", frames[0], snapshot_env(env)
        )
    if before != want and _ram_u8(env, HUD_CUR_ITEM) == before:
        return PrimitiveResult(
            False, "selected byte did not change", frames[0], snapshot_env(env)
        )
    closed = _close_menu(env, frames)
    if closed is None:
        return PrimitiveResult(False, "menu did not close", frames[0], snapshot_env(env))
    after = _ram_u8(env, HUD_CUR_ITEM)
    use = _ram_u8(env, 0x0303)
    if after != want or not _in_play(env):
        return PrimitiveResult(
            False,
            f"menu left hud=0x{after:02X}",
            frames[0],
            snapshot_env(env),
        )
    change = (
        f"0x{before:02X}->0x{after:02X}"
        if before != after
        else f"0x{after:02X} unchanged"
    )
    return PrimitiveResult(
        True,
        f"{key} hud {change} use=0x{use:02X} menu open {opened} close {closed}",
        frames[0],
        snapshot_env(env),
    )


# SwampPalaceEntry 2026-09-29. Phase left 0 at 159 frames after the placing
# Y. Link's box (x..x+15, y+8..y+23) lost HP at a 17px gap and not at 19.
BOMB_FUSE_FRAMES = 159
BOMB_BLAST_HIT_GAP = 17
BOMB_BLAST_MISS_GAP = 19
_BOMB_TYPE = 0x07
_DOORWAY = 0x006C
_BOMB_COUNT = 0xF343
_Y_USE = 0x0303
_BOW_ITEM = 0xF340
_ARROW_COUNT = 0xF377
# Link arrow and the halted arrow. Both count as one shot.
_ARROW_ANCILLA = frozenset({0x09, 0x0A})
# Hud slot to the name ``select_y_item`` takes. Shared slots keep one name.
_HUD_Y_NAME = {
    slot: name
    for name, (slot, _field) in _Y_ITEMS.items()
    if name not in {"bomb", "mushroom", "shovel"}
}
# link_bottle_info. Red is 3 and the drink writes 2. Blue is 5. Both open
# the heart refill (module 0x0E). Green (4) is magic only.
_HP_BOTTLE = frozenset({3, 5})
# kHudItemToItemOrg[0x10]. Lamp is hud 0x0B and use 0x09.
_BOTTLE_USE = 0x0B
_HEART_HP = 8
_REFILL_PAD = 90
_REFILL_PER_PIP = 24
_REFILL_CAP = 800


def _bottle_values(env: object) -> tuple[int, int, int, int]:
    inv = read_inventory(np.asarray(env.get_ram(), dtype=np.uint8))  # type: ignore[attr-defined]
    return tuple(int(getattr(inv, f"bottle_{i}")) for i in range(1, 5))  # type: ignore[return-value]


def _hp_now(env: object) -> tuple[int, int]:
    inv = read_inventory(np.asarray(env.get_ram(), dtype=np.uint8))  # type: ignore[attr-defined]
    return int(inv.hp), int(inv.max_hp)


def _whole_hearts(hp: int) -> int:
    if hp <= 0:
        return 0
    return (int(hp) + _HEART_HP - 1) // _HEART_HP


def worst_hit_hp(env: object) -> int:
    """Most contact damage among active sprites, in current-mail HP eighths."""
    armor = int(
        read_inventory(np.asarray(env.get_ram(), dtype=np.uint8)).armor  # type: ignore[attr-defined]
    )
    mail = armor if armor in (0, 1, 2) else 0
    worst = 0
    for sprite in active_sprites(env):
        if sprite.sprite_type == 0:
            continue
        try:
            amount = int(vulnerability(sprite.sprite_type).contact_damage[mail])
        except (TypeError, ValueError):
            continue
        if amount > worst:
            worst = amount
    return worst


def bottle_due(env: object) -> bool:
    """True with a red or blue potion at the last heart, or a lethal hit."""
    if not any(value in _HP_BOTTLE for value in _bottle_values(env)):
        return False
    if snapshot_env(env).game_mode == 0x12:
        return False
    hp, _max_hp = _hp_now(env)
    if _whole_hearts(hp) <= 1:
        return True
    return 0 < worst_hit_hp(env) >= hp


def drink_if_low(env: object) -> PrimitiveResult:
    """Select the bottle, drink, wait out the refill, restore the Y item.

    No-op when :func:`bottle_due` is false. The Y press is the only use.
    A red drink writes the bottle byte down in play; this never pokes it.
    """
    if not bottle_due(env):
        return PrimitiveResult(True, "bottle not due", 0, snapshot_env(env))
    frames = [0]
    if not _in_play(env):
        return _fail("not in play", frames, env)
    previous = _HUD_Y_NAME.get(_ram_u8(env, HUD_CUR_ITEM))
    hp0, max_hp = _hp_now(env)
    before = _bottle_values(env)
    picked = select_y_item(env, "bottle")
    frames[0] += int(picked.frames)
    if not picked.ok:
        return PrimitiveResult(False, picked.reason, frames[0], picked.snapshot)
    use = _ram_u8(env, _Y_USE)
    index = _ram_u8(env, BOTTLE_INDEX)
    if use != _BOTTLE_USE or not 1 <= index <= 4:
        return _fail(f"bottle use 0x{use:02X} index {index}", frames, env)
    _hold(env, "NONE", 1, frames)
    _hold(env, "Y", 1, frames)
    _hold(env, "NONE", 1, frames)
    limit = min(_REFILL_CAP, _REFILL_PAD + _REFILL_PER_PIP * max(1, max_hp - hp0))
    hp, now = hp0, before
    for _ in range(limit):
        now = _bottle_values(env)
        hp, max_hp = _hp_now(env)
        spent = any(old in _HP_BOTTLE and new < old for old, new in zip(before, now))
        if spent and hp >= max_hp and hp > hp0 and _in_play(env):
            break
        _hold(env, "NONE", 1, frames)
    else:
        return _fail(
            f"refill hp 0x{hp0:02X}->0x{hp:02X} bottles {before}->{now}",
            frames,
            env,
        )
    if previous and previous != "bottle":
        back = select_y_item(env, previous)
        frames[0] += int(back.frames)
        if not back.ok:
            return PrimitiveResult(
                False,
                f"restore {previous}: {back.reason}",
                frames[0],
                back.snapshot,
            )
    return PrimitiveResult(
        True,
        (
            f"hp 0x{hp0:02X}->0x{hp:02X} bottles {before}->{now} "
            f"restored {previous or 'none'}"
        ),
        frames[0],
        snapshot_env(env),
    )


_LINK_BITS = 0x0308
_CARRY_BIT = 0x80
_THROWN = 0xEC
# sprite_unk3. Carry ignores A until this byte is 3.
_SPRITE_UNK3 = 0x1FA1C
_OBJ_STATE = 0x0500
_OBJ_POS = 0x0540
_PUSH_LIMIT = 48


def _fail(reason: str, frames: list[int], env: object) -> PrimitiveResult:
    return PrimitiveResult(False, reason, frames[0], snapshot_env(env))


def _obj_word(env: object, addr: int) -> int:
    return _ram_u8(env, addr) | (_ram_u8(env, addr + 1) << 8)


def place_bomb(env: object) -> PrimitiveResult:
    """Place one bomb. The count is read only before the Y press.

    Refuses in a doorway (``$006C`` != 0). A new ancilla of type 0x07 is the
    placement. Fuse length and blast gap are the SwampPalaceEntry measurement.
    """
    frames = [0]
    if not _in_play(env):
        return _fail("not in play", frames, env)
    if _ram_u8(env, _DOORWAY) != 0:
        return _fail("doorway", frames, env)
    if _ram_u8(env, HUD_CUR_ITEM) != 0x04 and _ram_u8(env, _Y_USE) != 0x01:
        return _fail("bombs not selected", frames, env)
    count = _ram_u8(env, _BOMB_COUNT)
    if count == 0:
        return _fail("no bombs", frames, env)
    had = {row.slot for row in active_ancillae(env) if row.ancilla_type == _BOMB_TYPE}
    _hold(env, "NONE", 1, frames)
    _hold(env, "Y", 1, frames)
    born = None
    for _ in range(2):
        for row in active_ancillae(env):
            if row.ancilla_type == _BOMB_TYPE and row.slot not in had:
                born = row
                break
        if born is not None:
            break
        _hold(env, "NONE", 1, frames)
    if born is None:
        return _fail("no bomb ancilla", frames, env)
    return PrimitiveResult(
        True,
        (
            f"bomb ({born.x},{born.y}) count {count} "
            f"fuse {BOMB_FUSE_FRAMES} hit {BOMB_BLAST_HIT_GAP} "
            f"miss {BOMB_BLAST_MISS_GAP}"
        ),
        frames[0],
        snapshot_env(env),
    )


def shoot_arrow(env: object) -> PrimitiveResult:
    """Select the bow and fire once. The count is read only before Y.

    Refuses when bow ``$F340`` is 0, arrow count ``$F377`` is 0, or
    ``$006C`` is a doorway (that press does not start the draw).
    Success is a new ancilla 0x09 or 0x0A and a lower count. The draw
    is 3, 3, then 8 frames before the ancilla. The menu selector does
    not write RAM.
    """
    frames = [0]
    if not _in_play(env):
        return _fail("not in play", frames, env)
    if _ram_u8(env, _DOORWAY) != 0:
        return _fail("doorway", frames, env)
    if _ram_u8(env, _BOW_ITEM) == 0:
        return _fail("bow is not owned", frames, env)
    if _ram_u8(env, _ARROW_COUNT) == 0:
        return _fail("no arrows", frames, env)
    picked = select_y_item(env, "bow")
    frames[0] += int(picked.frames)
    if not picked.ok:
        return PrimitiveResult(False, picked.reason, frames[0], picked.snapshot)
    if _ram_u8(env, _DOORWAY) != 0:
        return _fail("doorway", frames, env)
    count = _ram_u8(env, _ARROW_COUNT)
    if count == 0:
        return _fail("no arrows", frames, env)
    had = {
        row.slot
        for row in active_ancillae(env)
        if row.ancilla_type in _ARROW_ANCILLA
    }
    _hold(env, "NONE", 1, frames)
    _hold(env, "Y", 1, frames)
    born = None
    after = count
    # Press frame plus the 3,3,8 draw. One Y; the rest are idle.
    for _ in range(24):
        after = _ram_u8(env, _ARROW_COUNT)
        for row in active_ancillae(env):
            if row.ancilla_type in _ARROW_ANCILLA and row.slot not in had:
                born = row
                break
        if born is not None and after < count:
            break
        _hold(env, "NONE", 1, frames)
    if born is None or after >= count:
        return _fail(f"no arrow ancilla count {count}->{after}", frames, env)
    return PrimitiveResult(
        True,
        (
            f"arrow 0x{born.ancilla_type:02X} ({born.x},{born.y}) "
            f"count {count}->{after}"
        ),
        frames[0],
        snapshot_env(env),
    )


_HAMMER_ITEM = 0xF34B
_STUN_TIMER = 0x0B58
_PEG_WORD = 0x4040
_HAMMER_WAIT = 16
_OBJ_SLOTS = 16


def _obj_tile(pos: int) -> tuple[int, int]:
    tile = (int(pos) & 0x3FFF) >> 1
    return tile & 63, (tile >> 6) & 63


def _attr_at(env: object, col: int, row: int) -> int:
    layer = _ram_u8(env, 0x00EE)
    if layer not in (0, 1):
        layer = 0
    grid = read_room_tiles(np.asarray(env.get_ram(), dtype=np.uint8), layer)  # type: ignore[attr-defined]
    if not (0 <= row < 64 and 0 <= col < 64):
        return 0
    return int(grid[row, col])


def _hammer_pegs(env: object) -> list[tuple[int, int, int, int]]:
    found: list[tuple[int, int, int, int]] = []
    for slot in range(_OBJ_SLOTS):
        if _obj_word(env, _OBJ_STATE + slot * 2) != _PEG_WORD:
            continue
        col, row = _obj_tile(_obj_word(env, _OBJ_POS + slot * 2))
        found.append((slot, col, row, _attr_at(env, col, row)))
    return found


def hammer(env: object) -> PrimitiveResult:
    """Select the hammer and swing once.

    Refuses in a doorway (``$006C`` != 0) and when ``$F34B`` is 0. A peg is
    an object whose replacement word is 0x4040. Success is that tile's
    attribute leaving 0x7N, or a stun timer ``$0B58`` rising from 0.
    """
    frames = [0]
    if not _in_play(env):
        return _fail("not in play", frames, env)
    if _ram_u8(env, _DOORWAY) != 0:
        return _fail("doorway", frames, env)
    if _ram_u8(env, _HAMMER_ITEM) == 0:
        return _fail("hammer is not owned", frames, env)
    picked = select_y_item(env, "hammer")
    frames[0] += int(picked.frames)
    if not picked.ok:
        return PrimitiveResult(False, picked.reason, frames[0], picked.snapshot)
    if _ram_u8(env, _DOORWAY) != 0:
        return _fail("doorway", frames, env)
    before = _hammer_pegs(env)
    stun0 = tuple(_ram_u8(env, _STUN_TIMER + slot) for slot in range(SPRITE_SLOTS))
    _hold(env, "NONE", 1, frames)
    _hold(env, "Y", 1, frames)
    hit: tuple[int, int, int, int, int] | None = None
    stunned: tuple[int, int] | None = None
    for _ in range(_HAMMER_WAIT):
        for slot, col, row, attr in before:
            now = _attr_at(env, col, row)
            if (attr & 0xF0) == 0x70 and (now & 0xF0) != 0x70:
                hit = (slot, col, row, attr, now)
                break
        if hit is None:
            for slot, old in enumerate(stun0):
                new = _ram_u8(env, _STUN_TIMER + slot)
                if old == 0 and new != 0:
                    stunned = (slot, new)
                    break
        if hit is not None or stunned is not None:
            break
        _hold(env, "NONE", 1, frames)
    if hit is not None:
        slot, col, row, attr, now = hit
        return PrimitiveResult(
            True,
            f"peg slot {slot} tile ({col},{row}) attr 0x{attr:02X}->0x{now:02X}",
            frames[0],
            snapshot_env(env),
        )
    if stunned is not None:
        slot, value = stunned
        return PrimitiveResult(
            True,
            f"stun slot {slot} $0B58=0x{value:02X}",
            frames[0],
            snapshot_env(env),
        )
    return _fail("hammer missed", frames, env)


_HOOK_ITEM = 0xF342
_HOOK_DRAG = 0x037E
_HOOK_TYPE = 0x1F
_HOOK_STATE = 0x13
_HOOK_WAIT = 48


def hookshot(env: object) -> PrimitiveResult:
    """Select the hookshot and fire once.

    Refuses in a doorway (``$006C`` != 0) and when ``$F342`` is 0.
    Reach is a new ancilla 0x1F. A pull is ``$037E`` bit 0 while Link's
    xy changes in state ``$5D`` 0x13. A retract with no drag misses.
    """
    frames = [0]
    if not _in_play(env):
        return _fail("not in play", frames, env)
    if _ram_u8(env, _DOORWAY) != 0:
        return _fail("doorway", frames, env)
    if _ram_u8(env, _HOOK_ITEM) == 0:
        return _fail("hookshot is not owned", frames, env)
    picked = select_y_item(env, "hookshot")
    frames[0] += int(picked.frames)
    if not picked.ok:
        return PrimitiveResult(False, picked.reason, frames[0], picked.snapshot)
    if _ram_u8(env, _DOORWAY) != 0:
        return _fail("doorway", frames, env)
    origin = snapshot_env(env)
    x0, y0 = int(origin.link_x), int(origin.link_y)
    had = {row.slot for row in active_ancillae(env) if row.ancilla_type == _HOOK_TYPE}
    _hold(env, "NONE", 1, frames)
    _hold(env, "Y", 1, frames)
    born = None
    dragged = False
    moved = None
    for _ in range(_HOOK_WAIT):
        for row in active_ancillae(env):
            if row.ancilla_type == _HOOK_TYPE and row.slot not in had:
                born = row
        if _ram_u8(env, _HOOK_DRAG) & 1:
            dragged = True
        now = snapshot_env(env)
        if (int(now.link_x), int(now.link_y)) != (x0, y0) and int(now.link_action) == _HOOK_STATE:
            moved = (int(now.link_x), int(now.link_y))
        if born is not None and dragged and moved is not None:
            break
        _hold(env, "NONE", 1, frames)
    if born is None:
        return _fail("no hookshot ancilla", frames, env)
    if not dragged or moved is None:
        return _fail(f"hookshot retract ({born.x},{born.y}) xy ({x0},{y0})", frames, env)
    x1, y1 = moved
    return PrimitiveResult(
        True,
        f"hookshot ({born.x},{born.y}) drag xy ({x0},{y0})->({x1},{y1}) $037E",
        frames[0],
        snapshot_env(env),
    )


_ROD_ITEM = 0xF345
_ROD_TYPE = 0x02
_MAGIC = 0xF36E
# AP-072: $F36E drops 16 on Y. $F37B was 0, so this is the normal column.
_ROD_COST = 16
_ROD_WAIT = 32


def fire_rod(env: object) -> PrimitiveResult:
    """Select the fire rod and fire once.

    Refuses in a doorway (``$006C`` != 0), when ``$F345`` is 0, or when
    ``$F36E`` is below 16. Success is a new ancilla 0x02 and magic lower
    by 16 (AP-072).
    """
    frames = [0]
    if not _in_play(env):
        return _fail("not in play", frames, env)
    if _ram_u8(env, _DOORWAY) != 0:
        return _fail("doorway", frames, env)
    if _ram_u8(env, _ROD_ITEM) == 0:
        return _fail("fire rod is not owned", frames, env)
    if _ram_u8(env, _MAGIC) < _ROD_COST:
        return _fail(f"magic {_ram_u8(env, _MAGIC)} < {_ROD_COST}", frames, env)
    picked = select_y_item(env, "fire_rod")
    frames[0] += int(picked.frames)
    if not picked.ok:
        return PrimitiveResult(False, picked.reason, frames[0], picked.snapshot)
    if _ram_u8(env, _DOORWAY) != 0:
        return _fail("doorway", frames, env)
    magic0 = _ram_u8(env, _MAGIC)
    had = {row.slot for row in active_ancillae(env) if row.ancilla_type == _ROD_TYPE}
    _hold(env, "NONE", 1, frames)
    _hold(env, "Y", 1, frames)
    born = None
    magic1 = magic0
    for _ in range(_ROD_WAIT):
        magic1 = _ram_u8(env, _MAGIC)
        for row in active_ancillae(env):
            if row.ancilla_type == _ROD_TYPE and row.slot not in had:
                born = row
        if born is not None and magic0 - magic1 == _ROD_COST:
            break
        _hold(env, "NONE", 1, frames)
    if born is None:
        return _fail(f"no fire rod ancilla magic {magic0}->{magic1}", frames, env)
    if magic0 - magic1 != _ROD_COST:
        return _fail(
            f"fire rod ({born.x},{born.y}) magic {magic0}->{magic1} not {_ROD_COST}",
            frames,
            env,
        )
    return PrimitiveResult(
        True,
        f"fire rod ({born.x},{born.y}) magic {magic0}->{magic1}",
        frames[0],
        snapshot_env(env),
    )


# AP-072 measured $F36E dropping 8 on Y. Ancilla 0x2C is the block. $0646
# rises only when that block covers 0x23, 0x24, 0x25, or 0x3B. A 0xB6 or
# 0xBC under it becomes sprite 0xED; $02F5 is Link standing on the platform.
# Facing $2F is 0, 2, 4, 6. sprite $0DE0 0..3 leaves a 0xB6 on the reverse press.
_SOMARIA_ITEM = 0xF350
_SOMARIA_COST = 8
_BLOCK_ANCILLA = 0x2C
_SWITCH_FLAG = 0x0646
_ON_PLATFORM = 0x02F5
_TILE_MASK = 0x00EC
_PIPE_START = frozenset({0xB6, 0xBC})
_SWITCH_ATTR = frozenset({0x23, 0x24, 0x25, 0x3B})
_PLATFORM_SPRITES = frozenset({0xED, 0xEF, 0xF0, 0xF1})
_FACING_PRESS = {0: "UP", 2: "DOWN", 4: "LEFT", 6: "RIGHT"}
_TRANSIT_PRESS = ("DOWN", "UP", "RIGHT", "LEFT")
_SOMARIA_WAIT = 32
_PUSH_SWITCH = 120
_RIDE_STAND = 90


def _tile_xy(env: object, x: int, y: int) -> tuple[int, int, int]:
    mask = _ram_u8(env, _TILE_MASK) | (_ram_u8(env, _TILE_MASK + 1) << 8)
    col = (int(x) & mask) >> 3
    row = (int(y) & mask) >> 3
    return col & 63, row & 63, _attr_at(env, col & 63, row & 63)


def _ahead_attr(env: object, button: str) -> int:
    snap = snapshot_env(env)
    col, row, _attr = _tile_xy(env, int(snap.link_x), int(snap.link_y))
    if button == "UP":
        row -= 1
    elif button == "DOWN":
        row += 1
    elif button == "LEFT":
        col -= 1
    else:
        col += 1
    return _attr_at(env, col, row)


def _pit_track(attr: int) -> bool:
    if attr == 0x20:
        return True
    return (attr & 0xF0) == 0xB0 and attr not in _PIPE_START


def _px_before_edge(x: int, y: int, button: str) -> int:
    """Pixels of the current 8px tile left before ``button`` leaves it."""
    if button == "UP":
        return int(y) & 7
    if button == "DOWN":
        return 7 - (int(y) & 7)
    if button == "LEFT":
        return int(x) & 7
    return 7 - (int(x) & 7)


def _arm_somaria(env: object, frames: list[int]) -> tuple[int, PrimitiveResult | None]:
    """Select the cane. Does not press Y and does not write the item byte."""
    if not _in_play(env):
        return 0, _fail("not in play", frames, env)
    if _ram_u8(env, _DOORWAY) != 0:
        return 0, _fail("doorway", frames, env)
    if _ram_u8(env, _SOMARIA_ITEM) == 0:
        return 0, _fail("somaria is not owned", frames, env)
    magic0 = _ram_u8(env, _MAGIC)
    if magic0 < _SOMARIA_COST:
        return 0, _fail(f"magic {magic0} < {_SOMARIA_COST}", frames, env)
    picked = select_y_item(env, "somaria")
    frames[0] += int(picked.frames)
    if not picked.ok:
        return 0, PrimitiveResult(False, picked.reason, frames[0], picked.snapshot)
    if _ram_u8(env, _DOORWAY) != 0:
        return 0, _fail("doorway", frames, env)
    return _ram_u8(env, _MAGIC), None


def _somaria_blocks(env: object) -> list[tuple[int, int, int]]:
    found: list[tuple[int, int, int]] = []
    for slot in range(ANCILLA_SLOTS):
        if _ram_u8(env, ANCILLA_TYPE + slot) != _BLOCK_ANCILLA:
            continue
        x = _ram_u8(env, ANCILLA_X_LOW + slot) | (_ram_u8(env, ANCILLA_X_HIGH + slot) << 8)
        y = _ram_u8(env, ANCILLA_Y_LOW + slot) | (_ram_u8(env, ANCILLA_Y_HIGH + slot) << 8)
        found.append((slot, x, y))
    return found


def _platform_sprites(env: object) -> list[SpriteSnapshot]:
    return [row for row in active_sprites(env) if row.sprite_type in _PLATFORM_SPRITES]


def somaria_block(env: object) -> PrimitiveResult:
    """Select the cane, make one block, and push it onto a switch.

    Refuses in a doorway (``$006C`` != 0), when ``$F350`` is 0, or when
    ``$F36E`` is below 8. Success is a new ancilla 0x2C, magic lower by 8,
    and ``$0646`` rising or the block's tile in 0x23, 0x24, 0x25, or 0x3B.
    The push is the facing already in ``$2F``. A pit or track stops the push.
    """
    frames = [0]
    magic0, failed = _arm_somaria(env, frames)
    if failed is not None:
        return failed
    had = {slot for slot, _x, _y in _somaria_blocks(env)}
    _hold(env, "NONE", 1, frames)
    _hold(env, "Y", 1, frames)
    born: tuple[int, int, int] | None = None
    magic1 = magic0
    for _ in range(_SOMARIA_WAIT):
        magic1 = _ram_u8(env, _MAGIC)
        for slot, x, y in _somaria_blocks(env):
            if slot not in had:
                born = (slot, x, y)
        if born is not None and magic0 - magic1 == _SOMARIA_COST:
            break
        _hold(env, "NONE", 1, frames)
    if born is None or magic0 - magic1 != _SOMARIA_COST:
        return _fail(f"no somaria block magic {magic0}->{magic1}", frames, env)
    button = _FACING_PRESS.get(_ram_u8(env, 0x002F))
    if button is None:
        return _fail(f"bad facing 0x{_ram_u8(env, 0x002F):02X}", frames, env)
    flag0 = _ram_u8(env, _SWITCH_FLAG)
    hit: tuple[int, int, int, int] | None = None
    for _ in range(_PUSH_SWITCH):
        if snapshot_env(env).game_mode == 0x12:
            return _fail("Link died", frames, env)
        blocks = _somaria_blocks(env)
        flag = _ram_u8(env, _SWITCH_FLAG)
        if blocks:
            col, row, attr = _tile_xy(env, blocks[0][1], blocks[0][2])
            if flag != flag0 or attr in _SWITCH_ATTR:
                hit = (col, row, attr, flag)
                break
        elif flag != flag0:
            hit = (-1, -1, 0, flag)
            break
        if _pit_track(_ahead_attr(env, button)):
            break
        _hold(env, button, 1, frames)
    if hit is None:
        where = ""
        blocks = _somaria_blocks(env)
        if blocks:
            col, row, attr = _tile_xy(env, blocks[0][1], blocks[0][2])
            where = f" tile ({col},{row}) attr 0x{attr:02X}"
        return _fail(f"block missed switch{where}", frames, env)
    col, row, attr, flag = hit
    return PrimitiveResult(
        True,
        (
            f"block tile ({col},{row}) attr 0x{attr:02X} "
            f"$0646={flag} magic {magic0}->{magic1}"
        ),
        frames[0],
        snapshot_env(env),
    )


def ride_platform(env: object) -> PrimitiveResult:
    """Place a somaria block on a 0xB6 or 0xBC pipe, stand, and press.

    The pipe is the live attribute under the new platform sprite. ``$02F5``
    is standing on it. At that junction the press is the reverse of ``$0DE0``.
    Success is the 8-point drop plus Link or the platform moving while ``$02F5``
    is set. This does not write the cane byte.
    """
    frames = [0]
    magic0, failed = _arm_somaria(env, frames)
    if failed is not None:
        return failed
    had = {row.slot for row in _platform_sprites(env)}
    _hold(env, "NONE", 1, frames)
    _hold(env, "Y", 1, frames)
    born: SpriteSnapshot | None = None
    magic1 = magic0
    pipe: tuple[int, int, int] = (0, 0, 0)
    for _ in range(_SOMARIA_WAIT):
        magic1 = _ram_u8(env, _MAGIC)
        for row in _platform_sprites(env):
            if row.slot in had:
                continue
            col, tile_row, attr = _tile_xy(env, row.x, row.y)
            live = attr if attr in _PIPE_START else _ram_u8(env, 0x0E90 + row.slot)
            if live in _PIPE_START:
                born = row
                pipe = (col, tile_row, live)
                break
        if born is not None and magic0 - magic1 == _SOMARIA_COST:
            break
        _hold(env, "NONE", 1, frames)
    if born is None or magic0 - magic1 != _SOMARIA_COST:
        return _fail(f"no somaria platform magic {magic0}->{magic1}", frames, env)
    for _ in range(_RIDE_STAND):
        if _ram_u8(env, _ON_PLATFORM):
            break
        if snapshot_env(env).game_mode == 0x12:
            return _fail("Link died", frames, env)
        still = [row for row in _platform_sprites(env) if row.slot == born.slot]
        if not still:
            return _fail("platform vanished", frames, env)
        plat = still[0]
        snap = snapshot_env(env)
        dx = int(plat.x) - int(snap.link_x)
        dy = int(plat.y) - int(snap.link_y)
        if abs(dx) + abs(dy) <= 8:
            _hold(env, "NONE", 1, frames)
            continue
        button = "DOWN" if abs(dy) >= abs(dx) and dy > 0 else "UP"
        if abs(dx) > abs(dy):
            button = "RIGHT" if dx > 0 else "LEFT"
        # The tile ahead can be pit while this tile still has floor.
        # Stop on the edge instead of stepping into the pit.
        if _pit_track(_ahead_attr(env, button)) and _px_before_edge(
            int(snap.link_x), int(snap.link_y), button
        ) < 2:
            _hold(env, "NONE", 1, frames)
            continue
        _hold(env, button, 1, frames)
        _col, _row, stepped = _tile_xy(
            env, int(snapshot_env(env).link_x), int(snapshot_env(env).link_y)
        )
        if _pit_track(stepped):
            return _fail("pit between Link and the platform", frames, env)
    if _ram_u8(env, _ON_PLATFORM) == 0:
        return _fail("not standing on platform", frames, env)
    still = [row for row in _platform_sprites(env) if row.slot == born.slot]
    if not still:
        return _fail("platform vanished", frames, env)
    plat = still[0]
    press = _TRANSIT_PRESS[int(plat.direction) & 3]
    x0, y0 = int(plat.x), int(plat.y)
    link0 = snapshot_env(env)
    moved = False
    for _ in range(48):
        _hold(env, press, 1, frames)
        if snapshot_env(env).game_mode == 0x12:
            return _fail("Link died", frames, env)
        still = [row for row in _platform_sprites(env) if row.slot == born.slot]
        link = snapshot_env(env)
        plat_xy = (int(still[0].x), int(still[0].y)) if still else (x0, y0)
        link_xy = (int(link.link_x), int(link.link_y))
        if plat_xy != (x0, y0) or link_xy != (int(link0.link_x), int(link0.link_y)):
            if _ram_u8(env, _ON_PLATFORM):
                moved = True
                break
    col, tile_row, attr = pipe
    if not moved:
        return _fail(
            f"junction {press} tile ({col},{tile_row}) attr 0x{attr:02X} did not move",
            frames,
            env,
        )
    return PrimitiveResult(
        True,
        (
            f"platform 0x{born.sprite_type:02X} tile ({col},{tile_row}) "
            f"attr 0x{attr:02X} press {press} $02F5 magic {magic0}->{magic1}"
        ),
        frames[0],
        snapshot_env(env),
    )


def lift(env: object) -> PrimitiveResult:
    """Tap A. Success is the carry bit on ``$0308``."""
    frames = [0]
    if not _in_play(env):
        return _fail("not in play", frames, env)
    if _ram_u8(env, _LINK_BITS) & _CARRY_BIT:
        return _fail("already carrying", frames, env)
    _hold(env, "NONE", 1, frames)
    _hold(env, "A", 1, frames)
    for _ in range(6):
        if _ram_u8(env, _LINK_BITS) & _CARRY_BIT:
            pots = [row for row in active_sprites(env) if row.sprite_type == _THROWN]
            where = f" ({pots[0].x},{pots[0].y})" if pots else ""
            return PrimitiveResult(True, f"lifted{where}", frames[0], snapshot_env(env))
        _hold(env, "NONE", 1, frames)
    return _fail("nothing lifted", frames, env)


def throw(env: object) -> PrimitiveResult:
    """Throw the carried pot or bush once its pose allows A.

    A before ``sprite_unk3`` reaches 3 is ignored. Success is the 0xEC sprite
    leaving state 10 or leaving the carry point.
    """
    frames = [0]
    if (_ram_u8(env, _LINK_BITS) & _CARRY_BIT) == 0:
        return _fail("not carrying", frames, env)
    pots = [row for row in active_sprites(env) if row.sprite_type == _THROWN]
    if not pots:
        return _fail("no carried sprite", frames, env)
    slot = pots[0].slot
    origin = (pots[0].x, pots[0].y)
    ready = False
    for _ in range(40):
        if _ram_u8(env, _SPRITE_UNK3 + slot) == 3:
            ready = True
            break
        _hold(env, "NONE", 1, frames)
    if not ready:
        return _fail("carry not ready", frames, env)
    held = [row for row in active_sprites(env) if row.slot == slot]
    origin = (held[0].x, held[0].y) if held else origin
    _hold(env, "NONE", 1, frames)
    _hold(env, "A", 1, frames)
    for _ in range(8):
        now = [row for row in active_sprites(env) if row.slot == slot]
        if not now or now[0].state != 10:
            landed = (now[0].x, now[0].y, now[0].state) if now else (*origin, 0)
            return PrimitiveResult(
                True, f"threw {origin}->{landed}", frames[0], snapshot_env(env)
            )
        _hold(env, "NONE", 1, frames)
    return _fail("still carried", frames, env)


def push_block(env: object, direction: str) -> PrimitiveResult:
    """Hold ``direction`` until a resting block (object state 0) changes.

    The windup is the block timer, longer than one step and shorter than a
    wall-drag cycle. Returns the frame the slot state or tile position changes.
    """
    frames = [0]
    button = direction.strip().upper()
    if button not in DIRECTION_BUTTONS:
        return _fail(f"bad direction {direction}", frames, env)
    if not _in_play(env):
        return _fail("not in play", frames, env)
    idle = {
        slot: _obj_word(env, _OBJ_POS + slot * 2)
        for slot in range(16)
        if _obj_word(env, _OBJ_STATE + slot * 2) == 0
    }
    if not idle:
        return _fail("no resting block", frames, env)
    for _ in range(_PUSH_LIMIT):
        _hold(env, button, 1, frames)
        for slot, pos in idle.items():
            state = _obj_word(env, _OBJ_STATE + slot * 2)
            moved = _obj_word(env, _OBJ_POS + slot * 2)
            if state != 0 or moved != pos:
                return PrimitiveResult(
                    True,
                    f"block slot {slot} in {frames[0]} state 0x{state:04X}",
                    frames[0],
                    snapshot_env(env),
                )
    return _fail(f"block did not move in {frames[0]}", frames, env)


# Room 0x05 after the boots receipt. Pose (2678, 8617), doorway $006C=0.
# A+direction. Frame 1 sets $5D=17 and takes one walk step ($5E=0).
# Frames 2-30 stay in 17 with $5E=16 and do not move. Frame 31 steps 4px,
# then 4px/frame while $5E stays 16. RIGHT into the tile east of the stop,
# attribute 0x27: frame 33 reaches the walk stop (2688, 8617) and clears
# $5E. Frame 34 is $5D=2. State 2 holds through frame 67 and rests at
# (2672, 8617). Isolated. Not continuous.
DASH_STATE = 17
DASH_CHARGE_FRAMES = 30
DASH_SPEED_PX = 4
DASH_SPEED_BYTE = 0x5E
DASH_SPEED_VALUE = 16
DASH_BONK_STATE = 2
DASH_WALL_ATTR = 0x27
DASH_WALL_STOP_XY = (2688, 8617)
DASH_BONK_REST_XY = (2672, 8617)
DASH_BONK_FRAMES = 34
_DASH_ITEM_SPRITE = 0x3B


def dash(env: object, direction: str, *, max_frames: int = 180) -> PrimitiveResult:
    """Hold A and one cardinal until the dash leaves state 17.

    Boots and ability bit 0x04 must already be set. This does not write
    them, and it does not press A while sprite 0x3B is in the room. The
    4px step is the frame after ``DASH_CHARGE_FRAMES``.
    """
    frames = [0]
    button = direction.strip().upper()
    if button not in DIRECTION_BUTTONS:
        return _fail(f"bad direction {direction}", frames, env)
    if not _in_play(env):
        return _fail("not in play", frames, env)
    if any(int(row.sprite_type) == _DASH_ITEM_SPRITE for row in active_sprites(env)):
        return _fail("sprite 0x3B", frames, env)
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    inv = read_inventory(ram)
    if int(inv.boots) == 0 or (int(inv.ability) & ABILITY_DASH) == 0:
        return _fail("no dash", frames, env)
    action = action_for("A", button)
    started = False
    fast = False
    last = snapshot_env(env)
    for _ in range(max_frames):
        step_frames(env, action, 1)
        frames[0] += 1
        snap = snapshot_env(env)
        if snap.game_mode == 0x12:
            return _fail("Link died", frames, env)
        if int(snap.link_state) == DASH_STATE:
            started = True
            moved = abs(snap.link_x - last.link_x) + abs(snap.link_y - last.link_y)
            if moved >= DASH_SPEED_PX:
                fast = True
            last = snap
            continue
        if started and fast:
            return PrimitiveResult(
                True,
                (
                    f"stop ({last.link_x},{last.link_y}) "
                    f"state {snap.link_state} ({snap.link_x},{snap.link_y})"
                ),
                frames[0],
                snap,
            )
        if started and frames[0] > DASH_CHARGE_FRAMES + 8:
            return _fail("dash stalled", frames, env)
        last = snap
    return _fail("dash timeout", frames, env)


# Save box is SELECT in play (module 0x07/0x09), not START. START opens items.
# Progress 0 never opens it. Message 0x0186, quit is choice 1.
_SAVE_MSG = 0x0186
_SPAWN_MSG = 0x0184
_TEXT_STATE = 0x1CD4
_TEXT_CHOICE = 0x1CE8
_TEXT_COUNT = 0x1CE9
_TEXT_MSG = 0x1CF0


def _msg(env: object) -> int:
    return _ram_u8(env, _TEXT_MSG) | (_ram_u8(env, _TEXT_MSG + 1) << 8)


def _text_settled(env: object, frames: list[int]) -> None:
    """Idle until the text lock ($1CE9) rises and falls, or the box closes."""
    saw = False
    for _ in range(100):
        count = _ram_u8(env, _TEXT_COUNT)
        if count:
            saw = True
        elif saw or _ram_u8(env, MODULE) not in (MENU_MODULE, 0x1B):
            return
        _hold(env, "NONE", 1, frames)


def _spawn_place(env: object) -> str:
    snap = snapshot_env(env)
    room = snap.room_id & 0xFF
    if snap.indoors and room == 0x04:
        return "links house"
    if snap.indoors and room == 0x12:
        return "sanctuary"
    if snap.indoors and room == 0x80:
        return "zelda cell"
    if snap.indoors and room == 0x55:
        return "secret entrance"
    if not snap.indoors and snap.screen_id == 0x5B and _ram_u8(env, 0x0FFF):
        return "pyramid"
    where = "in" if snap.indoors else "ow"
    return f"{where} room 0x{snap.room_id:04X} scr 0x{snap.screen_id:02X}"


def save_and_quit(env: object, choice: int = 0) -> PrimitiveResult:
    """Quit from the SELECT save box and continue until Link has control.

    ``choice`` is the spawn-menu row. It is used only when that menu appears
    (progress at least 2, light-world file). Row 0 continues in Link's house,
    row 1 in the Sanctuary. A dark-world file skips the menu. Progress 0
    refuses the box. Buttons only; this does not load a savestate.
    """
    frames = [0]
    if choice not in (0, 1):
        return _fail("spawn choice is 0 or 1", frames, env)
    if not _in_play(env):
        return _fail("not in play", frames, env)
    progress = _ram_u8(env, 0xF3C5)
    _hold(env, "NONE", 8, frames)
    _hold(env, "SELECT", 1, frames)
    _hold(env, "NONE", 2, frames)
    ready = False
    for _ in range(80):
        if (
            _ram_u8(env, _TEXT_COUNT) == 0
            and _ram_u8(env, MODULE) == MENU_MODULE
            and _ram_u8(env, 0x11) == 0x0B
        ):
            ready = True
            break
        if _ram_u8(env, MODULE) not in (MENU_MODULE, 0x07, 0x09):
            break
        _hold(env, "NONE", 1, frames)
    if not ready:
        if progress == 0:
            return _fail("progress 0 refuses save", frames, env)
        return _fail("save box did not open", frames, env)
    if _msg(env) != _SAVE_MSG:
        return _fail(f"save message 0x{_msg(env):04X}", frames, env)
    _hold(env, "DOWN", 1, frames)
    _hold(env, "NONE", 1, frames)
    _text_settled(env, frames)
    if _ram_u8(env, _TEXT_CHOICE) != 1:
        return _fail("quit row was not selected", frames, env)
    _hold(env, "B", 1, frames)
    _hold(env, "NONE", 1, frames)

    starts = 0
    file_presses = 0
    opened = False
    moved = False
    confirmed = False
    prev = None
    idle = 0
    for _ in range(1100):
        _hold(env, "NONE", 1, frames)
        snap = snapshot_env(env)
        mode = snap.game_mode
        now = (mode, snap.submodule, snap.room_id, snap.link_x, snap.link_y, _ram_u8(env, _TEXT_CHOICE))
        if now != prev:
            prev = now
            idle = 0
        else:
            idle += 1
        if mode in (0x00, 0x14) and idle == 20 and starts < 6:
            _hold(env, "START", 1, frames)
            starts += 1
            idle = 0
        elif mode == 0x01 and idle == 40 and file_presses < 3:
            _hold(env, "A", 1, frames)
            file_presses += 1
            idle = 0
        elif mode == 0x1B and not confirmed and _ram_u8(env, _TEXT_COUNT) == 0 and idle >= 2:
            if _msg(env) not in (_SPAWN_MSG, 0x000A):
                return _fail(f"spawn message 0x{_msg(env):04X}", frames, env)
            state = _ram_u8(env, _TEXT_STATE)
            if not opened:
                if state != 3:
                    _hold(env, "B", 1, frames)
                    _text_settled(env, frames)
                opened = True
                idle = 0
                prev = None
                continue
            if choice and not moved:
                _hold(env, "DOWN", 1, frames)
                _text_settled(env, frames)
                _hold(env, "NONE", 4, frames)
                moved = True
                idle = 0
                prev = None
                continue
            _hold(env, "NONE", 4, frames)
            _hold(env, "B", 1, frames)
            confirmed = True
            idle = 0
        if snap.has_control and mode in (0x07, 0x09) and frames[0] > 200:
            spawn = _ram_u8(env, 0xF3C8)
            return PrimitiveResult(
                True,
                (
                    f"{_spawn_place(env)} room 0x{snap.room_id:04X} "
                    f"xy=({snap.link_x},{snap.link_y}) "
                    f"spawn 0x{spawn:02X} progress 0x{progress:02X} "
                    f"choice {choice}"
                ),
                frames[0],
                snap,
            )
    return _fail(f"no control by {frames[0]}", frames, env)
