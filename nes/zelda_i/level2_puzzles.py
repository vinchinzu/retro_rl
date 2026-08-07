"""Level 2 (Moon) puzzle catalog — bomb walls, key doors, diamond solids.

Pure data for lab / recon runners. No emulator imports.

Geometry and outcomes are live-recon verified unless marked residual /
walkthrough-only. See ``docs/LEVEL2_ROUTE.md`` § Puzzle catalog.

Room IDs match ``level2_dungeon`` constants (duplicated here so this module
stays import-light and does not register room specs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# --- Room IDs (mirror level2_dungeon; keep independent) ---
ROOM_L2_ENTRY: Final = 0x7D
ROOM_L2_ROPES: Final = 0x6D
ROOM_L2_WEST_KEY: Final = 0x6C
ROOM_L2_EAST_KEY: Final = 0x7E
ROOM_L2_EAST_OF_ROPES: Final = 0x6E
ROOM_L2_COMPASS: Final = 0x6F
ROOM_L2_BOMB_N: Final = 0x5F
ROOM_L2_GORIYA_WEST: Final = 0x5E
ROOM_L2_ROPES_NORTH: Final = 0x4E
ROOM_L2_BOOM: Final = 0x4F
# Post-boom Dodongo tip (rr-n5i). Walkthrough "TF east of boss" is wrong live:
# kill opens LEFT only → TF room is WEST of boss (0x0d south-band maze LIVE).
ROOM_L2_BOSS: Final = 0x0E  # Dodongo type 0x32; bomb-N of 0x1e
ROOM_L2_TF: Final = 0x0D  # LEFT/WEST of 0x0e after kill — not east
LEVEL2_TRIFORCE_BIT: Final = 0x02
# RoomItemId on 0x0d after boss (dungeon_ids ROOM_ITEM 0x1B); collect residual.
ROOM_ITEM_L2_TF: Final = 0x1B
# Heart container on boss room after kill (ROOM_ITEM 0x1A).
ROOM_ITEM_L2_HC: Final = 0x1A

# cur_opened_doors bits (same as probe scripts / dungeon lab).
DOOR_RIGHT: Final = 0x01
DOOR_LEFT: Final = 0x02
DOOR_DOWN: Final = 0x04
DOOR_UP: Final = 0x08

# Diamond-east approach (nav_common.diamond_east_phase).
DOOR_Y_DEFAULT: Final = 141
DIAMOND_WALL_X: Final = 200
DIAMOND_BAND_7D: Final = 157  # 0x7d → 0x7e
DIAMOND_BAND_6E: Final = 113  # 0x6e → 0x6f (key)
# East wall opens only for door_y ≥ this (y≤133 never); poke-verified.
DOOR_Y_MIN_OPEN: Final = 137

# B-item selection notes (fixtures often already select bombs).
# Live recon sometimes shows selected_item == 0x01 with bombs present;
# probe scripts also poke 0x02. Prefer "already selected" over menu in lab.
B_ITEM_BOMB_PROBE: Final = 0x02


@dataclass(frozen=True)
class BombWall:
    """Bombable wall stand + expected open."""

    room: int
    stand: tuple[int, int]  # (x, y) place bomb
    face: str  # UP / DOWN / LEFT / RIGHT
    opens_to: int
    # After transition into opens_to, door bit often set toward origin.
    opened_door_bit: int | None = None
    live: bool = True
    notes: str = ""
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class KeyDoor:
    """Locked door that consumes one key on open."""

    room: int
    direction: str
    destination: int
    key_cost: int = 1
    # Optional diamond-east band for RIGHT exits blocked by solids.
    approach_band_y: int | None = None
    door_y: int = DOOR_Y_DEFAULT
    live: bool = True
    notes: str = ""
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiamondEast:
    """Mid-room diamond solids blocking naive y≈141 RIGHT."""

    room: int
    band_y: int
    destination: int
    requires_key: bool = False
    wall_x: int = DIAMOND_WALL_X
    door_y: int = DOOR_Y_DEFAULT
    notes: str = ""
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class SealedExit:
    """Direction that does not open under recon (no bomb / key / clear)."""

    room: int
    direction: str
    notes: str = ""
    evidence: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Bomb walls
# ---------------------------------------------------------------------------

# Verified LIVE: 0x6f north wall only. Dense stands y=96–105 with free pathing
# often miss; (120,101) UP+B is the reliable recon open (rr-ebe).
BOMB_WALL_6F_NORTH = BombWall(
    room=ROOM_L2_COMPASS,
    stand=(120, 101),
    face="UP",
    opens_to=ROOM_L2_BOMB_N,
    opened_door_bit=DOOR_DOWN,  # 0x5f often doors=4 after bomb entry
    live=True,
    notes=(
        "Walkthrough optional bomb-N shortcut. Place facing UP with bombs "
        "selected; wait blast ~60–90f then push UP. Opens 0x5f: 5× Gel 0x15 "
        "(TYPE-only) + map RoomItemId 0x17; doors often only DOWN=4; clear does "
        "not open RIGHT/UP (rr-fvt / l2_5f_policy.json)."
    ),
    evidence=(
        "recordings/l2_past6f_expand.json",
        "recordings/l2_5f_explore.json",
        "recordings/l2_5f_policy.json",
        "docs/LEVEL2_ROUTE.md",
    ),
)

# LIVE: 0x5f north wall → boom room 0x4f (rr-bsq / rr-ebe pure).
BOMB_WALL_5F_NORTH = BombWall(
    room=ROOM_L2_BOMB_N,
    stand=(120, 101),
    face="UP",
    opens_to=ROOM_L2_BOOM,
    opened_door_bit=DOOR_DOWN,  # 0x4f often doors=4 after bomb entry
    live=True,
    notes=(
        "Same stand as 0x6f bomb-N. Opens boom room 0x4f: 3× type 0x05 + "
        "RoomItemId 0x1e Magical Boomerang. Runner: run_level2_magic_boomerang."
    ),
    evidence=(
        "recordings/level2_magic_boomerang_isolated.json",
        "recordings/l2_cjf_expand.json",
    ),
)

# LIVE: 0x4f north wall → traps+Keese 0x3f (post-boom Dodongo path).
BOMB_WALL_4F_NORTH = BombWall(
    room=ROOM_L2_BOOM,
    stand=(120, 101),
    face="UP",
    opens_to=0x3F,  # ROOM_L2_TRAPS_KEESE
    opened_door_bit=DOOR_DOWN,
    live=True,
    notes="Post-boom path. Pure 2/2 Clean run_level2_bomb_north_4f.",
    evidence=("recordings/level2_bomb_north_4f_isolated.json",),
)

# LIVE: 0x1e north wall → Dodongo boss 0x0e (rr-n5i 2026-08-07).
# Walk-UP after Goriya clear sets doors=12 but physical solid (min_y≈117).
BOMB_WALL_1E_NORTH = BombWall(
    room=0x1E,  # ROOM_L2_GORIYA_BOMBS
    stand=(120, 101),
    face="UP",
    opens_to=0x0E,  # ROOM_L2_DODONGO
    opened_door_bit=DOOR_DOWN,
    live=True,
    notes=(
        "Critical rr-n5i unlock: doors bit UP|DOWN=12 after clear is a red "
        "herring; physical boss door is bomb-N @(120,101). Dodongo type 0x32; "
        "bomb-mouth → HC; LEFT→0x0d south-band TF LIVE assisted (RIGHT sealed)."
    ),
    evidence=(
        "recordings/l2_1e_up.json",
        "recordings/level2_dodongo.json",
        "recordings/l2_0d_tf_reach.json",
        "recordings/l2_tf02_encode.json",
    ),
)

# Catalog entry point for lab imports.
BOMB_WALLS: tuple[BombWall, ...] = (
    BOMB_WALL_6F_NORTH,
    BOMB_WALL_5F_NORTH,
    BOMB_WALL_4F_NORTH,
    BOMB_WALL_1E_NORTH,
)

# Negative bomb stands on 0x6f (full cardinal sweep failed except UP@101).
# Generic probe defaults (BOMB_STAND in probe_level2_past_6f) for reference.
BOMB_STAND_PROBE_DEFAULTS: dict[str, tuple[int, int]] = {
    "UP": (120, 109),
    "DOWN": (120, 173),
    "LEFT": (64, 141),
    "RIGHT": (176, 141),
}

BOMB_WALL_NEGATIVES_6F: tuple[tuple[str, tuple[int, int]], ...] = (
    # face, stand — all failed to open (l2_6f_exits / l2_6f_bombn)
    ("UP", (120, 100)),
    ("UP", (120, 105)),
    ("UP", (112, 100)),
    ("UP", (128, 100)),
    ("UP", (120, 96)),
    ("UP", (104, 105)),
    ("UP", (136, 105)),
    ("RIGHT", (192, 141)),
    ("RIGHT", (200, 141)),
    ("RIGHT", (184, 141)),
    ("RIGHT", (192, 133)),
    ("RIGHT", (192, 149)),
    ("RIGHT", (200, 133)),
    ("RIGHT", (200, 149)),
    ("DOWN", (120, 180)),
    ("DOWN", (120, 185)),
    ("DOWN", (112, 180)),
    ("DOWN", (128, 180)),
    ("DOWN", (120, 189)),
    ("LEFT", (48, 141)),
    ("LEFT", (64, 141)),
    ("LEFT", (32, 141)),
)

# ---------------------------------------------------------------------------
# Key doors
# ---------------------------------------------------------------------------

KEY_DOOR_6E_RIGHT = KeyDoor(
    room=ROOM_L2_EAST_OF_ROPES,
    direction="RIGHT",
    destination=ROOM_L2_COMPASS,
    key_cost=1,
    approach_band_y=DIAMOND_BAND_6E,
    door_y=DOOR_Y_DEFAULT,
    live=True,
    notes=(
        "Prefer WEST entry via 0x6d (south from 0x7e can stick ~y=181). "
        "Diamond-east band≈113 → wall x≥200 → S2 to door_y≥137 → pure RIGHT. "
        "Carry ≥2 keys into 0x6e so one remains after open."
    ),
    evidence=(
        "recordings/l2_6e_right_ok.json",
        "recordings/l2_east_open.json",
        "recordings/l2_6e_band_scan.json",
    ),
)

KEY_DOOR_5F_LEFT = KeyDoor(
    room=ROOM_L2_BOMB_N,
    direction="LEFT",
    destination=ROOM_L2_GORIYA_WEST,
    key_cost=1,
    approach_band_y=None,
    door_y=DOOR_Y_DEFAULT,
    live=True,
    notes=(
        "After bomb-N entry, doors bit often only DOWN=4; LEFT key opens 0x5e "
        "(5× Goriya 0x06). Approach y≈141 mid-height. keys observed 4→3."
    ),
    evidence=(
        "recordings/l2_past6f_expand.json",
        "recordings/l2_5f_explore.json",
    ),
)

# Walkthrough guessed key RIGHT for boom — LIVE path is bomb-UP → 0x4f instead.
KEY_DOOR_5F_RIGHT_RESIDUAL = KeyDoor(
    room=ROOM_L2_BOMB_N,
    direction="RIGHT",
    destination=0,  # sealed; boom is bomb-N → 0x4f
    key_cost=1,
    live=False,
    notes=(
        "RIGHT sealed (walk+bomb). Boom path is bomb-UP @ (120,101) → 0x4f, "
        "not key-RIGHT."
    ),
    evidence=(
        "recordings/l2_cjf_expand.json",
        "recordings/level2_magic_boomerang_isolated.json",
    ),
)

KEY_DOORS: tuple[KeyDoor, ...] = (
    KEY_DOOR_6E_RIGHT,
    KEY_DOOR_5F_LEFT,
    KEY_DOOR_5F_RIGHT_RESIDUAL,
)

KEY_DOORS_LIVE: tuple[KeyDoor, ...] = tuple(k for k in KEY_DOORS if k.live)

# ---------------------------------------------------------------------------
# Diamond solids (not block-push puzzles; mid-room geometry)
# ---------------------------------------------------------------------------

DIAMOND_EAST_7D = DiamondEast(
    room=ROOM_L2_ENTRY,
    band_y=DIAMOND_BAND_7D,
    destination=ROOM_L2_EAST_KEY,
    requires_key=False,
    notes="Entry-east open (not sealed). Naive y≈141 RIGHT sticks x≈128–176.",
    evidence=(
        "recordings/level2_clear7e_isolated.json",
        "recordings/l2_east_open.json",
    ),
)

DIAMOND_EAST_6E = DiamondEast(
    room=ROOM_L2_EAST_OF_ROPES,
    band_y=DIAMOND_BAND_6E,
    destination=ROOM_L2_COMPASS,
    requires_key=True,
    notes="Key door; band≈113 required (band scan: low y fails max_x).",
    evidence=(
        "recordings/l2_6e_right_ok.json",
        "recordings/l2_6e_band_scan.json",
    ),
)

DIAMOND_EAST_ROOMS: dict[int, int] = {
    ROOM_L2_ENTRY: DIAMOND_BAND_7D,
    ROOM_L2_EAST_OF_ROPES: DIAMOND_BAND_6E,
    # Probe also tried band 113 on 0x6f RIGHT; no live open yet.
    ROOM_L2_COMPASS: DIAMOND_BAND_6E,
}

DIAMOND_EAST: tuple[DiamondEast, ...] = (DIAMOND_EAST_7D, DIAMOND_EAST_6E)

# diamond_east_phase sequence (reference for docs / lab):
# free → band (band_y mid-x) → wall (x≥200) → door_y S2(LEFT×6,vert,RIGHT×10)
# → pure push RIGHT on door_y (no LEFT during push).
DIAMOND_EAST_SEQUENCE: tuple[str, ...] = (
    "free",
    "band",
    "wall",
    "door_y",
    "push",
)

# Push-block recon on 0x6f (center cluster) did not open a new door.
# Centers tried in probe_level2_past_6f expand_phase push_blocks:
PUSH_BLOCK_PROBE_CENTERS: tuple[tuple[int, int], ...] = (
    (120, 141),
    (136, 141),
    (104, 141),
    (120, 125),
    (120, 157),
)

# ---------------------------------------------------------------------------
# Post-boss triforce (0x0e → LEFT → 0x0d) — LIVE assisted geometry (rr-n5i)
# ---------------------------------------------------------------------------
# Walkthrough "TF east of boss" is wrong live: post-kill doors LEFT-only → 0x0d
# WEST of Dodongo. Collect is a **south-band maze** walk (not north band /
# green sprite). Push/bomb not required. Evidence: l2_0d_tf_reach.json LIVE.

L2_TF_PROBE_EVIDENCE: Final = "recordings/l2_0d_tf_reach.json"

# LIVE south-band waypoints on 0x0d (spawn ~(224,141) after LEFT from boss).
# (208,141) free east column → DOWN south band → LEFT → UP diamond maze.
L2_TF_COLLECT_WAYPOINTS: Final[tuple[tuple[int, int], ...]] = (
    (208, 141),
    (208, 189),
    (128, 189),
    (128, 149),
)
L2_TF_WAYPOINT_TOL: Final = 3
L2_TF_COLLECT_XY: Final = (128, 149)
# Collect hitbox (inclusive-ish); observed hit at y=149.
L2_TF_COLLECT_BOX: Final = ((112, 128), (140, 149))  # (x0,x1), (y0,y1)

# Push not required for LIVE collect.
L2_TF_PUSH_BLOCK_STAND: tuple[int, int] | None = None
L2_TF_PUSH_DIR: str | None = None

# Boss room: touch heart then exit LEFT (doors bit 0x02).
L2_BOSS_HC_STAND: Final = (120, 141)
L2_BOSS_EXIT_DOOR_Y: Final = DOOR_Y_DEFAULT  # 141 mid-height LEFT

# Geometry anchors (1-frame BFS free map).
L2_TF_ENTRY_ALCOVE: Final = (208, 141)  # free east column (not stuck x=224)
L2_TF_SPAWN_XY: Final = (224, 141)  # typical after LEFT from 0x0e
L2_TF_NORTH_CORRIDOR_Y: Final = 93  # free but NOT the collect path
L2_TF_SOUTH_BAND_Y: Final = 189
L2_TF_CHECKPOINT: Final = "Level2_0D_PostBoss"


@dataclass(frozen=True)
class PostBossTfPolicy:
    """Collect policy for 0x0d triforce (south-band maze)."""

    waypoints: tuple[tuple[int, int], ...] | None = None
    push_stand: tuple[int, int] | None = None
    push_dir: str | None = None
    collect_xy: tuple[int, int] | None = None
    collect_box: tuple[tuple[int, int], tuple[int, int]] | None = None
    tol: int = 3
    live: bool = False
    notes: str = ""
    evidence: tuple[str, ...] = ()


POST_BOSS_TF_POLICY = PostBossTfPolicy(
    waypoints=L2_TF_COLLECT_WAYPOINTS,
    push_stand=L2_TF_PUSH_BLOCK_STAND,
    push_dir=L2_TF_PUSH_DIR,
    collect_xy=L2_TF_COLLECT_XY,
    collect_box=L2_TF_COLLECT_BOX,
    tol=L2_TF_WAYPOINT_TOL,
    live=True,
    notes=(
        "LIVE assisted: 0x0e LEFT → 0x0d; south-band maze "
        "(208,141)→(208,189)→(128,189)→(128,149); idle until tf&0x02 / mode 18. "
        "North-band green sprite is a red herring. Push/bomb not required. "
        "Not Clean STATUS / natural-entry until pure 2/2 compose."
    ),
    evidence=(
        L2_TF_PROBE_EVIDENCE,
        "recordings/l2_0d_tf_reach_HANDOFF.md",
        "recordings/l2_0d_tf_reach_LIVE.png",
        "recordings/l2_tf02_encode.json",
    ),
)

SEALED_EXIT_BOSS_RIGHT = SealedExit(
    ROOM_L2_BOSS,
    "RIGHT",
    "walkthrough TF-east residual; live sealed (key/bomb/push fail); TF is LEFT→0x0d",
    evidence=(
        "recordings/level2_dodongo.json",
        "recordings/l2_boss_exits.json",
    ),
)

# ---------------------------------------------------------------------------
# Sealed / negative exits
# ---------------------------------------------------------------------------

SEALED_EXITS: tuple[SealedExit, ...] = (
    SealedExit(ROOM_L2_WEST_KEY, "LEFT", "sealed at recon"),
    SealedExit(ROOM_L2_WEST_KEY, "UP", "sealed at recon"),
    SealedExit(ROOM_L2_WEST_KEY, "DOWN", "sealed at recon"),
    SealedExit(ROOM_L2_ENTRY, "LEFT", "sealed at recon (raster)"),
    SealedExit(ROOM_L2_ROPES, "UP", "sealed at recon"),
    SealedExit(
        ROOM_L2_BOMB_N,
        "RIGHT",
        "sealed walk+bomb (rr-cjf); boom is bomb-UP → 0x4f not RIGHT",
        evidence=(
            "recordings/l2_cjf_expand.json",
            "recordings/l2_5f_policy.json",
        ),
    ),
    SEALED_EXIT_BOSS_RIGHT,
    # 0x5f UP without bomb is sealed; bomb-UP is LIVE (BOMB_WALL_5F_NORTH).
)

# ---------------------------------------------------------------------------
# Lab helpers (pure predicates — pass snapshot-like values)
# ---------------------------------------------------------------------------


def bomb_wall_for_room(room: int, face: str = "UP") -> BombWall | None:
    """Return catalog bomb wall for room+face, or None."""
    face_u = face.upper()
    for bw in BOMB_WALLS:
        if bw.room == room and bw.face == face_u:
            return bw
    return None


def diamond_band_for_room(room: int) -> int | None:
    """Open y-band for diamond-east RIGHT, or None if not catalogued."""
    return DIAMOND_EAST_ROOMS.get(room)


def key_door_for(room: int, direction: str) -> KeyDoor | None:
    """Return catalog key door for room+direction, or None."""
    d = direction.upper()
    for kd in KEY_DOORS:
        if kd.room == room and kd.direction == d:
            return kd
    return None


def is_at_bomb_stand(
    link_x: int,
    link_y: int,
    wall: BombWall,
    *,
    tol: int = 6,
) -> bool:
    """True if Link is within tol of wall.stand."""
    sx, sy = wall.stand
    return abs(link_x - sx) <= tol and abs(link_y - sy) <= tol


def bomb_wall_open_predicate(
    *,
    from_room: int,
    to_room: int,
    wall: BombWall | None = None,
) -> bool:
    """True if a room transition matches a catalog bomb-wall open.

    Lab use: after UP+B on 0x6f stand, ``from_room==0x6f and to_room==0x5f``;
    0x5f bomb-N → 0x4f boom also catalogued.
    """
    if wall is not None:
        return from_room == wall.room and to_room == wall.opens_to
    return any(
        bw.live and from_room == bw.room and to_room == bw.opens_to
        for bw in BOMB_WALLS
    )


def key_door_open_predicate(
    *,
    from_room: int,
    to_room: int,
    keys_before: int,
    keys_after: int,
    door: KeyDoor,
) -> bool:
    """True if transition + key delta matches a catalog key door."""
    if from_room != door.room or to_room != door.destination:
        return False
    if not door.live or door.destination == 0:
        return False
    return keys_before - keys_after == door.key_cost


__all__ = [
    "ROOM_L2_ENTRY",
    "ROOM_L2_ROPES",
    "ROOM_L2_WEST_KEY",
    "ROOM_L2_EAST_KEY",
    "ROOM_L2_EAST_OF_ROPES",
    "ROOM_L2_COMPASS",
    "ROOM_L2_BOMB_N",
    "ROOM_L2_GORIYA_WEST",
    "ROOM_L2_ROPES_NORTH",
    "ROOM_L2_BOOM",
    "ROOM_L2_BOSS",
    "ROOM_L2_TF",
    "LEVEL2_TRIFORCE_BIT",
    "ROOM_ITEM_L2_TF",
    "ROOM_ITEM_L2_HC",
    "DOOR_RIGHT",
    "DOOR_LEFT",
    "DOOR_DOWN",
    "DOOR_UP",
    "DOOR_Y_DEFAULT",
    "DIAMOND_WALL_X",
    "DIAMOND_BAND_7D",
    "DIAMOND_BAND_6E",
    "DOOR_Y_MIN_OPEN",
    "B_ITEM_BOMB_PROBE",
    "BombWall",
    "KeyDoor",
    "DiamondEast",
    "SealedExit",
    "PostBossTfPolicy",
    "BOMB_WALL_6F_NORTH",
    "BOMB_WALL_5F_NORTH",
    "BOMB_WALL_4F_NORTH",
    "BOMB_WALL_1E_NORTH",
    "BOMB_WALLS",
    "BOMB_STAND_PROBE_DEFAULTS",
    "BOMB_WALL_NEGATIVES_6F",
    "KEY_DOOR_6E_RIGHT",
    "KEY_DOOR_5F_LEFT",
    "KEY_DOOR_5F_RIGHT_RESIDUAL",
    "KEY_DOORS",
    "KEY_DOORS_LIVE",
    "DIAMOND_EAST_7D",
    "DIAMOND_EAST_6E",
    "DIAMOND_EAST_ROOMS",
    "DIAMOND_EAST",
    "DIAMOND_EAST_SEQUENCE",
    "PUSH_BLOCK_PROBE_CENTERS",
    "L2_TF_PROBE_EVIDENCE",
    "L2_TF_COLLECT_WAYPOINTS",
    "L2_TF_WAYPOINT_TOL",
    "L2_TF_COLLECT_XY",
    "L2_TF_COLLECT_BOX",
    "L2_TF_PUSH_BLOCK_STAND",
    "L2_TF_PUSH_DIR",
    "L2_BOSS_HC_STAND",
    "L2_BOSS_EXIT_DOOR_Y",
    "L2_TF_ENTRY_ALCOVE",
    "L2_TF_SPAWN_XY",
    "L2_TF_NORTH_CORRIDOR_Y",
    "L2_TF_SOUTH_BAND_Y",
    "L2_TF_CHECKPOINT",
    "POST_BOSS_TF_POLICY",
    "SEALED_EXIT_BOSS_RIGHT",
    "SEALED_EXITS",
    "bomb_wall_for_room",
    "diamond_band_for_room",
    "key_door_for",
    "is_at_bomb_stand",
    "bomb_wall_open_predicate",
    "key_door_open_predicate",
]
