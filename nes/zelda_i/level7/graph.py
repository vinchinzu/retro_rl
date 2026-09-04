"""Offline first-quest Level 7 room/door/stair graph.

Every room is a walkthrough hypothesis.  Source ids live in ``0x7xx`` so they
cannot be mistaken for live RAM ``$EB`` values (0x00–0x7F).  ``ram_id`` stays
None until a room is observed in play (ENTRY ``0x79``, N-path dest ``0x69``,
its east dest ``0x6A``, then ``0x6A`` east dest ``0x6B``).
Do not copy these ids into stop predicates.
"""

from __future__ import annotations

from dataclasses import dataclass

from zelda_i.anchors import SCREEN_LEVEL7_ENTRY_ROOM
from zelda_i.door_graph.core import (
    DoorDir,
    DungeonDoorGraph,
    GateKind,
    InventoryCaps,
    RoomExit,
)

EVIDENCE = "hypothesis"
ROUTE_ELIGIBLE = False
_HYP = "hypothesis"

# Source-only ids (not RAM).  Live promotion replaces ram_id, never these keys.
ENTRY = 0x700
MOLDORMS = 0x701
KEESE = 0x702
GORIYA_HINT = 0x703
OLD_MAN_NOSE = 0x704
DIGDOGGER_1 = 0x705
STALFOS_KEY = 0x706
GORIYA_BOMB_HUB = 0x707
KEESE_TRAPS = 0x708
ROPES_KEY = 0x709
DODONGOS_UPGRADE = 0x70A
BOMB_UPGRADE = 0x70B
GORIYA_COMPASS = 0x70C
COMPASS = 0x70D
GORIYA_BUBBLE = 0x70E
DIGDOGGER_2 = 0x70F
MOLDORM_KEY_OPT = 0x710
GORIYA_PRE_HUNGRY = 0x711
HUNGRY_GORIYA = 0x712
MAP = 0x713
MAP_EAST_LOCK = 0x714
HIDDEN_RUPEES = 0x715
GORIYA_POST_RUPEE = 0x716
GORIYA_KEY = 0x717
WEST_LOCK_SKIP = 0x718
CANDLE_PUSH = 0x719
RED_CANDLE_CELLAR = 0x71A
GORIYA_PRE_DIG = 0x71B
FORCED_DIGDOGGER = 0x71C
DODONGOS_BOSS_PATH = 0x71D
TIP_OF_NOSE = 0x71E
NOSE_CELLAR = 0x71F
PRE_BOSS = 0x720
AQUAMENTUS = 0x721
TRIFORCE = 0x722


@dataclass(frozen=True)
class Level7RoomHyp:
    """One unobserved L7 room.  ``ram_id`` is filled only from live ``$EB``."""

    source_id: int
    name: str
    ram_id: int | None = None
    role: str = ""
    evidence: str = EVIDENCE
    route_eligible: bool = ROUTE_ELIGIBLE


LEVEL7_ROOMS: tuple[Level7RoomHyp, ...] = (
    Level7RoomHyp(
        ENTRY,
        "entry",
        ram_id=SCREEN_LEVEL7_ENTRY_ROOM,
        role="public_level7_entry",
        evidence="fixture-live",
    ),
    Level7RoomHyp(
        MOLDORMS,
        "entry_north_goriya",
        ram_id=0x69,
        role="goriya_0x05_then_cleared_forever; live == source GORIYA_BOMB_HUB",
        evidence="fixture-live",
    ),
    Level7RoomHyp(
        KEESE,
        "keese_dark",
        ram_id=0x6A,
        role="keese_0x1b",
        evidence="fixture-live",
    ),
    Level7RoomHyp(
        GORIYA_HINT,
        "goriya_hint",
        ram_id=0x6B,
        role="goriya_0x05",
        evidence="fixture-live",
    ),
    Level7RoomHyp(
        OLD_MAN_NOSE,
        "old_man_tip_of_nose",
        ram_id=0x5B,
        role="live: bubble 0x40 + statue 0x50, dead-end (N/E/W/S walled); "
        "the north spur off 0x6B, NOT on the candle mainline",
        evidence="fixture-live",
    ),
    Level7RoomHyp(
        DIGDOGGER_1,
        "digdogger_optional",
        ram_id=0x6C,
        role="digdogger_0x38_plus_statue_0x55",
        evidence="fixture-live",
    ),
    Level7RoomHyp(
        STALFOS_KEY,
        "stalfos_key",
        ram_id=0x6D,
        role="key_plus_1; live: stalfos 0x2a + small_key 0x19, dead-end",
        evidence="fixture-live",
    ),
    Level7RoomHyp(GORIYA_BOMB_HUB, "goriya_bomb_hub", role="prefer_bomb_walls"),
    Level7RoomHyp(
        KEESE_TRAPS,
        "keese_traps",
        ram_id=0x68,
        role="live: 4 blade traps 0x49 (corners) + 4 keese 0x1b; "
        "OPEN N/S + bombed E; W shut",
        evidence="fixture-live",
    ),
    Level7RoomHyp(
        ROPES_KEY,
        "ropes_key",
        ram_id=0x78,
        role="live: ropes 0x28 + small_key 0x19 on the floor; dead-end "
        "(only UP back to 0x68); key not auto-picked",
        evidence="fixture-live",
    ),
    Level7RoomHyp(
        DODONGOS_UPGRADE,
        "dodongos_upgrade_path",
        ram_id=0x58,
        role="live: 3x invuln 0x31 (hp 240), room_item_id 0x0f, dark; Link "
        "spawns bottom (120,205); OPEN east -> 0x59; KEY north -> 0x48",
        evidence="fixture-live",
    ),
    Level7RoomHyp(
        BOMB_UPGRADE,
        "bomb_upgrade_16",
        ram_id=0x48,
        role="live: bubble 0x40 + 0x4f; old-man 'I BET YOU'D LIKE TO HAVE "
        "-100' bomb-capacity dead-end; do not write max_bombs",
        evidence="fixture-live",
    ),
    Level7RoomHyp(
        GORIYA_COMPASS,
        "goriya_to_compass",
        ram_id=0x59,
        role="live: goriya 0x05 + 0x06; OPEN west door from 0x58; UP door "
        "opens after kill-clear (cur_opened_doors bit 1)",
        evidence="fixture-live",
    ),
    Level7RoomHyp(COMPASS, "compass_stalfos"),
    Level7RoomHyp(
        GORIYA_BUBBLE,
        "goriya_keese_bubble",
        ram_id=0x49,
        role="live: goriya 0x05 + keese 0x1b + bubble residual 0x2b; entry "
        "(120,205) S mouth; L/R walled; DOWN -> 0x59; UP across the water "
        "moat needs the Stepladder and lands $EB=0x39",
        evidence="fixture-live",
    ),
    Level7RoomHyp(
        DIGDOGGER_2,
        "digdogger_skip",
        ram_id=0x39,
        role="live: digdogger 0x38 + statue 0x55; entry (120,205) S mouth; "
        "LEFT door OPEN on spawn (skip the fight) -> $EB=0x38; whistle=1",
        evidence="fixture-live",
    ),
    Level7RoomHyp(MOLDORM_KEY_OPT, "moldorm_key_optional"),
    Level7RoomHyp(
        GORIYA_PRE_HUNGRY,
        "goriya_pre_hungry",
        ram_id=0x38,
        role="live: goriya 0x05+0x06, diamond floor, compass room_item 0x0f "
        "uncollected; entry (208,141) E mouth; KEY-UP (keys 4->3) after "
        "rising the east pocket x=208 (interior y=149 is a diamond wall)",
        evidence="fixture-live",
    ),
    Level7RoomHyp(
        HUNGRY_GORIYA,
        "hungry_goriya",
        ram_id=0x28,
        role="live: GRUMBLE GRUMBLE NPC 0x36 + bubble 0x40; entry (120,205) "
        "S mouth; natural bait feed Food 1->0 then UP -> $EB=0x18",
        evidence="fixture-live",
    ),
    Level7RoomHyp(
        MAP,
        "map",
        ram_id=0x18,
        role="live: dark; map room_item 0x17; goriya 0x05 + keese 0x1b + "
        "bubble 0x2b; entry (120,189) from hungry UP; bomb-north stand "
        "(120,93) face UP -> $EB=0x08 (skip locked east)",
        evidence="fixture-live",
    ),
    Level7RoomHyp(MAP_EAST_LOCK, "map_east_fifth_lock", role="skip_lock"),
    Level7RoomHyp(
        HIDDEN_RUPEES,
        "hidden_rupees_off_map",
        ram_id=0x08,
        role="live: diamond cross; 0x35 cluster; entry (120,189) S mouth "
        "from 0x18 bomb-north; bomb-east stand (208,141) -> $EB=0x09",
        evidence="fixture-live",
    ),
    Level7RoomHyp(
        GORIYA_POST_RUPEE,
        "goriya_post_rupee",
        ram_id=0x09,
        role="live: goriya 0x05+0x06, room_item 0x0f, water north; entry "
        "(32,141) W mouth; south shutter is KILL_CLEAR (doors bit stays "
        "LEFT) then DOWN -> $EB=0x19; skip optional east key",
        evidence="fixture-live",
    ),
    Level7RoomHyp(GORIYA_KEY, "goriya_key", role="key_plus_1"),
    Level7RoomHyp(
        WEST_LOCK_SKIP,
        "west_lock_skip",
        ram_id=0x19,
        role="live: diamond floor, goriya 0x05; entry (120,93) N mouth; "
        "KEY-west skip; bomb-east stand (208,141) via south band -> $EB=0x1A",
        evidence="fixture-live",
    ),
    Level7RoomHyp(
        CANDLE_PUSH,
        "candle_block_push",
        ram_id=0x1A,
        role="live: 4-diamond plus + 0x68 at (96,144); entry (32,141) W "
        "mouth; kill-clear (NE goriya) then 0x68 UP (96,144)->(96,128); "
        "stairs at (128,141) -> cellar $EB=0x4A mode 9",
        evidence="fixture-live",
    ),
    Level7RoomHyp(
        RED_CANDLE_CELLAR,
        "red_candle_cellar",
        ram_id=0x4A,
        role="live: mode 9 two-ladder item cellar; Red Candle on the center "
        "pad; ADDR_CANDLE 0->2 NATURAL at ~(135,141); keese 0x1b; stairs "
        "return west-ladder UP -> play 0x1A (96,157)",
        evidence="fixture-live",
    ),
    Level7RoomHyp(
        GORIYA_PRE_DIG,
        "goriya_pre_forced_dig",
        ram_id=0x1B,
        role="live: goriya 0x05, open floor, entry (32,141) W mouth; "
        "KEY-east (diamond lock); bombs 8->7 from 0x1A east bomb",
        evidence="fixture-live",
    ),
    Level7RoomHyp(
        FORCED_DIGDOGGER,
        "forced_digdogger",
        ram_id=0x1C,
        role="live: digdogger 0x38 at (208,141), entry (16,141) W mouth; "
        "KEY-east from 0x1B (keys 3->2); north door KILL_CLEAR after shrink",
        evidence="fixture-live",
    ),
    Level7RoomHyp(
        DODONGOS_BOSS_PATH,
        "dodongos_boss_path",
        ram_id=0x0C,
        role="live: 3x 0x31 (dodongo-family), entry (120,205) S mouth; "
        "KILL-CLEAR north of forced Digdogger 0x1C",
        evidence="fixture-live",
    ),
    Level7RoomHyp(
        TIP_OF_NOSE,
        "tip_of_nose_wallmasters",
        ram_id=0x0D,
        role="live: 5x wallmaster 0x27 (plus-corners peel to west wall "
        "one-at-a-time) + bubbles 0x2b + 0x68 at (192,144); entry (32,141) "
        "W mouth is a grab trap (x=32 any y). Kill 5 then room_all_dead=1 "
        "2/2 (0d_wm_v10/0d_cleared). RIGHT-push slides 0x68 to (208,96) "
        "on the NE hole. ROM: AttrE secret=block_stairs; CheckWarps dest "
        "is cellar 0x7B AttrB; far side play 0x29. Walk-on live 2/2 "
        "(level7.stairs0d)",
        evidence="fixture-live",
    ),
    Level7RoomHyp(
        NOSE_CELLAR,
        "nose_cellar",
        ram_id=0x7B,
        role="ROM cellar 0x7B tunnel AttrA=0x29 AttrB=0x0D; live walk-on "
        "from 0x0D 2/2 (level7.stairs0d, B ladder x=$C0, spawn (192,93)). "
        "B→A cross live dest 0x29",
        evidence="fixture-live",
    ),
    Level7RoomHyp(
        PRE_BOSS,
        "pre_boss",
        ram_id=0x29,
        role="live: cellar 0x7B AttrA dest (96,157) 2/2; goriya 0x05/0x06; "
        "E bomb -> 0x2A",
        evidence="fixture-live",
    ),
    Level7RoomHyp(
        AQUAMENTUS,
        "aquamentus",
        ram_id=0x2A,
        role="live: type 0x3D W mouth (32,141); sword-only HC 3→4; "
        "E shutter -> 0x2B",
        evidence="fixture-live",
    ),
    Level7RoomHyp(
        TRIFORCE,
        "triforce_shard_7",
        ram_id=0x2B,
        role="live: diamond floor, W mouth (16,141); south-around to shard. "
        "OW leftover 0x42 (96,93) TF 0x40 on this recon pin (not Survival 0x7F)",
        evidence="fixture-live",
    ),
)

LEVEL7_ROOM_BY_ID: dict[int, Level7RoomHyp] = {r.source_id: r for r in LEVEL7_ROOMS}
FOOD_GATES: frozenset[int] = frozenset({HUNGRY_GORIYA})
# The fifth lock is the map-east KEY.  WEST_LOCK_SKIP is the bomb-east hall
# that avoids it; the room itself is on the preferred candle path.
FIFTH_LOCK_SKIPS: frozenset[int] = frozenset({MAP_EAST_LOCK})


@dataclass(frozen=True)
class Level7LedgerRow:
    """Hypothesized natural key/bomb delta.  Counts, not RAM writes."""

    room: int
    keys: int = 0
    bombs: int = 0
    food: int = 0
    note: str = ""


# Speed route: four dungeon keys, skip two locks via bombs (source).
LEVEL7_KEY_BOMB_LEDGER: tuple[Level7LedgerRow, ...] = (
    Level7LedgerRow(STALFOS_KEY, keys=1, note="stalfos drop"),
    Level7LedgerRow(GORIYA_BOMB_HUB, bombs=-1, note="bomb west, skip later lock"),
    Level7LedgerRow(ROPES_KEY, keys=1, note="ropes drop"),
    Level7LedgerRow(DODONGOS_UPGRADE, keys=-1, note="key north to bomb-upgrade old man"),
    Level7LedgerRow(GORIYA_PRE_HUNGRY, keys=-1, note="key north to hungry goriya"),
    Level7LedgerRow(HUNGRY_GORIYA, food=-1, note="natural bait consume"),
    Level7LedgerRow(MAP, bombs=-1, note="bomb north instead of fifth east lock"),
    Level7LedgerRow(HIDDEN_RUPEES, bombs=-1, note="bomb east"),
    Level7LedgerRow(GORIYA_KEY, keys=1, note="goriya drop"),
    Level7LedgerRow(CANDLE_PUSH, bombs=-1, note="bomb east to candle block room"),
    Level7LedgerRow(GORIYA_PRE_DIG, keys=-1, bombs=-1, note="bomb then key east"),
    Level7LedgerRow(DODONGOS_BOSS_PATH, bombs=-1, note="bomb east to tip of nose"),
    Level7LedgerRow(PRE_BOSS, bombs=-1, note="bomb east to aquamentus"),
)


def _e(
    direction: DoorDir,
    target: int | None,
    gate: GateKind = GateKind.OPEN,
    *,
    notes: str = "",
    verification: str = _HYP,
) -> RoomExit:
    return RoomExit(
        direction,
        target,
        gate,
        notes=notes or _HYP,
        verification=verification,
    )


def _open(direction: DoorDir, target: int, notes: str = "") -> RoomExit:
    return _e(direction, target, GateKind.OPEN, notes=notes)


def _back(src_dir: DoorDir, src: int) -> RoomExit:
    return _open(src_dir.opposite, src, notes="backtrack")


def _l7_exits() -> dict[int, tuple[RoomExit, ...]]:
    """Source topology.  Prefer BOMB over MAP_EAST_LOCK / WEST_LOCK_SKIP."""
    return {
        ENTRY: (
            _e(
                DoorDir.UP,
                MOLDORMS,
                notes="live dest $EB=0x69 goriya 0x05; dead: N path is Moldorms",
                verification="fixture-live",
            ),
            _e(
                DoorDir.RIGHT,
                None,
                notes="live PNG: east door present/shut; dest RAM unobserved",
                verification="probe_geometry",
            ),
        ),
        MOLDORMS: (
            _back(DoorDir.UP, ENTRY),
            _e(
                DoorDir.RIGHT,
                KEESE,
                notes=(
                    "live dest $EB=0x6A keese 0x1b; OPEN doorway — "
                    "dead: KEY/KILL_CLEAR, doors bit never sets"
                ),
                verification="fixture-live",
            ),
            _e(
                DoorDir.LEFT,
                KEESE_TRAPS,
                GateKind.BOMB,
                notes=(
                    "live: 0x69 west BOMB wall -> $EB=0x68; stand ~(44,141) "
                    "face LEFT; cur_opened_doors LEFT bit sets 2/2. This is "
                    "the candle-path branch (source GORIYA_BOMB_HUB LEFT). "
                    "0x69 has NO north exit (x-sweep 104..156 solid at y=93)"
                ),
                verification="fixture-live",
            ),
            _e(
                DoorDir.DOWN,
                ENTRY,
                GateKind.OPEN,
                notes="live: 0x69 DOWN -> 0x79 entry (backtrack)",
                verification="fixture-live",
            ),
        ),
        KEESE: (
            _e(
                DoorDir.RIGHT,
                GORIYA_HINT,
                notes=(
                    "live dest $EB=0x6B goriya 0x05; OPEN doorway — y=141 "
                    "centre band walls at x=48, cross the y=93 top corridor"
                ),
                verification="fixture-live",
            ),
            _open(DoorDir.LEFT, GORIYA_BOMB_HUB, "west after stalfos-key return"),
            _e(DoorDir.UP, COMPASS, GateKind.BOMB, notes="optional dark bomb"),
        ),
        GORIYA_HINT: (
            _e(
                DoorDir.LEFT,
                KEESE,
                GateKind.OPEN,
                notes="backtrack; live 0x6B LEFT->0x6A (224,141) OPEN 2/2",
                verification="fixture-live",
            ),
            _e(
                DoorDir.RIGHT,
                DIGDOGGER_1,
                GateKind.OPEN,
                notes=(
                    "live dest $EB=0x6C digdogger 0x38 + statue 0x55; OPEN "
                    "doorway — ride y=109 east, drop east column to y=141, "
                    "push RIGHT; skippable whistle-split spur"
                ),
                verification="fixture-live",
            ),
            _e(
                DoorDir.UP,
                OLD_MAN_NOSE,
                GateKind.KILL_CLEAR,
                notes=(
                    "live dest $EB=0x5B (bubble 0x40 + 0x50) 2/2; the north "
                    "door notch is at x~118 on the y=93 band (NOT x=128 — "
                    "that is solid). Opens after the six 0x6B goriya clear. "
                    "0x5B is a dead-end spur, not the candle mainline"
                ),
                verification="fixture-live",
            ),
        ),
        OLD_MAN_NOSE: (_back(DoorDir.UP, GORIYA_HINT),),
        DIGDOGGER_1: (
            _back(DoorDir.RIGHT, GORIYA_HINT),
            _e(
                DoorDir.RIGHT,
                STALFOS_KEY,
                GateKind.OPEN,
                notes=(
                    "live dest $EB=0x6D stalfos 0x2a + small_key 0x19 2/2; "
                    "ride y=141, bump the digdogger, RIGHT through the door"
                ),
                verification="fixture-live",
            ),
        ),
        STALFOS_KEY: (
            _e(
                DoorDir.LEFT,
                DIGDOGGER_1,
                GateKind.OPEN,
                notes="live: 0x6D is a dead-end, only LEFT -> 0x6C 2/2",
                verification="fixture-live",
            ),
        ),
        GORIYA_BOMB_HUB: (
            _open(DoorDir.RIGHT, KEESE),
            _e(DoorDir.LEFT, KEESE_TRAPS, GateKind.BOMB, notes="prefer bomb west"),
            _e(DoorDir.UP, DODONGOS_UPGRADE, GateKind.BOMB, notes="optional bomb north"),
        ),
        KEESE_TRAPS: (
            _e(
                DoorDir.RIGHT,
                GORIYA_BOMB_HUB,
                GateKind.BOMB,
                notes="live: 0x68 east is the bombed 0x69 wall (backtrack)",
                verification="fixture-live",
            ),
            _e(
                DoorDir.DOWN,
                ROPES_KEY,
                GateKind.OPEN,
                notes=(
                    "live dest $EB=0x78 ropes 0x28 + small_key 0x19 2/2; "
                    "peel x=160, drop y=141, align x=120, push DOWN; blade "
                    "traps 0x49 in corners — do not occupancy-walk (knockback "
                    "poisons the grid)"
                ),
                verification="fixture-live",
            ),
            _e(
                DoorDir.UP,
                DODONGOS_UPGRADE,
                GateKind.OPEN,
                notes=(
                    "live dest $EB=0x58 (dodongo 0x31) 2/2; align x=120, push "
                    "UP; 0x68 is dark w/ 4 blade traps 0x49 + 4 keese 0x1b"
                ),
                verification="fixture-live",
            ),
        ),
        ROPES_KEY: (_back(DoorDir.DOWN, KEESE_TRAPS),),
        DODONGOS_UPGRADE: (
            _e(
                DoorDir.DOWN,
                KEESE_TRAPS,
                GateKind.OPEN,
                notes="live: 0x58 DOWN -> 0x68 (backtrack)",
                verification="fixture-live",
            ),
            _e(
                DoorDir.UP,
                BOMB_UPGRADE,
                GateKind.KEY,
                notes=(
                    "live dest $EB=0x48 2/2; KEY door keys 4->3. Central "
                    "2-block mass walls x=120 around y=141 — east-around "
                    "(120,165)->(160,165)->y=93, align x=120, push UP. "
                    "Dodge 3x invuln 0x31. 0x48 is a 100-rupee bomb-capacity "
                    "old-man dead-end; do not write max_bombs"
                ),
                verification="fixture-live",
            ),
            _e(
                DoorDir.RIGHT,
                GORIYA_COMPASS,
                GateKind.OPEN,
                notes=(
                    "live dest $EB=0x59 goriya 0x05/0x06 2/2; OPEN door, keys "
                    "unchanged. Route: up the east-open column, x=200, drop "
                    "y=141, push RIGHT. 3x invuln 0x31 (hp240) roam — dodge"
                ),
                verification="fixture-live",
            ),
        ),
        BOMB_UPGRADE: (_back(DoorDir.UP, DODONGOS_UPGRADE),),
        GORIYA_COMPASS: (
            _back(DoorDir.RIGHT, DODONGOS_UPGRADE),
            _e(DoorDir.RIGHT, COMPASS, GateKind.KILL_CLEAR),
            _e(
                DoorDir.UP,
                GORIYA_BUBBLE,
                GateKind.KILL_CLEAR,
                notes=(
                    "live dest $EB=0x49 2/2; goriya 0x05/0x06 kill-clear sets "
                    "cur_opened_doors bit 3 (UP). Naive clear boxes Link at "
                    "(48,125) — perimeter waypoint micro: rise y~100, west "
                    "x~44, rise y~64, cross x=120, push UP (frame 2329)"
                ),
                verification="fixture-live",
            ),
        ),
        COMPASS: (_back(DoorDir.RIGHT, GORIYA_COMPASS),),
        GORIYA_BUBBLE: (
            _e(
                DoorDir.DOWN,
                GORIYA_COMPASS,
                GateKind.OPEN,
                notes="live: 0x49 DOWN -> 0x59 (backtrack), 1/1",
                verification="fixture-live",
            ),
            _e(
                DoorDir.UP,
                DIGDOGGER_2,
                GateKind.KILL_CLEAR,
                notes=(
                    "live dest $EB=0x39 2/2; kill-clear goriya 0x05 then walk "
                    "UP at x=120 across the full-width water moat (~y120, tile "
                    "0xF4). Stepladder required (ADDR_LADDER). Doors bit stays "
                    "0 (OPEN-like). Dest: digdogger 0x38 + statue 0x55, "
                    "(120,205) S mouth"
                ),
                verification="fixture-live",
            ),
        ),
        DIGDOGGER_2: (
            _back(DoorDir.UP, GORIYA_BUBBLE),
            _e(
                DoorDir.LEFT,
                GORIYA_PRE_HUNGRY,
                GateKind.OPEN,
                notes=(
                    "live dest $EB=0x38 2/2; skip Digdogger (LEFT already OPEN). "
                    "South mouth (120,205) -> x=120 UP to y=141 -> LEFT. Do not "
                    "hug the SW statue (boxes at (48,189)). Dest: goriya 0x05/0x06, "
                    "(208,141) E mouth"
                ),
                verification="fixture-live",
            ),
            _e(DoorDir.RIGHT, MOLDORM_KEY_OPT, GateKind.BOMB, notes="optional key"),
        ),
        MOLDORM_KEY_OPT: (_open(DoorDir.LEFT, DIGDOGGER_2),),
        GORIYA_PRE_HUNGRY: (
            _back(DoorDir.LEFT, DIGDOGGER_2),
            _e(
                DoorDir.UP,
                HUNGRY_GORIYA,
                GateKind.KEY,
                notes=(
                    "live dest $EB=0x28 2/2; KEY consume keys 4->3. Interior "
                    "y=149 diamond row blocks UP at x=120/104/88/200 — rise "
                    "the east mouth pocket x=208 to y=93, cross to x=120, "
                    "push UP. Dest: GRUMBLE GRUMBLE (120,205) S mouth"
                ),
                verification="fixture-live",
            ),
        ),
        HUNGRY_GORIYA: (
            _back(DoorDir.UP, GORIYA_PRE_HUNGRY),
            _e(
                DoorDir.UP,
                MAP,
                GateKind.OPEN,
                notes=(
                    "live dest $EB=0x18 2/2 after natural Food 1->0. Equip "
                    "already-owned Bait (B-slot 6), walk to (120,141), tap B. "
                    "NPC 0x36 despawns; north door opens. Dest: dark map room "
                    "room_item 0x17, goriya+keese"
                ),
                verification="fixture-live",
            ),
        ),
        MAP: (
            _back(DoorDir.UP, HUNGRY_GORIYA),
            _e(DoorDir.RIGHT, MAP_EAST_LOCK, GateKind.KEY, notes="fifth lock; skip"),
            _e(
                DoorDir.UP,
                HIDDEN_RUPEES,
                GateKind.BOMB,
                notes=(
                    "live dest $EB=0x08 2/2; stand (120,93) face UP; "
                    "cur_opened_doors UP bit sets; bombs 7->6"
                ),
                verification="fixture-live",
            ),
        ),
        MAP_EAST_LOCK: (_back(DoorDir.RIGHT, MAP),),
        HIDDEN_RUPEES: (
            _open(DoorDir.DOWN, MAP),
            _e(
                DoorDir.RIGHT,
                GORIYA_POST_RUPEE,
                GateKind.BOMB,
                notes=(
                    "live dest $EB=0x09 2/2; south-band x=200 then east "
                    "column to stand (208,141) face RIGHT; bombs 6->5"
                ),
                verification="fixture-live",
            ),
        ),
        GORIYA_POST_RUPEE: (
            _open(DoorDir.LEFT, HIDDEN_RUPEES),
            _e(DoorDir.RIGHT, GORIYA_KEY, GateKind.KILL_CLEAR),
            _e(
                DoorDir.DOWN,
                WEST_LOCK_SKIP,
                GateKind.KILL_CLEAR,
                notes=(
                    "live dest $EB=0x19 2/2; south shutter opens after "
                    "goriya 0x05/0x06 clear (doors bit stays LEFT). Dead: "
                    "DOWN is OPEN without the kill"
                ),
                verification="fixture-live",
            ),
        ),
        GORIYA_KEY: (_back(DoorDir.RIGHT, GORIYA_POST_RUPEE),),
        WEST_LOCK_SKIP: (
            _open(DoorDir.UP, GORIYA_POST_RUPEE),
            _e(DoorDir.LEFT, GORIYA_PRE_HUNGRY, GateKind.KEY, notes="fifth-lock sibling; skip"),
            _e(
                DoorDir.RIGHT,
                CANDLE_PUSH,
                GateKind.BOMB,
                notes=(
                    "live dest $EB=0x1A 2/2; south-around diamonds "
                    "(96,141)->(96,189)->(208,189)->(208,141) face RIGHT; "
                    "bombs 8->7 (goriya drops topped 5->8 in 0x09)"
                ),
                verification="fixture-live",
            ),
        ),
        CANDLE_PUSH: (
            _open(DoorDir.LEFT, WEST_LOCK_SKIP),
            _e(
                DoorDir.DOWN,
                RED_CANDLE_CELLAR,
                notes=(
                    "live dest $EB=0x4A mode 9 2/2 after 0x68 UP "
                    "(96,144)->(96,128) then (128/136,141) tile 0x71. "
                    "Dead: push while a goriya still lives. ADDR_CANDLE "
                    "0->2 walking onto the center pad from the east ladder"
                ),
                verification="fixture-live",
            ),
            _e(
                DoorDir.RIGHT,
                GORIYA_PRE_DIG,
                GateKind.BOMB,
                notes=(
                    "live dest $EB=0x1B 2/2; south-around "
                    "(96,189)->(208,189)->(208,141) face RIGHT; "
                    "bombs 8->7; dest goriya 0x05 (32,141) W mouth"
                ),
                verification="fixture-live",
            ),
        ),
        RED_CANDLE_CELLAR: (
            _e(
                DoorDir.UP,
                CANDLE_PUSH,
                notes=(
                    "live dest $EB=0x1A play 2/2 (4a_ret_v7/v8). Dead: walk "
                    "off the pad at y=141 (tile 243) as the return. Recipe: "
                    "RIGHT y=141 to east column x~192, LEFT+DOWN drop to "
                    "floor y=189, LEFT x=48, UP west ladder (tile 111) until "
                    "stairs. Leftover play (96,157) candle 2"
                ),
                verification="fixture-live",
            ),
        ),
        GORIYA_PRE_DIG: (
            _open(DoorDir.LEFT, CANDLE_PUSH),
            _e(
                DoorDir.RIGHT,
                FORCED_DIGDOGGER,
                GateKind.KEY,
                notes=(
                    "live dest $EB=0x1C 2/2; y=141 RIGHT spends a key "
                    "(3->2); dest digdogger 0x38 (16,141) W mouth. "
                    "Do not skip this fight (north is KILL_CLEAR)"
                ),
                verification="fixture-live",
            ),
        ),
        FORCED_DIGDOGGER: (
            _back(DoorDir.RIGHT, GORIYA_PRE_DIG),
            _e(
                DoorDir.UP,
                DODONGOS_BOSS_PATH,
                GateKind.KILL_CLEAR,
                notes=(
                    "live dest $EB=0x0C 2/2 (1c_wh_v3/v4); pause-select "
                    "recorder=5 (no ADDR_SELECTED_ITEM poke), 12xB shrinks "
                    "0x38 HP240 -> 3x 0x18 HP128, sword-kill, north. Dest "
                    "3x 0x31 (120,205) S mouth"
                ),
                verification="fixture-live",
            ),
        ),
        DODONGOS_BOSS_PATH: (
            _back(DoorDir.UP, FORCED_DIGDOGGER),
            _e(
                DoorDir.RIGHT,
                TIP_OF_NOSE,
                GateKind.BOMB,
                notes=(
                    "live dest $EB=0x0D 2/2; east-around the y=141 tile-181 "
                    "mass (120,165)->(200,165)->(200,141)->(208,141) face "
                    "RIGHT; bombs 7->6. Dead: y=141 centre RIGHT. Dest "
                    "wallmaster 0x27 + 0x68 (32,141) W mouth"
                ),
                verification="fixture-live",
            ),
        ),
        TIP_OF_NOSE: (
            _open(DoorDir.LEFT, DODONGOS_BOSS_PATH),
            _open(
                DoorDir.DOWN,
                NOSE_CELLAR,
                notes=(
                    "kill 5 wallmasters (room_all_dead=1 2/2); 16px RIGHT "
                    "on 0x68 (192,144) reveals the staircase at the "
                    "(208,96) cell; ring walk x=32 UP then y=96 east "
                    "steps on. ROM CheckWarps: cellar 0x7B AttrB; "
                    "InitMode9 spawns right ladder x=$C0. Live 2/2"
                ),
            ),
        ),
        NOSE_CELLAR: (
            _open(
                DoorDir.UP,
                TIP_OF_NOSE,
                notes="ROM CheckSubroom X>=$80 AttrB -> play 0x0D",
            ),
            _open(
                DoorDir.RIGHT,
                PRE_BOSS,
                notes="ROM: cellar 0x7B left/AttrA ladder x=$30 -> play 0x29",
            ),
        ),
        PRE_BOSS: (
            _open(DoorDir.LEFT, NOSE_CELLAR, notes="ROM 0x29 stairs -> cellar 0x7B"),
            _e(
                DoorDir.RIGHT,
                AQUAMENTUS,
                GateKind.BOMB,
                notes="ROM 0x29 E bomb / 0x2A W bomb",
            ),
        ),
        AQUAMENTUS: (
            _back(DoorDir.RIGHT, PRE_BOSS),
            _open(
                DoorDir.RIGHT,
                TRIFORCE,
                notes="ROM 0x2A E shutter then 0x2B W open after heart",
            ),
        ),
        TRIFORCE: (_back(DoorDir.RIGHT, AQUAMENTUS),),
    }


LEVEL7_HYPOTHESIS_GRAPH = DungeonDoorGraph.from_exits(
    _l7_exits(),
    level=7,
    name="level7_q1_hypothesis",
)


def ram_ids_observed() -> bool:
    return any(room.ram_id is not None for room in LEVEL7_ROOMS)


def path_uses_fifth_lock(path: tuple[RoomExit, ...] | None) -> bool:
    if not path:
        return False
    return any(exit_.target_room in FIFTH_LOCK_SKIPS for exit_ in path)


def path_requires_food(path: tuple[RoomExit, ...] | None) -> bool:
    if not path:
        return False
    return any(exit_.target_room in FOOD_GATES for exit_ in path)


def preferred_path(
    start: int,
    goal: int,
    caps: InventoryCaps | None = None,
) -> tuple[RoomExit, ...] | None:
    """BFS under Survival-like caps.  Bomb walls beat the fifth lock."""
    inv = caps if caps is not None else InventoryCaps(keys=4, bombs=8, can_clear=True)
    return LEVEL7_HYPOTHESIS_GRAPH.bfs_path(start, goal, inv)


def ledger_notes() -> list[str]:
    rows = []
    keys = bombs = food = 0
    for row in LEVEL7_KEY_BOMB_LEDGER:
        keys += row.keys
        bombs += row.bombs
        food += row.food
        rows.append(
            f"{LEVEL7_ROOM_BY_ID[row.room].name}: keys{row.keys:+d} bombs{row.bombs:+d}"
            f" food{row.food:+d} ({row.note})"
        )
    rows.append(f"net_hyp keys{keys:+d} bombs{bombs:+d} food{food:+d}")
    return rows


__all__ = [
    "AQUAMENTUS",
    "CANDLE_PUSH",
    "ENTRY",
    "EVIDENCE",
    "FIFTH_LOCK_SKIPS",
    "FOOD_GATES",
    "GORIYA_POST_RUPEE",
    "HIDDEN_RUPEES",
    "HUNGRY_GORIYA",
    "KEESE",
    "LEVEL7_HYPOTHESIS_GRAPH",
    "LEVEL7_KEY_BOMB_LEDGER",
    "LEVEL7_ROOMS",
    "MAP",
    "MAP_EAST_LOCK",
    "MOLDORMS",
    "RED_CANDLE_CELLAR",
    "ROUTE_ELIGIBLE",
    "TRIFORCE",
    "WEST_LOCK_SKIP",
    "Level7LedgerRow",
    "Level7RoomHyp",
    "ledger_notes",
    "path_requires_food",
    "path_uses_fifth_lock",
    "preferred_path",
    "ram_ids_observed",
]
