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
        role="live: reached via 0x69 west bomb wall; interior census pending",
        evidence="fixture-live",
    ),
    Level7RoomHyp(ROPES_KEY, "ropes_key", role="key_plus_1"),
    Level7RoomHyp(
        DODONGOS_UPGRADE,
        "dodongos_upgrade_path",
        ram_id=0x58,
        role="live: dodongo-family 0x31, room_item_id 0x0f, dark; Link spawns "
        "bottom (120,205); east exit not yet resolved (locked/kill-gated?)",
        evidence="fixture-live",
    ),
    Level7RoomHyp(BOMB_UPGRADE, "bomb_upgrade_16", role="optional_capacity"),
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
        "(120,205) S mouth; kill-clear opens the UP door bit but a full-width "
        "water moat (~y120) gates the UP path — needs the Stepladder; L/R "
        "walled; DOWN -> 0x59",
        evidence="fixture-live",
    ),
    Level7RoomHyp(DIGDOGGER_2, "digdogger_skip"),
    Level7RoomHyp(MOLDORM_KEY_OPT, "moldorm_key_optional"),
    Level7RoomHyp(GORIYA_PRE_HUNGRY, "goriya_pre_hungry"),
    Level7RoomHyp(HUNGRY_GORIYA, "hungry_goriya", role="requires_food"),
    Level7RoomHyp(MAP, "map"),
    Level7RoomHyp(MAP_EAST_LOCK, "map_east_fifth_lock", role="skip_lock"),
    Level7RoomHyp(HIDDEN_RUPEES, "hidden_rupees_off_map"),
    Level7RoomHyp(GORIYA_POST_RUPEE, "goriya_post_rupee"),
    Level7RoomHyp(GORIYA_KEY, "goriya_key", role="key_plus_1"),
    Level7RoomHyp(WEST_LOCK_SKIP, "west_lock_skip", role="skip_lock"),
    Level7RoomHyp(CANDLE_PUSH, "candle_block_push"),
    Level7RoomHyp(RED_CANDLE_CELLAR, "red_candle_cellar", role="public_level7_red_candle"),
    Level7RoomHyp(GORIYA_PRE_DIG, "goriya_pre_forced_dig"),
    Level7RoomHyp(FORCED_DIGDOGGER, "forced_digdogger"),
    Level7RoomHyp(DODONGOS_BOSS_PATH, "dodongos_boss_path"),
    Level7RoomHyp(TIP_OF_NOSE, "tip_of_nose_wallmasters", role="block_stairs"),
    Level7RoomHyp(NOSE_CELLAR, "nose_cellar"),
    Level7RoomHyp(PRE_BOSS, "pre_boss"),
    Level7RoomHyp(AQUAMENTUS, "aquamentus", role="heart_plus_1"),
    Level7RoomHyp(TRIFORCE, "triforce_shard_7", role="public_level7"),
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
            _open(DoorDir.DOWN, ROPES_KEY, notes="source; 0x68 has 4 blade traps 0x49"),
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
            _e(DoorDir.UP, BOMB_UPGRADE, GateKind.KEY, notes="north x=120 channel; keys 4->3"),
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
                    "live 0x49: kill-clear opens the UP door bit, but a "
                    "full-width water moat (~y120, tile 0xF4) walls the UP "
                    "path — the Stepladder is required to cross. L/R walled. "
                    "Dest DIGDOGGER_2 unobserved (recon fixture has no ladder)"
                ),
            ),
        ),
        DIGDOGGER_2: (
            _back(DoorDir.UP, GORIYA_BUBBLE),
            _open(DoorDir.LEFT, GORIYA_PRE_HUNGRY),
            _e(DoorDir.RIGHT, MOLDORM_KEY_OPT, GateKind.BOMB, notes="optional key"),
        ),
        MOLDORM_KEY_OPT: (_open(DoorDir.LEFT, DIGDOGGER_2),),
        GORIYA_PRE_HUNGRY: (
            _back(DoorDir.LEFT, DIGDOGGER_2),
            _e(DoorDir.UP, HUNGRY_GORIYA, GateKind.KEY),
        ),
        HUNGRY_GORIYA: (
            _back(DoorDir.UP, GORIYA_PRE_HUNGRY),
            _open(DoorDir.UP, MAP, notes="requires_food"),
        ),
        MAP: (
            _back(DoorDir.UP, HUNGRY_GORIYA),
            _e(DoorDir.RIGHT, MAP_EAST_LOCK, GateKind.KEY, notes="fifth lock; skip"),
            _e(DoorDir.UP, HIDDEN_RUPEES, GateKind.BOMB, notes="off-map bomb north"),
        ),
        MAP_EAST_LOCK: (_back(DoorDir.RIGHT, MAP),),
        HIDDEN_RUPEES: (
            _open(DoorDir.DOWN, MAP),
            _e(DoorDir.RIGHT, GORIYA_POST_RUPEE, GateKind.BOMB),
        ),
        GORIYA_POST_RUPEE: (
            _open(DoorDir.LEFT, HIDDEN_RUPEES),
            _e(DoorDir.RIGHT, GORIYA_KEY, GateKind.KILL_CLEAR),
            _open(DoorDir.DOWN, WEST_LOCK_SKIP),
        ),
        GORIYA_KEY: (_back(DoorDir.RIGHT, GORIYA_POST_RUPEE),),
        WEST_LOCK_SKIP: (
            _open(DoorDir.UP, GORIYA_POST_RUPEE),
            _e(DoorDir.LEFT, GORIYA_PRE_HUNGRY, GateKind.KEY, notes="fifth-lock sibling; skip"),
            _e(DoorDir.RIGHT, CANDLE_PUSH, GateKind.BOMB, notes="bomb east to candle"),
        ),
        CANDLE_PUSH: (
            _open(DoorDir.LEFT, WEST_LOCK_SKIP),
            _open(DoorDir.DOWN, RED_CANDLE_CELLAR, notes="stairs after left block push"),
            _e(DoorDir.RIGHT, GORIYA_PRE_DIG, GateKind.BOMB),
        ),
        RED_CANDLE_CELLAR: (
            _open(DoorDir.UP, CANDLE_PUSH, notes="stairs return"),
        ),
        GORIYA_PRE_DIG: (
            _open(DoorDir.LEFT, CANDLE_PUSH),
            _e(DoorDir.RIGHT, FORCED_DIGDOGGER, GateKind.KEY),
        ),
        FORCED_DIGDOGGER: (
            _back(DoorDir.RIGHT, GORIYA_PRE_DIG),
            _e(DoorDir.UP, DODONGOS_BOSS_PATH, GateKind.KILL_CLEAR, notes="must kill"),
        ),
        DODONGOS_BOSS_PATH: (
            _back(DoorDir.UP, FORCED_DIGDOGGER),
            _e(DoorDir.RIGHT, TIP_OF_NOSE, GateKind.BOMB),
        ),
        TIP_OF_NOSE: (
            _open(DoorDir.LEFT, DODONGOS_BOSS_PATH),
            _open(DoorDir.DOWN, NOSE_CELLAR, notes="push mid-right block; stairs"),
        ),
        NOSE_CELLAR: (
            _open(DoorDir.UP, TIP_OF_NOSE, notes="stairs"),
            _open(DoorDir.RIGHT, PRE_BOSS, notes="stairs far side"),
        ),
        PRE_BOSS: (
            _open(DoorDir.LEFT, NOSE_CELLAR, notes="stairs"),
            _e(DoorDir.RIGHT, AQUAMENTUS, GateKind.BOMB),
        ),
        AQUAMENTUS: (
            _back(DoorDir.RIGHT, PRE_BOSS),
            _open(DoorDir.RIGHT, TRIFORCE, notes="heart then east shard"),
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
    "Level7LedgerRow",
    "Level7RoomHyp",
    "ledger_notes",
    "path_requires_food",
    "path_uses_fifth_lock",
    "preferred_path",
    "ram_ids_observed",
]
