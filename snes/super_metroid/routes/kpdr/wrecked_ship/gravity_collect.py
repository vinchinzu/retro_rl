"""Post-Phantoon Attic→Gravity traversal + Gravity Suit collect.

Attic uses the newer human Gravity take as choreography: one Power Bomb at
the right entry, then a reactive aimed-beam sweep through every gray-door
enemy before leaving left. Later transit hops use settled human bodies and
stop on dest-room gs=8.
``play_gravity_collect`` stops when ``GRAVITY_MASK`` is set (320f tape,
collect+settle at 132f; the remaining 188f are morph idle).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from super_metroid.combat.enemies import Enemy, Intent, choose, list_enemies
from super_metroid.paths import GAME_DIR, SCRATCH_STATE_DIR
from super_metroid.plm import session_ram, snapshot_plms
from super_metroid.ram import GRAVITY_MASK, SuperMetroidState
from super_metroid.routes.controller_common import (
    ensure_morph,
    hold,
    is_morph,
    require_room,
    select_weapon,
    unmorph,
)
from super_metroid.routes.kpdr.room_ids import (
    ROOM_BOWLING,
    ROOM_GRAVITY,
    ROOM_HOMING_GEEMER,
    ROOM_PANCAKES,
    ROOM_WEST_OCEAN,
    ROOM_WS_ATTIC,
)
from super_metroid.routes.runtime import ControllerSession, Split
from super_metroid.routes.skills.charge_shot import session_beam_charge
from super_metroid.routes.skills.knockback import (
    escape_knockback_spin,
    is_knockback,
)

S23_HOPS = GAME_DIR / "tasks" / "full_start_v1_segments" / "s23" / "hops"
ATTIC_HOP_BODY = S23_HOPS / "hop_03_Attic.json"
ATTIC_GUIDE_BODY = (
    GAME_DIR
    / "tasks"
    / "gravity_path_v2"
    / "gravity_path_v2_take01_hops"
    / "hop_03_Attic.json"
)
WEST_OCEAN_HOP_BODY = S23_HOPS / "hop_04_West_Ocean.json"
WEST_OCEAN_GUIDE_BODY = (
    GAME_DIR
    / "tasks"
    / "gravity_path_v2"
    / "gravity_path_v2_take01_hops"
    / "hop_04_West_Ocean.json"
)
PANCAKES_HOP_BODY = S23_HOPS / "hop_05_Pancakes_and_Wavers_Room.json"
HOMING_GEEMER_HOP_BODY = S23_HOPS / "hop_06_Homing_Geemer_Room.json"
BOWLING_HOP_BODY = S23_HOPS / "hop_07_Bowling_Alley.json"
BOWLING_GUIDE_BODY = (
    GAME_DIR
    / "tasks"
    / "gravity_path_v2"
    / "gravity_path_v2_take01_hops"
    / "hop_07_Bowling_Alley.json"
)
GRAVITY_HOP_BODY = S23_HOPS / "hop_08_Gravity_Suit_Room.json"
BOWLING_DROP_JSON = SCRATCH_STATE_DIR / "bowling_drop_trigger.json"
TAPE_BODY_FRAMES = 320
COLLECT_FRAMES = 132
CANDIDATE_ID = "controller:gravity_collect"
PARENT_TAPE_ID = "tape:s23_gravity"
_SETTLE = 120

# Live IDs from gravity_path_v2 take01. Coverns (0xEA3F) and the linked
# off-map Kihunter remnant do not count toward the gray-door lock.
ATTIC_KIHUNTER_ID = 0xEB3F
ATTIC_KIHUNTER_WINGS_ID = 0xEB7F
ATTIC_ATOMIC_ID = 0xE9FF
ATTIC_REQUIRED_ENEMY_IDS = frozenset(
    {ATTIC_KIHUNTER_ID, ATTIC_KIHUNTER_WINGS_ID, ATTIC_ATOMIC_ID}
)
ATTIC_INTENT = Intent(engage=ATTIC_REQUIRED_ENEMY_IDS)
ATTIC_COMBAT_BUDGET = 4_200
ATTIC_POWER_BOMB_FUSE = 130
ATTIC_WEST_OCEAN_SETTLE = 420
WEST_OCEAN_ENTRY_RUN_FRAMES = 8
WEST_OCEAN_PANCAKES_SETTLE = 240
# s23 pancakes body crosses the right door and stops in dest gs=11.
# Human Homing Geemer settle is 122f; the 120f generic idle is RED.
PANCAKES_HOMING_GEEMER_SETTLE = 240
# s23 homing-geemer body ends in dest gs=11. Human Bowling settle is 186f;
# the 180f idle is RED at (39,395).
HOMING_GEEMER_BOWLING_SETTLE = 240
# v2 bowling dwell is 2,852f and ends in dest gs=11. Live dest settle is
# 120f at Gravity (216,130) p26. Chozo ride drops morph through the
# mid→bottom screen at y=512; the ride bot is live 0xF0FF beside Samus.
BOWLING_GRAVITY_SETTLE = 240
BOWLING_FLOOR_Y = 512
BOWLING_RIDE_BOT_ID = 0xF0FF
LAST_BOWLING_DROP: dict[str, object] | None = None

__all__ = [
    "ATTIC_HOP_BODY",
    "ATTIC_GUIDE_BODY",
    "BOWLING_DROP_JSON",
    "BOWLING_FLOOR_Y",
    "BOWLING_GRAVITY_SETTLE",
    "BOWLING_GUIDE_BODY",
    "BOWLING_HOP_BODY",
    "BOWLING_RIDE_BOT_ID",
    "CANDIDATE_ID",
    "COLLECT_FRAMES",
    "GRAVITY_HOP_BODY",
    "HOMING_GEEMER_HOP_BODY",
    "LAST_BOWLING_DROP",
    "PANCAKES_HOP_BODY",
    "PARENT_TAPE_ID",
    "TAPE_BODY_FRAMES",
    "WEST_OCEAN_HOP_BODY",
    "WEST_OCEAN_GUIDE_BODY",
    "WEST_OCEAN_ENTRY_RUN_FRAMES",
    "WEST_OCEAN_PANCAKES_SETTLE",
    "PANCAKES_HOMING_GEEMER_SETTLE",
    "HOMING_GEEMER_BOWLING_SETTLE",
    "attic_required_enemies",
    "is_bowling_floor_drop",
    "load_gravity_body",
    "load_s23_body",
    "play_attic_to_west_ocean",
    "play_bowling_to_gravity",
    "play_gravity_collect",
    "play_homing_geemer_to_bowling",
    "play_pancakes_to_homing_geemer",
    "play_west_ocean_to_pancakes",
    "require_gravity_collected",
    "snapshot_bowling_drop",
]


def attic_required_enemies(enemies: Sequence[Enemy]) -> tuple[Enemy, ...]:
    """Door-counted live Attic enemies, excluding Coverns and dead links."""
    return tuple(
        enemy
        for enemy in enemies
        if int(enemy.enemy_id) in ATTIC_REQUIRED_ENEMY_IDS
        and 24 < int(enemy.x) < 1_700
        and 32 < int(enemy.y) < 260
    )


def _has_gravity(state: SuperMetroidState) -> bool:
    return bool(state.collected_items & GRAVITY_MASK)


def _arrived(state: SuperMetroidState, dest_room: int) -> bool:
    return (
        int(state.room_id) == int(dest_room)
        and int(state.game_state) == 8
        and int(state.door_transition) == 0
    )


def is_bowling_floor_drop(state: SuperMetroidState, prev_y: int) -> bool:
    """Morph crossing the mid→bottom screen while the Chozo ride lowers."""
    if int(state.room_id) != ROOM_BOWLING:
        return False
    if not is_morph(int(state.pose)):
        return False
    y = int(state.samus_y)
    return int(prev_y) < BOWLING_FLOOR_Y <= y


def snapshot_bowling_drop(
    session: ControllerSession, prev_y: int
) -> dict[str, object]:
    """Pose / kinematics + live enemies and PLMs at the floor-drop frame."""
    state = session.state
    ram = session_ram(session)
    enemies = tuple(
        {
            "slot": int(enemy.slot),
            "id": int(enemy.enemy_id),
            "id_hex": f"0x{int(enemy.enemy_id):04X}",
            "xy": [int(enemy.x), int(enemy.y)],
            "hp": int(enemy.hp),
        }
        for enemy in (list_enemies(session) if ram is not None else ())
    )
    plms = tuple(
        {
            **row,
            "id_hex": f"0x{int(row['id']):04X}",
            "inst_hex": f"0x{int(row['inst']):04X}",
        }
        for row in (snapshot_plms(ram) if ram is not None else ())
    )
    ride_bot = next(
        (row for row in enemies if int(row["id"]) == BOWLING_RIDE_BOT_ID),
        None,
    )
    return {
        "frame": int(session.frame),
        "room": f"0x{int(state.room_id):04X}",
        "xy": [int(state.samus_x), int(state.samus_y)],
        "prev_y": int(prev_y),
        "pose": int(state.pose),
        "gs": int(getattr(state, "game_state", 8)),
        "dt": int(getattr(state, "door_transition", 0)),
        "movement_type": int(getattr(state, "movement_type", 0)),
        "velocity_y": int(getattr(state, "velocity_y", 0)),
        "vertical_direction": int(getattr(state, "vertical_direction", 0)),
        "ride_bot_id": f"0x{BOWLING_RIDE_BOT_ID:04X}",
        "ride_bot": ride_bot,
        "enemies": enemies,
        "plms": plms,
    }


def _write_bowling_drop(row: dict[str, object]) -> None:
    BOWLING_DROP_JSON.parent.mkdir(parents=True, exist_ok=True)
    BOWLING_DROP_JSON.write_text(
        json.dumps(row, indent=2) + "\n", encoding="utf-8"
    )


def _settled_west_ocean_entry(state: SuperMetroidState) -> bool:
    """Human-tape entry seat: grounded after the long left-door transition."""
    return (
        _arrived(state, ROOM_WEST_OCEAN)
        and int(state.pose) in (1, 2, 9, 10)
        and abs(int(state.velocity_y)) <= 1
        and 120 <= int(state.samus_y) <= 160
    )


def load_s23_body(path: Path | str) -> tuple[tuple[int, ...], ...]:
    """SNES-12 frames from a materialized s23 hop body."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"s23 hop body missing: {source}")
    data = json.loads(source.read_text(encoding="utf-8"))
    raw = data.get("frames") or data.get("raw_buttons")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"no frames in {source}")
    out: list[tuple[int, ...]] = []
    for index, row in enumerate(raw):
        if not isinstance(row, (list, tuple)) or len(row) != 12:
            raise ValueError(f"invalid SNES-12 frame {index} in {source}")
        out.append(tuple(int(v) for v in row))
    return tuple(out)


def load_gravity_body(path: Path | str | None = None) -> tuple[tuple[int, ...], ...]:
    """SNES-12 frames from the s23 Gravity hop body."""
    return load_s23_body(GRAVITY_HOP_BODY if path is None else path)


def _play_s23_to_room(
    session: ControllerSession,
    *,
    label: str,
    start_room: int,
    dest_room: int,
    body: Sequence[Sequence[int]],
    settle: int = _SETTLE,
) -> SuperMetroidState:
    require_room(session, start_room, label)
    if _arrived(session.state, dest_room):
        return session.state
    for row in body:
        if _arrived(session.state, dest_room):
            return session.state
        session.step(np.array(row, dtype=np.int8), f"{label}_tape")
    for _ in range(settle):
        if _arrived(session.state, dest_room):
            return session.state
        hold(session, 1, reason=f"{label}_settle")
    state = session.state
    if not _arrived(state, dest_room):
        raise TimeoutError(
            f"{label}: expected 0x{dest_room:04X} gs=8, "
            f"got 0x{int(state.room_id):04X} gs={int(state.game_state)} "
            f"dt={int(state.door_transition)} {state}"
        )
    return state


def play_attic_to_west_ocean(session: ControllerSession) -> SuperMetroidState:
    """Power Bomb, kill every gray-door enemy, then leave left to West Ocean.

    Choreography follows ``ATTIC_GUIDE_BODY`` but enemy choice is live-RAM
    reactive so natural Main Shaft entry does not inherit the tape's RNG.
    """
    label = "attic_to_west_ocean"
    require_room(session, ROOM_WS_ATTIC, label)

    hold(session, 12, reason=f"{label}_settle")
    select_weapon(session, 3)
    ensure_morph(session)
    hold(session, 8, "X", reason=f"{label}_power_bomb")
    hold(session, ATTIC_POWER_BOMB_FUSE, reason=f"{label}_power_bomb_fuse")
    unmorph(session)

    for _ in range(ATTIC_COMBAT_BUDGET):
        state = session.state
        if _settled_west_ocean_entry(state):
            return state
        if int(state.room_id) == ROOM_WEST_OCEAN:
            for _ in range(ATTIC_WEST_OCEAN_SETTLE):
                if _settled_west_ocean_entry(session.state):
                    return session.state
                hold(session, 1, reason=f"{label}_door_settle")
            state = session.state
            raise TimeoutError(
                f"{label}: West Ocean entry did not ground after the door: {state}"
            )
        if int(state.room_id) != ROOM_WS_ATTIC:
            raise RuntimeError(
                f"{label}: unexpected room 0x{int(state.room_id):04X}: {state}"
            )

        enemies = attic_required_enemies(list_enemies(session))
        if is_knockback(state):
            target = min(
                enemies,
                key=lambda enemy: abs(int(enemy.x) - int(state.samus_x)),
                default=None,
            )
            direction = (
                "LEFT"
                if target is None or int(target.x) < int(state.samus_x)
                else "RIGHT"
            )
            escape_knockback_spin(
                session,
                prefer_dir=direction,
                run_frames=4,
                spin_frames=16,
                label=label,
            )
            continue

        if not enemies:
            hold(session, 1, "LEFT", "B", "X", reason=f"{label}_exit")
            continue

        choice = choose(
            int(state.samus_x),
            int(state.samus_y),
            int(state.facing),
            enemies,
            ATTIC_INTENT,
            movement_type=int(state.movement_type),
            charge=session_beam_charge(session),
            velocity_y=int(state.velocity_y),
            fire_range_px=96,
        )
        buttons = choice.buttons if choice.buttons is not None else ("LEFT", "B")
        if buttons:
            hold(session, 1, *buttons, reason=f"{label}_engage")
        else:
            hold(session, 1, reason=f"{label}_engage_wait")

    state = session.state
    raise TimeoutError(
        f"{label}: kill-all/left-door budget exhausted in "
        f"0x{int(state.room_id):04X} ({state.samus_x},{state.samus_y}) "
        f"kills={state.enemies_killed}/{state.num_enemies}"
    )


def play_west_ocean_to_pancakes(session: ControllerSession) -> SuperMetroidState:
    """Restore the human entry momentum, then follow the Gravity-v2 maze body.

    The human anchor starts at the same far-right door seat with run momentum,
    while the natural Attic leave is settled and grounded.  Eight ``B+LEFT``
    frames reproduce that handoff without changing the body's total duration.
    """
    label = "west_ocean_to_pancakes"
    require_room(session, ROOM_WEST_OCEAN, label)
    if _arrived(session.state, ROOM_PANCAKES):
        return session.state
    hold(
        session,
        WEST_OCEAN_ENTRY_RUN_FRAMES,
        "LEFT",
        "B",
        reason=f"{label}_entry_run",
    )
    body = load_s23_body(WEST_OCEAN_GUIDE_BODY)
    return _play_s23_to_room(
        session,
        label=label,
        start_room=ROOM_WEST_OCEAN,
        dest_room=ROOM_PANCAKES,
        body=body[WEST_OCEAN_ENTRY_RUN_FRAMES:],
        settle=WEST_OCEAN_PANCAKES_SETTLE,
    )


def play_pancakes_to_homing_geemer(session: ControllerSession) -> SuperMetroidState:
    """s23 416f ends in dest gs=11; idle until Homing Geemer gs=8."""
    return _play_s23_to_room(
        session,
        label="pancakes_to_homing_geemer",
        start_room=ROOM_PANCAKES,
        dest_room=ROOM_HOMING_GEEMER,
        body=load_s23_body(PANCAKES_HOP_BODY),
        settle=PANCAKES_HOMING_GEEMER_SETTLE,
    )


def play_homing_geemer_to_bowling(session: ControllerSession) -> SuperMetroidState:
    """s23 98f ends in dest gs=11; idle until Bowling gs=8 at ~(39,395)."""
    return _play_s23_to_room(
        session,
        label="homing_geemer_to_bowling",
        start_room=ROOM_HOMING_GEEMER,
        dest_room=ROOM_BOWLING,
        body=load_s23_body(HOMING_GEEMER_HOP_BODY),
        settle=HOMING_GEEMER_BOWLING_SETTLE,
    )


def play_bowling_to_gravity(session: ControllerSession) -> SuperMetroidState:
    """v2 2852f ends in dest gs=11; snapshot morph crossing y=512."""
    global LAST_BOWLING_DROP
    LAST_BOWLING_DROP = None
    label = "bowling_to_gravity"
    require_room(session, ROOM_BOWLING, label)
    if _arrived(session.state, ROOM_GRAVITY):
        return session.state
    body = load_s23_body(BOWLING_GUIDE_BODY)
    prev_y = int(session.state.samus_y)
    for row in body:
        if _arrived(session.state, ROOM_GRAVITY):
            return session.state
        session.step(np.array(row, dtype=np.int8), f"{label}_tape")
        state = session.state
        if LAST_BOWLING_DROP is None and is_bowling_floor_drop(state, prev_y):
            LAST_BOWLING_DROP = snapshot_bowling_drop(session, prev_y)
            if session_ram(session) is not None:
                _write_bowling_drop(LAST_BOWLING_DROP)
        prev_y = int(state.samus_y)
    for _ in range(BOWLING_GRAVITY_SETTLE):
        if _arrived(session.state, ROOM_GRAVITY):
            return session.state
        hold(session, 1, reason=f"{label}_settle")
        state = session.state
        if LAST_BOWLING_DROP is None and is_bowling_floor_drop(state, prev_y):
            LAST_BOWLING_DROP = snapshot_bowling_drop(session, prev_y)
            if session_ram(session) is not None:
                _write_bowling_drop(LAST_BOWLING_DROP)
        prev_y = int(state.samus_y)
    state = session.state
    if not _arrived(state, ROOM_GRAVITY):
        raise TimeoutError(
            f"{label}: expected 0x{ROOM_GRAVITY:04X} gs=8, "
            f"got 0x{int(state.room_id):04X} gs={int(state.game_state)} "
            f"dt={int(state.door_transition)} {state}"
        )
    return state


def play_gravity_collect(
    session: ControllerSession,
    *,
    frames: Sequence[Sequence[int]] | None = None,
) -> SuperMetroidState:
    """Natural Gravity PLM collect. Leave as soon as the bit is set."""
    require_room(session, ROOM_GRAVITY, "gravity_collect")
    if _has_gravity(session.state):
        return session.state
    body = frames if frames is not None else load_gravity_body()
    for row in body:
        if _has_gravity(session.state):
            return session.state
        session.step(np.array(row, dtype=np.int8), "gravity_tape")
    if not _has_gravity(session.state):
        raise TimeoutError(
            f"gravity_collect: items still 0x{session.state.collected_items:04X} "
            f"after {len(body)} frames: {session.state}"
        )
    return session.state


def require_gravity_collected(
    session: ControllerSession, splits: list[Split] | None = None, result: object = None
) -> None:
    del splits, result
    if not _has_gravity(session.state):
        raise RuntimeError(
            f"Gravity not collected: items=0x{session.state.collected_items:04X}"
        )
