"""Clean Main Shaft → Early Supers and Bridge Missile → Dachora.

The master arrives at the Main Shaft elevator with zero missiles. The upper
right door (``0x9BC8``) takes five, so this branch:

1. Follows the human descent to the first ledge, then snipes the three upper
   Zeelas. The drop an enemy leaves is rolled when it dies, so every shot
   timing is tried against the ROM and one whose drop is a missile (+2) is kept.
2. Collects each drop, walks to the door and shoots it open.

The mockball keeps running momentum under all three gates. A wall jump
reaches the shelf above the right chute, then Samus crosses to the tank
and returns via the Bridge Missile pack. ROM lookahead only selects
the Main Shaft farming inputs.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from super_metroid.routes.kpdr.brinstar.big_pink import _IDLE, _Lab, _act, _u16
from super_metroid.routes.runtime import ControllerSession

ROOM_MAIN_SHAFT = 0x9AD9
ROOM_EARLY_SUPERS = 0x9BC8
ROOM_DACHORA = 0x9CB3

HUMAN_TAPE = "tasks/full_start_v1_segments/s6/tape.json"
HUMAN_SPAN = (7160, 8770)  # board row 39 slice of that tape
HUMAN_IDLE = 4  # the V6 build retimed this slice by four idle frames
LEDGE_FRAME = 600  # human frame on the first ledge, above the left-wall Zeela

_ZEELA = 0xDC7F
_ADDR_ENEMY0 = 0x0F78
_ADDR_PROJ_ID, _ADDR_PROJ_X, _ADDR_PROJ_Y, _ADDR_PROJ_LIST = 0x1997, 0x1A4B, 0x1A93, 0x1B47
_DROP_HEADER = 0xF345
_SPAWN_END = 0xED8D  # drop pointers below this are still the pop-in animation


def _human_slice() -> np.ndarray:
    """Board row 39's human buttons, shots kept: the Zeela fight needs them."""
    from super_metroid.human_tape.hops import load_task_json
    from super_metroid.paths import GAME_DIR

    data = load_task_json(GAME_DIR / HUMAN_TAPE)
    first, last = HUMAN_SPAN
    body = np.asarray(data["frames"], dtype=np.int8)[first : last + 1].copy()
    return np.concatenate([np.zeros((HUMAN_IDLE, 12), dtype=np.int8), body])


def zeelas(env: Any) -> list[tuple[int, int, int]]:
    """Live Zeelas as ``(x, y, hp)``."""
    ram = env.get_ram()
    out = []
    for slot in range(16):
        base = _ADDR_ENEMY0 + slot * 0x40
        if _u16(ram, base) == _ZEELA:
            out.append((_u16(ram, base + 2), _u16(ram, base + 6), _u16(ram, base + 0x14)))
    return out


def drops(env: Any) -> list[tuple[str, int, int]]:
    """Enemy drops as ``(kind, x, y)``; ``pop`` while the spawn animation plays."""
    ram = env.get_ram()
    out = []
    for slot in range(18):
        if _u16(ram, _ADDR_PROJ_ID + 2 * slot) != _DROP_HEADER:
            continue
        loop = _u16(ram, _ADDR_PROJ_LIST + 2 * slot)
        kind = ("pop" if loop < _SPAWN_END else "small" if loop < 0xEDA3 else "big" if loop < 0xEDB9
                else "missile" if loop < 0xEDCF else "other")
        out.append((kind, _u16(ram, _ADDR_PROJ_X + 2 * slot), _u16(ram, _ADDR_PROJ_Y + 2 * slot)))
    return out


_FAN: dict[str, Any] = {}


def _init_fan(inherited_env: Any) -> None:
    """A fork gets its own emulator; close its inherited copy first."""
    from super_metroid.dev.common import make_dev_env

    inherited_env.close()
    env = make_dev_env()
    env.reset()
    _FAN["lab"] = _Lab(env)


def _try_snipe(task: tuple[bytes, int, tuple[str, ...], int | None, int, int, int, int, int]) -> tuple[int, int] | None:
    """One (spot, aim, wait) row of the grid; returns ``(frames, delay)`` of its best kill."""
    base, alive, aim, spot, wait, settle, period, shots, limit = task
    lab = _FAN["lab"]
    held = [name for name in aim if name != "X"]
    best: tuple[int, int] | None = None
    for delay in range(12):
        lab.restore(base)
        health, killed, ready, frames, died_at = lab.health(), None, None, 0, None
        before = zeelas(lab.env)
        for frame in range(limit):
            if best is not None and frames >= best[0]:
                break
            state = lab.state()
            if ready is None and (spot is None or abs(state.samus_x - spot) <= 2):
                ready = frame
            k = -1 if ready is None else frame - ready - wait - delay
            if ready is None:
                action = _act("RIGHT" if state.samus_x < spot else "LEFT")
            elif k >= 0 and k % period == 0 and k < shots:
                action = _act(*held, "X")
            else:
                action = _act(*held) if held else _IDLE
            lab.env.step(action)
            frames += 1
            if lab.health() < health:
                break
            now = zeelas(lab.env)
            if killed is None and len(now) < alive:
                killed = frame
                hurt = min(before, key=lambda z: z[2])  # the one the shots wore down
                died_at = (hurt[0], hurt[1])
            before = now
            if killed is not None and frame >= killed + settle:
                # The new drop sits where the Zeela died; older drops are elsewhere.
                near = [kind for kind, x, y in drops(lab.env)
                        if died_at is not None and abs(x - died_at[0]) <= 24 and abs(y - died_at[1]) <= 40]
                if near == ["missile"]:
                    best = (killed + 1, delay)
                break
    return best


def _plan_for(lab: _Lab, base: bytes, aim: tuple[str, ...], spot: int | None, wait: int, delay: int,
              frames: int, period: int, shots: int) -> list[np.ndarray]:
    """Rebuild the button list of a winning try by replaying it on ``lab``."""
    held = [name for name in aim if name != "X"]
    lab.restore(base)
    plan, ready = [], None
    for frame in range(frames):
        state = lab.state()
        if ready is None and (spot is None or abs(state.samus_x - spot) <= 2):
            ready = frame
        k = -1 if ready is None else frame - ready - wait - delay
        if ready is None:
            action = _act("RIGHT" if state.samus_x < spot else "LEFT")
        elif k >= 0 and k % period == 0 and k < shots:
            action = _act(*held, "X")
        else:
            action = _act(*held) if held else _IDLE
        lab.env.step(action)
        plan.append(action)
    return plan


def _snipe(lab: _Lab, aims: list[tuple[str, ...]], waits: range, *,
           spots: tuple[int | None, ...] = (None,), settle: int = 40, period: int = 6,
           shots: int = 60, limit: int = 220, workers: int = 16) -> tuple[list[np.ndarray], dict[str, Any]] | None:
    """Fewest frames to kill one Zeela with a missile drop, never losing energy.

    Each try walks to ``spot`` (an x, or stays put), holds ``aim`` for ``wait``
    frames, then pulses X. The drop kind is rolled at the kill frame, so most
    (wait, delay) pairs give energy; the ones that give a missile are found by
    trying each on the ROM, a grid row per fork worker.
    """
    import multiprocessing

    base = lab.blob()
    alive = len(zeelas(lab.env))
    tasks = [(base, alive, aim, spot, wait, settle, period, shots, limit)
             for spot in spots for aim in aims for wait in waits]
    with multiprocessing.get_context("fork").Pool(workers, initializer=_init_fan, initargs=(lab.env,)) as pool:
        results = pool.map(_try_snipe, tasks)
    winners = [(res[0], task, res[1]) for task, res in zip(tasks, results) if res is not None]
    if not winners:
        lab.restore(base)
        return None
    frames, task, delay = min(winners, key=lambda w: w[0])
    aim, spot, wait = task[2], task[3], task[4]
    plan = _plan_for(lab, base, aim, spot, wait, delay, frames, period, shots)
    lab.restore(base)
    return plan, {"spot": spot, "aim": aim, "wait": wait, "delay": delay}


def _collect(lab: _Lab, *, limit: int = 260) -> None:
    """Walk to the missile drop and jump for it; stop on the +2."""
    before = lab.state().missiles
    rise_left, prev_y, still = 0, None, 0
    for _ in range(limit):
        state = lab.state()
        if state.missiles > before:
            return
        spots = [(x, y) for kind, x, y in drops(lab.env) if kind in ("pop", "missile")]
        if not spots:
            raise RuntimeError("Main Shaft: the missile drop is gone")
        tx, ty = min(spots, key=lambda s: abs(s[0] - state.samus_x))
        dx, above = tx - state.samus_x, state.samus_y - ty
        still = still + 1 if state.samus_y == prev_y else 0
        prev_y = state.samus_y
        grounded = still >= 2
        names = []
        if abs(dx) > 3:
            names.append("RIGHT" if dx > 0 else "LEFT")
        if rise_left > 0:
            names.append("A")
            rise_left -= 1
        elif grounded and above > 14 and abs(dx) <= 16:
            names.append("A")
            rise_left = 22
        lab.commit([_act(*names) if names else _IDLE])
    raise RuntimeError("Main Shaft: missile drop was not collected")


def _macros() -> list[np.ndarray]:
    """Directional, run and jump holds a beam can chain four frames at a time."""
    out = []
    for direction in ((), ("LEFT",), ("RIGHT",)):
        for jump in ((), ("A",)):
            for run in ((), ("B",)):
                out.append(_act(*direction, *jump, *run) if (direction or jump or run) else _IDLE)
    return out


def _reach_drop(lab: _Lab, *, width: int = 20, chunk: int = 4, depth: int = 50) -> None:
    """Beam over movement macros until the live missile drop is picked up.

    For drops across a gap, where walking and one jump are not enough. Children
    that lose energy are dropped; the score is the distance to the drop.
    """
    base = lab.blob()
    before = lab.state().missiles
    macros = _macros()
    beam: list[tuple[float, bytes, list[np.ndarray]]] = [(0.0, base, [])]
    for _ in range(depth):
        kids, seen = [], set()
        for _, blob, path in beam:
            for action in macros:
                lab.restore(blob)
                if not lab.play([action] * chunk):
                    continue
                state = lab.state()
                spots = [(x, y) for kind, x, y in drops(lab.env) if kind in ("pop", "missile")]
                if state.missiles > before:
                    lab.restore(base)
                    lab.commit(path + [action] * chunk)
                    return
                if not spots:
                    continue
                tx, ty = min(spots, key=lambda sp: abs(sp[0] - state.samus_x) + abs(sp[1] - state.samus_y))
                score = abs(tx - state.samus_x) + 1.5 * abs(ty - state.samus_y) + 0.2 * (len(path) + chunk)
                key = (state.samus_x // 4, state.samus_y // 4, state.pose, state.velocity_y)
                if key not in seen:
                    seen.add(key)
                    kids.append((score, lab.blob(), path + [action] * chunk))
        if not kids:
            break
        kids.sort(key=lambda kid: kid[0])
        beam = kids[:width]
    lab.restore(base)
    raise RuntimeError("Main Shaft: no jump chain reaches the missile drop")


DOOR_MISSILES = 5
DOOR_X = 219  # stand here and the shots reach the red cap at x=224
DOOR_BAND = (1120, 1190)  # the cap spans block rows 70-73


def _select_missiles(lab: _Lab) -> None:
    if lab.state().selected_item != 1:
        lab.commit([_act("SELECT")] + [_IDLE] * 20)


def _open_door(lab: _Lab, *, limit: int = 700) -> None:
    """Drop down the right column, shoot the red cap five times, walk through.

    The column's far side (x>=208) is a ledge at y=1067, so the drop is made
    from x=194..205 and the walk to the door starts on the floor below it.
    """
    before = lab.state().missiles
    if before < DOOR_MISSILES:
        raise RuntimeError(f"Main Shaft: {before} missiles at the red door, need {DOOR_MISSILES}")
    _select_missiles(lab)
    at_door = 0
    for _ in range(limit):
        state = lab.state()
        if state.room_id == ROOM_EARLY_SUPERS and state.game_state == 8 and not state.door_transition:
            return
        if state.room_id != ROOM_MAIN_SHAFT:
            lab.commit([_IDLE])
        elif state.samus_y < DOOR_BAND[0]:
            names = ["LEFT"] if state.samus_x > 205 else ["RIGHT"] if state.samus_x < 194 else []
            lab.commit([_act(*names) if names else _IDLE])
        elif state.samus_x < DOOR_X - 8:
            lab.commit([_act("RIGHT")])
        else:
            # One press per eight frames; the cap opens on the fifth hit and
            # Samus walks on through with RIGHT held.
            shoot = at_door % 8 == 0 and state.missiles > before - DOOR_MISSILES
            at_door += 1
            lab.commit([_act("RIGHT", "X") if shoot else _act("RIGHT")])
    raise RuntimeError("Main Shaft: never settled in the Early Supers room")


def _hold(lab: _Lab, count: int, *names: str) -> None:
    lab.commit([_act(*names) if names else _IDLE] * count)


def _farm_door_ammo(lab: _Lab) -> None:
    """Work across the upper Zeelas from the elevator landing."""
    lab.commit(_human_slice()[:LEDGE_FRAME])
    shots = (
        ([("LEFT", "DOWN")], range(0, 13, 4), (None,)),
        ([("X",)], range(0, 121, 4), (90, 70)),
        ([("UP",)], range(0, 161, 4), (None, 90)),
    )
    for index, (aims, waits, spots) in enumerate(shots):
        if lab.state().missiles >= DOOR_MISSILES:
            return
        result = _snipe(lab, aims, waits, spots=spots)
        if result is None:
            raise RuntimeError(f"Main Shaft: no usable missile drop from Zeela {index + 1}")
        lab.commit(result[0])
        if index == 0:
            _collect(lab)
    while lab.state().missiles < DOOR_MISSILES:
        _reach_drop(lab)


def _mockball(lab: _Lab) -> None:
    """Jump out of the entrance pocket and roll under the three gates."""
    _hold(lab, 10)
    for count, keys in (
        (4, ("RIGHT",)), (4, ("RIGHT", "DOWN")), (8, ("LEFT",)),
        (4, ("RIGHT", "A")), (20, ("A",)), (4, ()),
    ):
        _hold(lab, count, *keys)
    if not 78 <= lab.state().samus_x <= 91:
        raise RuntimeError("Early Supers: missed the entrance runway")

    _hold(lab, 40, "RIGHT", "B")
    _hold(lab, 3, "RIGHT", "B", "A")
    _hold(lab, 7, "RIGHT", "B")
    _hold(lab, 1, "B", "A", "DOWN")
    _hold(lab, 2, "B", "A")
    _hold(lab, 1, "B", "A", "DOWN")

    # Samus lands during the morph animation. Check that the ball still
    # carries run momentum after the last shutter, on the far ledge.
    _hold(lab, 91, "RIGHT", "B", "A")
    state = lab.state()
    if state.room_id != ROOM_EARLY_SUPERS or not (
        670 <= state.samus_x <= 691 and state.samus_y <= 365
        and state.pose in (29, 30) and state.momentum_x >= 2
    ):
        raise RuntimeError(f"Early Supers: mockball missed the far ledge at ({state.samus_x}, {state.samus_y})")


def _collect_supers(lab: _Lab) -> None:
    """Shoot the chute block, wall-jump onto the shelf, and run to the tank."""
    before = lab.state().max_super_missiles
    _hold(lab, 8, "UP", "Y")
    for frame in range(50):
        x = lab.state().samus_x
        if x < 684:
            _hold(lab, 1, "RIGHT")
        elif x > 688:
            _hold(lab, 1, "LEFT")
        else:
            _hold(lab, 1, "UP", *(("X",) if frame % 10 == 0 else ()))

    _hold(lab, 1, "RIGHT")
    _hold(lab, 20, "RIGHT", "A")
    _hold(lab, 2, "LEFT")
    _hold(lab, 1, "LEFT", "A")
    _hold(lab, 40, "A")
    _hold(lab, 48)
    shelf = lab.state()
    if not (657 <= shelf.samus_x <= 668 and 214 <= shelf.samus_y <= 224):
        raise RuntimeError("Early Supers: missed the shelf above the right chute")

    for frame in range(340):
        state = lab.state()
        if state.max_super_missiles == before + 5:
            return
        jump = frame < 48 or (
            225 <= state.samus_x <= 310 and (state.vertical_direction == 1 or frame % 2 == 0)
        )
        keys = ["LEFT", "B"]
        if jump:
            keys.append("A")
        if frame % 6 == 0:
            keys.append("X")
        _hold(lab, 1, *keys)
    raise RuntimeError("Early Supers: did not collect the Super Missile tank")


def _wait_pickup(lab: _Lab) -> None:
    for frame in range(400):
        _hold(lab, 1, *(("A",) if frame % 5 == 0 else ()))
        state = lab.state()
        if frame > 100 and state.game_state == 8 and state.pose not in (0, 138):
            return


def _return_with_bridge_missiles(lab: _Lab) -> None:
    """Bomb down the left chute, pick up the Missile pack, and exit left."""
    before = lab.state().max_missiles
    _wait_pickup(lab)
    _hold(lab, 1, "DOWN")
    _hold(lab, 5)
    _hold(lab, 1, "DOWN")
    _hold(lab, 12)
    for frame in range(300):
        state = lab.state()
        if state.samus_y >= 420 and not state.vertical_direction:
            break
        keys = []
        if state.samus_x > 73:
            keys.append("LEFT")
            if frame % 38 == 0:
                keys.append("X")
        elif state.samus_x < 70:
            keys.append("RIGHT")
        elif frame % 40 == 0:
            keys.append("X")
        _hold(lab, 1, *keys)
    else:
        raise RuntimeError("Early Supers: did not descend the left chute")

    for frame in range(450):
        if lab.state().max_missiles == before + 5:
            break
        _hold(lab, 1, "RIGHT", *(("X",) if frame % 38 == 0 else ()))
    else:
        raise RuntimeError("Early Supers: did not collect Bridge Missile")

    _wait_pickup(lab)
    for frame in range(350):
        if lab.state().samus_x <= 98:
            break
        _hold(lab, 1, "LEFT", *(("X",) if frame % 38 == 0 else ()))
    # The left hatch is one jump above the bomb-block corridor.
    _hold(lab, 8, "UP")
    for frame in range(420):
        state = lab.state()
        if state.room_id == ROOM_MAIN_SHAFT and state.game_state == 8 and not state.door_transition:
            return
        if state.room_id != ROOM_EARLY_SUPERS:
            _hold(lab, 1)
            continue
        keys = ["RIGHT" if frame < 10 else "LEFT"]
        if frame >= 10 and frame % 36 < 12:
            keys.append("A")
        if frame >= 10 and frame % 12 == 5:
            keys.append("X")
        _hold(lab, 1, *keys)
    raise RuntimeError("Early Supers: did not return to Main Shaft")


def _enter_dachora(lab: _Lab) -> None:
    """Steer between the shaft ledges and save Supers by using five Missiles."""
    starting_missiles = lab.state().missiles
    if starting_missiles < DOOR_MISSILES:
        raise RuntimeError("Dachora: need five Missiles after the Early Supers detour")

    for frame in range(700):
        state = lab.state()
        if state.room_id == ROOM_DACHORA and state.game_state == 8 and not state.door_transition:
            return
        x, y = state.samus_x, state.samus_y
        keys: tuple[str, ...] = ()
        if state.room_id == ROOM_MAIN_SHAFT and state.game_state == 8:
            if y < 1650:
                if y < 1175:
                    lo, hi = 0, 124
                elif y < 1450:
                    lo, hi = 154, 165
                elif y < 1560:
                    lo, hi = 172, 182
                else:
                    lo, hi = 115, 127
                keys = ("RIGHT",) if x < lo else ("LEFT",) if x > hi else ()
            elif y != 1675 or x < 180:
                keys = ("RIGHT",)
            elif state.selected_item == 0 and state.missiles > starting_missiles - DOOR_MISSILES:
                keys = ("SELECT",)
            else:
                fire = state.missiles > starting_missiles - DOOR_MISSILES and frame % 12 == 0
                keys = ("RIGHT", "X") if fire else ("RIGHT",)
        _hold(lab, 1, *keys)
    raise RuntimeError("Main Shaft: never settled in Dachora Room")


def play_early_supers_to_dachora(session: ControllerSession) -> SimpleNamespace:
    """Take the Early Supers route from the real Green Brinstar elevator leave."""
    if session.state.room_id != ROOM_MAIN_SHAFT:
        raise ValueError("Early Supers requires the Green Brinstar Main Shaft entry")
    env = session.env
    lab = _Lab(env)
    entry = lab.blob()
    try:
        _farm_door_ammo(lab)
        _select_missiles(lab)
        _open_door(lab)
        _mockball(lab)
        _collect_supers(lab)
        _return_with_bridge_missiles(lab)
        _enter_dachora(lab)
    finally:
        env.em.set_state(entry)
    for action in lab.actions:
        session.step(action, "early_supers_to_dachora")
    return SimpleNamespace(fight=SimpleNamespace(rollout_restores=lab.restores + 1))
