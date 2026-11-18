"""Clean Big Pink leave: board row 41, ``0x9D19 -> 0x9D9C``.

The master arrives with zero missiles and the pink door at ``(987, 139)``
takes five. The human tape had refills on, so it never farmed. This leave:

1. Farms Zeb drops on the pipe ledge. A shot delay is chosen by trying each
   against the ROM and keeping one whose drop rolls a missile (+2). The pipe
   re-arms when the room reloads, so the farm leaves by the west door and
   returns.
2. Follows the human climb with ROM lookahead. Only when a Sidehopper is near
   does it branch (fire, wait, step); every branch that loses energy is
   dropped.
3. Shoots the door five times and walks through.

Buttons only. Every choice is a restore-and-try; nothing writes RAM.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np

from retro_harness.actions import buttons
from super_metroid.ram import parse_env_state
from super_metroid.routes.runtime import ControllerSession

ROOM_BIG_PINK = 0x9D19
ROOM_KIHUNTER = 0x9D9C
DOOR_MISSILES = 5
FARM_TARGET = 8  # four drops; the climb may spend two killing a Sidehopper

HUMAN_TAPE = "tasks/full_start_v1_segments/s6/tape.json"
HUMAN_SPAN = (9911, 11360)  # board row 41 slice of that tape
LEDGE_FRAME = 352  # human frame standing on the Zeb pipe ledge
CLIMB_START = 138  # first non-idle human frame
DOOR_X = 987

_IDLE = np.zeros(12, dtype=np.int8)
_X, _SELECT = 9, 2  # SNES-12 columns the climb masks
_ZEB, _SIDEHOPPER, _REO = 0xF193, 0xD93F, 0xD27F
_HOSTILE = (_ZEB, _SIDEHOPPER, _REO)
_DROP_HEADER = 0xF345  # enemy-drop projectile; its list says what it is
_ADDR_PROJ_ID, _ADDR_PROJ_X, _ADDR_PROJ_Y, _ADDR_PROJ_LIST = 0x1997, 0x1A4B, 0x1A93, 0x1B47
_ADDR_ENEMY0 = 0x0F78  # 0x40 bytes per slot: id, x(+2), y(+6), hp(+0x14)


def _act(*names: str) -> np.ndarray:
    return np.asarray(buttons(*names), dtype=np.int8)


def _u16(ram: Any, address: int) -> int:
    return int(ram[address]) | (int(ram[address + 1]) << 8)


def _hostiles(env: Any) -> tuple[tuple[int, int, int, int], ...]:
    """Every live Zeb, Sidehopper and Reo as ``(species, x, y, hp)``."""
    ram = env.get_ram()
    out = []
    for slot in range(16):
        base = _ADDR_ENEMY0 + slot * 0x40
        species = _u16(ram, base)
        if species in _HOSTILE:
            out.append((species, _u16(ram, base + 2), _u16(ram, base + 6), _u16(ram, base + 0x14)))
    return tuple(out)


def _hostiles_of(env: Any, species: int) -> list[tuple[int, int, int, int]]:
    return [row for row in _hostiles(env) if row[0] == species]


def _drops(env: Any) -> list[tuple[str, int, int]]:
    """Live enemy drops as ``(kind, x, y)``; kind is read off the pickup loop."""
    ram = env.get_ram()
    out = []
    for slot in range(18):
        if _u16(ram, _ADDR_PROJ_ID + 2 * slot) != _DROP_HEADER:
            continue
        loop = _u16(ram, _ADDR_PROJ_LIST + 2 * slot)
        kind = ("small" if 0xED8D <= loop < 0xEDA3 else "big" if 0xEDA3 <= loop < 0xEDB9
                else "missile" if 0xEDB9 <= loop < 0xEDCF else "other")
        out.append((kind, _u16(ram, _ADDR_PROJ_X + 2 * slot), _u16(ram, _ADDR_PROJ_Y + 2 * slot)))
    return out


def _human_climb() -> tuple[np.ndarray, np.ndarray]:
    """Board row 41's human buttons and where the human stood after each.

    Shots and item select are removed: the human fired and cycled weapons for
    reasons that no longer hold (refills on), and neither changes where Samus
    walks.
    """
    from super_metroid.human_tape.hops import load_task_json
    from super_metroid.paths import GAME_DIR

    data = load_task_json(GAME_DIR / HUMAN_TAPE)
    first, last = HUMAN_SPAN
    body = np.asarray(data["frames"], dtype=np.int8)[first : last + 1].copy()
    body[:, _X] = 0
    body[:, _SELECT] = 0
    where = np.asarray([(row["x"], row["y"]) for row in data["trace"][first : last + 1]], dtype=np.int32)
    return body, where


class _Lab:
    """Search on the live env; every try restores, so only chosen buttons commit."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.restores = 0
        self.actions: list[np.ndarray] = []

    def state(self) -> Any:
        return parse_env_state(self.env, mode="nav")

    def health(self) -> int:
        return _u16(self.env.get_ram(), 0x09C2)

    def blob(self) -> bytes:
        return self.env.em.get_state()

    def restore(self, blob: bytes) -> None:
        self.env.em.set_state(blob)
        self.restores += 1

    def play(self, actions: Any, *, hit_free: bool = True) -> bool:
        """Step actions; return False (stopping) on the first energy loss."""
        health = self.health()
        for action in actions:
            self.env.step(action)
            if hit_free and self.health() < health:
                return False
        return True

    def commit(self, actions: Any) -> None:
        for action in actions:
            self.env.step(action)
            self.actions.append(np.asarray(action, dtype=np.int8))


def _seat(lab: _Lab, human: np.ndarray) -> None:
    """Reach the ledge seat from the door: idle k frames, then the human climb.

    The idle count is searched because the door-exit walk and the master's
    last pose leave different settling times.
    """
    base = lab.blob()
    climb = list(human[CLIMB_START:LEDGE_FRAME]) + [_IDLE] * 17
    for k in range(0, CLIMB_START + 1):
        lab.restore(base)
        actions = [_IDLE] * k + climb
        if lab.play(actions):
            state = lab.state()
            if state.room_id == ROOM_BIG_PINK and state.samus_y < 560 and state.samus_x > 600:
                lab.restore(base)
                lab.commit(actions)
                return
    lab.restore(base)
    raise RuntimeError("Big Pink: no idle prefix reaches the Zeb ledge")


def _zeb_missile(lab: _Lab) -> None:
    """Shoot the pipe Zeb at a delay whose drop the ROM rolls as a missile."""
    base = lab.blob()
    zebs = len(_hostiles_of(lab.env, _ZEB))
    for delay in range(20):
        lab.restore(base)
        pattern = [_act("X") if f >= delay and (f - delay) % 6 == 0 else _IDLE for f in range(110)]
        killed = None
        for frame, action in enumerate(pattern):
            lab.env.step(action)
            if killed is None and len(_hostiles_of(lab.env, _ZEB)) < zebs:
                killed = frame
        settled = _drops(lab.env)
        if killed is not None and settled and settled[0][0] == "missile":
            lab.restore(base)
            lab.commit(pattern[: killed + 1])
            return
    lab.restore(base)
    raise RuntimeError("Big Pink: no Zeb shot delay rolls a missile")


def _collect_missile(lab: _Lab) -> None:
    before = lab.state().missiles
    for _ in range(150):
        state = lab.state()
        if state.missiles > before:
            return
        target = [x for kind, x, _ in _drops(lab.env) if kind == "missile"]
        if not target or abs(state.samus_x - target[0]) <= 4:
            lab.commit([_IDLE])
        else:
            lab.commit([_act("RIGHT" if state.samus_x < target[0] else "LEFT")])
    raise RuntimeError("Big Pink: dropped missile was not collected")


def _reload_room(lab: _Lab) -> None:
    """Fall to the floor, leave by the west door and come back (re-arms the pipe)."""
    for _ in range(400):
        state = lab.state()
        if state.samus_y >= 651 and state.samus_x <= 560:
            break
        lab.commit([_act("LEFT")])
    for target, direction in ((ROOM_BIG_PINK, "LEFT"), (None, "RIGHT")):
        for i in range(600):
            lab.commit([_act(direction, "X") if i % 8 < 2 else _act(direction)])
            state = lab.state()
            moved = state.room_id != ROOM_BIG_PINK if target else state.room_id == ROOM_BIG_PINK
            if moved and state.game_state == 8 and not state.door_transition:
                break
        else:
            raise RuntimeError(f"Big Pink: door {direction} never settled")
        lab.commit([_IDLE] * 10)


def _hostile_near(lab: _Lab) -> bool:
    state = lab.state()
    return any(abs(x - state.samus_x) < 260 and abs(y - state.samus_y) < 200
               for _, x, y, _ in _hostiles(lab.env))


_BRANCHES = {
    "idle": [_IDLE], "LEFT": [_act("LEFT")], "RIGHT": [_act("RIGHT")],
    "LEFT+B": [_act("LEFT", "B")], "RIGHT+B": [_act("RIGHT", "B")],
    "LEFT+A": [_act("LEFT", "A")], "RIGHT+A": [_act("RIGHT", "A")], "A": [_act("A")],
    "fire": [_act("X"), _IDLE, _IDLE, _IDLE],
    "fire+LEFT": [_act("LEFT", "X"), _act("LEFT"), _act("LEFT"), _act("LEFT")],
    "fire+RIGHT": [_act("RIGHT", "X"), _act("RIGHT"), _act("RIGHT"), _act("RIGHT")],
    "fire+LEFT+UP": [_act("LEFT", "UP", "X")] + [_act("LEFT", "UP")] * 3,
    "fire+RIGHT+UP": [_act("RIGHT", "UP", "X")] + [_act("RIGHT", "UP")] * 3,
    "fire+jump": [_act("A", "X")] + [_act("A")] * 3,
    "back": [_act("RIGHT", "B")] * 4, "back+jump": [_act("RIGHT", "A", "B")] * 4,
}


_FAN: dict[str, Any] = {}


def _init_climb(inherited_env: Any, human: np.ndarray, where: np.ndarray) -> None:
    """A fork gets its own emulator; close its inherited copy first."""
    from super_metroid.dev.common import make_dev_env

    inherited_env.close()
    env = make_dev_env()
    env.reset()
    _FAN.update(lab=_Lab(env), human=human, where=where)


def _fan_node(task: tuple[bytes, int, int, int]) -> tuple[list[Any], int]:
    """Children of one beam state: follow the human, and near hostiles branch."""
    blob, j, spent_from, chunk = task
    lab, human, where = _FAN["lab"], _FAN["human"], _FAN["where"]
    start_restores = lab.restores
    lab.restore(blob)
    options = [("follow", [human[min(j + i, len(human) - 1)] for i in range(chunk)], j + chunk)]
    if _hostile_near(lab):
        options += [(name, acts, j) for name, acts in _BRANCHES.items()]
    kids = []
    for _, acts, next_j in options:
        lab.restore(blob)
        if not lab.play(acts):
            continue
        state = lab.state()
        hostiles = _hostiles(lab.env)
        hoppers = [hp for species, _, _, hp in hostiles if species == _SIDEHOPPER]
        # Lagging the human by a running start is free; being somewhere else is not.
        off = abs(state.samus_x - int(where[next_j - 1][0])) + abs(state.samus_y - int(where[next_j - 1][1]))
        score = (2 * next_j - min(400, max(0, off - 48)) - 25 * (spent_from - state.missiles)
                 + 150 * (4 - len(hoppers)) - 0.5 * sum(hoppers))
        key = (state.samus_x, state.samus_y, state.pose, state.velocity_x, state.velocity_y, next_j, hostiles)
        kids.append((score, lab.blob(), next_j, list(acts), (state.samus_x, state.samus_y, state.health, state.missiles), key))
    return kids, lab.restores - start_restores


def _follow_human(lab: _Lab, human: np.ndarray, where: np.ndarray, *, width: int = 16, workers: int = 16,
                  progress: Callable[[dict[str, Any]], None] | None = None) -> None:
    """Beam over the human climb; branch only near hostiles, never lose energy.

    Fork workers expand beam states in parallel; every child is a ROM try from
    its parent's exact frame.
    """
    import multiprocessing

    chunk = 4
    origin = lab.blob()
    spent_from = lab.state().missiles
    beam = [(0.0, origin, LEDGE_FRAME, [])]
    try:
        with multiprocessing.get_context("fork").Pool(
            workers, initializer=_init_climb, initargs=(lab.env, human, where)
        ) as pool:
            for _ in range(600):
                rows = pool.map(_fan_node, [(blob, j, spent_from, chunk) for _, blob, j, _ in beam])
                lab.restores += sum(row[1] for row in rows)
                kids, seen = [], set()
                for (_, _, _, path), (children, _) in zip(beam, rows):
                    for score, blob, next_j, acts, summary, key in children:
                        if key not in seen:
                            seen.add(key)
                            kids.append((score, blob, next_j, path + acts, summary))
                if not kids:
                    raise RuntimeError("Big Pink: every climb branch loses energy")
                kids.sort(key=lambda kid: -kid[0])
                # Niches: states that wait or back off are worth keeping even though
                # the dodging front scores higher, or the beam locks into a dodge loop.
                beam, cells = [], {}
                for kid in kids:
                    cell = (kid[2] // 16, kid[4][0] // 24, kid[4][1] // 32)
                    if cells.get(cell, 0) < 2:
                        cells[cell] = cells.get(cell, 0) + 1
                        beam.append(kid[:4])
                    if len(beam) == width:
                        break
                best = kids[0]
                if progress is not None:
                    progress({"j": best[2], "score": round(best[0]), "x": best[4][0], "y": best[4][1],
                              "health": best[4][2], "missiles": best[4][3], "beam": len(beam), "blob": best[1]})
                if best[4][0] >= DOOR_X - 50 and best[4][1] <= 150:
                    lab.restore(origin)
                    lab.commit(best[3])
                    if lab.blob() != best[1]:
                        raise RuntimeError("Big Pink: climb buttons did not replay exactly")
                    return
            raise RuntimeError("Big Pink: climb search ran out of steps")
    except BaseException:
        lab.restore(origin)
        raise


def _open_door_and_leave(lab: _Lab) -> None:
    """Five missiles at the pink door, then walk through and settle."""
    before = lab.state().missiles
    if before < DOOR_MISSILES:
        raise RuntimeError(f"Big Pink: {before} missiles at the pink door, need {DOOR_MISSILES}")
    for i in range(800):
        state = lab.state()
        if state.room_id == ROOM_KIHUNTER and state.game_state == 8 and not state.door_transition:
            return
        firing = state.room_id == ROOM_BIG_PINK and state.samus_x >= DOOR_X - 7 and before - state.missiles < DOOR_MISSILES
        lab.commit([_act("RIGHT", "X") if firing and i % 8 == 0 else _act("RIGHT")])
    raise RuntimeError("Big Pink: never settled in the Kihunter room")


def _select_missiles(lab: _Lab) -> None:
    if lab.state().selected_item != 1:
        lab.commit([_act("SELECT")] + [_IDLE] * 20)


def play_clean_big_pink_leave(
    session: ControllerSession, *, progress: Callable[[dict[str, Any]], None] | None = None,
) -> SimpleNamespace:
    """Farm missiles, climb past the Sidehoppers, open the pink door, enter 0x9D9C.

    Searches on ``session.env``, restores the entry frame, then commits only the
    chosen buttons through ``session.step``. A failure commits nothing.
    """
    env = getattr(session, "env", None)
    if env is None or session.state.room_id != ROOM_BIG_PINK:
        raise ValueError("Big Pink leave requires a live session in 0x9D19")
    entry = env.em.get_state()
    lab = _Lab(env)
    try:
        human, where = _human_climb()
        origin = lab.state()
        drops = 0
        while True:
            _seat(lab, human)
            _zeb_missile(lab)
            _collect_missile(lab)
            drops += 1
            if lab.state().missiles >= FARM_TARGET:
                break
            _reload_room(lab)
        _select_missiles(lab)
        _follow_human(lab, human, where, progress=progress)
        _open_door_and_leave(lab)
        actions, restores, final = lab.actions, lab.restores, lab.state()
        if final.health < origin.health:
            raise RuntimeError("Big Pink leave lost energy")
    finally:
        env.em.set_state(entry)
    for action in actions:
        session.step(action, "big_pink_leave")
    return SimpleNamespace(
        fight=SimpleNamespace(rollout_restores=restores),
        report={"drops": drops, "committed_frames": len(actions),
                "missiles": session.state.missiles, "health": session.state.health},
    )
