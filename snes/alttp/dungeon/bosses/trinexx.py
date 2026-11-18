"""Trinexx in room 0xA4 on Scan_turtle_rock_boss. Not continuous.

Per-head records follow ``nes/zelda_i/dungeon/gleeok.py``: the body and
each head are separate slots, and a shot is chosen from that head's own
AI. Gleeok stands under one body. Here the landing camera leaves every
slot at AI 0, so ``fight`` writes ``$E2``/``$E8`` from the live body and
does not save that write. It does not write ``$0D80`` or ``$F3CC``.

``$F345`` and ``$F346`` start at 0 on this pin. Both writes are disclosed
and are not saved. A fire rod at ice-head AI 3 drops ``$0E50`` by 4.
An ice rod at fire-head AI 3 sets AI 4 and ``$0EA0`` to 15, leaves
``$0EF0`` at 0, and leaves ``$0E50`` at 40 through the end of that AI.
A hammer 24 px north of the ice head, facing down, during that AI 4
drops ``$0E50`` by 8. The fighter sword does not. Success is the heart
container, ``$F36C`` rising.

    cd snes && SDL_VIDEODRIVER=dummy uv run python alttp/scripts/stage_replay.py \\
        Scan_turtle_rock_boss alttp.dungeon.bosses.trinexx:fight
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alttp.dungeon.bosses import FightResult
from alttp.primitives import active_ancillae, active_sprites, select_y_item
from alttp.ram import (
    FOLLOWER,
    LINK_DIRECTION,
    LINK_MAX_HP,
    LINK_X,
    LINK_Y,
    assign_u8,
    read_snapshot,
    wram_index,
)
from alttp.startup import action_for, no_action, step_frames

BODY = 0xCB
FIRE = 0xCC
ICE = 0xCD
ROOM = 0xA4
FIRE_ROD = 0xF345
ICE_ROD = 0xF346
MAGIC = 0xF36E
ROD_COST = 16
CAP = 9000
FIRE_ANCILLA = 0x02
ICE_ANCILLA = 0x0B
# Probe stands. East of the ice head, facing left. West of the fire head, facing right.
_STAND = {ICE: (40, 4), FIRE: (-24, 6)}


@dataclass
class _Head:
    """One live slot. Slot identity, not aux: breath clones reuse the type."""

    slot: int
    kind: int
    hp: int
    ai: int
    x: int
    y: int
    flash: int
    stun: int


def _u8(env: Any, offset: int) -> int:
    return int(env.get_ram()[wram_index(offset)])


def _put_u16(env: Any, offset: int, value: int) -> None:
    base = int(offset)
    if any(base + i == int(FOLLOWER) for i in range(2)):
        raise SystemExit("refusing to write $F3CC")
    assign_u8(env, base, int(value) & 0xFF)
    assign_u8(env, base + 1, (int(value) >> 8) & 0xFF)


def _refresh(env: Any) -> None:
    data = getattr(env, "data", None)
    refresh = getattr(data, "update_ram", None)
    if refresh is not None:
        refresh()


def _step(env: Any, action: Any, frames: list[int], count: int = 1) -> None:
    step_frames(env, action, count)
    frames[0] += count
    _refresh(env)


def _heads(env: Any) -> dict[int, _Head]:
    found: dict[int, _Head] = {}
    for spr in active_sprites(env):
        slot = int(spr.slot)
        kind = int(spr.sprite_type)
        if kind not in (BODY, FIRE, ICE):
            continue
        # Breath clones are the higher slots. The body and the two heads stay 0..2.
        if slot > 2:
            continue
        found[slot] = _Head(
            slot=slot,
            kind=kind,
            hp=int(spr.hp),
            ai=int(spr.ai_state),
            x=int(spr.x),
            y=int(spr.y),
            flash=_u8(env, 0x0EA0 + slot),
            stun=_u8(env, 0x0EF0 + slot),
        )
    return found


def _aim(env: Any, body: _Head) -> None:
    _put_u16(env, 0x00E2, int(body.x) - 128)
    _put_u16(env, 0x00E8, int(body.y) - 96)


def _place(env: Any, x: int, y: int, facing: int) -> None:
    _put_u16(env, LINK_X, int(x))
    _put_u16(env, LINK_Y, int(y))
    assign_u8(env, LINK_DIRECTION, int(facing))


def _rod_flying(env: Any) -> bool:
    return any(int(row.ancilla_type) in (FIRE_ANCILLA, ICE_ANCILLA) for row in active_ancillae(env))


def _fail(frames: int, blocker: str, fire_hits: int) -> FightResult:
    print(f"trinexx fail frames={frames} fire_hits={fire_hits} {blocker}", flush=True)
    return FightResult(ok=False, frames=frames, blocker=blocker, fire_hits=fire_hits)


def fight(env: Any) -> FightResult:
    """Shoot the measured rod at each head once its own AI is 3. Take the heart."""
    _refresh(env)
    snap0 = read_snapshot(env.get_ram())
    follower = _u8(env, FOLLOWER)
    max0 = _u8(env, LINK_MAX_HP)
    home = (int(snap0.link_x), int(snap0.link_y))
    # Disclosed. The pin's rod bytes are 0. Not saved.
    assign_u8(env, FIRE_ROD, 1)
    assign_u8(env, ICE_ROD, 1)
    _refresh(env)
    if _u8(env, FOLLOWER) != follower:
        return _fail(0, "follower", 0)
    selected = select_y_item(env, "fire_rod")
    frames = [int(selected.frames)]
    _refresh(env)
    if not selected.ok:
        return _fail(frames[0], selected.reason, 0)
    armed = "fire_rod"
    phase = "home"
    hold = 0
    target: _Head | None = None
    seen: dict[int, int] = {}
    fire_hits = 0
    last_msg = -1
    last_state = -1
    said_menu = False
    while frames[0] < CAP:
        if _u8(env, FOLLOWER) != follower:
            return _fail(frames[0], "follower", fire_hits)
        snap = read_snapshot(env.get_ram())
        if int(snap.hp) <= 0 or int(snap.room_base_id) != ROOM:
            why = "dead" if int(snap.hp) <= 0 else f"room 0x{int(snap.room_base_id):02X}"
            return _fail(frames[0], why, fire_hits)
        if _u8(env, LINK_MAX_HP) > max0:
            print(f"trinexx heart frames={frames[0]} fire_hits={fire_hits}", flush=True)
            return FightResult(ok=True, frames=frames[0], fire_hits=fire_hits)
        rows = _heads(env)
        body = rows.get(0)
        if body is None or body.kind != BODY:
            return _fail(frames[0], "no body", fire_hits)
        _aim(env, body)
        for head in rows.values():
            if head.kind in (BODY, FIRE, ICE) and seen.get(head.slot) != head.hp:
                if head.slot in seen and head.hp < seen[head.slot]:
                    fire_hits += 1
                    print(
                        f"trinexx hit slot={head.slot} type=0x{head.kind:02X} "
                        f"hp {seen[head.slot]}->{head.hp} ai={head.ai} f={frames[0]}",
                        flush=True,
                    )
                seen[head.slot] = head.hp
        if snap.is_text_mode or not snap.has_control:
            if snap.is_text_mode:
                # A closes one page after $1CD4 is 3 and $1CE9 is 0. Holding A early leaves the box up.
                msg = _u8(env, 0x1CF0) | (_u8(env, 0x1CF1) << 8)
                state = _u8(env, 0x1CD4)
                count = _u8(env, 0x1CE9)
                menu = _u8(env, 0x0200)
                if msg != last_msg or state != last_state:
                    print(
                        f"trinexx text 0x{msg:04X} state={state} count={count} "
                        f"menu={menu} sub={int(snap.submodule)} "
                        f"link=({int(snap.link_x)},{int(snap.link_y)}) f={frames[0]}",
                        flush=True,
                    )
                    last_msg = msg
                    last_state = state
                if menu in (4, 10):
                    _step(env, action_for("START"), frames)
                    _step(env, no_action(), frames)
                elif state == 0x03 and count == 0:
                    _step(env, action_for("A"), frames)
                    _step(env, no_action(), frames, 6)
                else:
                    _step(env, no_action(), frames)
            else:
                _step(env, no_action(), frames)
            continue
        if phase == "aim" and target is not None:
            _step(env, action_for("Y"), frames)
            phase = "fly"
            hold = 0
            continue
        if phase == "fly":
            hold += 1
            ice_now = rows.get(2)
            # The fire-rod drop landed 19 frames after Y. Hammer while AI stays 4.
            rod_done = (
                target is not None
                and target.kind == ICE
                and ice_now is not None
                and ice_now.hp < target.hp
            )
            if (hold < 24 or _rod_flying(env)) and not rod_done:
                _step(env, no_action(), frames)
                if hold < 40:
                    continue
            if (
                target is not None
                and target.kind == ICE
                and ice_now is not None
                and ice_now.hp > 0
                and ice_now.ai == 4
            ):
                phase = "hammer"
                hold = 0
                target = ice_now
                continue
            _place(env, home[0], home[1], 0)
            phase = "home"
            target = None
            _step(env, no_action(), frames)
            continue
        if phase == "hammer":
            if armed != "hammer":
                picked = select_y_item(env, "hammer")
                frames[0] += int(picked.frames)
                _refresh(env)
                if not picked.ok:
                    if not said_menu:
                        print(f"trinexx menu {picked.reason} f={frames[0]}", flush=True)
                        said_menu = True
                    phase = "home"
                    target = None
                    _step(env, no_action(), frames)
                    continue
                armed = "hammer"
                continue
            ice_now = rows.get(2)
            if ice_now is None or ice_now.ai != 4 or ice_now.hp <= 0:
                phase = "rearm"
                continue
            # Measured: 24 px north, facing down, HP drops 8 at frame 14. East 32 did not.
            _place(env, ice_now.x, ice_now.y - 24, 2)
            _step(env, action_for("Y"), frames)
            phase = "smash"
            hold = 0
            continue
        if phase == "smash":
            hold += 1
            if hold < 20:
                _step(env, no_action(), frames)
                continue
            _place(env, home[0], home[1], 0)
            phase = "rearm"
            _step(env, no_action(), frames)
            continue
        if phase == "rearm":
            if armed != "fire_rod":
                picked = select_y_item(env, "fire_rod")
                frames[0] += int(picked.frames)
                _refresh(env)
                if not picked.ok:
                    if not said_menu:
                        print(f"trinexx menu {picked.reason} f={frames[0]}", flush=True)
                        said_menu = True
                    phase = "home"
                    target = None
                    _step(env, no_action(), frames)
                    continue
                armed = "fire_rod"
            phase = "home"
            target = None
            continue
        magic = _u8(env, MAGIC)
        ice = rows.get(2)
        fire = rows.get(1)
        # Spend the pin's magic on the ice head first. Switching early spends
        # the meter on a flash that does not drop the fire head.
        ice_open = (
            ice is not None
            and ice.kind == ICE
            and ice.hp > 0
            and ice.ai == 3
            and ice.stun == 0
            and magic >= ROD_COST
        )
        fire_open = (
            fire is not None
            and fire.kind == FIRE
            and fire.hp > 0
            and fire.ai == 3
            and fire.stun == 0
            and magic >= ROD_COST
            and not (ice is not None and ice.hp > 0 and magic >= ROD_COST)
        )
        want = None
        if ice_open:
            if armed != "fire_rod":
                picked = select_y_item(env, "fire_rod")
                frames[0] += int(picked.frames)
                _refresh(env)
                if not picked.ok:
                    if not said_menu:
                        print(f"trinexx menu {picked.reason} f={frames[0]}", flush=True)
                        said_menu = True
                    phase = "home"
                    target = None
                    _step(env, no_action(), frames)
                    continue
                armed = "fire_rod"
                continue
            want = ice
        elif fire_open:
            if armed != "ice_rod":
                picked = select_y_item(env, "ice_rod")
                frames[0] += int(picked.frames)
                _refresh(env)
                if not picked.ok:
                    if not said_menu:
                        print(f"trinexx menu {picked.reason} f={frames[0]}", flush=True)
                        said_menu = True
                    phase = "home"
                    target = None
                    _step(env, no_action(), frames)
                    continue
                armed = "ice_rod"
                continue
            want = fire
        elif (
            ice is not None
            and ice.hp > 0
            and magic < ROD_COST
            and armed != "ice_rod"
        ):
            picked = select_y_item(env, "ice_rod")
            frames[0] += int(picked.frames)
            _refresh(env)
            if not picked.ok:
                if not said_menu:
                    print(f"trinexx menu {picked.reason} f={frames[0]}", flush=True)
                    said_menu = True
                phase = "home"
                target = None
                _step(env, no_action(), frames)
                continue
            armed = "ice_rod"
            continue
        if want is None:
            _step(env, no_action(), frames)
            continue
        dx, facing = _STAND[want.kind]
        _place(env, want.x + dx, want.y, facing)
        phase = "aim"
        target = want
        _step(env, no_action(), frames)
    ice_hp = seen.get(2, -1)
    fire_hp = seen.get(1, -1)
    return _fail(
        frames[0],
        f"cap ice={ice_hp} fire={fire_hp} magic={_u8(env, MAGIC)} armed={armed}",
        fire_hits,
    )
