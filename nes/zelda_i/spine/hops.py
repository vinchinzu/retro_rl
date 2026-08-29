"""One hop catalog + attach loop for the Survival spine.

A hop is a row: through-id, stop name, stages, success. ``attach_hops``
runs the row list and stops on fail or a ``through`` match. Per-level
``continue_*`` and ``*_stages``/``*_success`` pairs are rows, not functions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from zelda_i.dungeon.engine import DungeonRoomSpec, GenericDungeonRoomController
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot, read_snapshot

Stages = Sequence[tuple[str, Any, int]]
StageFn = Callable[[], Stages]
SuccessFn = Callable[..., bool]
Hook = Callable[..., None]


@dataclass(frozen=True)
class SpineHop:
    """One Survival stop. ``success(snap)`` or ``success(snap, keys_before=)``."""

    through: str
    stop: str
    stages: StageFn | Stages
    success: SuccessFn
    dedicated: bool = False
    capture_keys: bool = False
    before: Hook | None = None
    after: Hook | None = None


def play_ready(
    snap: ZeldaSnapshot,
    *,
    level: int,
    screen: int | None = None,
    not_screen: int | None = None,
    mode: int = PLAY_MODE,
    allow_transition: bool = False,
    tf_bit: int | None = None,
    tf_eq: int | None = None,
    spec: DungeonRoomSpec | None = None,
    rod: bool = False,
    compass_bit: int | None = None,
    map_bit: int | None = None,
    item: str | None = None,
    item_min: int = 1,
    keys_before: int | None = None,
    keys_cmp: str | None = None,
) -> bool:
    """Shared dungeon enter/clear stop. Replaces copy-pasted ``*_success``."""
    if snap.level != level or snap.mode != mode:
        return False
    if not allow_transition and snap.transitioning:
        return False
    if screen is not None and snap.screen != screen:
        return False
    if not_screen is not None and snap.screen == not_screen:
        return False
    if tf_bit is not None and not (snap.triforce & tf_bit):
        return False
    if tf_eq is not None and snap.triforce != tf_eq:
        return False
    if spec is not None and spec.live_enemies(snap):
        return False
    if rod and not snap.rod:
        return False
    if compass_bit is not None and (snap.compass & compass_bit) == 0:
        return False
    if map_bit is not None and (snap.map & map_bit) == 0:
        return False
    if item is not None and int(getattr(snap, item, 0)) < item_min:
        return False
    if keys_cmp == "gt" and keys_before is not None and not (snap.keys > keys_before):
        return False
    if keys_cmp == "lt" and keys_before is not None and not (snap.keys < keys_before):
        return False
    return True


def ready(
    *,
    level: int,
    screen: int | None = None,
    not_screen: int | None = None,
    mode: int = PLAY_MODE,
    allow_transition: bool = False,
    tf_bit: int | None = None,
    tf_eq: int | None = None,
    spec: DungeonRoomSpec | None = None,
    rod: bool = False,
    compass_bit: int | None = None,
    map_bit: int | None = None,
    item: str | None = None,
    item_min: int = 1,
    keys_cmp: str | None = None,
) -> SuccessFn:
    """Bind ``play_ready`` as a hop success callable."""

    def ok(snap: ZeldaSnapshot, *, keys_before: int | None = None) -> bool:
        return play_ready(
            snap,
            level=level,
            screen=screen,
            not_screen=not_screen,
            mode=mode,
            allow_transition=allow_transition,
            tf_bit=tf_bit,
            tf_eq=tf_eq,
            spec=spec,
            rod=rod,
            compass_bit=compass_bit,
            map_bit=map_bit,
            item=item,
            item_min=item_min,
            keys_before=keys_before,
            keys_cmp=keys_cmp,
        )

    return ok


def fight_stage(name: str, spec: DungeonRoomSpec) -> tuple[str, Any, int]:
    return (name, GenericDungeonRoomController(spec=spec), spec.max_frames)


def _stages(hop: SpineHop) -> Stages:
    return hop.stages() if callable(hop.stages) else hop.stages


def attach_hops(
    env,
    run,
    hops: Sequence[SpineHop],
    *,
    through: str,
    run_stages,
    **hop_kw,
) -> None:
    """Run hops in order. Stops on stage fail, success miss, or through-match."""
    for hop in hops:
        if hop.dedicated and through != hop.through:
            continue
        if hop.before is not None:
            hop.before(env, run)
        keys_before = None
        if hop.capture_keys:
            keys_before = int(read_snapshot(env.get_ram()).keys)
        if not run_stages(env, run, _stages(hop), **hop_kw):
            return
        snap = read_snapshot(env.get_ram())
        if hop.capture_keys:
            run.success = bool(hop.success(snap, keys_before=keys_before))
        else:
            run.success = bool(hop.success(snap))
        if hop.after is not None:
            hop.after(env, run, snap)
        if not run.success:
            run.failed_stage = hop.stop
            return
        if through == hop.through:
            return
