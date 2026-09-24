"""Survival-spine L6 catalog + continue. Hop rows live in hops/suffix modules."""

from __future__ import annotations

from zelda_i.anchors import TF_BIT_L5
from zelda_i.level6.dungeon import ROOM_7A_SPEC
from zelda_i.level6.hops import l6_prefix
from zelda_i.level6.overworld import (
    LEVEL6,
    LEVEL6_EAST_KEY_ROOM,
    LEVEL6_ENTRY_ROOM,
)
from zelda_i.level6.spine_suffix import l6_suffix_hops
from zelda_i.ram import ZeldaSnapshot
from zelda_i.spine.hops import attach_hops, play_ready

__all__ = [
    "L6_STOPS",
    "L6_THROUGH",
    "continue_level6_spine",
    "level6_east_key_success",
    "level6_entry_success",
    "run_level6_from_entrance",
]


def level6_entry_success(snap: ZeldaSnapshot, *, whistle: int) -> bool:
    """Room-ready Dragon entry 0x79 with L5 inventory. Do not grant Whistle.

    Named form of the ``level6-entry`` hop predicate (``_entry_ok`` in
    ``level6.hops``). ``whistle`` is supplied by the caller because it is not a
    snapshot field.
    """
    return (
        play_ready(
            snap,
            level=LEVEL6,
            screen=LEVEL6_ENTRY_ROOM,
            tf_bit=TF_BIT_L5,
            item="raft",
        )
        and snap.ladder > 0
        and whistle >= 1
    )


def level6_east_key_success(snap: ZeldaSnapshot, *, keys_before: int) -> bool:
    """Cleared 0x7a with a natural key pickup. Do not UP to Old Man 0x6a.

    Named form of the ``level6-east-key`` hop predicate
    (``ok6(screen=LEVEL6_EAST_KEY_ROOM, spec=ROOM_7A_SPEC, keys_cmp="gt", ...)``).
    """
    return play_ready(
        snap,
        level=LEVEL6,
        screen=LEVEL6_EAST_KEY_ROOM,
        spec=ROOM_7A_SPEC,
        tf_bit=TF_BIT_L5,
        keys_before=keys_before,
        keys_cmp="gt",
    )


def _l6_rows():
    return l6_prefix(None) + l6_suffix_hops()


L6_THROUGH = tuple(hop.through for hop in _l6_rows())
L6_STOPS = {hop.through: hop.stop for hop in _l6_rows()}


def continue_level6_spine(
    env,
    run,
    *,
    through: str,
    run_stages,
    room_timer=None,
    assist=None,
    on_frame=None,
) -> None:
    """Attach L6 suffix after L5 TF. Mutates ``run``; caller returns it."""
    attach_hops(
        env,
        run,
        l6_prefix(env) + l6_suffix_hops(),
        through=through,
        run_stages=run_stages,
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    )


def run_level6_from_entrance(
    env,
    obs,
    *,
    assist=None,
    through: str = "level6",
    on_frame=None,
    room_timer=None,
    poke_arrows: bool = False,
):
    """Fixture-live L6 from play 0x79. Skip the OW entry hop. No pokes.

    ``poke_arrows`` defaults False; the continuous spine buys arrows at 0x4A.
    """
    from zelda_i.route.chain import run_controller_stage

    class _Run:
        def __init__(self):
            self.through = through
            self.success = True
            self.stages = []
            self.end_frame = 0
            self.failed_stage = None
            self.obs = obs
            self.allow_pokes = False

    def run_stages(env, run, stages, **kw):
        del kw
        for name, controller, max_frames in stages:
            next_obs, stage = run_controller_stage(
                env,
                run.obs,
                name=name,
                controller=controller,
                max_frames=max_frames,
                assist=assist,
                on_frame=on_frame,
                room_timer=room_timer,
                frame_base=run.end_frame,
            )
            run.obs = next_obs
            run.stages.append(stage)
            run.end_frame = stage.end_frame
            if not stage.success:
                run.success = False
                run.failed_stage = name
                return False
        return True

    run = _Run()
    interior = tuple(
        hop
        for hop in l6_prefix(env, require_prior_tf=False)
        + l6_suffix_hops(poke_arrows=poke_arrows, require_prior_tf=False)
        if hop.through != "level6-entry"
    )
    attach_hops(
        env,
        run,
        interior,
        through=through,
        run_stages=run_stages,
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    )
    return run
