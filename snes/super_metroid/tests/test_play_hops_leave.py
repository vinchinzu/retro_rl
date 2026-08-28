"""play_hops leave glance: leftover on LeaveMiss. No emulator."""

from __future__ import annotations

import pytest

from super_metroid.hop_glance import LeaveMiss, final_from_state
from super_metroid.leave_specs import WS_BASEMENT_TO_PHANTOON, WS_ENTRANCE_TO_MAIN
from super_metroid.routes.kpdr.post_ice_spine import POST_ICE_SPINE
from super_metroid.routes.kpdr.room_ids import (
    ROOM_PHANTOON,
    ROOM_WS_BASEMENT,
    ROOM_WS_ENTRANCE,
    ROOM_WS_MAIN,
)
from super_metroid.routes.kpdr.spine_types import SpineHop
from super_metroid.routes.tips import play_hops


class _State:
    def __init__(
        self,
        *,
        room_id: int,
        x: int,
        y: int,
        pose: int,
        game_state: int = 8,
        door_transition: int = 0,
        health: int = 299,
    ) -> None:
        self.room_id = room_id
        self.samus_x = x
        self.samus_y = y
        self.pose = pose
        self.game_state = game_state
        self.door_transition = door_transition
        self.health = health


class _Session:
    def __init__(self, state: _State) -> None:
        self.state = state
        self.frame = 0
        self.transitions: list[object] = []


def _hop(*, play, leave=None, to_room: int = ROOM_WS_MAIN) -> SpineHop:
    return SpineHop(
        "ws_entrance_to_main",
        play,
        ROOM_WS_ENTRANCE,
        to_room,
        "Wrecked Ship Main Shaft",
        "phantoon",
        use_transition_split=False,
        leave=leave,
    )


def test_play_hops_wrong_room_raises_leave_miss_with_leftover() -> None:
    session = _Session(_State(room_id=ROOM_WS_ENTRANCE, x=57, y=139, pose=1))

    def _land_wrong(sess: _Session) -> None:
        sess.state = _State(room_id=0xCD13, x=39, y=128, pose=1)
        sess.frame = 12

    with pytest.raises(LeaveMiss) as caught:
        play_hops(session, [], [_hop(play=_land_wrong, leave=WS_ENTRANCE_TO_MAIN)])
    err = caught.value
    assert err.hop_id == "ws_entrance_to_main"
    assert err.leftover["xy"] == [39, 128]
    assert err.leftover["pose"] == 1
    assert err.leftover["gs"] == 8
    assert err.leftover["dt"] == 0
    assert "health" in err.leftover
    assert "room" in err.leftover
    assert "expected Wrecked Ship Main Shaft 0xCAF6, got 0xCD13" in str(err)
    assert err.leftover == final_from_state(session.state)


def test_play_hops_wrong_pose_raises_leave_miss_not_room_only() -> None:
    session = _Session(_State(room_id=ROOM_WS_ENTRANCE, x=57, y=139, pose=1))

    def _land_morph(sess: _Session) -> None:
        sess.state = _State(room_id=ROOM_WS_MAIN, x=1063, y=907, pose=29)

    with pytest.raises(LeaveMiss) as caught:
        play_hops(session, [], [_hop(play=_land_morph, leave=WS_ENTRANCE_TO_MAIN)])
    err = caught.value
    assert err.leftover["xy"] == [1063, 907]
    assert err.leftover["pose"] == 29
    assert any("not stand" in m for m in err.misses)
    assert "0xCAF6" in err.leftover["room"]


def test_play_hops_no_leave_spec_still_emits_leftover_on_wrong_room() -> None:
    session = _Session(_State(room_id=ROOM_WS_ENTRANCE, x=57, y=139, pose=1))

    def _land_wrong(sess: _Session) -> None:
        sess.state = _State(room_id=0xCA08, x=960, y=139, pose=9)

    hop = _hop(play=_land_wrong, leave=None, to_room=ROOM_WS_MAIN)
    assert hop.leave is None
    with pytest.raises(LeaveMiss) as caught:
        play_hops(session, [], [hop])
    err = caught.value
    assert isinstance(err, RuntimeError)
    assert "expected Wrecked Ship Main Shaft 0xCAF6, got 0xCA08" in str(err)
    assert err.leftover["xy"] == [960, 139]
    assert err.leftover["pose"] == 9
    assert err.leftover["gs"] == 8


def test_play_hops_pass_when_leave_glances() -> None:
    session = _Session(_State(room_id=ROOM_WS_ENTRANCE, x=57, y=139, pose=1))

    def _land_ok(sess: _Session) -> str:
        sess.state = _State(room_id=ROOM_WS_MAIN, x=1063, y=907, pose=9)
        return "ok"

    assert play_hops(session, [], [_hop(play=_land_ok, leave=WS_ENTRANCE_TO_MAIN)]) == "ok"


def test_play_hops_morph_in_basement_exit_raises_leave_miss() -> None:
    """Already-green basement→Phantoon hop: morph in the door is a glance miss."""
    spine = next(h for h in POST_ICE_SPINE if h.hop_id == "ws_basement_to_phantoon")
    assert spine.leave is WS_BASEMENT_TO_PHANTOON
    session = _Session(_State(room_id=ROOM_WS_BASEMENT, x=657, y=92, pose=24))

    def _land_morph(sess: _Session) -> None:
        sess.state = _State(room_id=ROOM_PHANTOON, x=39, y=124, pose=29)

    hop = SpineHop(
        spine.hop_id,
        _land_morph,
        spine.from_room,
        spine.to_room,
        spine.room_label,
        spine.tip_id,
        use_transition_split=False,
        leave=spine.leave,
    )
    with pytest.raises(LeaveMiss) as caught:
        play_hops(session, [], [hop])
    err = caught.value
    assert err.hop_id == "ws_basement_to_phantoon"
    assert err.leftover["xy"] == [39, 124]
    assert err.leftover["pose"] == 29
    assert err.leftover["gs"] == 8
    assert err.leftover == final_from_state(session.state)
    assert any("not door" in m for m in err.misses)


def test_play_hops_basement_exit_spin_glances() -> None:
    spine = next(h for h in POST_ICE_SPINE if h.hop_id == "ws_basement_to_phantoon")
    session = _Session(_State(room_id=ROOM_WS_BASEMENT, x=657, y=92, pose=24))

    def _land_spin(sess: _Session) -> str:
        sess.state = _State(room_id=ROOM_PHANTOON, x=39, y=124, pose=81)
        return "ok"

    hop = SpineHop(
        spine.hop_id,
        _land_spin,
        spine.from_room,
        spine.to_room,
        spine.room_label,
        spine.tip_id,
        use_transition_split=False,
        leave=spine.leave,
    )
    assert play_hops(session, [], [hop]) == "ok"
