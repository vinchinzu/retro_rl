"""Offline tests for multi-truth anchors and Segment registry."""

from __future__ import annotations

from alttp.opening_route.anchors import (
    STATE_SEMANTICS,
    TIP_ANCHOR_ORDER,
    anchor_by_id,
    match_anchors,
    opening_anchors,
    resolve_continuous_tip_node,
)
from alttp.opening_route.segment import get_segment, list_segments, segment_registry
from alttp.paths import (
    FIGHTER_SWORD_STATE,
    HYRULE_CASTLE_GROUNDS_STATE,
    STATE_SEMANTIC_NAMES,
)
from alttp.ram import (
    HYRULE_CASTLE_MAIN_HALL_ROOM,
    SANCTUARY_ROOM,
    SECRET_PASSAGE_ROOM,
    ZELDA_CELL_ROOM,
    AlttpSnapshot,
)


def _snap(**kwargs: object) -> AlttpSnapshot:
    base: dict[str, object] = dict(
        game_mode=0x07,
        submodule=0x00,
        room_id=0,
        indoors=False,
        screen_id=0x1B,
        link_x=2000,
        link_y=1600,
        link_direction=0,
        link_action=0,
        camera_x=0,
        camera_y=0,
        dark_world=False,
        sword_level=0,
        lamp_level=0,
        num_keys=0xFF,
        follower=0,
    )
    base.update(kwargs)
    return AlttpSnapshot(**base)  # type: ignore[arg-type]


def test_opening_anchors_nonempty_and_unique_ids() -> None:
    anchors = opening_anchors()
    assert len(anchors) >= 6
    ids = [a.anchor_id for a in anchors]
    assert len(ids) == len(set(ids))
    tiers = {a.tier for a in anchors}
    assert "route" in tiers
    assert "approach" in tiers
    assert "trigger" in tiers


def test_grounds_spawn_matches_controllable_screen() -> None:
    a = anchor_by_id("HyruleCastle_GroundsSpawn_Controllable")
    assert a is not None
    assert a.matches(_snap(screen_id=0x1B, indoors=False, game_mode=0x09))
    assert not a.matches(_snap(screen_id=0x2C, indoors=False))


def test_secret_hole_approach_position_window() -> None:
    a = anchor_by_id("HyruleCastle_SecretPassageApproach")
    assert a is not None
    near = _snap(
        screen_id=0x1B, indoors=False, link_x=2430, link_y=1704, game_mode=0x09
    )
    far = _snap(screen_id=0x1B, indoors=False, link_x=1000, link_y=1000, game_mode=0x09)
    assert a.matches(near)
    assert not a.matches(far)


def test_fighter_sword_anchor_requires_inventory() -> None:
    a = anchor_by_id("HyruleCastle_SecretEntrance_FighterSword")
    assert a is not None
    bare = _snap(indoors=True, room_id=SECRET_PASSAGE_ROOM, sword_level=0)
    armed = _snap(indoors=True, room_id=SECRET_PASSAGE_ROOM, sword_level=1)
    assert not a.matches(bare)
    assert a.matches(armed)


def test_courtyard_pocket_anchor() -> None:
    a = anchor_by_id("HyruleCastle_Courtyard_SecretStairsPocket")
    assert a is not None
    pocket = _snap(
        indoors=False,
        screen_id=0x1B,
        link_x=2248,
        link_y=1755,
        sword_level=1,
        game_mode=0x09,
    )
    assert a.matches(pocket)
    assert any(
        m.anchor_id == "HyruleCastle_Courtyard_SecretStairsPocket"
        for m in match_anchors(pocket)
    )


def test_main_door_approach_and_hall_anchors() -> None:
    approach = anchor_by_id("HyruleCastle_MainDoorApproach")
    assert approach is not None
    at_door = _snap(
        indoors=False,
        screen_id=0x1B,
        link_x=2040,
        link_y=1790,
        sword_level=1,
        game_mode=0x09,
    )
    assert approach.matches(at_door)
    hall = anchor_by_id("HyruleCastle_MainHall")
    assert hall is not None

    indoor = _snap(
        indoors=True,
        room_id=HYRULE_CASTLE_MAIN_HALL_ROOM,
        sword_level=1,
        game_mode=0x07,
    )
    assert hall.matches(indoor)
    assert "pocket_to_main_hall" in list_segments()


def test_zelda_and_sanctuary_anchors() -> None:
    zelda = anchor_by_id("HyruleCastle_ZeldaCell")
    assert zelda is not None
    assert zelda.matches(
        _snap(indoors=True, room_id=ZELDA_CELL_ROOM, sword_level=1, game_mode=0x07)
    )
    assert zelda.graph_node_id == "room_80"

    sanc = anchor_by_id("HyruleCastle_Sanctuary")
    assert sanc is not None
    assert sanc.matches(_snap(indoors=True, room_id=SANCTUARY_ROOM, game_mode=0x07))
    assert sanc.graph_node_id == "sanctuary"


def test_state_semantics_cover_key_dev_states() -> None:
    assert HYRULE_CASTLE_GROUNDS_STATE in STATE_SEMANTICS
    assert FIGHTER_SWORD_STATE in STATE_SEMANTICS
    assert "bridge" not in STATE_SEMANTICS[HYRULE_CASTLE_GROUNDS_STATE].lower() or (
        "NOT" in STATE_SEMANTICS[HYRULE_CASTLE_GROUNDS_STATE]
        or "not" in STATE_SEMANTICS[HYRULE_CASTLE_GROUNDS_STATE]
    )
    assert STATE_SEMANTIC_NAMES[HYRULE_CASTLE_GROUNDS_STATE].startswith("HyruleCastle_")


def test_segment_registry_has_continuous_segments() -> None:
    reg = segment_registry()
    assert "castle_to_sword" in reg
    assert "sword_to_secret_entrance_clear" in reg
    assert "pocket_to_main_hall" in reg
    assert "castle_dungeon_prefix" in reg
    assert "main_hall_to_zelda" in reg
    assert "escort_to_sanctuary" in reg
    assert set(list_segments()) == set(reg)
    seg = get_segment("castle_to_sword")
    assert seg.exit.verification == "continuous"
    assert seg.entry.graph_node_id == "castle_grounds"

    planned = get_segment("main_hall_to_zelda")
    assert planned.exit.verification == "planned"
    assert planned.entry.graph_node_id == "room_61"
    dungeon = get_segment("castle_dungeon_prefix")
    assert dungeon.exit.verification == "continuous"
    assert dungeon.exit.graph_node_id == "room_50"
    escort = get_segment("escort_to_sanctuary")
    assert escort.exit.verification == "planned"
    assert escort.exit.graph_node_id == "sanctuary"


def test_tip_resolution_most_specific() -> None:
    assert TIP_ANCHOR_ORDER[0] == "HyruleCastle_NW_0x50"
    assert TIP_ANCHOR_ORDER[1] == "HyruleCastle_MainWest_0x60"
    assert "HyruleCastle_MainHall" in TIP_ANCHOR_ORDER

    grounds = _snap(screen_id=0x1B, indoors=False, game_mode=0x09)
    assert resolve_continuous_tip_node(grounds) == "castle_grounds"

    sword = _snap(
        indoors=True,
        room_id=SECRET_PASSAGE_ROOM,
        sword_level=1,
        link_x=2803,
        link_y=2680,
        game_mode=0x07,
    )
    assert resolve_continuous_tip_node(sword) == "room_55_sword"

    # y-threshold chamber split when only fighter-sword (no position) matches
    south_y = _snap(
        indoors=True,
        room_id=SECRET_PASSAGE_ROOM,
        sword_level=1,
        link_x=2500,  # outside south position window
        link_y=2900,
        game_mode=0x07,
    )
    assert resolve_continuous_tip_node(south_y) == "room_55_south"

    pocket = _snap(
        indoors=False,
        screen_id=0x1B,
        link_x=2248,
        link_y=1755,
        sword_level=1,
        game_mode=0x09,
    )
    assert resolve_continuous_tip_node(pocket) == "courtyard_secret_pocket"

    hall = _snap(
        indoors=True,
        room_id=HYRULE_CASTLE_MAIN_HALL_ROOM,
        sword_level=1,
        game_mode=0x07,
    )
    assert resolve_continuous_tip_node(hall) == "room_61"

    from alttp.ram import HYRULE_CASTLE_MAIN_WEST_ROOM, HYRULE_CASTLE_NW_ROOM

    west = _snap(
        indoors=True,
        room_id=HYRULE_CASTLE_MAIN_WEST_ROOM,
        sword_level=1,
        game_mode=0x07,
    )
    assert resolve_continuous_tip_node(west) == "room_60"

    nw = _snap(
        indoors=True,
        room_id=HYRULE_CASTLE_NW_ROOM,
        sword_level=1,
        game_mode=0x07,
    )
    assert resolve_continuous_tip_node(nw) == "room_50"

    cell = _snap(
        indoors=True,
        room_id=ZELDA_CELL_ROOM,
        sword_level=1,
        game_mode=0x07,
    )
    assert resolve_continuous_tip_node(cell) == "room_80"

    sanc = _snap(indoors=True, room_id=SANCTUARY_ROOM, game_mode=0x07)
    assert resolve_continuous_tip_node(sanc) == "sanctuary"


def test_entry_rejection_does_not_call_play() -> None:
    """play_checked with enforce_entry rejects without invoking play_fn."""
    import numpy as np

    from alttp.opening_route.segment import ScriptSegment
    from alttp.ram import DARK_WORLD_FLAG, INDOORS, MODULE, ROOM_ID, SUBMODULE

    called: list[str] = []

    def _should_not_run(env: object, **kwargs: object) -> object:
        called.append("play")
        raise AssertionError("play_fn must not run on entry reject")

    seg = get_segment("castle_to_sword")
    spy = ScriptSegment(
        segment_id=seg.segment_id,
        play_fn=_should_not_run,  # type: ignore[arg-type]
        entry=seg.entry,
        exit=seg.exit,
        label=seg.label,
        graph_edge_id=seg.graph_edge_id,
    )

    class _FakeEnv:
        def get_ram(self) -> np.ndarray:
            # Wrong place: indoors secret passage — not castle grounds.
            ram = np.zeros(0x20000, dtype=np.uint8)
            ram[MODULE] = 0x07
            ram[SUBMODULE] = 0x00
            ram[INDOORS] = 1
            ram[DARK_WORLD_FLAG] = 0
            ram[ROOM_ID] = SECRET_PASSAGE_ROOM & 0xFF
            return ram

    evidence = spy.play_checked(_FakeEnv(), source="test")
    assert evidence.ok is False
    assert evidence.phase == "entry_rejected"
    assert evidence.frames == 0
    assert "entry requirement not met" in evidence.blocker
    assert called == []
    assert "play_fn not called" in evidence.notes[0]

    # enforce_entry=False must call play_fn.
    raised = False
    try:
        spy.play_checked(_FakeEnv(), source="test", enforce_entry=False)
    except AssertionError:
        raised = True
    assert raised
    assert called == ["play"]


def test_main_hall_to_zelda_offline_acceptance_paths() -> None:
    import numpy as np

    from alttp.opening_route.main_hall_to_zelda import (
        evaluate_acceptance,
        run_from_main_hall,
    )
    from alttp.ram import (
        DARK_WORLD_FLAG,
        EQUIP_SWORD,
        FOLLOWER,
        HYRULE_CASTLE_MAIN_WEST_ROOM,
        INDOORS,
        MODULE,
        ROOM_ID,
        SUBMODULE,
        wram_index,
    )

    class _HallEnv:
        def __init__(self, *, room: int, sword: int = 1, follower: int = 0) -> None:
            self.room = room
            self.sword = sword
            self.follower = follower

        def get_ram(self) -> np.ndarray:
            ram = np.zeros(0x20000, dtype=np.uint8)
            ram[MODULE] = 0x07
            ram[SUBMODULE] = 0x00
            ram[INDOORS] = 1
            ram[DARK_WORLD_FLAG] = 0
            ram[ROOM_ID] = self.room & 0xFF
            ram[ROOM_ID + 1] = (self.room >> 8) & 0xFF
            ram[wram_index(EQUIP_SWORD)] = self.sword
            ram[wram_index(FOLLOWER)] = self.follower
            return ram

    rescued = _HallEnv(room=ZELDA_CELL_ROOM, follower=1)
    result2 = run_from_main_hall(rescued, source="test")
    assert result2.ok is True
    assert result2.phase == "zelda_rescued"

    west = _HallEnv(room=HYRULE_CASTLE_MAIN_WEST_ROOM, follower=0)
    result_w = run_from_main_hall(west, source="test")
    assert result_w.ok is False
    assert result_w.phase == "left_main_hall_west"
    assert "0x60" in result_w.blocker or "B1" in result_w.blocker

    cell = _HallEnv(room=ZELDA_CELL_ROOM, follower=0)
    result_c = run_from_main_hall(cell, source="test")
    assert result_c.ok is False
    assert result_c.phase == "in_zelda_cell"

    acc = evaluate_acceptance(
        _snap(
            indoors=True,
            room_id=HYRULE_CASTLE_MAIN_HALL_ROOM,
            sword_level=1,
            game_mode=0x07,
        )
    )
    assert acc["main_hall"] is True
    assert acc["zelda_follower"] is False
    assert acc["left_main_hall_west"] is False
    acc_w = evaluate_acceptance(
        _snap(
            indoors=True,
            room_id=HYRULE_CASTLE_MAIN_WEST_ROOM,
            sword_level=1,
            game_mode=0x07,
        )
    )
    assert acc_w["left_main_hall_west"] is True


def test_escort_scaffold_offline() -> None:
    import numpy as np

    from alttp.opening_route.escort_to_sanctuary import run_from_escort
    from alttp.ram import (
        DARK_WORLD_FLAG,
        FOLLOWER,
        INDOORS,
        LINK_ITEM_LAMP,
        MODULE,
        ROOM_ID,
        SUBMODULE,
        wram_index,
    )

    class _EscortEnv:
        def __init__(
            self,
            *,
            room: int = 0x80,
            follower: int = 1,
            lamp: int = 1,
        ) -> None:
            self.room = room
            self.follower = follower
            self.lamp = lamp

        def get_ram(self) -> np.ndarray:
            ram = np.zeros(0x20000, dtype=np.uint8)
            ram[MODULE] = 0x07
            ram[SUBMODULE] = 0x00
            ram[INDOORS] = 1
            ram[DARK_WORLD_FLAG] = 0
            ram[ROOM_ID] = self.room & 0xFF
            ram[wram_index(FOLLOWER)] = self.follower
            ram[wram_index(LINK_ITEM_LAMP)] = self.lamp
            return ram

    env = _EscortEnv()
    result = run_from_escort(env, source="test")
    assert result.ok is False
    assert result.phase == "not_implemented"
    assert "not implemented" in result.blocker.lower()

    sanc = _EscortEnv(room=SANCTUARY_ROOM)
    result2 = run_from_escort(sanc, source="test")
    assert result2.ok is True
    assert result2.phase == "sanctuary_reached"
