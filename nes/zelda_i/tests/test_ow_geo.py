"""The overworld lattice rung (``OverworldPathController._rung_geo``).

It declines when the fall-through align-then-push would already walk to the
hop's edge on floor. That judgement must model the fall-through as it is: a
horizontal hop with no ``align_y`` pushes along Link's own row.
"""

from __future__ import annotations

from zelda_i.overworld.graph import ScreenHop
from zelda_i.overworld.path import OverworldPathController


def _nodes_0x5b_south_west() -> frozenset[tuple[int, int]]:
    """0x5B's west side from the live $6530 lattice (BFS_5B, 2026-09-23):
    rows 85..189 run to the west edge; rows 197..213 are only the x=48..80
    column Link climbs out of 0x6B on."""
    open_rows = {(x, y) for x in range(0, 88, 8) for y in range(85, 190, 8)}
    column = {(x, y) for x in range(48, 88, 8) for y in range(197, 214, 8)}
    return frozenset(open_rows | column)


def test_unaligned_left_hop_is_not_clear_from_a_row_that_is_rock_west() -> None:
    """2026-09-23 chain: back on 0x5B at (48, 205) for the 0x5A LEFT hop,
    the rung declined (row 189 reaches the edge), the fall-through pushed
    LEFT on row 205 into rock, and ``unstick_wait`` held 9750 frames."""
    nodes = _nodes_0x5b_south_west()
    hop = ScreenHop(0x5A, "LEFT")
    goals = {n for n in nodes if n[0] == 0}
    assert not OverworldPathController._geo_direct_clear(hop, nodes, (48, 205), goals)
    # The same hop from an open row is still left to the fall-through.
    assert OverworldPathController._geo_direct_clear(hop, nodes, (48, 181), goals)


def test_aligned_left_hop_is_clear_when_the_align_row_reaches_the_edge() -> None:
    """With ``align_y`` the fall-through does move to that row first."""
    nodes = _nodes_0x5b_south_west()
    hop = ScreenHop(0x5A, "LEFT", align_y=189)
    goals = {n for n in nodes if n[0] == 0 and n[1] == 189}
    assert OverworldPathController._geo_direct_clear(hop, nodes, (48, 205), goals)


def _snap(x: int, y: int, screen: int = 0x5C):
    from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

    return ZeldaSnapshot(
        mode=PLAY_MODE, level=0, screen=screen, next_screen=screen, link_x=x, link_y=y,
        facing=8, sword=2, bombs=0, rupees=0, keys=0, health=0x44, triforce=0, compass=0,
        dialog_timer=0, colliding_tile=0, room_item_id=0, room_all_dead=0, room_obj_count=0,
        cur_opened_doors=0, open_doorway_mask=0, objects=(),
    )


def test_maze_walk_never_idles_on_a_stuck_count() -> None:
    """Last-heart run 29 (2026-09-24): knocked back to (192, 85) on 0x5C,
    the maze follower's unstick idled; idling keeps Link still, so the
    count only grew: 19830 frames of ``maze_unstick_wait`` before timeout."""
    from retro_harness.nes import nes_idle_action
    from zelda_i.overworld.graph import LEVEL2_5C_MAZE_WAYPOINTS

    ctrl = OverworldPathController(
        hops=(ScreenHop(0x5D, "RIGHT", y_band_lo=120, y_band_hi=140),),
        maze_waypoints=LEVEL2_5C_MAZE_WAYPOINTS,
    )
    ctrl.maze_wp_index = 11
    ctrl.stuck = 10_000
    act = ctrl._follow_maze(_snap(192, 85))
    assert list(act.action) != list(nes_idle_action())


def test_horizontal_push_aligns_until_the_slide_lands_on_the_target_row() -> None:
    """0x37, align_y=140 (run 29): DOWN stopped at 135 ("within 5"), the
    RIGHT push slid Link back onto row 133, and the two swapped for 12500
    frames. The push waits until Link's nearest turn row is 141."""
    from retro_harness.controls import NES_BUTTON_NAME_TO_INDEX
    from zelda_i.overworld.common import align_and_push

    def press(y: int) -> str:
        act = align_and_push(_snap(144, y, 0x37), direction="RIGHT", reason="hop0", align_y=140)
        held = [d for d in ("UP", "DOWN", "LEFT", "RIGHT") if act.action[NES_BUTTON_NAME_TO_INDEX[d]]]
        assert len(held) == 1, held
        return held[0]

    assert press(133) == "DOWN"
    assert press(135) == "DOWN"
    assert press(138) == "RIGHT"
    assert press(141) == "RIGHT"
    assert press(150) == "UP"


def test_hop_list_resumes_after_the_screen_link_starts_on() -> None:
    """The L4 walk starts on 0x74, or on 0x64 after a potion restock there;
    with ``resume_on_screen`` the same hop list picks up at 0x65."""
    from zelda_i.level4.overworld import LEVEL4_HOPS_FROM_POST_L3

    targets = [h.target for h in LEVEL4_HOPS_FROM_POST_L3]
    for screen, expect in ((0x74, 0), (0x64, targets.index(0x64) + 1)):
        ctrl = OverworldPathController(hops=LEVEL4_HOPS_FROM_POST_L3, resume_on_screen=True)
        ctrl.step(_snap(112, 93, screen))
        assert ctrl.hop_index == expect, hex(screen)


def test_0x79_east_skirt_comes_down_to_the_exit_rows_from_above() -> None:
    """Run 19 pre-l1: a rupee scoop left Link at (192, 109) and the skirt
    pressed RIGHT into rock for 28184 frames. From x=192 only rows 133/141
    reach 0x79's east edge (live $6530 lattice)."""
    from retro_harness.controls import NES_BUTTON_NAME_TO_INDEX
    from zelda_i.overworld.shop_p7 import make_shop_p7_walk_controller

    ctrl = make_shop_p7_walk_controller()

    def press(y: int) -> str:
        act = ctrl._leave_79_east(_snap(192, y, 0x79))
        return next(d for d in ("UP", "DOWN", "LEFT", "RIGHT") if act.action[NES_BUTTON_NAME_TO_INDEX[d]])

    assert press(109) == "DOWN"
    assert press(133) == "RIGHT"
    assert press(157) == "UP"


def test_hop_unstick_walks_the_lattice_once_the_wiggle_is_spent(monkeypatch) -> None:
    """``unstick_wiggle`` idled once its nudge was spent; idling keeps Link
    still, so ``stuck`` only grew (0x1E ``letter``: 2528 frames). Past the
    wiggle the rung walks the ROM lattice to the hop's exit band."""
    import zelda_i.overworld.path as path
    from retro_harness.nes import nes_idle_action

    calls = []

    def band_step(env, snap, direction, lo, hi):
        calls.append((direction, lo, hi))
        return "DOWN"

    monkeypatch.setattr(path, "ow_edge_band_step", band_step)
    ctrl = OverworldPathController(hops=(ScreenHop(0x5D, "RIGHT", y_band_lo=120, y_band_hi=140),))
    ctrl.stuck = 10_000
    ctrl._hop = ctrl.hops[0]  # the ladder sets it each frame
    act = ctrl._rung_unstick(_snap(96, 85))
    assert list(act.action) != list(nes_idle_action())
    assert calls == [("RIGHT", 120, 140)]


def test_post_l6_walk_hands_a_stall_to_the_unstick_rung() -> None:
    """``post_l6_path_stuck_wait`` idled at the top of the hop ladder
    (priority 10), ahead of every rung that could move Link."""
    from retro_harness.nes import nes_idle_action
    from zelda_i.level7.pond import PostLevel6OverworldController

    ctrl = PostLevel6OverworldController()
    ctrl.stuck = ctrl.stuck_threshold + 1
    hop = ctrl.hops[0]
    act = ctrl._extra_hop_action(_snap(96, 85, 0x10), hop)
    assert act is None or list(act.action) != list(nes_idle_action())
