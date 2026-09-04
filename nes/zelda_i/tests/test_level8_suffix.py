"""L8 Gleeok-suffix composition stays inert. No emulator. No RAM writes.

These tests go red if someone composes the suffix, or greens ``L8_THROUGH``,
without a natural predecessor (rr-8t4.3 / rr-6o7.1) and a measured
``Level8ClearEndpoint``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from retro_harness.nes import nes_idle_action

from zelda_i.level8.dungeon import (
    UNOBSERVED_LEVEL8_CLEAR,
    UNOBSERVED_LEVEL8_TOPOLOGY,
    Level8ClearEndpoint,
    level8_clear_stop,
)
from zelda_i.level8.hops import _clear_stages, l8_hops
from zelda_i.level8.passage import CELLAR_ROOM, SOURCE_ROOM, SPAWN_XY
from zelda_i.level8.path import UnverifiedLevel8PathController
from zelda_i.level8.spine import continue_level8_spine
from zelda_i.level8.suffix import (
    CELLAR_2F_SETTLE_FRAMES,
    FIXTURE_LINEAGE_LEVEL8_SUFFIX,
    LEVEL8_SUFFIX_GATES,
    SUFFIX_PREREQUISITE_BEADS,
    Level8Cellar2FSettleController,
    Level8SuffixLineage,
    make_cellar_2f_settle_controller,
    suffix_blockers,
    suffix_stages,
)
from zelda_i.ram import (
    ADDR_HEALTH,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MAGIC_KEY,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    PASSAGE_MODE,
    PLAY_MODE,
    read_snapshot,
)

IDLE = list(nes_idle_action())
# A composable lineage exists only in this test: nothing in the tree builds one.
_COMPOSED = Level8SuffixLineage(
    evidence="test_only", route_eligible=True, natural_predecessor=True
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PASSAGE_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 8)
    ram[ADDR_SCREEN] = fields.get("screen", CELLAR_ROOM)
    ram[ADDR_LINK_X] = fields.get("x", SPAWN_XY[0])
    ram[ADDR_LINK_Y] = fields.get("y", SPAWN_XY[1])
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x7F)
    ram[ADDR_MAGIC_KEY] = fields.get("magic_key", 1)
    ram[ADDR_HEALTH] = fields.get("health", 0x33)
    return ram


def _env(ram: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(get_ram=lambda: ram)


class _StageRecorder:
    def __init__(self, *, ok: bool = True) -> None:
        self.rows: list[tuple[str, object, int]] = []
        self.ok = ok

    def __call__(self, env, run, stages, **_kw) -> bool:
        self.rows.extend(stages)
        return self.ok

    def names(self) -> list[str]:
        return [name for name, _, _ in self.rows]


def test_default_lineage_is_not_composable() -> None:
    assert FIXTURE_LINEAGE_LEVEL8_SUFFIX.evidence == "fixture_lineage"
    assert FIXTURE_LINEAGE_LEVEL8_SUFFIX.route_eligible is False
    assert FIXTURE_LINEAGE_LEVEL8_SUFFIX.natural_predecessor is False
    assert FIXTURE_LINEAGE_LEVEL8_SUFFIX.composable() is False
    assert suffix_stages() == ()
    assert suffix_blockers() == (
        "post_l7_handoff_unmeasured",
        "suffix_lineage_not_route_eligible",
    )
    assert SUFFIX_PREREQUISITE_BEADS == ("rr-8t4.3", "rr-6o7.1")


def test_half_a_lineage_still_refuses() -> None:
    """route_eligible without a natural predecessor must not compose."""
    for lineage in (
        Level8SuffixLineage(route_eligible=True),
        Level8SuffixLineage(natural_predecessor=True),
    ):
        assert lineage.composable() is False
        assert suffix_stages(lineage=lineage) == ()
        assert suffix_blockers(lineage)


def test_clear_chapter_default_rows_are_the_fail_closed_three() -> None:
    rows = _clear_stages(topology=UNOBSERVED_LEVEL8_TOPOLOGY)
    assert [name for name, _, _ in rows] == [
        "level8_return_passage",
        "level8_four_head_gleeok",
        "level8_heart_shard_leave",
    ]
    passage = rows[0][1]
    assert isinstance(passage, UnverifiedLevel8PathController)
    passage.step(read_snapshot(_ram(mode=PLAY_MODE, screen=0x3E)))
    assert passage.failed is True


def test_composed_rows_are_the_ordered_suffix() -> None:
    rows = _clear_stages(topology=UNOBSERVED_LEVEL8_TOPOLOGY, suffix=_COMPOSED)
    assert [name for name, _, _ in rows] == [
        gate.stage for gate in LEVEL8_SUFFIX_GATES
    ]
    assert "level8_four_head_gleeok" in [name for name, _, _ in rows]
    assert not any(
        isinstance(ctl, UnverifiedLevel8PathController) for _, ctl, _ in rows
    )
    for _, ctl, budget in rows:
        assert budget >= 1
        assert getattr(ctl, "route_eligible", False) is False


def test_settle_sits_between_the_stairs_and_the_cross() -> None:
    """P1/P2 was measured from the settled pin, never from the K3/K4 arrival."""
    names = [gate.stage for gate in LEVEL8_SUFFIX_GATES]
    stairs = names.index("level8_return_passage_stairs_3f")
    settle = names.index("level8_return_passage_cellar_2f_settle")
    cross = names.index("level8_return_passage_cross_2f")
    assert stairs < settle < cross


def test_suffix_gate_rooms_form_a_chain() -> None:
    gates = [g for g in LEVEL8_SUFFIX_GATES if "settle" not in g.stage]
    for prev, nxt in zip(gates, gates[1:]):
        # Each row either continues in the room the previous row arrived in
        # (the Gleeok fight, the shard walk) or leaves it through one gate.
        assert (
            nxt.dest_room == prev.dest_room
            or nxt.origin_room == prev.dest_room
        ), f"{prev.stage} -> {nxt.stage} is not contiguous"


def test_cellar_settle_presses_nothing_and_needs_the_full_idle() -> None:
    ctl = make_cellar_2f_settle_controller()
    assert isinstance(ctl, Level8Cellar2FSettleController)
    ram = _ram(x=208, y=141)
    before = ram.copy()
    for _ in range(8):
        act = ctl.step(read_snapshot(ram))
        assert list(act.action) == IDLE
    assert np.array_equal(ram, before), "settle must not write RAM"
    assert not ctl.success and not ctl.failed
    # Even parked on the settled pose, the idle budget must elapse first.
    settled = _ram()
    ctl2 = make_cellar_2f_settle_controller()
    for _ in range(CELLAR_2F_SETTLE_FRAMES - 1):
        ctl2.step(read_snapshot(settled))
    assert not ctl2.success
    ctl2.step(read_snapshot(settled))
    assert ctl2.success is True
    assert ctl2.report()["writes"] == 0


def test_cellar_settle_fails_back_on_the_source_room() -> None:
    ctl = make_cellar_2f_settle_controller()
    ctl.step(read_snapshot(_ram(mode=PLAY_MODE, screen=SOURCE_ROOM, x=32, y=141)))
    assert ctl.failed is True
    assert "returned_source_0x3f" in ctl.notes


def test_fixture_lineage_endpoint_cannot_green_the_clear_stop() -> None:
    """The post-shard OW pin is fixture-lineage; it may not be the endpoint."""
    assert not UNOBSERVED_LEVEL8_CLEAR.complete()
    fixture_endpoint = Level8ClearEndpoint(
        level=0,
        screen=0x6D,
        mode=PLAY_MODE,
        incoming_heart_containers=3,
        outgoing_heart_containers=4,
        evidence="fixture_lineage",
        route_eligible=False,
    )
    assert fixture_endpoint.complete() is False
    snap = read_snapshot(_ram(level=0, mode=PLAY_MODE, screen=0x6D, triforce=0xFF))
    assert level8_clear_stop(snap, magic_key=1, endpoint=fixture_endpoint) is False


def _clear_hop(env, **kw):
    hop = l8_hops(env, **kw)[2]
    assert hop.through == "level8" and hop.stop == "level8_triforce_0x80"
    return hop


def test_composed_suffix_still_cannot_green_through_level8() -> None:
    """Post-shard RAM + every suffix row present must still fail the stop."""
    ram = _ram(level=0, mode=PLAY_MODE, screen=0x6D, triforce=0xFF)
    hop = _clear_hop(_env(ram), suffix=_COMPOSED)
    names = [name for name, _, _ in hop.stages()]
    assert names[-1] == "level8_heart_shard_leave_2c"
    assert hop.success(read_snapshot(ram)) is False


def test_default_spine_call_keeps_the_fail_closed_clear_chapter() -> None:
    """The wired spine attaches the fail-closed rows, never the suffix."""
    ram = _ram(level=0, mode=PLAY_MODE, screen=0x6D, triforce=0xFF)
    run = SimpleNamespace(success=True, failed_stage=None)
    rows = _StageRecorder(ok=True)
    continue_level8_spine(_env(ram), run, through="level8", run_stages=rows)
    # attach_hops stops at the entry chapter: the seam is still blocked.
    assert run.success is False
    assert run.failed_stage == "level8_entry_live"

    hop = _clear_hop(_env(ram))
    names = [name for name, _, _ in hop.stages()]
    assert names == [
        "level8_return_passage",
        "level8_four_head_gleeok",
        "level8_heart_shard_leave",
    ]
    assert "level8_return_passage_east_3e" not in names
    assert hop.success(read_snapshot(ram)) is False


def test_hop_rows_default_to_the_fixture_lineage() -> None:
    from inspect import signature

    for fn in (l8_hops, continue_level8_spine):
        param = signature(fn).parameters["suffix"]
        assert param.default is FIXTURE_LINEAGE_LEVEL8_SUFFIX
