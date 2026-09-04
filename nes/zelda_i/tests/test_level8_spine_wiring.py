"""The L8 seam is reachable from the Survival spine but still fails closed.

``--through level8-entry`` must be a valid spine stop (no "unknown spine stop"
error) and must still refuse: the default handoff is unmeasured and the L8
topology is not route-eligible, so no L8 stop can green. The disclosed recon
packets are reachable only through an explicit non-default parameter. No
emulator.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from zelda_i.level8.dungeon import (
    LIVE_RECON_LEVEL8_TOPOLOGY,
    UNOBSERVED_LEVEL8_TOPOLOGY,
    Level8Topology,
)
from zelda_i.level8.entry import (
    LIVE_RECON_BUSH_BURN_TARGET,
    UNMEASURED_POST_L7_HANDOFF,
    UNVERIFIED_BUSH_BURN_TARGET,
)
from zelda_i.level8.hops import l8_hops
from zelda_i.level8.spine import (
    L8_STOPS,
    L8_THROUGH,
    LIVE_RECON_L8_OVERRIDES,
    continue_level8_spine,
)
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOMBS,
    ADDR_BOW,
    ADDR_CANDLE,
    ADDR_HEALTH,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MAGIC_KEY,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_SELECTED_ITEM,
    ADDR_SWORD,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.spine.survival import SPINE_THROUGH, SpineRun, run_survival_spine

CANDLE_RED, B_ITEM_CANDLE = 2, 4
RECON_ENTRY_ROOM = 0x7E


def _ram(**fields: int) -> np.ndarray:
    """Post-burn L8 entry RAM by default: the best case the stop may see."""
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 8)
    ram[ADDR_SCREEN] = fields.get("screen", RECON_ENTRY_ROOM)
    ram[ADDR_LINK_X] = fields.get("x", 120)
    ram[ADDR_LINK_Y] = fields.get("y", 205)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x7F)
    ram[ADDR_SWORD] = fields.get("sword", 3)
    ram[ADDR_HEALTH] = fields.get("health", 0xBB)
    ram[ADDR_KEYS] = fields.get("keys", 4)
    ram[ADDR_BOMBS] = fields.get("bombs", 8)
    ram[ADDR_CANDLE] = fields.get("candle", CANDLE_RED)
    ram[ADDR_SELECTED_ITEM] = fields.get("selected", B_ITEM_CANDLE)
    ram[ADDR_BOW] = fields.get("bow", 1)
    ram[ADDR_ARROWS] = fields.get("arrows", 1)
    ram[ADDR_MAGIC_KEY] = fields.get("magic_key", 0)
    return ram


def _env(ram: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(get_ram=lambda: ram)


def _run() -> SimpleNamespace:
    return SimpleNamespace(success=True, failed_stage=None)


class _StageRecorder:
    """Stand-in for ``_run_stages``: records rows, never touches an emulator."""

    def __init__(self, *, step: bool = False, ok: bool = True) -> None:
        self.rows: list[tuple[str, Any, int]] = []
        self.step = step
        self.ok = ok

    def __call__(self, env, run, stages, **_kw) -> bool:
        for name, controller, max_frames in stages:
            self.rows.append((name, controller, max_frames))
            if not self.step:
                continue
            bind = getattr(controller, "bind_env", None)
            if callable(bind):
                bind(env)
            controller.step(read_snapshot(env.get_ram()))
            if getattr(getattr(controller, "phase", None), "name", "") == "FAILED":
                run.success = False
                run.failed_stage = name
                return False
        return self.ok

    def names(self) -> list[str]:
        return [name for name, _, _ in self.rows]


def test_level8_through_targets_are_wired_spine_stops() -> None:
    assert L8_THROUGH == ("level8-entry", "level8-magic-key", "level8")
    for through in L8_THROUGH:
        assert through in SPINE_THROUGH
        run = SpineRun(through=through, success=False, boot_frames=0)
        assert run.report()["stop"] == L8_STOPS[through]


def test_through_level8_entry_is_not_an_unknown_spine_stop() -> None:
    """The seam is reachable: the stop check passes, the assist check bites."""
    try:
        run_survival_spine(None, None, assist=None, through="level8-entry")
    except ValueError as exc:
        assert "unknown spine stop" not in str(exc)
        assert "UnlimitedHealthAssist" in str(exc)
    else:
        raise AssertionError("missing assist must raise")

    try:
        run_survival_spine(None, None, assist=None, through="level8-book")
    except ValueError as exc:
        assert "unknown spine stop" in str(exc)
    else:
        raise AssertionError("a genuinely unknown stop must raise")


def test_default_entry_chapter_refuses_on_unmeasured_handoff() -> None:
    """First wired stage fails closed before any input, on defaults."""
    ram = _ram(level=0, screen=0x6D, x=48, y=93)
    run = _run()
    stages = _StageRecorder(step=True)
    continue_level8_spine(
        _env(ram), run, through="level8-entry", run_stages=stages
    )
    assert stages.names() == ["level8_post_l7_to_bush"]
    approach = stages.rows[0][1]
    assert not UNMEASURED_POST_L7_HANDOFF.complete()
    assert approach.handoff is UNMEASURED_POST_L7_HANDOFF
    assert "post_l7_handoff_unmeasured" in approach.notes
    assert run.success is False
    assert run.failed_stage == "level8_post_l7_to_bush"


def test_entry_stop_refuses_even_with_recon_topology() -> None:
    """Fixture recon is not route evidence: the stop stays False either way."""
    env = _env(_ram())
    snap = read_snapshot(env.get_ram())
    assert not UNOBSERVED_LEVEL8_TOPOLOGY.route_eligible
    assert not LIVE_RECON_LEVEL8_TOPOLOGY.route_eligible
    for topology in (UNOBSERVED_LEVEL8_TOPOLOGY, LIVE_RECON_LEVEL8_TOPOLOGY):
        for hop in l8_hops(env, topology=topology):
            assert hop.success(snap) is False


def test_wired_chapter_cannot_green_even_if_every_stage_passes() -> None:
    """Stages green + L8 RAM present still fails: the stop predicate refuses."""
    for overrides in ({}, LIVE_RECON_L8_OVERRIDES):
        run = _run()
        continue_level8_spine(
            _env(_ram()),
            run,
            through="level8-entry",
            run_stages=_StageRecorder(ok=True),
            **overrides,
        )
        assert run.success is False
        assert run.failed_stage == L8_STOPS["level8-entry"]


def test_recon_overrides_are_an_explicit_opt_in_path() -> None:
    """The disclosed constants reach the burn stage only when passed in."""
    assert LIVE_RECON_L8_OVERRIDES == {
        "burn_target": LIVE_RECON_BUSH_BURN_TARGET,
        "topology": LIVE_RECON_LEVEL8_TOPOLOGY,
    }
    assert LIVE_RECON_BUSH_BURN_TARGET.complete()
    assert not LIVE_RECON_BUSH_BURN_TARGET.route_eligible
    assert not UNVERIFIED_BUSH_BURN_TARGET.complete()

    default_rows = _StageRecorder()
    continue_level8_spine(
        _env(_ram()), _run(), through="level8-entry", run_stages=default_rows
    )
    recon_rows = _StageRecorder()
    continue_level8_spine(
        _env(_ram()),
        _run(),
        through="level8-entry",
        run_stages=recon_rows,
        **LIVE_RECON_L8_OVERRIDES,
    )
    assert default_rows.names() == recon_rows.names()
    default_burn = dict(zip(default_rows.names(), default_rows.rows))
    recon_burn = dict(zip(recon_rows.names(), recon_rows.rows))
    assert default_burn["level8_burn_bush_enter"][1].target is (
        UNVERIFIED_BUSH_BURN_TARGET
    )
    assert recon_burn["level8_burn_bush_enter"][1].target is (
        LIVE_RECON_BUSH_BURN_TARGET
    )


def test_default_spine_call_passes_no_recon() -> None:
    """``run_survival_spine`` forwards nothing to L8 unless asked."""
    from inspect import signature

    param = signature(run_survival_spine).parameters["level8_overrides"]
    assert param.default is None


def test_magic_key_stop_needs_a_0_to_1_rise_not_an_owned_key() -> None:
    """A pin that already carries MK=1 must not satisfy the MK chapter.

    Uses a hand-built route-eligible topology that exists only here: the
    tree's own topologies both keep ``route_eligible=False``, so this is the
    stop's *second* gate, not a way to green anything.
    """
    topology = Level8Topology(
        entry_room=RECON_ENTRY_ROOM,
        magic_key_room=0x1F,
        evidence="test_only",
        route_eligible=True,
    )
    ram = _ram(screen=0x1F, magic_key=1)
    env, run = _env(ram), _run()
    hop = [h for h in l8_hops(env, topology=topology) if h.through == "level8-magic-key"]
    assert len(hop) == 1
    hop = hop[0]

    # Chapter entered with the key already owned: no natural acquisition.
    hop.before(env, run)
    assert hop.success(read_snapshot(ram)) is False

    # Chapter entered with MK 0, key acquired during the stages: greens.
    zero = _ram(screen=0x1F, magic_key=0)
    env2 = _env(zero)
    hop2 = [
        h for h in l8_hops(env2, topology=topology)
        if h.through == "level8-magic-key"
    ][0]
    hop2.before(env2, _run())
    zero[ADDR_MAGIC_KEY] = 1
    assert hop2.success(read_snapshot(zero)) is True


def test_magic_key_hop_captures_the_key_before_its_stages() -> None:
    hop = [h for h in l8_hops(_env(_ram())) if h.through == "level8-magic-key"][0]
    assert hop.before is not None
