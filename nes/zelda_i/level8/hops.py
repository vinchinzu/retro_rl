"""Named Level 8 chapter factories and Survival ``SpineHop`` rows."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from zelda_i.level8.dungeon import (
    UNOBSERVED_LEVEL8_CLEAR,
    UNOBSERVED_LEVEL8_TOPOLOGY,
    Level8ClearEndpoint,
    Level8Topology,
    level8_clear_stop,
    level8_entry_stop,
    level8_magic_key_stop,
)
from zelda_i.level8.entry import (
    BURN_MAX_FRAMES,
    SELECT_MAX_FRAMES,
    UNMEASURED_POST_L7_HANDOFF,
    UNVERIFIED_BUSH_BURN_TARGET,
    BushBurnTarget,
    PostLevel7Handoff,
    make_burn_level8_bush_controller,
    make_post_l7_to_bush_controller,
    make_select_red_candle_controller,
)
from zelda_i.level8.path import (
    Level8NorthManhandlaController,
    UnverifiedLevel8PathController,
    make_blue_gohma_controller,
    make_darknut_key_controller,
    make_four_head_gleeok_controller,
    make_gleeok_passage_controller,
    make_magic_key_stairs_controller,
    make_north_manhandla_controller,
    make_shard_leave_controller,
)
from zelda_i.level8.suffix import (
    FIXTURE_LINEAGE_LEVEL8_SUFFIX,
    Level8SuffixLineage,
    make_cellar_2f_settle_controller,
    suffix_stages,
)
from zelda_i.level8.overworld import L7_POND_TO_LEVEL8_BUSH_HOPS
from zelda_i.overworld.graph import ScreenHop
from zelda_i.ram import ADDR_CANDLE, ADDR_MAGIC_KEY, ZeldaSnapshot, read_snapshot, read_u8
from zelda_i.spine.hops import SpineHop

Stage = tuple[str, object, int]
ControllerFactory = Callable[[], object]


def make_entry_to_magic_key_controller() -> Level8NorthManhandlaController:
    """First magic-key sub-stage; chapters still use named sub-stages."""
    return make_north_manhandla_controller()


def make_magic_key_to_shard_controller() -> UnverifiedLevel8PathController:
    return make_gleeok_passage_controller()


def _stage(name: str, factory: ControllerFactory) -> Stage:
    controller = factory()
    return (name, controller, int(getattr(controller, "max_frames", 1)))


def _entry_stages(
    *,
    handoff: PostLevel7Handoff,
    post_l7_hops: tuple[ScreenHop, ...],
    burn_target: BushBurnTarget,
):
    approach = make_post_l7_to_bush_controller(
        handoff=handoff,
        hops=post_l7_hops,
    )
    select = make_select_red_candle_controller()
    burn = make_burn_level8_bush_controller(target=burn_target)
    return (
        ("level8_post_l7_to_bush", approach, approach.max_frames),
        ("level8_select_red_candle", select, SELECT_MAX_FRAMES),
        ("level8_burn_bush_enter", burn, BURN_MAX_FRAMES),
    )


def _magic_key_stages(*, topology: Level8Topology):
    return (
        _stage("level8_north_manhandla_bomb", make_north_manhandla_controller),
        _stage("level8_darknut_key_up", make_darknut_key_controller),
        _stage(
            "level8_blue_gohma",
            lambda: make_blue_gohma_controller(topology=topology),
        ),
        _stage("level8_magic_key_stairs", make_magic_key_stairs_controller),
    )


def _clear_stages(
    *,
    topology: Level8Topology,
    suffix: Level8SuffixLineage = FIXTURE_LINEAGE_LEVEL8_SUFFIX,
):
    """Fail-closed clear chapter, or the ordered suffix once it composes.

    ``suffix_stages`` is empty for every lineage the tree owns today (the
    frontier pin is fixture-lineage), so the default rows are unchanged and
    ``level8_return_passage`` still refuses on the first frame.  A composable
    lineage swaps in ``level8.suffix.LEVEL8_SUFFIX_GATES``; even then
    ``level8_clear_stop`` still needs a measured ``Level8ClearEndpoint``.
    """
    composed = suffix_stages(lineage=suffix)
    if composed:
        return composed
    return (
        _stage("level8_return_passage", make_gleeok_passage_controller),
        _stage(
            "level8_four_head_gleeok",
            lambda: make_four_head_gleeok_controller(topology=topology),
        ),
        _stage("level8_heart_shard_leave", make_shard_leave_controller),
    )


def l8_hops(
    env,
    *,
    handoff: PostLevel7Handoff = UNMEASURED_POST_L7_HANDOFF,
    post_l7_hops: tuple[ScreenHop, ...] = L7_POND_TO_LEVEL8_BUSH_HOPS,
    burn_target: BushBurnTarget = UNVERIFIED_BUSH_BURN_TARGET,
    topology: Level8Topology = UNOBSERVED_LEVEL8_TOPOLOGY,
    clear_endpoint: Level8ClearEndpoint = UNOBSERVED_LEVEL8_CLEAR,
    suffix: Level8SuffixLineage = FIXTURE_LINEAGE_LEVEL8_SUFFIX,
) -> tuple[SpineHop, ...]:
    """Build fresh L8 rows. Defaults are intentionally non-executable."""

    def entry_ok(snap: ZeldaSnapshot, **_) -> bool:
        return level8_entry_stop(
            snap,
            candle=read_u8(env.get_ram(), ADDR_CANDLE),
            topology=topology,
        )

    # ``level8_magic_key_stop`` degrades to "owns a Magical Key" without a
    # before-count, so a pin that already carries MK=1 would satisfy the
    # chapter.  Latch the count at the top of the chapter and require 0->1.
    magic_key_before: dict[str, int] = {}

    def capture_magic_key(hop_env, _run) -> None:
        magic_key_before["value"] = int(read_u8(hop_env.get_ram(), ADDR_MAGIC_KEY))

    def magic_key_ok(snap: ZeldaSnapshot, **_) -> bool:
        return level8_magic_key_stop(
            snap,
            magic_key=read_u8(env.get_ram(), ADDR_MAGIC_KEY),
            topology=topology,
            magic_key_before=magic_key_before.get("value"),
        )

    # Containers at the chapter start, read live (the measured 12 -> 13 is
    # one run's history; a continuous run that skipped another heart
    # arrives with fewer). A resume inside the chapter keeps the endpoint.
    containers: dict[str, int] = {}

    def capture_containers(hop_env, _run) -> None:
        containers["in"] = int(read_snapshot(hop_env.get_ram()).heart_containers)

    def clear_ok(snap: ZeldaSnapshot, **_) -> bool:
        endpoint = clear_endpoint
        if "in" in containers and endpoint.complete():
            endpoint = replace(
                endpoint,
                incoming_heart_containers=containers["in"],
                outgoing_heart_containers=containers["in"] + 1,
            )
        return level8_clear_stop(
            snap,
            magic_key=read_u8(env.get_ram(), ADDR_MAGIC_KEY),
            endpoint=endpoint,
        )

    return (
        SpineHop(
            "level8-entry",
            "level8_entry_live",
            lambda: _entry_stages(
                handoff=handoff,
                post_l7_hops=post_l7_hops,
                burn_target=burn_target,
            ),
            entry_ok,
        ),
        SpineHop(
            "level8-magic-key",
            "level8_magic_key_natural",
            lambda: _magic_key_stages(topology=topology),
            magic_key_ok,
            before=capture_magic_key,
        ),
        SpineHop(
            "level8",
            "level8_triforce_0x80",
            lambda: _clear_stages(topology=topology, suffix=suffix),
            clear_ok,
            before=capture_containers,
        ),
    )


__all__ = [
    "l8_hops",
    "make_cellar_2f_settle_controller",
    "make_blue_gohma_controller",
    "make_entry_to_magic_key_controller",
    "make_four_head_gleeok_controller",
    "make_magic_key_to_shard_controller",
]
