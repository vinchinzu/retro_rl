from __future__ import annotations

from types import SimpleNamespace

from super_metroid.autopilot import ReactivePolicyRegistry
from super_metroid.ram import HI_JUMP_MASK
from super_metroid.reactive_policy import (
    PolicyVariant,
    ReactivePolicyRunner,
    ReactiveRoomPolicy,
    ReferenceSample,
    ReferenceTrajectory,
)
from super_metroid.room_policy_tools import mark_takeovers_verified, mark_verified


def _state(x: int, *, items: int = 0, room: int = 0x96BA):
    return SimpleNamespace(
        room_id=room,
        samus_x=x,
        samus_y=100,
        velocity_x=0,
        velocity_y=0,
        momentum_x=0,
        pose=1,
        facing=8,
        movement_type=0,
        vertical_direction=0,
        collected_items=items,
    )


def _sample(x: int, button: int, *, hold: int = 1) -> ReferenceSample:
    action = [0] * 12
    action[button] = 1
    return ReferenceSample.from_state(_state(x), action, frames=hold)


def _variant(
    variant_id: str,
    *,
    required: int = 0,
    forbidden: int = 0,
) -> PolicyVariant:
    trajectory = ReferenceTrajectory(
        trajectory_id=f"{variant_id}_take",
        samples=(_sample(0, 6, hold=3), _sample(100, 7)),
    )
    return PolicyVariant(
        variant_id=variant_id,
        trajectories=(trajectory,),
        required_items=required,
        forbidden_items=forbidden,
    )


def test_runner_attaches_midroom_and_caches_timed_span() -> None:
    runner = ReactivePolicyRunner(_variant("base"))

    target = runner.resume(_state(2))
    assert target.sample_index == 0
    assert runner.action(_state(2)).tolist()[6] == 1
    assert runner.has_held_action
    assert runner.continue_action().tolist()[6] == 1
    assert runner.continue_action().tolist()[6] == 1
    assert not runner.has_held_action

    target = runner.resume(_state(94))
    assert target.sample_index == 1
    assert runner.action(_state(94)).tolist()[7] == 1


def test_policy_round_trip_selects_equipment_variant(tmp_path) -> None:
    policy = ReactiveRoomPolicy(
        policy_id="climb",
        room_id=0x96BA,
        from_room_id=0x975C,
        exit_room_id=0x92FD,
        variants=(
            _variant("base", forbidden=HI_JUMP_MASK),
            _variant("hi_jump", required=HI_JUMP_MASK),
        ),
        status="verified_live_anchor",
    )
    path = policy.save(tmp_path / "climb.json")
    loaded = ReactiveRoomPolicy.load(path)

    assert loaded.select_variant(0).variant_id == "base"
    assert loaded.select_variant(HI_JUMP_MASK).variant_id == "hi_jump"
    assert loaded.to_dict() == policy.to_dict()


def test_registry_requires_verified_status_and_prefers_entry_specific_policy() -> None:
    generic = ReactiveRoomPolicy(
        policy_id="generic",
        room_id=0x96BA,
        exit_room_id=0x92FD,
        variants=(_variant("base"),),
        status="verified_live_anchor",
    )
    specific = ReactiveRoomPolicy(
        policy_id="specific",
        room_id=0x96BA,
        from_room_id=0x975C,
        exit_room_id=0x92FD,
        variants=(_variant("base"),),
        status="verified_live_anchor",
    )
    candidate = ReactiveRoomPolicy(
        policy_id="candidate",
        room_id=0x96BA,
        from_room_id=0x975C,
        exit_room_id=0x92FD,
        variants=(_variant("base"),),
    )

    selected = ReactivePolicyRegistry((generic, candidate, specific)).select(
        _state(0), from_room_id=0x975C, route_id="kpdr"
    )

    assert selected is not None
    assert selected[0].policy_id == "specific"


def test_each_equipment_variant_needs_its_own_verification() -> None:
    policy = ReactiveRoomPolicy(
        policy_id="climb",
        room_id=0x96BA,
        exit_room_id=0x92FD,
        variants=(_variant("base"), _variant("hi_jump", required=HI_JUMP_MASK)),
    )
    base_report = {
        "green": True,
        "dual": True,
        "anchor": "base.state",
        "runs": [
            {"variant": "base", "frames": 100, "fps": 500},
            {"variant": "base", "frames": 100, "fps": 510},
        ],
    }
    policy = mark_verified(policy, base_report)
    assert policy.status == "candidate"

    hi_report = {
        **base_report,
        "anchor": "hi.state",
        "runs": [
            {"variant": "hi_jump", "frames": 90, "fps": 520},
            {"variant": "hi_jump", "frames": 90, "fps": 530},
        ],
    }
    policy = mark_verified(policy, hi_report)
    assert policy.status == "verified_live_anchor"

    takeover_report = {
        "ok": True,
        "anchor": "hi.state",
        "runs": [
            {
                "variant": "hi_jump",
                "takeover_point": 50,
                "perturb_frames": 4,
                "autopilot_frames": 40,
                "fps": 600,
                "adapter_frames": 0,
                "room": "0x92FD",
            }
        ],
    }
    policy = mark_takeovers_verified(policy, takeover_report)
    assert policy.meta["takeoverVerification"]["hi_jump"]["green"]
