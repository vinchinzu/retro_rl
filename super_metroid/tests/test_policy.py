from __future__ import annotations

from dataclasses import replace

import numpy as np

from super_metroid.policy import PolicySegment, StateRequirement, load_policy
from super_metroid.ram import GameplayPhase, MORPH_BALL_MASK, parse_state


def test_imported_policy_has_valid_actions_and_provenance() -> None:
    segment = PolicySegment(
        "elevator_return",
        "elevator_to_pit.json",
        StateRequirement(),
        StateRequirement(),
        "elevator_return",
    )

    actions, metadata = load_policy(segment)

    assert len(actions) == 352
    assert all(action.shape == (12,) for action in actions)
    assert len(metadata["source_sha256"]) == 64
    assert metadata["source_slice"] == "raw_buttons"


def test_state_requirement_reports_natural_entry_mismatch() -> None:
    state = replace(
        parse_state(np.zeros(0x10000, dtype=np.uint8)),
        room_id=0x975C,
        phase=GameplayPhase.ORDINARY_GAMEPLAY,
        collected_items=MORPH_BALL_MASK,
        max_missiles=5,
        samus_x=700,
        samus_y=187,
    )
    requirement = StateRequirement(
        room_id=0x975C,
        phases=frozenset({GameplayPhase.ORDINARY_GAMEPLAY}),
        collected_items_mask=MORPH_BALL_MASK,
        minimum_ammo_capacities=(10, 0, 0),
        x_range=(692, 694),
        y_range=(187, 187),
    )

    failures = requirement.failures(state)

    assert any("ammo capacities" in failure for failure in failures)
    assert any("x 700" in failure for failure in failures)
