"""Clean-track infra: artifact stems, integrity flags, assist-off behavior."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from super_metroid.assist import UnlimitedAmmoAssist, UnlimitedResourcesAssist
from super_metroid.ram import GameplayPhase, parse_state
from super_metroid.routes.continuous import (
    default_artifact_paths,
    default_tip_artifact_paths,
    default_tip_room_timing_path,
)
from super_metroid.routes.runtime import (
    Split,
    assist_integrity,
    clean_artifact_stem,
    default_artifacts,
    evaluate_integrity,
    resolve_clean_resources,
    resource_writes_zero,
)


class FakeData:
    def __init__(self) -> None:
        self.writes: list[tuple[str, int]] = []

    def set_value(self, key: str, value: int) -> None:
        self.writes.append((key, value))


def _state():
    ram = np.zeros(0x10000, dtype=np.uint8)
    return parse_state(ram)


def test_clean_artifact_stem_appends_once() -> None:
    assert clean_artifact_stem("bombs") == "bombs_clean"
    assert (
        clean_artifact_stem("bombs_clean")
        == "bombs_clean"
    )


def test_resolve_clean_resources_from_assist_flags() -> None:
    assert resolve_clean_resources(unlimited_energy=True, unlimited_ammo=True) is False
    assert resolve_clean_resources(unlimited_energy=False, unlimited_ammo=True) is False
    assert resolve_clean_resources(unlimited_energy=True, unlimited_ammo=False) is False
    assert resolve_clean_resources(unlimited_energy=False, unlimited_ammo=False) is True
    # Explicit override wins over assist flags.
    assert (
        resolve_clean_resources(
            unlimited_energy=True,
            unlimited_ammo=True,
            require_clean_resources=True,
        )
        is True
    )
    assert (
        resolve_clean_resources(
            unlimited_energy=False,
            unlimited_ammo=False,
            require_clean_resources=False,
        )
        is False
    )


def test_default_artifacts_assisted_vs_clean() -> None:
    video, report = default_artifacts("bombs")
    assert video.name == "bombs.mp4"
    assert report.name == "bombs.json"

    c_video, c_report = default_artifacts("bombs", clean=True)
    assert c_video.name == "bombs_clean.mp4"
    assert c_report.name == "bombs_clean.json"
    assert c_video.name != video.name
    assert c_report.name != report.name


def test_default_tip_artifact_paths_clean_isolated() -> None:
    video, report = default_tip_artifact_paths("bombs")
    assert video.name == "bombs.mp4"
    assert report.name == "bombs.json"

    c_video, c_report = default_tip_artifact_paths("bombs", clean=True)
    assert c_video.name == "bombs_clean.mp4"
    assert c_report.name == "bombs_clean.json"

    # Primary assisted default tip stems (verified Wave Beam tip).
    tip_v, tip_r = default_artifact_paths()
    assert tip_v.name == "wave.mp4"
    assert tip_r.name == "wave.json"
    clean_v, clean_r = default_artifact_paths(clean=True)
    assert clean_v.name == "wave_clean.mp4"
    assert clean_r.name == "wave_clean.json"


def test_default_tip_room_timing_clean_stem() -> None:
    normal = default_tip_room_timing_path("supers")
    assert normal.name == "supers_room_timing.json"
    clean = default_tip_room_timing_path("supers", clean=True)
    assert clean.name == "supers_clean_room_timing.json"


def test_resource_writes_zero_when_assists_disabled() -> None:
    assist = UnlimitedResourcesAssist(
        unlimited_energy=False, unlimited_ammo=False
    )
    data = FakeData()
    state = replace(
        _state(),
        frame=10,
        phase=GameplayPhase.ORDINARY_GAMEPLAY,
        health=40,
        max_health=99,
        missiles=1,
        max_missiles=5,
    )
    assist.apply(data, state)
    assert data.writes == []
    flags = resource_writes_zero(assist)
    assert all(flags.values())
    integrity = assist_integrity(assist, require_clean_resources=True)
    assert integrity["clean_resources_zero"] is True
    assert integrity["energy_writes_zero"] is True
    assert integrity["missiles_writes_zero"] is True


def test_resource_writes_zero_fails_when_ammo_refilled() -> None:
    assist = UnlimitedAmmoAssist(enabled=True)
    data = FakeData()
    state = replace(
        _state(),
        frame=10,
        phase=GameplayPhase.ORDINARY_GAMEPLAY,
        missiles=1,
        max_missiles=5,
    )
    assist.apply(data, state)
    assert data.writes == [("missiles", 5)]
    integrity = assist_integrity(assist, require_clean_resources=True)
    assert integrity["missiles_writes_zero"] is False
    assert integrity["clean_resources_zero"] is False


def test_evaluate_integrity_clean_mode_requires_zero_resource_writes() -> None:
    assist = UnlimitedAmmoAssist(enabled=True)
    data = FakeData()
    state = replace(
        _state(),
        frame=10,
        phase=GameplayPhase.ORDINARY_GAMEPLAY,
        missiles=1,
        max_missiles=5,
    )
    assist.apply(data, state)

    success, integrity = evaluate_integrity(
        required_splits=("morph_ball",),
        splits=[Split("morph_ball", 100, 0x9E9F)],
        transitions=[],
        final_conditions={"ok": True},
        assist=assist,
        video_evidence_payload=None,
        require_transitions=False,
        require_clean_resources=True,
    )
    assert success is False
    assert integrity["clean_resources_zero"] is False

    assist_off = UnlimitedAmmoAssist(enabled=False)
    success_ok, integrity_ok = evaluate_integrity(
        required_splits=("morph_ball",),
        splits=[Split("morph_ball", 100, 0x9E9F)],
        transitions=[],
        final_conditions={"ok": True},
        assist=assist_off,
        video_evidence_payload=None,
        require_transitions=False,
        require_clean_resources=True,
    )
    assert success_ok is True
    assert integrity_ok["clean_resources_zero"] is True


def test_assisted_integrity_does_not_require_clean_resources() -> None:
    assist = UnlimitedAmmoAssist(enabled=True)
    data = FakeData()
    state = replace(
        _state(),
        frame=10,
        phase=GameplayPhase.ORDINARY_GAMEPLAY,
        missiles=1,
        max_missiles=5,
    )
    assist.apply(data, state)

    success, integrity = evaluate_integrity(
        required_splits=("morph_ball",),
        splits=[Split("morph_ball", 100, 0x9E9F)],
        transitions=[],
        final_conditions={"ok": True},
        assist=assist,
        video_evidence_payload=None,
        require_transitions=False,
        require_clean_resources=False,
    )
    assert success is True
    assert "clean_resources_zero" not in integrity


def test_run_to_accepts_clean_flags_on_early_tips() -> None:
    """Morph/bombs accept clean kwargs; capability flags gate optional args."""
    import inspect

    from super_metroid.routes.continuous import run_bombs, run_morph, run_to

    for fn in (run_morph, run_bombs, run_to):
        sig = inspect.signature(fn)
        assert "unlimited_energy" in sig.parameters
        assert "unlimited_ammo" in sig.parameters
        assert "require_clean_resources" in sig.parameters

    # Room-timing still rejected on morph via ContinuousTip capability flags
    # (no inspect.signature filtering on the runner itself).
    with pytest.raises(ValueError, match="room timing"):
        run_to(
            "morph",
            room_timing_path="/tmp/x.json",
            unlimited_energy=False,
            unlimited_ammo=False,
        )
