"""Unit and opt-in ROM coverage for the first-item policy product."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sm_rando import morph_policy
from sm_rando.paths import GAME, GAME_DIR
from super_metroid.routes.kpdr.room_ids import ROOM_MORPH


def test_morph_policy_targets_sm_rando_integration_and_clean_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls: dict[str, object] = {}
    sentinel = object()
    integration_dir = tmp_path / "SMRando-Snes"
    integration_dir.mkdir()
    (integration_dir / "rom.sfc").touch()

    def fake_make_env(*args, **kwargs):
        calls["make_env"] = (args, kwargs)
        return sentinel

    def fake_run_tip(tip_id, **kwargs):
        calls["run_tip"] = (tip_id, kwargs)
        assert kwargs["env_factory"]() is sentinel
        return SimpleNamespace(success=True)

    monkeypatch.setattr(morph_policy, "make_env", fake_make_env)
    monkeypatch.setattr(morph_policy, "run_tip", fake_run_tip)
    monkeypatch.setattr(morph_policy, "INTEGRATION_DIR", integration_dir)

    report = morph_policy.run_morph_policy(
        video_path="proof.mp4",
        report_path="proof.json",
    )

    assert report.success is True
    assert calls["make_env"] == (
        (GAME, "NONE", GAME_DIR),
        {"render_mode": "rgb_array"},
    )
    tip_id, kwargs = calls["run_tip"]
    assert tip_id == "morph"
    assert kwargs["rom_path"] == integration_dir / "rom.sfc"
    assert kwargs["video_path"] == "proof.mp4"
    assert kwargs["report_path"] == "proof.json"
    assert kwargs["unlimited_energy"] is False
    assert kwargs["unlimited_ammo"] is False
    assert kwargs["require_clean_resources"] is True


@pytest.mark.rom
@pytest.mark.rom_smoke
def test_sm_rando_policy_reaches_first_item(tmp_path) -> None:
    import stable_retro as retro

    if not hasattr(retro.data.Integrations, "CUSTOM"):
        pytest.skip("stable_retro test stub cannot execute ROM smoke")
    report = morph_policy.run_morph_policy(
        video_path=None,
        report_path=tmp_path / "policy_to_morph.json",
    )

    assert report.success is True
    assert report.outcome == "morph_ball_acquired"
    assert report.final_state["room_id"] == ROOM_MORPH
    assert report.final_state["morph_ball"] is True
    assert report.state_loads == 0
    assert report.progression_writes == 0
