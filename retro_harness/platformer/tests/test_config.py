"""Tests for level config and registry (no ROM needed)."""

import pytest
from retro_harness.platformer.level_config import (
    PlatformerRAM,
    LevelConfig,
    LEVEL_REGISTRY,
    get_level_config,
    list_levels,
    register_level,
)

# Import levels to trigger registration (game-owned LevelConfig packs)
import donkey_kong_country.platformer_levels  # noqa: F401
import smb.platformer_levels  # noqa: F401
import smb3.platformer_levels  # noqa: F401
import SMW.platformer_levels  # noqa: F401
import super_metroid.platformer_levels  # noqa: F401


def test_dkc_winkys_registered():
    config = get_level_config("dkc_winkys_walkway")
    assert config.display_name == "Winky's Walkway"
    assert config.game_name == "DonkeyKongCountry-Snes"
    assert config.target_level_id == 0xD9


def test_alias_lookup():
    c1 = get_level_config("winkys")
    c2 = get_level_config("dkc_winkys")
    c3 = get_level_config("dkc_winkys_walkway")
    assert c1 is c2 is c3


def test_unknown_level_raises():
    with pytest.raises(KeyError, match="Unknown level"):
        get_level_config("nonexistent_level")


def test_list_levels_deduplicates():
    levels = list_levels()
    ids = [l.level_id for l in levels]
    assert len(ids) == len(set(ids))


def test_ram_to_schema():
    ram = PlatformerRAM(
        camera_x=(0x00B2, "u16"),
        player_x=(0x00B4, "u16"),
        lives=(0x0575, "u8"),
    )
    schema = ram.to_schema()
    assert "camera_x" in schema.fields
    assert "player_x" in schema.fields
    assert "lives" in schema.fields
    # None fields should not appear
    assert "camera_y" not in schema.fields


def test_level_config_game_dir():
    config = get_level_config("winkys")
    assert config.game_dir.name == "donkey_kong_country"
    assert config.game_dir.is_absolute()


def test_level_config_runs_dir():
    config = get_level_config("winkys")
    assert "runs" in str(config.runs_dir)


def test_bk2_to_env_default():
    config = get_level_config("winkys")
    assert config.bk2_to_env == [11 - i for i in range(12)]
    assert config.selftest_expect_death is True


# -- Super Metroid tests -----------------------------------------------------


def test_sm_alias_lookup():
    c1 = get_level_config("climb_return")
    c2 = get_level_config("sm_climb_up")
    c3 = get_level_config("sm_climb_return")
    assert c1 is c2 is c3


def test_sm_landing_site_registered():
    config = get_level_config("sm_landing_site")
    assert config.progress_axis == "waypoints"  # uses auto-generated waypoints
    assert len(config.waypoints) >= 3  # multi-screen room needs waypoints
    assert config.death_signals == ["health_zero"]
    assert config.selftest_expect_death is False


def test_sm_all_12_segments_registered():
    sm_ids = [
        "sm_landing_site", "sm_parlor_descent", "sm_climb_descent",
        "sm_pit_room_descent", "sm_elevator_descent", "sm_morph_ball_collect",
        "sm_morph_ball_return", "sm_elevator_return", "sm_pit_room_return",
        "sm_climb_return", "sm_parlor_to_flyway", "sm_flyway_to_torizo",
    ]
    for level_id in sm_ids:
        config = get_level_config(level_id)
        assert config.game_name == "SuperMetroid-Snes"


def test_sm_ram_schema():
    config = get_level_config("sm_climb_return")
    schema = config.ram_schema
    assert "player_x" in schema.fields
    assert "player_y" in schema.fields
    assert "level_id" in schema.fields
    assert "health" in schema.fields
