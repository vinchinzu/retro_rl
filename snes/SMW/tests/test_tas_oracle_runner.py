"""Offline tests for isolated BizHawk oracle launch configuration."""

from __future__ import annotations

import json

from SMW.tas.oracle_runner import build_command, prepare_bizhawk_config


def test_prepare_config_isolates_snes_runtime_paths(tmp_path) -> None:
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "PreferredCores": {"SNES": "BSNES"},
                "PathEntries": {
                    "Paths": [
                        {"System": "SNES", "Type": "Save RAM", "Path": "old"},
                        {"System": "SNES", "Type": "Savestates", "Path": "old"},
                        {"System": "NES", "Type": "Save RAM", "Path": "keep"},
                    ]
                },
            }
        )
    )
    destination = tmp_path / "run" / "config.json"

    prepare_bizhawk_config(source, destination, tmp_path / "run")
    config = json.loads(destination.read_text())

    assert config["PreferredCores"]["SNES"] == "BSNES"
    assert config["RunLuaDuringTurbo"] is True
    assert config["PlayMovieMatchHash"] is True
    paths = config["PathEntries"]["Paths"]
    assert paths[0]["Path"].endswith("run/save_ram")
    assert paths[1]["Path"].endswith("run/states")
    assert paths[2]["Path"] == "keep"


def test_build_command_keeps_rom_last_and_userdata_explicit(tmp_path) -> None:
    paths = {
        name: tmp_path / name
        for name in ("bizhawk", "config", "movie", "script", "rom", "out")
    }
    command = build_command(
        bizhawk=paths["bizhawk"],
        config=paths["config"],
        movie=paths["movie"],
        lua=paths["script"],
        rom=paths["rom"],
        output_dir=paths["out"],
        target_levels=3,
        max_frames=12345,
    )

    assert command[:2] == ["xvfb-run", "-a"]
    assert command[-1] == str(paths["rom"].resolve())
    assert any("target_levels:3" in item for item in command)
    assert any("max_frames:12345" in item for item in command)
