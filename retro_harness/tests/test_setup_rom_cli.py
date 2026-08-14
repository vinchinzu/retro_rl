"""Tests for retro_harness.setup_rom_cli."""

from __future__ import annotations

from pathlib import Path

from retro_harness.setup_rom_cli import main_setup_rom, setup_and_print


def test_setup_and_print_calls_setup_game_rom(tmp_path: Path, monkeypatch, capsys) -> None:
    rom = tmp_path / "game.nes"
    rom.write_bytes(b"NES")

    def fake_setup(**kwargs):
        assert kwargs["shared_zip"] == tmp_path / "rom.zip"
        assert kwargs["game_dir"] == tmp_path
        assert kwargs["integration_name"] == "Demo-Nes"
        return rom

    monkeypatch.setattr("retro_harness.setup_rom_cli.setup_game_rom", fake_setup)
    out = setup_and_print(
        shared_zip=tmp_path / "rom.zip",
        game_dir=tmp_path,
        integration_name="Demo-Nes",
    )
    assert out == rom
    assert "ROM ready:" in capsys.readouterr().out
    assert main_setup_rom(
        shared_zip=tmp_path / "rom.zip",
        game_dir=tmp_path,
        integration_name="Demo-Nes",
    ) == 0
