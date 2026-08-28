"""Practice start-manifest extract: category order + Save Station records."""

from __future__ import annotations

from pathlib import Path

import pytest

from super_metroid.paths import SHARED_PRACTICE_ROM
from super_metroid.practice_repertoire.export_catalog import (
    parse_data_file,
    parse_menu_file,
)
from super_metroid.practice_repertoire.source_extract import (
    parse_teleports,
    validate_category_order,
)

_STEMS = ("kpdr20", "kpdr25", "prkd19")


def _mainmenu(tmp_path: Path, *, stems: tuple[str, ...] = _STEMS, extra: str = "") -> Path:
    entries = "\n".join(f"        dw #PresetsMenu{stem.title()}" for stem in stems)
    text = (
        "preset_category_submenus:\n"
        "{\n"
        f"{entries}\n"
        "        dw #$0000\n"
        "}\n"
        f"{extra}\n"
    )
    path = tmp_path / "mainmenu.asm"
    path.write_text(text, encoding="utf-8")
    return path


def test_validate_category_order_ok(tmp_path: Path) -> None:
    path = _mainmenu(tmp_path)
    validate_category_order(path, _STEMS)


def test_validate_category_order_drift_fails(tmp_path: Path) -> None:
    path = _mainmenu(tmp_path, stems=("kpdr25", "kpdr20", "prkd19"))
    with pytest.raises(ValueError, match="category inventory drifted"):
        validate_category_order(path, _STEMS)


def test_validate_category_order_missing_block(tmp_path: Path) -> None:
    path = tmp_path / "mainmenu.asm"
    path.write_text("no category block here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="preset_category_submenus missing"):
        validate_category_order(path, _STEMS)


def _teleport_menu(tmp_path: Path) -> Path:
    """52 Save Stations actions: 7 areas, last area gets 10 stations."""

    lines = ["preset_category_submenus:\n{\n        dw #PresetsMenuKpdr20\n        dw #$0000\n}\n"]
    counts = (7, 7, 7, 7, 7, 7, 10)
    n = 0
    for area, count in enumerate(counts):
        for station in range(count):
            n += 1
            selector = (area << 8) | station
            lines.append(
                f'tel_area{area}_s{station}:\n'
                f'        %cm_jsl("Station {n}", #action_teleport, #${selector:04X})\n'
            )
    assert n == 52
    path = tmp_path / "mainmenu.asm"
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _station_rom(tmp_path: Path) -> Path:
    """Headerless LoROM: pointer table at $80:C4B5 → file 0x44B5."""

    rom = bytearray(0x10000)
    # Area pointers: each area's records start at 0x5000 + area * 0x100, LoROM bit set.
    for area in range(7):
        pointer = 0x8000 | (0x5000 + area * 0x100)
        off = 0x44B5 + 2 * area
        rom[off : off + 2] = pointer.to_bytes(2, "little")
        for station in range(10):
            rec = 0x5000 + area * 0x100 + 14 * station
            room = 0x91F8 + area
            ddb = 0x1800 + station
            words = (room, ddb, 0, 0x0100, 0x0200, 0x0030, 0x0040)
            for i, word in enumerate(words):
                rom[rec + 2 * i : rec + 2 * i + 2] = word.to_bytes(2, "little")
    path = tmp_path / "practice.sfc"
    path.write_bytes(bytes(rom))
    return path


def test_parse_teleports_load_station_records(tmp_path: Path) -> None:
    menu = _teleport_menu(tmp_path)
    rom = _station_rom(tmp_path)
    rows = parse_teleports(menu, rom)
    assert len(rows) == 52
    first = rows[0]
    assert first["id"] == "teleport/crateria/station_1"
    assert first["kind"] == "teleport"
    assert first["parameterized_state"] is True
    assert first["room_id"] == 0x91F8
    assert first["ddb"] == 0x1800
    assert first["x"] == (0x0100 + 0x80 + 0x0040) & 0xFFFF
    assert first["y"] == (0x0200 + 0x0030) & 0xFFFF
    last = rows[-1]
    assert last["area"] == "ceres"
    assert last["station_index"] == 9
    assert last["room_id"] == 0x91F8 + 6
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids))


def test_parse_teleports_rejects_wrong_count(tmp_path: Path) -> None:
    path = tmp_path / "mainmenu.asm"
    path.write_text(
        "preset_category_submenus:\n{\n        dw #$0000\n}\n"
        'tel_only:\n        %cm_jsl("Only", #action_teleport, #$0000)\n',
        encoding="utf-8",
    )
    rom = _station_rom(tmp_path)
    with pytest.raises(ValueError, match="expected 52 Save Stations"):
        parse_teleports(path, rom)


def test_parse_data_file_resolves_parent(tmp_path: Path) -> None:
    path = tmp_path / "foo_data.asm"
    path.write_text(
        "preset_foo_root:\n"
        "    dw #$0000\n"
        "    dw $079B, $9E9F\n"
        "    dw $0AF6, $0080\n"
        "    dw #$FFFF\n"
        "\n"
        "preset_foo_child:\n"
        "    dw #preset_foo_root\n"
        "    dw $0AF6, $0100\n"
        "    dw #$FFFF\n",
        encoding="utf-8",
    )
    definitions = parse_data_file(path)
    assert definitions["preset_foo_root"]["parent"] is None
    assert definitions["preset_foo_root"]["room_id"] == 0x9E9F
    assert definitions["preset_foo_root"]["x"] == 0x0080
    assert definitions["preset_foo_child"]["parent"] == "preset_foo_root"
    assert definitions["preset_foo_child"]["room_id"] == 0x9E9F
    assert definitions["preset_foo_child"]["x"] == 0x0100


def test_parse_menu_file_data_label(tmp_path: Path) -> None:
    path = tmp_path / "foo_menu.asm"
    path.write_text(
        "PresetsMenuFoo:\n"
        "    dw #presets_goto_foo_crateria\n"
        "    dw #$0000\n"
        '    %cm_header("FOO")\n'
        "\n"
        "presets_goto_foo_crateria:\n"
        '    %cm_submenu("Crateria", #presets_submenu_foo_crateria)\n'
        "\n"
        "presets_submenu_foo_crateria:\n"
        "    dw #presets_foo_crateria_morph\n"
        "    dw #$0000\n"
        "\n"
        "presets_foo_crateria_morph:\n"
        '    %cm_preset("Morph", #preset_names_morph, #preset_foo_crateria_morph)\n',
        encoding="utf-8",
    )
    title, areas, sessions = parse_menu_file(path, "foo")
    assert title == "FOO"
    assert areas[0]["id"] == "crateria"
    assert sessions[0]["id"] == "foo/crateria/morph"
    assert sessions[0]["data_label"] == "preset_foo_crateria_morph"
    assert sessions[0]["name"] == "Morph"


@pytest.mark.skipif(
    not SHARED_PRACTICE_ROM.is_file(),
    reason="practice ROM not built (run setup_practice_rom.py)",
)
def test_live_practice_rom_has_52_teleports(tmp_path: Path) -> None:
    # Live check needs upstream mainmenu; skip if the exporter cache is absent.
    cache = Path("/tmp/sm_practice/presets")
    mainmenu = cache.parent / "mainmenu.asm"
    if not mainmenu.is_file():
        mainmenu = cache / "mainmenu.asm"
    if not mainmenu.is_file():
        pytest.skip("practice-hack mainmenu.asm cache missing")
    rows = parse_teleports(mainmenu, SHARED_PRACTICE_ROM)
    assert len(rows) == 52
    assert all(row["kind"] == "teleport" for row in rows)
