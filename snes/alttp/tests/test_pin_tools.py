"""Development pins preserve their source and retain auditable provenance."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from alttp.scripts import audit_pins, scan_tape, whatif_pin
from alttp.spine.tape import read_pin_provenance, write_pin_provenance
from retro_harness.env import read_state_bytes, state_path, write_state_bytes


@pytest.mark.parametrize(
    "alias", ["same", "suffix", "destination_symlink", "source_symlink"]
)
def test_whatif_rejects_source_alias_before_writes(tmp_path, monkeypatch, alias) -> None:
    monkeypatch.setattr(whatif_pin, "GAME_DIR", tmp_path)
    source = "Whatif_baseline"
    destination = source
    path = state_path(tmp_path, whatif_pin.GAME, source)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"original state")
    sidecar = write_pin_provenance(path, source_run="baseline", frame=42, kind="whatif")
    original_provenance = sidecar.read_bytes()
    if alias == "suffix":
        source += ".state"
    elif alias in ("destination_symlink", "source_symlink"):
        alias_name = "Whatif_alias"
        state_path(tmp_path, whatif_pin.GAME, alias_name).symlink_to(path)
        if alias == "destination_symlink":
            destination = alias_name
        else:
            source = alias_name

    def unexpected_write(*args):
        pytest.fail("source alias reached inventory writes")

    monkeypatch.setattr(whatif_pin, "assign_field", unexpected_write)
    with pytest.raises(SystemExit, match="source"):
        whatif_pin.write_whatif(
            object(), source, [whatif_pin.parse_set("sword=2")], destination
        )
    assert path.read_bytes() == b"original state"
    assert sidecar.read_bytes() == original_provenance


def test_scan_pin_is_classified_by_auditor(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(audit_pins, "GAME_DIR", tmp_path)
    monkeypatch.setattr(scan_tape, "describe", lambda env: ["settled room 0x81"])
    name = "Scan_0x81"
    path = state_path(tmp_path, audit_pins.GAME, name)
    path.parent.mkdir(parents=True)
    env = SimpleNamespace(em=SimpleNamespace(get_state=lambda: b"scan state"))
    scan_tape._save_pin(
        env,
        path.parent,
        name,
        {"tape": "escape.tape.npz", "frame": 42, "location": ["in", 0x81], "pin": name},
    )
    assert read_state_bytes(path) == b"scan state"
    assert audit_pins.provenance_kind(name) == "scan"
    data = read_pin_provenance(path)
    assert data is not None
    assert data["source_run"] == "scan:escape.tape.npz"
    assert data["frame"] == 42
    assert data["tape"] == "escape.tape.npz"
    assert data["location"] == ["in", 0x81]
    assert data["pin"] == name
    assert data["git_sha"]
    assert isinstance(data["dirty"], bool)


@pytest.mark.rom
def test_whatif_copy_preserves_source_on_rom(tmp_path, monkeypatch) -> None:
    from alttp.ram import read_inventory, read_snapshot

    env = whatif_pin.open_pin("FighterSword")
    monkeypatch.setattr(whatif_pin, "GAME_DIR", tmp_path)
    source = "Whatif_baseline"
    path = state_path(tmp_path, whatif_pin.GAME, source)
    try:
        write_state_bytes(path, env.em.get_state())
        sidecar = write_pin_provenance(path, source_run="baseline", frame=0, kind="whatif")
        original_state = path.read_bytes()
        original_provenance = sidecar.read_bytes()
        original_ram = np.asarray(env.get_ram(), dtype=np.uint8).copy()
        sets = [whatif_pin.parse_set("sword=2")]
        with pytest.raises(SystemExit, match="source"):
            whatif_pin.write_whatif(env, source, sets, source)
        np.testing.assert_array_equal(env.get_ram(), original_ram)
        whatif_pin.write_whatif(env, source, sets, "Whatif_copy")
        assert read_inventory(env.get_ram()).sword == 2
        assert read_snapshot(env.get_ram()).follower == read_snapshot(original_ram).follower
        assert path.read_bytes() == original_state
        assert sidecar.read_bytes() == original_provenance
        provenance = read_pin_provenance(state_path(tmp_path, whatif_pin.GAME, "Whatif_copy"))
        assert provenance["kind"] == "whatif"
    finally:
        env.close()


@pytest.mark.rom
def test_scan_replayed_rom_tape_has_auditable_pins(tmp_path, monkeypatch) -> None:
    from alttp.paths import BOOT_STATE, INTEGRATION_DIR
    from alttp.spine.tape import ButtonTape, ram_crc, sha1_file
    from alttp.startup import (
        build_blank_sram, build_boot_env, create_fresh_profile,
        inject_sram_into_env, load_fresh_profile_slot_one, no_action,
        open_file_select, snapshot_env, wait_for_title_screen,
    )

    env = build_boot_env(render_mode=None)
    tape = ButtonTape(
        boot_state=BOOT_STATE, rom_sha1=sha1_file(INTEGRATION_DIR / "rom.sfc")
    )
    crcs = []
    try:
        env.reset()
        inject_sram_into_env(env, build_blank_sram())
        original_step = env.step

        def record_step(action):
            result = original_step(action)
            tape.record(action)
            crcs.append(ram_crc(env.get_ram()))
            return result

        env.step = record_step
        assert wait_for_title_screen(env).phase == "title"
        open_file_select(env)
        create_fresh_profile(env)
        load_fresh_profile_slot_one(env)
        assert snapshot_env(env).has_control
        env.step(no_action())
        path = tape.save("scan_test", ram=env.get_ram(), directory=tmp_path)
    finally:
        env.close()
    with np.load(path, allow_pickle=False) as packed:
        arrays = {key: packed[key] for key in packed.files}
    meta = json.loads(str(arrays["meta"].item()))
    meta["sram_blank_at_boot"] = True
    arrays["meta"] = np.asarray(json.dumps(meta))
    arrays["frame_crc"] = np.asarray(crcs, dtype=np.uint32)
    np.savez_compressed(path, **arrays)

    monkeypatch.setattr(audit_pins, "GAME_DIR", tmp_path)
    out = state_path(tmp_path, audit_pins.GAME, "unused").parent
    report = scan_tape.scan_tape(path, out)
    assert report["ok"]
    assert report["played"] == len(crcs)
    assert report["loads"] == report["writes"] == report["desynced"] == 0
    assert report["pins"]
    for row in report["pins"]:
        assert audit_pins.provenance_kind(row["pin"]) == "scan"
        data = read_pin_provenance(out / f"{row['pin']}.state")
        assert data["tape"] == path.name
        assert data["frame"] == row["frame"]
        assert data["location"] == row["location"]
