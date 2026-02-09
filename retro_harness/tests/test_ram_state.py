"""Tests for retro_harness.ram_state module."""
import numpy as np
import pytest

from retro_harness.ram_state import (
    RAMSchema,
    RAMWatcher,
    read_u8,
    read_u16,
    read_u16_be,
    read_s8,
    read_s16,
)


@pytest.fixture
def sample_ram():
    ram = np.zeros(256, dtype=np.uint8)
    ram[0x10] = 42          # u8
    ram[0x20] = 0xCD        # u16 LE low byte
    ram[0x21] = 0xAB        # u16 LE high byte -> 0xABCD
    ram[0x30] = 0xAB        # u16 BE high byte
    ram[0x31] = 0xCD        # u16 BE low byte -> 0xABCD
    ram[0x40] = 200         # s8 -> -56
    ram[0x50] = 0x01        # s16 LE -> 0x8001 = -32767
    ram[0x51] = 0x80
    return ram


class TestReaders:
    def test_read_u8(self, sample_ram):
        assert read_u8(sample_ram, 0x10) == 42

    def test_read_u16(self, sample_ram):
        assert read_u16(sample_ram, 0x20) == 0xABCD

    def test_read_u16_be(self, sample_ram):
        assert read_u16_be(sample_ram, 0x30) == 0xABCD

    def test_read_s8_negative(self, sample_ram):
        assert read_s8(sample_ram, 0x40) == -56

    def test_read_s8_positive(self, sample_ram):
        assert read_s8(sample_ram, 0x10) == 42

    def test_read_s16_negative(self, sample_ram):
        assert read_s16(sample_ram, 0x50) == -32767

    def test_read_s16_positive(self):
        ram = np.zeros(4, dtype=np.uint8)
        ram[0] = 0x39
        ram[1] = 0x05
        assert read_s16(ram, 0) == 0x0539


class TestRAMSchema:
    def test_basic_read(self, sample_ram):
        schema = RAMSchema({
            "val_u8": (0x10, "u8"),
            "val_u16": (0x20, "u16"),
        })
        result = schema.read(sample_ram)
        assert result == {"val_u8": 42, "val_u16": 0xABCD}

    def test_read_one(self, sample_ram):
        schema = RAMSchema({"val": (0x10, "u8")})
        assert schema.read_one(sample_ram, "val") == 42

    def test_fields(self):
        schema = RAMSchema({"a": (0, "u8"), "b": (1, "u16")})
        assert schema.fields == ["a", "b"]

    def test_from_dict(self, sample_ram):
        schema = RAMSchema.from_dict({"x": (0x10, "u8")})
        assert schema.read_one(sample_ram, "x") == 42

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown type"):
            RAMSchema({"bad": (0, "float32")})

    def test_all_types(self, sample_ram):
        schema = RAMSchema({
            "a": (0x10, "u8"),
            "b": (0x20, "u16"),
            "c": (0x30, "u16_be"),
            "d": (0x40, "s8"),
            "e": (0x50, "s16"),
        })
        result = schema.read(sample_ram)
        assert result == {"a": 42, "b": 0xABCD, "c": 0xABCD, "d": -56, "e": -32767}


class TestRAMWatcher:
    def test_first_call_no_changes(self, sample_ram):
        schema = RAMSchema({"val": (0x10, "u8")})
        watcher = RAMWatcher(schema)
        changes = watcher.update(sample_ram)
        assert changes == {}

    def test_detect_change(self, sample_ram):
        schema = RAMSchema({"val": (0x10, "u8")})
        watcher = RAMWatcher(schema)
        watcher.update(sample_ram)

        sample_ram[0x10] = 99
        changes = watcher.update(sample_ram)
        assert changes == {"val": (42, 99)}

    def test_no_change_no_report(self, sample_ram):
        schema = RAMSchema({"val": (0x10, "u8")})
        watcher = RAMWatcher(schema)
        watcher.update(sample_ram)
        changes = watcher.update(sample_ram)
        assert changes == {}
