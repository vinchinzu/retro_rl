"""Harvest editor snapshot decode helpers without stable-retro imports."""

from __future__ import annotations

import base64

import numpy as np

from harvest.paths import GAME, PROJECT_DIR, STATES_DIR

ADDR_X = 0x00D6
ADDR_Y = 0x00D8
ADDR_TOOL = 0x0921
ADDR_TILEMAP = 0x0022
GAME_ID = GAME
MAP_NAMES = {
    0x00: "Farm",
    0x01: "Farm",
    0x02: "Farm",
    0x03: "Farm",
    0x0C: "Path",
    0x04: "Town",
    0x1C: "Shop",
    0x24: "Animal Shop",
    0x15: "House",
    0x16: "House L1",
    0x17: "House L2",
    0x18: "Shed",
    0x19: "Barn",
    0x1A: "Coop",
    0x26: "Shed",
    0x27: "Barn",
    0x28: "Coop",
}
CAPTURE_DIR = PROJECT_DIR / "debug_alignment" / "editor_exports"
HOT_SAVE_PATH = STATES_DIR / "editor_hot.state"
DEFAULT_STATE_PATH = STATES_DIR / "latest.state"


def map_name(tilemap_id: int) -> str:
    return MAP_NAMES.get(tilemap_id, f"0x{tilemap_id:02X}")


def decode_wram(snapshot: dict[str, object], *, copy: bool = False) -> np.ndarray:
    raw = snapshot.get("wramRaw")
    if isinstance(raw, (bytes, bytearray, memoryview)):
        view = np.frombuffer(raw, dtype=np.uint8)
        return view.copy() if copy else view
    encoded = snapshot.get("wramBase64")
    if not isinstance(encoded, str):
        return np.zeros(0, dtype=np.uint8)
    view = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
    return view.copy() if copy else view


def snapshot_obs(snapshot: dict[str, object]) -> np.ndarray | None:
    raw = snapshot.get("frameRgb24Raw")
    if raw is None:
        encoded = snapshot.get("frameRgb24Base64")
        if not isinstance(encoded, str):
            return None
        raw = base64.b64decode(encoded)
    else:
        raw = bytes(raw)
    width = int(snapshot.get("frameWidth") or 256)
    height = int(snapshot.get("frameHeight") or 224)
    expected = width * height * 3
    if len(raw) != expected:
        return None
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()


def tilemap_id_from_ram(ram: bytes) -> int:
    if ADDR_TILEMAP < len(ram):
        return int(ram[ADDR_TILEMAP])
    return 0
