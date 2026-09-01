"""Scratch: decode the raw WRAM windows written by ``tas_wram`` / ``our_wram``.

Field names beyond ``super_metroid.ram`` are the ones the wall-jump contract
needs: hitbox radii ($0AFE/$0B00) and the direct-page controller latches.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

SLICE_BASE = {"dp": 0x0000, "samus": 0x0A00, "room": 0x0780, "gs": 0x0990}

# SNES joypad bitmask, high byte first (B Y sel start up down left right / A X L R).
BTN_BITS = {
    "B": 0x8000, "Y": 0x4000, "SELECT": 0x2000, "START": 0x1000,
    "UP": 0x0800, "DOWN": 0x0400, "LEFT": 0x0200, "RIGHT": 0x0100,
    "A": 0x0080, "X": 0x0040, "L": 0x0020, "R": 0x0010,
}


class Ram:
    """Sparse WRAM view stitched from the captured slices."""

    def __init__(self, blobs: dict[str, str]) -> None:
        self.parts = {
            name: (SLICE_BASE[name], bytes.fromhex(hx)) for name, hx in blobs.items()
        }

    def u8(self, addr: int) -> int:
        for base, buf in self.parts.values():
            if base <= addr < base + len(buf):
                return buf[addr - base]
        raise KeyError(hex(addr))

    def u16(self, addr: int) -> int:
        return self.u8(addr) | (self.u8(addr + 1) << 8)

    def s16(self, addr: int) -> int:
        v = self.u16(addr)
        return v - 0x10000 if v & 0x8000 else v


def fields(ram: Ram) -> dict:
    return {
        "room": ram.u16(0x079B),
        "gs": ram.u16(0x0998),
        "pose": ram.u16(0x0A1C),
        "face": ram.u8(0x0A1E),
        "mt": ram.u8(0x0A1F),
        "a94": ram.u8(0x0A94),
        "a96": ram.u16(0x0A96),
        "a28": ram.u16(0x0A28),
        "x": ram.u16(0x0AF6),
        "xs": ram.u16(0x0AF8),
        "y": ram.u16(0x0AFA),
        "ys": ram.u16(0x0AFC),
        "xr": ram.u16(0x0AFE),
        "yr": ram.u16(0x0B00),
        "vy": ram.u16(0x0B2E),
        "vys": ram.u16(0x0B2C),
        "vd": ram.u16(0x0B36),
        "vx": ram.u16(0x0B42),
        "vxs": ram.u16(0x0B44),
        "mx": ram.u16(0x0B46),
        "mxs": ram.u16(0x0B48),
    }


def btn_mask(names: list[str]) -> int:
    return sum(BTN_BITS[n] for n in names if n in BTN_BITS)


def load(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    out = []
    for row in data["rows"]:
        ram = Ram(row["ram"])
        rec = {"f": row["f"], "btn": row["btn"], **fields(ram)}
        rec["_ram"] = ram
        out.append(rec)
    return out


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "tas_wram.json"
    rows = load(path)
    lo = int(sys.argv[2]) if len(sys.argv) > 2 else rows[0]["f"]
    hi = int(sys.argv[3]) if len(sys.argv) > 3 else rows[-1]["f"]
    print(f"{'f':>6} {'room':>6} {'x':>4}.{'':2} {'y':>4} {'pose':>4} "
          f"{'mt':>3} {'fc':>3} {'xr':>3} {'yr':>3} {'vd':>3} {'vy':>5} "
          f"{'vys':>5} {'a94':>4} {'a96':>4} {'a28':>6} buttons")
    for r in rows:
        if not (lo <= r["f"] <= hi):
            continue
        print(
            f"{r['f']:6d} {r['room']:#06x} {r['x']:4d}.{r['xs'] >> 8:02d} "
            f"{r['y']:4d} {r['pose']:4d} {r['mt']:3d} {r['face']:3d} "
            f"{r['xr']:3d} {r['yr']:3d} {r['vd']:3d} "
            f"{r['vy']:5d} {r['vys']:5d} {r['a94']:4d} {r['a96']:4d} {r['a28']:6d} "
            f"{' '.join(r['btn'])}"
        )


if __name__ == "__main__":
    main()
