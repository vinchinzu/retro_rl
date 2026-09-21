"""Parse Zelda Dungeon overworld walkthrough maps (16×8, cyan overlay).

Map-1.png is The Gathering §1.1: start 0x77 to the coast bomb cave 0x6F.
The cyan line is the screen sequence. Live 0x79 east is the south beach
(``overworld.shop_p7``), not this overlay's centre lane (the rocky bowl).

https://www.zeldadungeon.net/Zelda01/Walkthrough/01/Map-1.png
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from zelda_i.overworld.common import (
    EDGE_EAST_X,
    EDGE_NORTH_Y,
    EDGE_SOUTH_Y,
    EDGE_WEST_X,
)
from zelda_i.overworld.graph import (
    OVERWORLD_COLS,
    OVERWORLD_ROWS,
    SCREEN_START,
    ScreenHop,
    direction_between,
    screen_id,
)
from zelda_i.paths import GAME_DIR

MAP1_URL = "https://www.zeldadungeon.net/Zelda01/Walkthrough/01/Map-1.png"
MAP1_PATH = GAME_DIR / "refs" / "zd" / "Map-1.png"

# Derived once from the cyan overlay with :func:`parse_overworld_map`.  Keep
# the small route datum in source control instead of requiring the ignored,
# third-party PNG at runtime.  ``MAP1_PATH`` and the parser remain available
# for manually checking a newly downloaded copy of the walkthrough image.
_MAP1_SCREENS = (0x77, 0x78, 0x79, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E, 0x7F, 0x6F)
_MAP1_LANES = {
    0x6F: (107, 160),
    0x77: (165, 133),
    0x78: (121, 130),
    0x79: (118, 131),
    0x7A: (117, 131),
    0x7B: (132, 131),
    0x7C: (121, 130),
    0x7D: (123, 130),
    0x7E: (123, 130),
    0x7F: (82, 118),
}

_CYAN_FRAC = 0.02
_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class PaintedRoute:
    """Ordered screens along a ZD cyan overlay, plus hop table and lanes."""

    screens: tuple[int, ...]
    hops: tuple[ScreenHop, ...]
    dest: int
    lanes: dict[int, tuple[int, int]]
    source: str

    def hops_from(self, screen: int) -> tuple[ScreenHop, ...]:
        """Hops that leave ``screen`` along the painted path (not the arrival)."""
        i = self.screens.index(screen)
        return self.hops[i:]


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def read_png_rgb(path: Path) -> np.ndarray:
    """8-bit RGB/RGBA PNG → HxWx3 uint8. Stdlib only (no display)."""
    with path.open("rb") as fh:
        if fh.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"not a PNG: {path}")
        width = height = color_type = None
        idat = bytearray()
        while True:
            n_raw = fh.read(4)
            if len(n_raw) < 4:
                break
            (n,) = struct.unpack(">I", n_raw)
            ctype = fh.read(4)
            data = fh.read(n)
            fh.read(4)
            if ctype == b"IHDR":
                width, height, bit_depth, color_type, _c, _f, inter = struct.unpack(
                    ">IIBBBBB", data
                )
                if bit_depth != 8 or color_type not in (2, 6) or inter != 0:
                    raise ValueError(
                        f"unsupported PNG {bit_depth}/{color_type}/{inter}"
                    )
            elif ctype == b"IDAT":
                idat.extend(data)
            elif ctype == b"IEND":
                break
    if width is None or height is None or color_type is None:
        raise ValueError(f"no IHDR: {path}")
    raw = zlib.decompress(bytes(idat))
    bpp = 3 if color_type == 2 else 4
    stride = width * bpp
    rows: list[bytes] = []
    i = 0
    prev = bytearray(stride)
    for _ in range(height):
        filt = raw[i]
        i += 1
        row = bytearray(raw[i : i + stride])
        i += stride
        if filt == 1:
            for x in range(stride):
                row[x] = (row[x] + (row[x - bpp] if x >= bpp else 0)) & 255
        elif filt == 2:
            for x in range(stride):
                row[x] = (row[x] + prev[x]) & 255
        elif filt == 3:
            for x in range(stride):
                left = row[x - bpp] if x >= bpp else 0
                row[x] = (row[x] + ((left + prev[x]) // 2)) & 255
        elif filt == 4:
            for x in range(stride):
                left = row[x - bpp] if x >= bpp else 0
                up = prev[x]
                ul = prev[x - bpp] if x >= bpp else 0
                row[x] = (row[x] + _paeth(left, up, ul)) & 255
        elif filt != 0:
            raise ValueError(f"PNG filter {filt}")
        rows.append(bytes(row))
        prev = row
    rgb = np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(height, width, bpp)
    return np.ascontiguousarray(rgb[:, :, :3])


def _cyan_mask(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    # Overlay is vivid teal, not the darker ocean.
    return (g > 140) & (b > 140) & (r < 120) & (g > r + 40) & (b > r + 40)


def _nes_xy(fx: float, fy: float) -> tuple[int, int]:
    x = int(round(EDGE_WEST_X + fx * (EDGE_EAST_X - EDGE_WEST_X)))
    y = int(round(EDGE_NORTH_Y + fy * (EDGE_SOUTH_Y - EDGE_NORTH_Y)))
    return x, y


def _cell_lane(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    return float(xs.mean()) / max(w, 1), float(ys.mean()) / max(h, 1)


def _painted_cells(mask: np.ndarray) -> dict[int, tuple[int, int]]:
    h, w = mask.shape
    ch, cw = h / OVERWORLD_ROWS, w / OVERWORLD_COLS
    lanes: dict[int, tuple[int, int]] = {}
    for row in range(OVERWORLD_ROWS):
        for col in range(OVERWORLD_COLS):
            y0, y1 = int(row * ch), int((row + 1) * ch)
            x0, x1 = int(col * cw), int((col + 1) * cw)
            cell = mask[y0:y1, x0:x1]
            if cell.size == 0 or float(cell.mean()) < _CYAN_FRAC:
                continue
            frac = _cell_lane(cell)
            if frac is None:
                continue
            lanes[screen_id(col, row)] = _nes_xy(*frac)
    return lanes


def _painted_neighbors(screen: int, painted: set[int]) -> tuple[int, ...]:
    col, row = screen & 0x0F, (screen >> 4) & 0x0F
    out: list[int] = []
    for dr, dc in _NEIGHBORS:
        rr, cc = row + dr, col + dc
        if not (0 <= rr < OVERWORLD_ROWS and 0 <= cc < OVERWORLD_COLS):
            continue
        sid = screen_id(cc, rr)
        if sid in painted:
            out.append(sid)
    return tuple(out)


def _path_from_start(painted: set[int], start: int) -> tuple[int, ...]:
    """The cyan overlay is a simple corridor. Walk it from ``start``."""
    if start not in painted:
        raise ValueError(f"start {start:#04x} is not on the overlay")
    prev: int | None = None
    cur = start
    path = [cur]
    seen = {cur}
    while True:
        nxt = [n for n in _painted_neighbors(cur, painted) if n != prev]
        if not nxt:
            break
        if len(nxt) != 1:
            raise ValueError(
                f"overlay branches at {cur:#04x}: {[hex(n) for n in nxt]}"
            )
        cur = nxt[0]
        if cur in seen:
            raise ValueError(f"overlay loops at {cur:#04x}")
        seen.add(cur)
        path.append(cur)
        prev = path[-2]
    return tuple(path)


def _hops_for(screens: tuple[int, ...], lanes: dict[int, tuple[int, int]]) -> tuple[ScreenHop, ...]:
    hops: list[ScreenHop] = []
    for src, dest in zip(screens, screens[1:]):
        d = direction_between(src, dest)
        if d is None:
            raise ValueError(f"non-adjacent {src:#04x} -> {dest:#04x}")
        x, y = lanes.get(src) or lanes[dest]
        if d in ("LEFT", "RIGHT"):
            hops.append(ScreenHop(dest, d, align_y=y))
        else:
            hops.append(ScreenHop(dest, d, align_x=x))
    return tuple(hops)


def parse_overworld_map(
    path: Path,
    *,
    start: int = SCREEN_START,
    source: str | None = None,
) -> PaintedRoute:
    """Cyan overlay → screen sequence, NES-lane hops, dest = far endpoint."""
    rgb = read_png_rgb(path)
    lanes = _painted_cells(_cyan_mask(rgb))
    painted = set(lanes)
    painted.add(start)
    screens = _path_from_start(painted, start)
    hops = _hops_for(screens, lanes)
    return PaintedRoute(
        screens=screens,
        hops=hops,
        dest=screens[-1],
        lanes=lanes,
        source=source or str(path),
    )


def mirror_screen_hops(
    start: int, hops: tuple[ScreenHop, ...]
) -> tuple[ScreenHop, ...]:
    """Westbound mirror: each forward hop's alignment, opposite direction."""
    screens = (start,) + tuple(h.target for h in hops)
    out: list[ScreenHop] = []
    for i in range(len(screens) - 1, 0, -1):
        d = direction_between(screens[i], screens[i - 1])
        if d is None:
            raise ValueError("mirror of a non-grid hop")
        fwd = hops[i - 1]
        if d in ("LEFT", "RIGHT"):
            out.append(ScreenHop(screens[i - 1], d, align_y=fwd.align_y))
        else:
            out.append(ScreenHop(screens[i - 1], d, align_x=fwd.align_x))
    return tuple(out)


@lru_cache(maxsize=1)
def map1_route() -> PaintedRoute:
    """The route derived from The Gathering Map-1.png cyan overlay.

    The source image is third-party reference material and ``*.png`` is
    intentionally ignored by the repository, so production route lookup
    must not depend on that local file being present.
    """
    lanes = dict(_MAP1_LANES)
    return PaintedRoute(
        screens=_MAP1_SCREENS,
        hops=_hops_for(_MAP1_SCREENS, lanes),
        dest=_MAP1_SCREENS[-1],
        lanes=lanes,
        source=MAP1_URL,
    )
