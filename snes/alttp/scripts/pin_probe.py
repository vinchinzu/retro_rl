"""Look inside one ALTTP save state. Load only; no RAM writes.

``--press DIR:N`` holds that button for N frames and reprints Link and the
sprites. The pose after a press is what the ROM allows there.

``--chests`` lists chest slots in this room: tile from ``$06E0``, the opened
bit on ``$0402``, the matching save-word bit, and the z3 item for that
chest-table record.

``--doors`` prints each occupied WRAM slot (``$1980`` / ``$19A0`` /
``$19C0``) as direction, type, and the map destination. Slots 8–15 are
the other layer of the same door, not a second exit. A room ``0x81``
pin has no west slot; the gold lock is object ``0x18`` on room ``0x80``.

``--tiles`` prints the indoor attribute grid around Link. ``@`` sits on
Link's tile; the byte under it is on the header line. Overworld collision
does not use this table.

``--fixture`` writes ``tests/fixtures/room_<id>.json`` (tiles, sprites, Link)
for offline tests. It does not step.

    cd snes && SDL_VIDEODRIVER=dummy uv run python alttp/scripts/pin_probe.py CastleMain
    cd snes && SDL_VIDEODRIVER=dummy uv run python alttp/scripts/pin_probe.py EasternPalaceEntry
    cd snes && SDL_VIDEODRIVER=dummy uv run python alttp/scripts/pin_probe.py CastleMain --press LEFT:40
    cd snes && SDL_VIDEODRIVER=dummy uv run python alttp/scripts/pin_probe.py CastleMain --tiles
    cd snes && SDL_VIDEODRIVER=dummy uv run python alttp/scripts/pin_probe.py CastleMain --fixture
    cd snes && SDL_VIDEODRIVER=dummy uv run python alttp/scripts/pin_probe.py CastleB1Key --chests
    cd snes && SDL_VIDEODRIVER=dummy uv run python alttp/scripts/pin_probe.py CastleZeldaB1West --doors
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from alttp.dungeon.rom_scan import chest_lines, door_lines
from alttp.dungeon.tilemap import TILE_COLS, TILE_ROWS, read_room_tiles, tile_kind
from alttp.paths import GAME, GAME_DIR
from alttp.primitives import active_sprites
from alttp.ram import (
    INVENTORY_FIELDS,
    read_inventory,
    read_snapshot,
    read_u16,
    room_label,
)
from alttp.sprite_ids import sprite_name
from alttp.startup import action_for, no_action
from retro_harness.env import make_env, read_state_bytes, state_path
from retro_harness.segment_runner import configure_headless

# Blank dungeon-key sentinel. Same 0xFF the HUD uses when the counter is off.
_KEYS_BLANK = 0xFF
# Always shown, even at 0, so a bare sword pin is not an empty line.
_ALWAYS = frozenset({"sword", "shield", "hp", "max_hp", "keys", "rupees"})

# tilemap_location_calc_mask. CastleMain reads $01F8.
_TILE_MASK = 0x00EC
# Room-border byte on CastleMain (tilemap LEGEND). Interior walls are other bytes.
_BORDER = 0x01
# Spike and conveyor were not on an indoor pin. Bytes are snesrev
# TileDetect_ExecuteInner: 0x44 spike, 0x68..0x6B up/down/left/right, 0x40 grass.
_SPIKE = 0x44
_GRASS = 0x40
# CastleMain statues and pit rims. AP-048: 0x27 blocks the same as 0x04.
_BLOCK = 0x27
_CONVEYOR = {0x68: "^", 0x69: "v", 0x6A: "<", 0x6B: ">"}
# Chamfered corners on CastleMain. Direction is the diagonal those cells form.
_SLOPE = {
    0x10: "\\",
    0x18: "\\",
    0x11: "/",
    0x19: "/",
    0x12: "/",
    0x1A: "/",
    0x13: "\\",
    0x1B: "\\",
}
# snesrev stair cases that are not in tilemap.LEGEND. 0x30 is the north mouth.
_STAIRS = frozenset({0x1C, 0x1E, 0x1F, 0x22, *range(0x30, 0x38), 0x3D, 0x3E, 0x3F})
_KIND_GLYPH = {
    "floor": ".",
    "wall": "#",
    "pit": "p",
    "shallow_water": "~",
    "deep_water": "=",
    "stairs": "s",
    "door": "d",
    "ledge": "L",
    "ice": "i",
}
_LEGEND = (
    "legend . floor  # wall  B block  p pit  ~ shallow-water  = deep-water  "
    "s stairs  d door  * spike  ^v<> conveyor  L ledge  /\\ slope  "
    "i ice  g grass  ? unknown  @ link"
)
# Offline dumps. ``room_61.json`` is CastleMain.
FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
_LINK_KEYS = ("x", "y", "layer", "dir", "state", "hp")


def inventory_summary(ram: np.ndarray) -> str:
    """Non-zero inventory fields, plus sword, shield, hearts, keys, rupees."""
    inv = read_inventory(ram)
    parts: list[str] = []
    for field in INVENTORY_FIELDS:
        value = int(getattr(inv, field.name))
        if field.name not in _ALWAYS and value == 0:
            continue
        if field.name == "keys" and value == _KEYS_BLANK:
            parts.append("keys=blank")
            continue
        parts.append(f"{field.name}={value}")
    return " ".join(parts)


def describe(env: object) -> list[str]:
    """Module, place, Link, inventory, and every active sprite. No writes."""
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    snap = read_snapshot(ram)
    if snap.indoors:
        place = (
            f"room=0x{snap.room_id:04X} {room_label(snap.room_base_id)} "
            f"dungeon=0x{snap.dungeon_id:02X} floor=0x{snap.floor:02X}"
        )
    else:
        place = f"screen=0x{snap.screen_id:02X}"
    lines = [
        (
            f"module=0x{snap.game_mode:02X} submodule=0x{snap.submodule:02X} "
            f"{place}"
        ),
        (
            f"link=({snap.link_x},{snap.link_y}) layer={snap.layer} "
            f"dir=0x{snap.link_direction:02X} state={snap.link_state} "
            f"hp=0x{snap.hp:02X}"
        ),
        f"inventory {inventory_summary(ram)}",
    ]
    sprites = active_sprites(env)
    if not sprites:
        lines.append("sprites (none)")
    for sprite in sprites:
        lines.append(
            f"  slot {sprite.slot:2d} {sprite_name(sprite.sprite_type):20s} "
            f"state=0x{sprite.state:02X} hp={sprite.hp} ({sprite.x},{sprite.y})"
        )
    return lines


def resolve_state(name: str) -> Path:
    path = Path(name)
    if path.suffix == ".state" and path.is_file():
        return path
    found = state_path(GAME_DIR, GAME, name)
    if not found.is_file():
        raise SystemExit(f"missing pin {name}")
    return found


def open_pin(name: str) -> object:
    """Boot headless and load the pin bytes. Does not step or write RAM."""
    configure_headless()
    path = resolve_state(name)
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    env.reset()
    env.em.set_state(read_state_bytes(path))
    return env


def hold_press(env: object, spec: str) -> None:
    """Hold one ``DIR:N`` (or ``NONE:N``) for N frames. Controller only."""
    name, sep, count = spec.partition(":")
    if not sep:
        raise SystemExit(f"press must be DIR:N, got {spec!r}")
    try:
        frames = int(count)
    except ValueError:
        raise SystemExit(f"press must be DIR:N, got {spec!r}") from None
    if frames < 0:
        raise SystemExit(f"press frames must be >= 0, got {spec!r}")
    button = name.strip().upper()
    action = no_action() if button in ("", "NONE") else action_for(button)
    for _ in range(frames):
        env.step(action)  # type: ignore[attr-defined]


def _glyph(value: int) -> str:
    """One character for an attribute byte. Unknown stays ``?``."""
    kind = tile_kind(value)
    if kind != "unknown":
        return _KIND_GLYPH.get(kind, "?")
    if value == _SPIKE:
        return "*"
    if value == _GRASS:
        return "g"
    if value in _CONVEYOR:
        return _CONVEYOR[value]
    if value in _SLOPE:
        return _SLOPE[value]
    if value == _BLOCK:
        return "B"
    if value in _STAIRS:
        return "s"
    # 0x80..0xAF and 0xF0..0xFF are the door families. 0x8E/0xF0 are already doors.
    if 0x80 <= value <= 0xAF or value >= 0xF0:
        return "d"
    return "?"


def _link_tile(x: int, y: int, mask: int) -> tuple[int, int]:
    """Column, row in the 64×64 grid. ``x`` is shifted; ``y`` stays pixels."""
    local_x = int(x) & int(mask)
    local_y = int(y) & int(mask)
    return (local_x >> 3) & 63, (local_y >> 3) & 63


def _room_window(grid: np.ndarray, col: int, row: int) -> tuple[int, int, int, int]:
    """Rows and cols that are not the outer ``0x01`` border, plus one cell."""
    body = grid != _BORDER
    body[row, col] = True
    ys, xs = np.nonzero(body)
    rows, cols = grid.shape
    return (
        max(0, int(ys.min()) - 1),
        min(rows - 1, int(ys.max()) + 1),
        max(0, int(xs.min()) - 1),
        min(cols - 1, int(xs.max()) + 1),
    )


def tile_lines(env: object) -> list[str]:
    """ASCII attribute grid on Link's layer. Reads RAM. No writes."""
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    snap = read_snapshot(ram)
    if not snap.indoors:
        return ["tiles (outdoors; the $7F room grid is not this screen)"]
    mask = read_u16(ram, _TILE_MASK)
    grid = read_room_tiles(ram, snap.layer)
    col, row = _link_tile(snap.link_x, snap.link_y, mask)
    value = int(grid[row, col])
    r0, r1, c0, c1 = _room_window(grid, col, row)
    lines = [
        (
            f"tiles layer={snap.layer} mask=${mask:04X} "
            f"link_tile=({col},{row}) byte=0x{value:02X} {tile_kind(value)}"
        ),
        _LEGEND,
        "     " + "".join(f"{c // 10:d}" if c % 10 == 0 else " " for c in range(c0, c1 + 1)),
        "     " + "".join(str(c % 10) for c in range(c0, c1 + 1)),
    ]
    for r in range(r0, r1 + 1):
        cells = [
            "@" if r == row and c == col else _glyph(int(grid[r, c]))
            for c in range(c0, c1 + 1)
        ]
        lines.append(f"{r:3d}  " + "".join(cells))
    return lines


def fixture_path(room_id: int) -> Path:
    """``tests/fixtures/room_<id>.json`` for a room base id."""
    return FIXTURES / f"room_{int(room_id) & 0xFF:02x}.json"


def capture_fixture(env: object) -> dict:
    """Tiles, sprites, and Link for the loaded indoor room. No writes."""
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    snap = read_snapshot(ram)
    if not snap.indoors:
        raise SystemExit("fixture needs an indoor room")
    grid = read_room_tiles(ram, snap.layer)
    sprites = [
        {
            "slot": sprite.slot,
            "name": sprite_name(sprite.sprite_type),
            "type": sprite.sprite_type,
            "state": sprite.state,
            "hp": sprite.hp,
            "x": sprite.x,
            "y": sprite.y,
        }
        for sprite in active_sprites(env)
    ]
    return {
        "room": snap.room_base_id,
        "room_hex": f"0x{snap.room_base_id:02X}",
        "layer": snap.layer,
        "link": {
            "x": snap.link_x,
            "y": snap.link_y,
            "layer": snap.layer,
            "dir": snap.link_direction,
            "state": snap.link_state,
            "hp": snap.hp,
        },
        "sprites": sprites,
        "tiles": [[int(v) for v in row] for row in grid],
    }


def write_fixture(env: object) -> Path:
    """Write the capture next to the other offline fixtures."""
    data = capture_fixture(env)
    path = fixture_path(int(data["room"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")
    return path


def load_room_fixture(path: Path | str) -> dict:
    """Load a ``--fixture`` dump. ``tiles`` is 64×64. No ROM."""
    src = Path(path)
    data = json.loads(src.read_text())
    for key in ("room", "link", "sprites", "tiles"):
        if key not in data:
            raise ValueError(f"{src} missing {key}")
    tiles = data["tiles"]
    if (
        not isinstance(tiles, list)
        or len(tiles) != TILE_ROWS
        or any(not isinstance(row, list) or len(row) != TILE_COLS for row in tiles)
    ):
        raise ValueError(f"{src} tiles must be {TILE_ROWS}x{TILE_COLS}")
    link = data["link"]
    if not isinstance(link, dict) or any(key not in link for key in _LINK_KEYS):
        raise ValueError(f"{src} link missing one of {_LINK_KEYS}")
    if not isinstance(data["sprites"], list):
        raise ValueError(f"{src} sprites must be a list")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", help="save-state name (CastleMain) or a .state path")
    parser.add_argument(
        "--press",
        nargs="*",
        default=(),
        metavar="DIR:N",
        help="hold DIR (or NONE) for N frames, then reprint Link and sprites",
    )
    parser.add_argument(
        "--chests",
        action="store_true",
        help="list chest slots, opened bits, and the z3 item at each",
    )
    parser.add_argument(
        "--doors",
        action="store_true",
        help="print each live door slot as direction, type, and destination",
    )
    parser.add_argument(
        "--tiles",
        action="store_true",
        help="print the attribute grid around Link, with a kind legend",
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="write tests/fixtures/room_<id>.json (tiles, sprites, Link)",
    )
    args = parser.parse_args(argv)
    env = open_pin(args.state)
    try:
        print("\n".join(describe(env)))
        if args.chests:
            print("\n".join(chest_lines(env)))
        if args.doors:
            print("\n".join(door_lines(env)))
        if args.tiles:
            print("\n".join(tile_lines(env)))
        if args.fixture:
            print(f"wrote {write_fixture(env)}")
        for press in args.press:
            hold_press(env, press)
            print(f"-- after {press}")
            print("\n".join(describe(env)))
            if args.chests:
                print("\n".join(chest_lines(env)))
            if args.doors:
                print("\n".join(door_lines(env)))
            if args.tiles:
                print("\n".join(tile_lines(env)))
    finally:
        env.close()  # type: ignore[attr-defined]
    return 0


if __name__ == "__main__":
    sys.exit(main())
