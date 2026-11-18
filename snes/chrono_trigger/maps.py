"""ROM exit graph and observed field maps; derived data stays local.

Exit records describe geometry, not event prerequisites. Attribute bytes
are exported without claiming that a tile is traversable at every level.
"""

from collections import deque
from pathlib import Path
import hashlib
import json

from .ram import Snapshot, word

GAME_DIR = Path(__file__).resolve().parent
GENERATED = GAME_DIR / "maps" / "generated"
ROM_SHA1 = "de5822f4f2f7a55acb8926d4c0eaa63d5d989312"
NAMES = {
    0x000: "Load screen", 0x001: "Crono's house downstairs",
    0x002: "Crono's bedroom", 0x005: "Millennial Fair south",
    0x008: "Lucca's exhibit", 0x1B7: "Millennial Fair north / Leene's Bell",
    0x070: "Truce Canyon", 0x071: "Truce Canyon gate clearing (600 AD)",
    0x1F0: "Overworld (1000 AD)", 0x1F1: "Overworld (600 AD)",
    0x1F2: "Overworld (2300 AD)",
}


def checked_rom(data: bytes) -> bytes:
    """Strip a copier header and require the layout the decoder understands."""
    if len(data) % 0x8000 == 512:
        data = data[512:]
    digest = hashlib.sha1(data).hexdigest()
    if digest != ROM_SHA1:
        raise ValueError(f"expected Chrono Trigger US v1.0 ({ROM_SHA1}); got {digest}")
    return data


def exit_graph(rom: bytes) -> dict:
    """Decode the 512 location records and their seven-byte exit records.

    Layout from ChronoRET ram_engine.inc: LocRom $F60000,
    ExitRom_LocOfs / ExitRom $E50000. SNES HiROM -> file = bus & $3FFFFF.
    """
    rom = checked_rom(rom)
    locations = []
    for loc in range(512):
        base = 0x360000 + loc * 14
        start, end = word(rom, 0x250000 + loc * 2), word(rom, 0x250002 + loc * 2)
        if not 0x402 <= start <= end <= 0x10000 or (end - start) % 7:
            raise ValueError(f"invalid exit pointer range for location {loc:03x}")
        exits = []
        for offset in range(start, end, 7):
            x, y, span, lo, hi, dx, dy = rom[0x250000 + offset:0x250007 + offset]
            exits.append({
                "tile": [x, y], "extra_tiles": span & 0x7F,
                "vertical": bool(span & 0x80), "destination": lo | ((hi & 1) << 8),
                "arrival_tile": [dx, dy], "facing_flags": hi,
                "verified": False,
            })
        locations.append({
            "id": loc, "name": NAMES.get(loc, f"Location {loc:03X}"),
            "map_id": word(rom, base + 4), "event_id": word(rom, base + 8),
            "camera_limits": list(rom[base + 10:base + 14]), "exits": exits,
        })
    return {"rom_sha1": ROM_SHA1, "kind": "static_geometry_not_progression", "locations": locations}


def geometry_path(graph: dict, start: int, goal: int) -> list[int] | None:
    """Shortest exit-only path. Events/doors/era gates still require verification."""
    rows = {row["id"]: row for row in graph["locations"]}
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        loc, path = queue.popleft()
        if loc == goal:
            return path
        for edge in rows.get(loc, {}).get("exits", []):
            dest = edge["destination"]
            if dest not in visited:
                visited.add(dest)
                queue.append((dest, path + [dest]))
    return None


SLOPE_SHIFT = bytes.fromhex("82020282810101818303038381010181830303838202")
SLOPE_BASE = bytes.fromhex("3D03033D7D03C33D3E23031D3DC3037D1E03233D3D03")
SLOPE_SIDE = bytes.fromhex("0000FFFF0000FFFF0000FFFF0000FFFF0000FFFF0000")
DIRS = {"RIGHT": (1, 0), "LEFT": (-1, 0), "DOWN": (0, 1), "UP": (0, -1)}


class Field:
    """Leader collision of the loaded field map, as the ROM tests it.

    Port of C0:8A6D (box corners), C0:9AD3 (tile half by shape),
    C0:9C37 (level test) and C0:9C5C (commit, exit tiles). Pixels here are
    Obj_Pos / 16. Slides around corners are not modelled, so every planned
    step is one the game also accepts.
    """

    def __init__(self, ram: bytes, rom: bytes, refused: set | None = None):
        self.ram, self.location = ram, word(ram, 0x100)
        # (x, y, button) steps the game refused although this model allowed them.
        self.refused = refused or set()
        self.mask = (ram[0x1D1E], ram[0x1D1F])
        self.width, self.height = ram[0xBCB] * 16, ram[0xBCD] * 16
        start, end = word(rom, 0x250000 + self.location * 2), word(rom, 0x250002 + self.location * 2)
        self.exits = [rom[0x250000 + o:0x250007 + o] for o in range(start, end, 7)]
        party = {word(ram, 0x197 + i) for i in (0, 2, 4)}
        self.solids = [
            (word(ram, 0x1800 + s), word(ram, 0x1880 + s), s // 2)
            for s in range(2, min(ram[0x12000], 63) * 2 + 2, 2)
            if ram[0xF00 + s] and not ram[0x1100 + s] & 0x80 and s not in party and ram[0x1B01 + s] & 1
        ]
        self._half: dict[tuple[int, int], tuple[int, bool]] = {}

    def index(self, x: int, y: int) -> int:
        return ((y >> 4) & self.mask[1]) << 8 | ((x >> 4) & self.mask[0])

    def half(self, x: int, y: int) -> tuple[int, bool]:
        """(level, no_level) of the half tile under a pixel."""
        key = (x, y)
        if key in self._half:
            return self._half[key]
        i = self.index(x, y)
        shape, b = self.ram[0x30C0 + i] >> 2, self.ram[0x7040 + i]
        fx, fy = (x & 15) << 4, (y & 15) << 4
        if shape == 0:
            second = False
        elif shape == 1:
            second = True
        elif shape < 0x18:
            n = shape - 2
            t = fx >> (SLOPE_SHIFT[n] & 0x7F)
            if SLOPE_SHIFT[n] & 0x80:
                t ^= 0xFF
            t = (SLOPE_BASE[n] + t - (fy >> 2)) & 0xFF
            second = bool((SLOPE_SIDE[n] ^ t) & 0x80)
        else:
            left, top = fx < 0x80, fy < 0x80
            second = {0x18: left, 0x19: top, 0x1A: left and not top, 0x1B: not left and not top,
                      0x1C: not left and top, 0x1D: left and top}.get(shape, False)
        result = ((b >> 3) & 3, bool(b & 0x20)) if second else (b & 3, bool(b & 4))
        self._half[key] = result
        return result

    def blocked(self, x: int, y: int, level: int, corner: bool = True) -> bool:
        if corner and not self.ram[0x70C0 + self.index(x, y)] & 0x80:
            return True  # treasure tile
        tile, no_level = self.half(x, y)
        if no_level:
            return level == 3
        return tile == 0 or (tile != 3 and tile != level and level != 3)

    def touches(self, x: int, y: int, ignore: int | None = None) -> bool:
        return any(abs(ox - x * 16) < 0xF0 and abs(oy - y * 16) < 0xF0  # 14 px in the ROM, plus one for fractions
                   for ox, oy, obj in self.solids if obj != ignore)

    def step(self, x: int, y: int, level: int, name: str, ignore: int | None = None):
        """One pixel in a direction: the new (x, y, level), or None when blocked."""
        dx, dy = DIRS[name]
        nx, ny = x + dx, y + dy
        if (x, y, name) in self.refused:
            return None
        if not (0 <= nx < self.width and 0 <= ny < self.height):
            return None
        # Slides: attribute A bits 2-3 push the leader (0.5 / 1 / 2 px a frame) in the
        # direction of bits 0-1. A walking step of 1 px cannot climb the faster two.
        push = self.ram[0x7000 + self.index(x, y)]
        if push & 0x08 and (dx, dy) == ((0, 1), (0, -1), (1, 0), (-1, 0))[push & 3]:
            return None
        if dx:
            if self.touches(x + 8 * dx, y - 4, ignore):
                return None
            corners = ((x + 8 * dx, y), (x + 8 * dx, y - 7))
        else:
            edge = y + 1 if dy > 0 else y - 8
            if self.touches(x, edge, ignore):
                return None
            corners = ((x - 7, edge), (x + 7, edge))
        if any(self.blocked(cx, cy, level) for cx, cy in corners) or self.blocked(nx, ny, level, False):
            return None
        tile, no_level = self.half(nx, ny)
        return nx, ny, level if no_level else tile

    def exit_at(self, x: int, y: int) -> int | None:
        """Destination location of the exit tile under a pixel, as C0:9C5C matches it."""
        value = self.ram[0x7080 + self.index(x, y)]
        if value & 0x80 or value >= len(self.exits):
            return None
        ex, ey, _, lo, hi, _, _ = self.exits[value]
        if (ex == x >> 4 and abs(ey - (y >> 4)) < 0x3F) or ey == y >> 4:
            return lo | (hi & 1) << 8
        return None

    def exit_tile(self, x: int, y: int) -> tuple[int, int]:
        """First tile of the exit record under a pixel (records span several tiles)."""
        return tuple(self.exits[self.ram[0x7080 + self.index(x, y)]][:2])

    def path(self, x: int, y: int, level: int, done, *, ignore: int | None = None,
             limit: int = 200000):
        """Shortest list of (button, x, y) legs on an 8-pixel lattice.

        done(x, y, exit_destination) -> truthy ends the search at that
        pixel. Returns (legs, reached) where reached is every settled node,
        or (None, reached) when no node satisfies done.
        """
        start = (x, y, level)
        parent = {start: None}
        queue = deque([start])
        while queue and len(parent) < limit:
            node = queue.popleft()
            for name, (dx, dy) in DIRS.items():
                cx, cy, cl = node
                # Lattice x = 0 mod 8, y = 7 mod 8 keeps the 14 x 8 box inside one tile row.
                count = ((-cx if dx > 0 else cx) % 8 or 8) if dx else (((7 - cy) if dy > 0 else (cy - 7)) % 8 or 8)
                hit = False
                for _ in range(count):
                    moved = self.step(cx, cy, cl, name, ignore)
                    if moved is None:
                        break
                    cx, cy, cl = moved
                    if done(cx, cy, self.exit_at(cx, cy)):
                        hit = True
                        break
                else:
                    moved = True
                if not hit and (moved is None or self.exit_at(cx, cy) is not None):
                    continue  # blocked, or an exit tile that is not wanted
                nxt = (cx, cy, cl)
                if nxt in parent:
                    continue
                parent[nxt] = (node, name)
                if hit:
                    legs = []
                    while parent[nxt]:
                        prev, button = parent[nxt]
                        if legs and legs[0][0] == button:
                            legs[0] = (button, legs[0][1], legs[0][2])
                        else:
                            legs.insert(0, (button, nxt[0], nxt[1]))
                        nxt = prev
                    return legs, parent
                queue.append(nxt)
        return None, parent


class World:
    """Overworld 8-pixel walk lattice (C2:3567 / C2:39C3) and entrance triggers."""

    def __init__(self, ram: bytes):
        self.ram = ram
        self.entrances = []
        for i in range(min(ram[0x1B38], 128)):
            a = 0x7E00 + i * 7
            if ram[a] & 0x80:
                self.entrances.append(((ram[a] & 127, ram[a + 1] & 63), word(ram, a + 3) & 511))

    def prop(self, col: int, row: int) -> int:
        col, row = col % 192, row % 128
        meta = self.ram[0x5800 + (row >> 1) * 96 + (col >> 1)]
        value = self.ram[0x7000 + meta * 2 + (row & 1)]
        return value & 15 if col & 1 else value >> 4

    def free(self, x: int, y: int) -> bool:
        col, row = x >> 3, y >> 3
        return not (self.prop(col - 1, row - 1) | self.prop(col, row - 1)) & 3

    def trigger(self, x: int, y: int) -> bool:
        col, row = x >> 3, y >> 3
        return bool(self.prop(col - 1, row - 1) & self.prop(col, row - 1) & 4)

    def path(self, x: int, y: int, done):
        start = (x, y)
        parent = {start: None}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if done(*node):
                legs = []
                while parent[node]:
                    prev, button = parent[node]
                    if legs and legs[0][0] == button:
                        pass
                    else:
                        legs.insert(0, (button, node[0], node[1]))
                    node = prev
                return legs, parent
            for name, (dx, dy) in DIRS.items():
                nxt = ((node[0] + dx * 8) % 1536, (node[1] + dy * 8) % 1024)
                if nxt not in parent and self.free(*nxt):
                    parent[nxt] = (node, name)
                    queue.append(nxt)
        return None, parent


class Atlas:
    """Current run's sampled tiles, raw field planes, and observed transitions."""

    def __init__(self):
        self.scenes: dict[int, dict] = {}
        self.transitions: list[dict] = []
        self.previous: tuple[int, float, float] | None = None

    def sample(self, state: Snapshot, ram: bytes, frame: int) -> None:
        if not state.ready or state.x is None or state.y is None:
            return
        if state.location < 0x1F0 and not state.control:
            return
        loc = state.location
        if self.previous and self.previous[0] != loc:
            self.transitions.append({"from": self.previous[0], "to": loc, "frame": frame,
                                     "departure": list(self.previous[1:]),
                                     "arrival": [state.x, state.y]})
        self.previous = (loc, state.x, state.y)
        scene = self.scenes.setdefault(loc, {"id": loc, "name": NAMES.get(loc, f"Location {loc:03X}"),
                                             "visited": set(), "last_position": []})
        scene["visited"].add((int(state.x // 16), int(state.y // 16)))
        scene["last_position"] = [state.x, state.y]
        if 0x1F0 <= loc < 0x1F8:
            scene.update(width=96, height=64)
            scene["layer1"] = [list(ram[0x4000 + y * 96:0x4000 + (y + 1) * 96]) for y in range(64)]
            triggers = []
            for i in range(min(ram[0x1B38], 128)):
                a = 0x7E00 + i * 7
                triggers.append({"tile": [ram[a] & 127, ram[a + 1] & 63],
                                 "active": bool(ram[a] & 128),
                                 "destination": word(ram, a + 3) & 511,
                                 "arrival_tile": list(ram[a + 5:a + 7])})
            scene["triggers"] = triggers
        elif loc < 0x1F0:
            width, height = ram[0xBCB], ram[0xBCD]
            if not 1 <= width <= 64 or not 1 <= height <= 64:
                return
            scene.update(width=width, height=height)
            for name, base in (("layer1", 0x3000), ("attributes_a", 0x7000),
                               ("attributes_b", 0x7040), ("exit_indices", 0x7080)):
                scene[name] = [list(ram[base + y * 256:base + y * 256 + width]) for y in range(height)]

    def write(self, output: Path) -> None:
        output.mkdir(parents=True, exist_ok=True)
        scenes = [{**scene, "visited": sorted(scene["visited"])} for scene in self.scenes.values()]
        payload = {"kind": "observed_trace_and_raw_tile_attributes", "scenes": scenes,
                   "transitions": self.transitions}
        (output / "atlas.json").write_text(json.dumps(payload, indent=2) + "\n")
        # Local, self-contained inspection; no network or third-party JS.
        page = '''<!doctype html><meta charset="utf-8"><title>Chrono Trigger atlas</title>
<style>body{background:#171b24;color:#eef;font:16px system-ui;margin:30px}canvas{image-rendering:pixelated;border:1px solid #777}select{font:inherit}pre{white-space:pre-wrap}</style>
<h1>Chrono Trigger observed atlas</h1><p>Green: visited tiles. Blue: exit tiles.
Dark: both half-tile “no level” bits set. Gray: other raw attributes.
Raw geometry does not prove a route is traversable. Hover for RAM values.</p>
<select id="scene"></select><p id="info"></p><canvas id="map"></canvas><pre id="detail"></pre>
<script>const data=DATA, sel=document.querySelector('#scene'), c=document.querySelector('#map'),
ctx=c.getContext('2d');for(const s of data.scenes){let o=document.createElement('option');o.textContent=s.id.toString(16).padStart(3,'0')+' '+s.name;sel.append(o)}
function draw(){const s=data.scenes[sel.selectedIndex];if(!s)return;const w=s.width||64,h=s.height||64;c.width=w*12;c.height=h*12;
for(let y=0;y<h;y++)for(let x=0;x<w;x++){let b=s.attributes_b?.[y]?.[x];ctx.fillStyle=b===undefined?'#222':(b&36)===36?'#111':'#656e80';if(s.exit_indices?.[y]?.[x]<128)ctx.fillStyle='#347ddb';ctx.fillRect(x*12,y*12,11,11)}
ctx.fillStyle='#347ddb';for(const t of s.triggers||[])if(t.active)ctx.fillRect(t.tile[0]*12,t.tile[1]*12,11,11);
ctx.fillStyle='#6f6';for(const [x,y]of s.visited)ctx.fillRect(x*12,y*12,11,11);
document.querySelector('#info').textContent=`${s.name}: ${s.visited.length} visited tiles; last position ${s.last_position}`}
sel.onchange=draw;c.onmousemove=e=>{const s=data.scenes[sel.selectedIndex],r=c.getBoundingClientRect(),x=Math.floor((e.clientX-r.left)*c.width/r.width/12),y=Math.floor((e.clientY-r.top)*c.height/r.height/12);
document.querySelector('#detail').textContent=JSON.stringify({tile:[x,y],layer1:s.layer1?.[y]?.[x],a:s.attributes_a?.[y]?.[x],b:s.attributes_b?.[y]?.[x],exit:s.exit_indices?.[y]?.[x]},null,2)};draw();</script>'''
        (output / "atlas.html").write_text(page.replace("DATA", json.dumps(payload).replace("<", "\\u003c")))
