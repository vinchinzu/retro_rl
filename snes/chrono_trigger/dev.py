"""Route authoring tools. Practice only: they load pins and (explore) emulator states.

  uv run python -m snes.chrono_trigger.dev scout PIN ['[rows]'] [--map] [--save NAME] [--picks]
  uv run python -m snes.chrono_trigger.dev explore PIN TARGET_LOC [--limit N] [--avoid LOC,LOC]
  uv run python -m snes.chrono_trigger.dev script LOC [text filter]
  uv run python -m snes.chrono_trigger.dev keys PIN A:2 -:30 UP+B:10 ... [--save NAME]

scout plays optional rows from a pin, then prints where the party is, every exit or
entrance (and whether the collision map reaches it), every object, and a tile map.
explore searches exit hops breadth-first and prints the rows that reach a location.
script decodes a location's event script (needs refs/temporal-redux).
keys presses raw buttons and prints menu, battle and position state after each step.
"""

import argparse
from collections import deque
import contextlib
import io
import json
from pathlib import Path
import sys

from .bot import GAME_DIR, Runner
from .maps import Field, World
from .ram import observe, read_wram, word

REDUX = GAME_DIR / "refs" / "temporal-redux" / "sourcefiles"
OUT = GAME_DIR / "recordings" / "dev"


def names() -> dict[int, str]:
    try:
        sys.path.insert(0, str(REDUX))
        from jetsoftime.ctenums import LocID
        return {int(v): v.name for v in LocID}
    except ImportError:
        return {}


def describe(r: Runner, show_map: bool) -> None:
    s, ram, known = r.state, r.ram, names()
    label = lambda d: f"{d:03X}:{known.get(d, '?')}"
    worn = [ram[0x2600 + c * 0x50 + 0x27:0x2600 + c * 0x50 + 0x2B].hex() for c in s.party if c < 7]
    print(f"{label(s.location)} pos=({s.x},{s.y}) level={ram[0x155]} story={s.story:02X} party={list(s.party)} "
          f"hp={s.hp}/{s.max_hp} mp={s.mp}/{s.max_mp} gold={s.gold} control={s.control} message={s.message} "
          f"mode={s.mode}\n  items={[f'{i:02X}x{r.stock(i)}' for i in s.items]} helmet/armor/weapon/accessory={worn}")
    if s.battle:
        print("  battle", s.battle_phase, [(b.slot, b.identity, b.hp, b.max_hp, b.mp) for b in s.battlers],
              "techs", [sorted(r.techs(i)) for i in range(3)])
        return
    if s.x is None:
        return
    x, y = int(s.x), int(s.y)
    if s.location >= 0x1F0:
        world = World(ram)
        _, seen = world.path(x, y, lambda px, py: False)
        for dest in sorted({d for _, d in world.entrances}):
            tiles = [t for t, d in world.entrances if d == dest]
            ok = any((px >> 4, py >> 4) in tiles and world.trigger(px, py) for px, py in seen)
            print("  entrance", label(dest), tiles, "REACHABLE" if ok else "")
        return
    field, level = Field(ram, r.rom), ram[0x155]
    _, seen = field.path(x, y, level, lambda px, py, dest: False)
    for i, e in enumerate(field.exits):
        dest, tile = e[3] | (e[4] & 1) << 8, (e[0], e[1])
        legs, _ = field.path(x, y, level, lambda px, py, d: d == dest and field.exit_tile(px, py) == tile)
        print(f"  exit {i} tile={list(tile)} span={e[2] & 0x7F}{'v' if e[2] & 0x80 else 'h'} -> {label(dest)}",
              "REACHABLE" if legs is not None else "")
    party = {word(ram, 0x197 + i) for i in (0, 2, 4)}
    for slot in range(0, min(ram[0x12000], 63) * 2 + 2, 2):
        ox, oy = word(ram, 0x1800 + slot) // 16, word(ram, 0x1880 + slot) // 16
        if ox < 4000:
            shown = ram[0xF00 + slot] and not ram[0x1100 + slot] & 0x80
            print(f"  obj {slot // 2} pos=({ox},{oy})", "solid" if ram[0x1B01 + slot] & 1 else "",
                  "PARTY" if slot in party else "", "" if shown else "hidden")
    if show_map:
        tiles = {(px >> 4, py >> 4) for px, py, _ in seen}
        for ty in range(field.height // 16):
            line = "".join(
                "@" if (tx, ty) == (x >> 4, y >> 4) else "#" if (tx, ty) in tiles
                else "E" if ram[0x7080 + (ty << 8 | tx)] < 0x80 else "T" if ram[0x70C0 + (ty << 8 | tx)] < 0x80
                else "." for tx in range(field.width // 16))
            if line.strip("."):
                print(f"{ty:3d} {line}")


def screenshot(r: Runner) -> None:
    import pygame
    OUT.mkdir(parents=True, exist_ok=True)
    pygame.image.save(pygame.surfarray.make_surface(r.env.render().swapaxes(0, 1)), str(OUT / "screen.png"))


def explore(r: Runner, target: int, limit: int, avoid: set[int]) -> list[dict] | None:
    def restore(blob):
        r.env.em.set_state(blob)
        r.env.data.update_ram()
        r.ram = read_wram(r.env)
        r.state = observe(r.ram)

    def options():
        s = r.state
        field, found = Field(r.ram, r.rom), []
        for e in field.exits:
            dest, tile = e[3] | (e[4] & 1) << 8, (e[0], e[1])
            if dest not in avoid and field.path(
                    int(s.x), int(s.y), r.ram[0x155],
                    lambda px, py, d: d == dest and field.exit_tile(px, py) == tile)[0] is not None:
                found.append({"kind": "exit", "to": dest, "tile": list(tile)})
        return found

    key = lambda: (r.state.location, int(r.state.x) >> 5, int(r.state.y) >> 5)
    seen, queue, tried = {key()}, deque([(r.env.em.get_state(), [])]), 0
    while queue and tried < limit:
        blob, rows = queue.popleft()
        restore(blob)
        for row in options() if r.state.location < 0x1F0 else []:
            restore(blob)
            result = r.play_row(dict(row, timeout=4000))
            tried += 1
            s = r.state
            print(f"{' ' * len(rows)}{row['to']:03X} {row['tile']} {result.reason} -> {s.location:03X} "
                  f"({s.x},{s.y}) hp={s.hp}", file=sys.stderr)
            if not result.passed or s.x is None:
                continue
            if s.location == target:
                return rows + [row]
            if key() not in seen:
                seen.add(key())
                queue.append((r.env.em.get_state(), rows + [row]))
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["scout", "explore", "script", "keys"])
    parser.add_argument("args", nargs="*")
    parser.add_argument("--map", action="store_true")
    parser.add_argument("--picks", action="store_true")
    parser.add_argument("--save")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--avoid", default="")
    a = parser.parse_args(argv)
    if a.command == "script":
        sys.path.insert(0, str(REDUX))
        from jetsoftime import ctevent
        event = ctevent.Event.from_rom_location((GAME_DIR / "roms" / "Chrono Trigger.sfc").read_bytes(),
                                                int(a.args[0], 0))
        text = io.StringIO()
        with contextlib.redirect_stdout(text):
            event.print_human_readable_fn()
        want = a.args[1].lower() if len(a.args) > 1 else ""
        print("\n".join(line for line in text.getvalue().splitlines() if want in line.lower()))
        return 0
    r = Runner(state=a.args[0], output=OUT)
    try:
        if a.command == "explore":
            rows = explore(r, int(a.args[1], 0), a.limit, {int(v, 0) for v in a.avoid.split(",") if v})
            print(json.dumps(rows) if rows else "not found")
        elif a.command == "keys":
            for step in a.args[1:]:
                held, frames = step.split(":")
                for _ in range(int(frames)):
                    r.tick(tuple(n for n in held.split("+") if n != "-"))
                s, ram = r.state, r.ram
                print(f"{step:10} pos=({s.x},{s.y}) control={s.control} message={s.message} mode={s.mode} "
                      f"menu={ram[0x67]:02X} cursor={ram[0x54]} battle={s.battle_phase} actor={s.command_actor} "
                      f"submenu={ram[0x95DB]} target={s.target_slot} gold={s.gold}")
        else:
            for i, row in enumerate(json.loads(a.args[1]) if len(a.args) > 1 else []):
                result = r.play_row(row)
                print(i, row.get("id", row["kind"]), "ok" if result.passed else "FAIL", result.reason, result.frames)
                if not result.passed:
                    break
        describe(r, a.map)
        if a.picks:
            print(*[f"{p['frame']} slot{p['actor']} {p['pick']} hp={p['hp']}" for p in r.picks[-60:]], sep="\n")
        if a.save:
            r.save_pin(a.save)
        screenshot(r)
    finally:
        r.env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
