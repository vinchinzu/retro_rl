"""One bounded route Composer for autoplay, pin probes, and input tapes.

All gameplay goes through Runner.tick. No RAM writes, automatic restores,
health refills, warps, or guessed ending predicate.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import zipfile

from retro_harness.env import GameSpec, link_rom_into_integration, read_state_bytes
from retro_harness.snes import buttons

from .maps import Atlas, DIRS, Field, World, GAME_DIR, GENERATED, ROM_SHA1, checked_rom, exit_graph, geometry_path
from .ram import Snapshot, observe, read_wram, word
from .route import OPENING, ROUTE

GAME = GameSpec("ChronoTrigger-Snes", GAME_DIR)
ROM = GAME_DIR / "roms" / "Chrono Trigger.sfc"

# Battle defaults by character id (0 Crono, 1 Marle, 2 Lucca, 3 Robo, 4 Frog, 5 Ayla, 6 Magus).
# A row's "battle" object overrides: heal_below (percent), spend, techs, focus (enemy ids),
# escape (False for mandatory bosses), stall (frames), flee (force escape), ether_below (MP), slots.
HEALS = {1: (12, 9), 3: (29, 26), 4: (36, 33), 5: (41,)}
STRIKES = {0: (4, 1, 2), 1: (11,), 2: (19, 17), 3: (27, 25), 4: (35, 34), 5: (48, 42, 43), 6: (50, 51)}
GOAL_KEYS = {"story_min", "party_contains", "item_contains", "flag", "flag_clear", "rested",
             "x_min", "x_max", "y_min", "y_max"}
TONICS = (0xBF, 0xBE, 0xBD)
REVIVE = 0xC7

def setup_rom(source: Path) -> None:
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            members = [n for n in archive.namelist() if Path(n).suffix.lower() in {".sfc", ".smc"}]
            if len(members) != 1:
                raise ValueError("ROM archive must contain exactly one SNES ROM")
            data = archive.read(members[0])
    else:
        data = source.read_bytes()
    data = checked_rom(data)
    ROM.parent.mkdir(parents=True, exist_ok=True)
    ROM.write_bytes(data)
    link_rom_into_integration(ROM, GAME.states_dir)
    print(f"ROM ready: {ROM.relative_to(GAME_DIR)} (US v1.0, copier header removed if present)")


@dataclass
class Outcome:
    passed: bool
    reason: str
    frames: int
    state: Snapshot


class Runner:
    def __init__(self, *, state: str | None = None, output: Path, headed: bool = False):
        if not ROM.is_file():
            raise FileNotFoundError("Run the setup command with your local ROM first")
        self.rom = checked_rom(ROM.read_bytes())
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.env = GAME.make_env(render_mode="human" if headed else "rgb_array")
        self.env.reset()
        self.state_loads = 0
        if state:
            path = GAME.state_path(state)
            if not path.is_file():
                self.env.close()
                raise FileNotFoundError(f"missing start pin: {path}")
            self.env.em.set_state(read_state_bytes(path))
            self.env.data.update_ram()
            self.state_loads = 1
        self.start_state = state
        self.frame = 0
        self.ram = read_wram(self.env)
        self.state = observe(self.ram)
        self.atlas = Atlas()
        self.rows: list[dict] = []
        self.tape: list[dict] = []
        self.battle_events: list[dict] = []
        self._battle_key = None
        self._turn = None
        self.refused: dict[int, set] = {}
        self._watch = None
        self._pressed = 0
        self.picks: list[dict] = []
        self.started = time.monotonic()
        self.atlas.sample(self.state, self.ram, self.frame)
        self.capture_battle()

    def capture_battle(self) -> None:
        s = self.state
        fields = ("mode", "battle_phase", "battlers", "hp", "max_hp", "mp", "max_mp",
                  "command_actor", "command_row", "target_slot", "last_battle_result", "gate_battle_won")
        key = (self.ram[0x500:0x504].hex(), *(getattr(s, k) for k in fields))
        previous = self._battle_key
        if key != previous and (s.battle or previous and (
            previous[1] == "battle" or previous[-1] != s.gate_battle_won
        )):
            record = s.record()
            self.battle_events.append({"frame": self.frame, "nmi": key[0],
                                       **{k: record[k] for k in fields}})
        self._battle_key = key

    def tick(self, names: tuple[str, ...]) -> None:
        result = self.env.step(buttons(*names))
        self.frame += 1
        if self.tape and self.tape[-1]["buttons"] == list(names):
            self.tape[-1]["frames"] += 1
        else:
            self.tape.append({"buttons": list(names), "frames": 1})
        self.ram = read_wram(self.env)
        self.state = observe(self.ram)
        self.capture_battle()
        if self.frame % 8 == 0:
            self.atlas.sample(self.state, self.ram, self.frame)
        if result[2] or result[3]:
            raise RuntimeError("emulator ended before a verified route goal")

    def confirm(self) -> tuple[str, ...]:
        if self.frame % 24 >= 2:
            return ()
        if self.state.naming or self.frame < 300 and self.start_state is None:
            return ("START",)
        return ("A",)

    def matches(self, goal: dict) -> bool:
        for key, value in goal.items():
            if key in {"x_min", "x_max", "y_min", "y_max"}:
                at = self.state.x if key[0] == "x" else self.state.y
                if at is None or (at < value if key.endswith("min") else at > value):
                    return False
            elif key in {"flag", "flag_clear"}:
                # [offset, mask] in the event flag block at $7F0000.
                if bool(self.ram[0x10000 + value[0]] & value[1]) != (key == "flag"):
                    return False
            elif key == "rested":
                if (self.state.hp, self.state.mp) != (self.state.max_hp, self.state.max_mp):
                    return False
            elif key == "story_min":
                if self.state.story < value:
                    return False
            elif key == "party_contains":
                if value not in self.state.party:
                    return False
            elif key == "item_contains":
                if value not in self.state.items:
                    return False
            elif getattr(self.state, key) != value:
                return False
        return True

    def plan(self, row: dict):
        """Legs for a routed row from the live collision map, or None."""
        s, kind = self.state, row["kind"]
        x, y = int(s.x), int(s.y)
        if s.location >= 0x1F0:
            world = World(self.ram)
            if kind == "exit":
                cells = {tile for tile, dest in world.entrances if dest == row["to"]}
                if "tile" in row:
                    cells = {tuple(row["tile"])}
                return world.path(x, y, lambda px, py: (px >> 4, py >> 4) in cells and world.trigger(px, py))[0]
            at = tuple(row["at"])
            return world.path(x, y, lambda px, py: (px, py) == at)[0]
        field = Field(self.ram, self.rom, self.refused.get(s.location))
        level = self.ram[0x155]
        ignore = None
        seen = {}

        def route(done):
            nonlocal seen
            legs, seen = field.path(x, y, level, done, ignore=ignore)
            return legs

        def nearest(at, score, within):
            best = min(seen, key=lambda n: score(abs(n[0] - at[0]), abs(n[1] - at[1])))
            if within is not None and score(abs(best[0] - at[0]), abs(best[1] - at[1])) > within:
                return None
            return [] if best[:2] == (x, y) else route(lambda px, py, dest: (px, py) == best[:2])

        if kind == "exit":
            tile = row.get("tile")
            legs = route(lambda px, py, dest: dest == row["to"] and (
                tile is None or all(want in (None, got) for want, got in zip(tile, field.exit_tile(px, py)))))
        elif kind == "talk":
            at = self.subject(row)
            if row.get("chase"):
                ignore = row.get("object")
                route(lambda px, py, dest: False)
                return nearest(at, lambda dx, dy: dx + dy, row.get("reach", 40))
            route(lambda px, py, dest: False)
            # Stand squarely in front of the object, close enough for A to reach it.
            # Not on top of it either: A looks one step ahead of the leader.
            legs = nearest(at, lambda dx, dy: max(dx, dy) + 4 * min(dx, dy) + (99 if max(dx, dy) < 12 else 0),
                           row.get("reach", 40))
        else:
            at = tuple(row["at"])
            legs = [] if (x, y) == at else route(lambda px, py, dest: (px, py) == at)
            if legs is None:
                legs = nearest(at, lambda dx, dy: dx + dy, row.get("reach", 24 if row.get("via") else None))
        if legs is not None or not row.get("via"):
            return legs
        # A door or event line is in the way: walk to the first listed push point
        # that is reachable and lean on it. drive() routes again while pushing.
        for vx, vy, button in row["via"]:
            legs = nearest((vx, vy), lambda dx, dy: dx + dy, 16)
            if legs is not None:
                return legs + [(button, -1 if button == "LEFT" else 1 << 20, -1 if button == "UP" else 1 << 20)]
        return None

    def subject(self, row: dict) -> tuple[int, int]:
        """Pixel a talk row faces: an object, or "at" for chests and fixed switches."""
        if "at" in row:
            return tuple(row["at"])
        slot = row["object"] * 2
        return word(self.ram, 0x1800 + slot) // 16, word(self.ram, 0x1880 + slot) // 16

    def techs(self, pc: int) -> set[int]:
        """Tech ids a battle slot can pay for now (Tech_MenuEntry, 20 seven-byte records per PC)."""
        base, found = 0x1A80 + pc * 0x8C, set()
        mp = {b.slot: b.mp for b in self.state.battlers}
        for entry in (self.ram[base + i * 7:base + i * 7 + 7] for i in range(20)):
            # Costs per PC slot sit at +3, +5, +6; $80 and up means that slot takes no part.
            costs = {slot: entry[at] for slot, at in ((0, 3), (1, 5), (2, 6)) if entry[at] < 0x80}
            if entry[0] not in (0, 0xFF) and all(mp.get(slot, 0) >= cost for slot, cost in costs.items()):
                found.add(entry[0])
        return found

    def choose(self, actor: int, banned: set, orders: dict) -> tuple:
        """(menu, id, target slot) for the PC whose command menu is open."""
        pcs = [b for b in self.state.battlers if b.slot < 3 and b.present]
        foes = [b for b in self.state.battlers if b.slot >= 3 and b.present and b.hp > 0]
        me = next(b for b in pcs if b.slot == actor)
        known = self.techs(actor)
        items = {self.ram[0x1580 + i * 5] for i in range(250)}
        down = [b for b in pcs if b.hp == 0]
        if down and ("item", REVIVE) not in banned and REVIVE in items:
            return ("item", REVIVE, down[0].slot)
        hurt = min((b for b in pcs if b.hp), key=lambda b: b.hp / max(b.max_hp, 1))
        low = hurt.hp * 100 <= hurt.max_hp * orders.get("heal_below", 45)
        # Top up with spells before the last ordinary enemy falls: there is no free healing after.
        mop_up = len(foes) == 1 and foes[0].hp < 100 and hurt.hp * 100 <= hurt.max_hp * 75
        tonic = next((i for i in TONICS[::-1] if i in items and ("item", i) not in banned), None)
        if mop_up and tonic and self.stock(tonic) >= 5:
            return ("item", tonic, hurt.slot)  # cheap and plentiful: keep MP for bosses
        if low or mop_up:
            for tech in HEALS.get(me.identity, ()):
                if tech in known and ("tech", tech) not in banned:
                    return ("tech", tech, hurt.slot)
        if low:
            missing = hurt.max_hp - hurt.hp
            for item in (TONICS if missing > 120 else TONICS[::-1]):
                if item in items and ("item", item) not in banned:
                    return ("item", item, hurt.slot)
        if me.mp < orders.get("ether_below", 0):
            for item in (0xC0, 0xC1, 0xC2):
                if item in items and ("item", item) not in banned:
                    return ("item", item, actor)
        focus = None
        if foes:
            wanted = orders.get("focus", ())
            slots = orders.get("slots", ())
            focus = min(foes, key=lambda b: (slots.index(b.slot) if b.slot in slots else len(slots),
                                             wanted.index(b.identity) if b.identity in wanted else len(wanted),
                                             b.hp)).slot
        # MP only comes back at inns and shelters: by default it is kept for bosses.
        spend = orders.get("spend", any(b.max_hp >= 300 for b in foes))
        if spend:
            for tech in orders.get("techs", {}).get(str(me.identity), STRIKES.get(me.identity, ())):
                if tech in known and ("tech", tech) not in banned:
                    return ("tech", tech, focus)
        return ("attack", 0, focus)

    def fight(self, orders: dict | None = None) -> tuple[str, ...]:
        # Some enemies cannot be hurt by what the party has (the 2300 AD Shadows before
        # magic). When enemy HP has not moved for a long while, hold L+R and run.
        orders = orders or {}
        foes = sum(b.hp for b in self.state.battlers if b.slot >= 3)
        watch = self._watch
        if watch is None or self.frame - watch["seen"] > 30:
            watch = self._watch = {"hp": foes, "since": self.frame, "seen": self.frame, "flee": False}
        watch["seen"] = self.frame
        if foes != watch["hp"]:
            watch.update(hp=foes, since=self.frame)
        if orders.get("flee") or (orders.get("escape", True)
                                 and self.frame - watch["since"] > orders.get("stall", 2400)):
            if not watch["flee"]:
                self.picks.append({"frame": self.frame, "actor": None, "pick": ["flee", 0, None],
                                   "hp": [b.hp for b in self.state.battlers]})
            watch["flee"] = True
        if watch["flee"] and self.state.battle_phase != "victory":
            return ("L", "R")
        names = self.command(orders)
        if names:
            self._pressed = self.frame
        return names

    def command(self, orders: dict | None = None) -> tuple[str, ...]:
        """Battle menus from RAM: Attack / Tech / Item rows, list cursors, target cursor."""
        s, ram = self.state, self.ram
        tap = self.frame % 8 < 2
        if s.battle_phase == "victory":
            return self.confirm()
        if s.command_actor is None:
            self._turn = None
            return ()
        turn = self._turn
        if not turn or turn["actor"] != s.command_actor:
            turn = self._turn = {"actor": s.command_actor, "banned": set(), "moves": 0, "pick": None}
        if not tap:
            return ()
        if turn["pick"] is None:
            turn["pick"] = self.choose(s.command_actor, turn["banned"], orders or {})
            self.picks.append({"frame": self.frame, "actor": s.command_actor, "pick": list(turn["pick"]),
                               "hp": [b.hp for b in s.battlers]})
            turn["moves"] = 0
        menu, wanted, target = turn["pick"]
        row = {"attack": 0, "tech": 1, "item": 2}[menu]

        def give_up():
            turn["banned"].add((menu, wanted))
            turn["pick"] = None
            return ("B",)

        if s.battle_phase == "target":
            turn["moves"] += 1
            if target is None or s.target_slot == target or not ram[0x960A] or turn["moves"] > 14:
                turn["pick"] = None  # a refused confirm is chosen again next tap
                return ("A",)
            return ("RIGHT",)
        submenu = ram[0x95DB]
        if submenu == 0:
            at = ram[0x95DC + s.command_actor]
            return ("A",) if at == row else ("DOWN",) if at < row else ("UP",)
        if submenu != row:
            return ("B",)
        if submenu == 1 and self.frame - self._pressed < 14:
            return ()  # the copy of the entry under the list cursor lags the cursor
        turn["moves"] += 1
        if turn["moves"] > (30 if submenu == 1 else 270):
            return give_up()
        if submenu == 1:
            if ram[0x9EE3] != wanted:
                return ("DOWN",)
            # Flags bit 7: not enough MP (or a partner is not ready) once the row is graded.
            if ram[0xA099] and ram[0x9EE5] & 0x80:
                return give_up()
            return ("A",)
        # The item list keeps inventory order with empty records between entries.
        at = ram[0x95E6] + ram[0x95E5]
        index = next((i for i in range(250) if ram[0x1580 + 5 * i] == wanted), None)
        if index is None:
            return give_up()
        return ("A",) if at == index else ("DOWN",) if at < index else ("UP",)

    def drive(self, row: dict) -> Outcome:
        """Routed rows: exit / go / talk. Dialogue and battles on the way are handled here."""
        start, kind = self.frame, row["kind"]
        timeout = row.get("timeout", 6000)
        goal = row.get("goal") or ({"location": row["to"], "ready": True} if kind == "exit" else None)
        legs = None
        where = subject = None
        stalled = replans = stable = fought = 0
        # An exit into the same location (tower floors) is done once the screen has faded and returned.
        hop = kind == "exit" and "goal" not in row and row["to"] == self.state.location
        faded = spoke = False
        while self.frame - start < timeout:
            s, age = self.state, self.frame - start
            faded |= not s.ready
            spoke |= bool(s.message)
            if kind == "talk" and "goal" not in row:
                # A chest or a one-line NPC: done when its message has come and gone.
                goal = {"message": 0, "control": True, "ready": True} if spoke else {"location": -1}
            if hop:
                goal = {"ready": True} if faded else {"location": -1}
            if s.battle_phase == "defeat" or s.battle and s.last_battle_result == 1:
                return Outcome(False, "party_defeated", age, s)
            live = (not s.battle and not s.menu and not s.message and s.ready and s.x is not None
                    and (s.location >= 0x1F0 or s.control)
                    # Object dialogue can temporarily show an overworld during a
                    # cutscene (Arris's recording). There is no NPC to approach there.
                    and not (kind == "talk" and "object" in row and s.location >= 0x1F0))
            if goal:
                satisfied = self.matches(goal)
            else:
                satisfied = live and legs == [] and kind == "go"
            stable = stable + 1 if satisfied else 0
            if stable >= (20 if goal else 1):
                return Outcome(True, "goal", age, s)
            if satisfied:
                self.tick(())
                continue
            if s.battle:
                names, legs = self.fight(row.get("battle")), None
                fought += 1
                if fought > row.get("battle_timeout", 60000):
                    return Outcome(False, "battle_timeout", age, s)
                start += 1  # the row's timeout budgets travel and dialogue, not combat
            elif s.naming or s.message:
                names = self.confirm()
                if s.message and "choice" in row and self.frame % 24 == 12:
                    names = (row["choice"],)
            elif not live:
                names = ()
            else:
                here = (s.location, int(s.x), int(s.y))
                if kind == "talk" and row.get("chase") and self.frame % 4 == 0:
                    at = self.subject(row)
                    if at != subject:
                        legs, subject = None, at
                if legs and legs[-1][1] in (-1, 1 << 20) and self.frame % 30 == 0:
                    legs = None  # leaning on a door: see whether the way is open yet
                if legs is None or stalled >= 45:
                    if stalled >= 45:
                        replans += 1
                        if replans > row.get("replans", 40):
                            return Outcome(False, "movement_stalled", age, s)
                        if legs and s.location < 0x1F0 and legs[0][0] in DIRS and replans % 2 == 0:
                            # Twice refused at the same spot: an unmodelled blocker. Route around it.
                            self.refused.setdefault(s.location, set()).add((here[1], here[2], legs[0][0]))
                    legs, stalled = self.plan(row), 0
                    if legs is None:
                        return Outcome(False, "no_path", age, s)
                while legs and {"RIGHT": here[1] >= legs[0][1], "LEFT": here[1] <= legs[0][1],
                                "DOWN": here[2] >= legs[0][2], "UP": here[2] <= legs[0][2]}[legs[0][0]]:
                    if here[1:] != legs[0][1:]:
                        legs = self.plan(row)  # slid or overshot: route again from here
                        if legs is None:
                            return Outcome(False, "no_path", age, s)
                        break
                    legs.pop(0)
                if legs:
                    names = (legs[0][0],)
                elif kind == "talk":
                    at = self.subject(row)
                    dx, dy = at[0] - here[1], at[1] - here[2]
                    direction = ("DOWN" if dy > 0 else "UP") if abs(dy) > abs(dx) else ("RIGHT" if dx > 0 else "LEFT")
                    facing = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}[direction]
                    names = (direction,) if self.ram[0x1600 + word(self.ram, 0x197)] != facing else self.confirm()
                    if max(abs(dx), abs(dy)) > 40:
                        legs = None  # the NPC walked away
                elif kind == "exit":
                    # Overworld entrances open on A; a field exit that did not fire wants a push.
                    names = self.confirm() if s.location >= 0x1F0 else tuple(row.get("push", ()))
                    stalled += 1
                else:
                    names = tuple(row.get("push", ()))
                stalled = stalled + 1 if here == where and legs else 0 if legs else stalled
                where = here
                if kind == "talk" and row.get("chase"):
                    # Catch moving NPCs with run + confirm. Walk the last pixels of
                    # each leg so a two-pixel running step cannot skip a narrow turn.
                    if legs and names and max(abs(here[1] - legs[0][1]), abs(here[2] - legs[0][2])) > 3:
                        names += ("B",)
                    at = self.subject(row)
                    if "A" not in names and max(abs(s.x - at[0]), abs(s.y - at[1])) <= 48 and self.frame % 4 == 0:
                        names += ("A",)
            if not s.battle and not s.menu:
                names += tuple(row.get("hold", ()))
            self.tick(names)
        return Outcome(False, "timeout", self.frame - start, self.state)

    def stock(self, item: int) -> int:
        return sum(self.ram[0x2500 + i] for i in range(242) if self.ram[0x2400 + i] == item)

    def pulse(self, *names: str, wait: int = 14) -> None:
        for held in (names, names, *[()] * wait):
            self.tick(held)

    def shop(self, row: dict) -> Outcome:
        """Talk to a merchant, raise each [item, stock] in row["buy"] to that stock, and leave.

        Shop menu (mode $0C): A on the top row opens the buy list; A on an entry opens a
        count window where UP adds ten and RIGHT adds one; A buys and returns to the list.
        """
        start = self.frame
        opened = self.drive({**row, "kind": "talk", "goal": {"menu": True}})
        if not opened.passed:
            return opened
        for _ in range(200):
            self.tick(())
            if self.ram[0x67] == 0x0C and self.ram[0x104A]:
                break
        else:
            return Outcome(False, "not_a_shop", self.frame - start, self.state)
        for _ in range(40):
            self.tick(())
        ids = list(self.ram[0x1000:0x1000 + self.ram[0x104A]])
        self.pulse("A", wait=30)
        for item, want in row["buy"]:
            if item not in ids:
                return Outcome(False, "not_sold_here", self.frame - start, self.state)
            price = word(self.ram, 0x6E00 + item * 2)
            for _ in range(4):
                have = self.stock(item)
                count = min(want - have, self.state.gold // max(price, 1), 99 - have)
                if count <= 0:
                    break
                for _ in range(40):
                    at = self.ram[0x1040] + self.ram[0x1041]
                    if at == ids.index(item):
                        break
                    self.pulse("DOWN" if at < ids.index(item) else "UP")
                self.pulse("A", wait=30)
                for _ in range((count - 1) // 10):
                    self.pulse("UP")
                for _ in range((count - 1) % 10):
                    self.pulse("RIGHT")
                total = int.from_bytes(self.ram[0x1044:0x1047], "little")
                if total != price * count:
                    self.pulse("B", wait=30)  # count window did not follow: close it and retry
                    continue
                self.pulse("A", wait=40)
            if self.stock(item) < min(want, 99) and not row.get("partial"):
                return Outcome(False, "purchase_failed", self.frame - start, self.state)
        for _ in range(12):
            if not self.state.menu:
                break
            self.pulse("B", wait=40)
        done = self.play_row({"kind": "confirm", "timeout": 1200,
                              "goal": {"menu": False, "control": True, "ready": True, "message": 0}})
        return Outcome(done.passed, done.reason, self.frame - start, self.state)

    def outfit(self, row: dict) -> Outcome:
        """Equip the highest-numbered weapon, helmet and armor each member can use.

        Menu page 1: the cursor ($54) starts on the member list; A moves it to that
        member's slots (0 weapon, 1 helmet, 2 armor, 3 accessory), where $0F00 lists
        what the inventory offers for the slot; A enters the list, A again equips.
        Within one kind of gear the item ids rise with strength.
        """
        start = self.frame

        def until(test, limit=240):
            for _ in range(limit):
                if test():
                    return True
                self.tick(())
            return False

        def cursor(value):
            for _ in range(8):
                if self.ram[0x54] == value:
                    return True
                self.pulse("DOWN" if self.ram[0x54] < value else "UP", wait=20)
            return self.ram[0x54] == value

        self.pulse("X", wait=4)
        if not until(lambda: self.state.menu and self.ram[0x67] == 0):
            return Outcome(False, "menu_did_not_open", self.frame - start, self.state)
        for _ in range(40):
            self.tick(())
        self.pulse("A", wait=40)
        changed = []
        members = [c for c in self.ram[0x2980:0x2983] if c < 7]
        for index, char in enumerate(members):
            if not cursor(index):
                break
            self.pulse("A", wait=30)
            for slot, offset in ((0, 0x29), (1, 0x27), (2, 0x28)):
                if not cursor(slot):
                    break
                for _ in range(20):
                    self.tick(())
                worn = self.ram[0x2600 + char * 0x50 + offset]
                offered = list(self.ram[0xF00:0xF00 + min(self.ram[0xFC6], 60)])
                if not offered or max(offered) <= worn:
                    continue
                self.pulse("A", wait=30)
                for _ in range(offered.index(max(offered))):
                    self.pulse("DOWN", wait=20)
                self.pulse("A", wait=30)
                if self.ram[0x2600 + char * 0x50 + offset] == max(offered):
                    changed.append((char, max(offered)))
                else:
                    self.pulse("B", wait=30)
            self.pulse("B", wait=30)
        for _ in range(8):
            if not self.state.menu:
                break
            self.pulse("B", wait=40)
        self.equipped = changed
        done = self.play_row({"kind": "wait", "timeout": 600, "frames": 1,
                              "goal": {"menu": False, "control": True, "ready": True, "message": 0}})
        return Outcome(done.passed, done.reason, self.frame - start, self.state)

    def play_row(self, row: dict) -> Outcome:
        if row["kind"] in {"exit", "go", "talk"}:
            return self.drive(row)
        if row["kind"] == "outfit":
            return self.outfit(row)
        if row["kind"] == "shop":
            return self.shop(row)
        start = self.frame
        kind = row["kind"]
        timeout = row.get("timeout", 1800)
        goal = row.get("goal")
        target = row.get("target")
        arrived = False
        stable = 0
        previous = None
        unchanged = 0
        unknown_menu = 0
        frames = row.get("frames", 0)
        enemies_seen: set[int] = set()
        enemies_defeated: set[int] = set()
        party_survived = False
        while self.frame - start < timeout:
            s = self.state
            age = self.frame - start
            if kind == "battle":
                if s.battlers:
                    party_survived = any(b.slot < 3 and b.present and b.hp > 0 for b in s.battlers)
                for b in s.battlers:
                    if b.slot >= 3:
                        enemies_seen.add(b.slot)
                        if b.hp == 0:
                            enemies_defeated.add(b.slot)
                if s.battle_phase == "defeat" or s.last_battle_result == 1:
                    return Outcome(False, "party_defeated", age, s)
            near = target is not None and s.ready and s.x is not None and s.y is not None and (
                abs(s.x - target[0]) <= 1.5 and abs(s.y - target[1]) <= 1.5)
            arrived |= near
            satisfied = self.matches(goal) if goal else near if kind == "walk" else age >= frames
            if kind == "battle":
                satisfied = (satisfied and party_survived and bool(enemies_seen) and enemies_seen <= enemies_defeated
                             and not s.battle and s.ready and s.control and s.last_battle_result == 0)
            stable = stable + 1 if satisfied else 0
            if stable >= (20 if goal else 1):
                return Outcome(True, "goal", age, s)
            if satisfied:
                # Settle the leave predicate without starting another conversation
                # or moving away from the goal during its confirmation window.
                self.tick(())
                continue
            if kind in {"walk", "touch", "interact"} and s.ready and not s.menu and s.location != row["location"]:
                if not goal or s.location != goal.get("location"):
                    return Outcome(False, "unexpected_location", age, s)
            if kind == "confirm":
                names = self.confirm()
            elif kind == "press":
                names = tuple(row.get("buttons", []))
                if row.get("pulse") and age % 24 >= 2:
                    names = ()
            elif kind == "wait":
                names = ()
            elif kind == "battle":
                if s.battle:
                    if s.battle_phase == "command" and s.command_row != 0:
                        names = ("UP",) if self.frame % 24 < 2 else ()
                    elif s.battle_phase in {"command", "target", "victory"}:
                        names = self.confirm()
                    elif s.battle_phase == "submenu":
                        return Outcome(False, "unexpected_battle_submenu", age, s)
                    else:
                        names = ()
                else:
                    names = self.confirm() if s.message else ()
            elif kind in {"walk", "touch", "interact"}:
                unknown_menu = unknown_menu + 1 if s.menu and not s.naming else 0
                if unknown_menu > 120:
                    return Outcome(False, "unexpected_menu", age, s)
                if s.battle:
                    if not goal or not (goal.get("battle") or goal.get("battle_phase")):
                        return Outcome(False, "unexpected_battle", age, s)
                    names = ()
                elif s.menu and not s.naming:
                    names = ()
                elif s.naming or s.message:
                    names = self.confirm()
                elif not s.ready or s.location != row["location"] or s.location < 0x1F0 and not s.control:
                    names = ()
                elif kind == "walk" and arrived and goal:
                    names = tuple(row.get("push", []))
                elif s.x is None or s.y is None:
                    names = ()
                else:
                    if kind in {"touch", "interact"}:
                        slot = row["object"] * 2
                        target = [word(self.ram, 0x1800 + slot) / 16,
                                  word(self.ram, 0x1880 + slot) / 16]
                    dx, dy = target[0] - s.x, target[1] - s.y
                    axes = [(dx, "RIGHT", "LEFT"), (dy, "DOWN", "UP")]
                    if row.get("axis") == "y" or kind in {"touch", "interact"} and abs(dy) > abs(dx):
                        axes.reverse()
                    names = ()
                    for delta, positive, negative in axes:
                        if abs(delta) > 1.5:
                            names = (positive if delta > 0 else negative,)
                            break
                    if kind == "interact" and max(abs(dx), abs(dy)) <= 24 and min(abs(dx), abs(dy)) <= 6:
                        # Face the NPC before a pulsed confirm. Facing 0/1/2/3 = up/down/left/right.
                        direction = ("DOWN" if dy > 0 else "UP") if abs(dy) > abs(dx) else ("RIGHT" if dx > 0 else "LEFT")
                        slot = word(self.ram, 0x197)
                        facing = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}[direction]
                        names = (direction,) if self.ram[0x1600 + slot] != facing else self.confirm()
                key = (s.location, s.x, s.y)
                unchanged = unchanged + 1 if key == previous and names and not s.menu and not s.message else 0
                previous = key
                if unchanged >= 180:
                    return Outcome(False, "movement_stalled", age, s)
            else:
                raise ValueError(f"unknown row kind: {kind}")
            self.tick(names)
        return Outcome(False, "timeout", self.frame - start, self.state)

    def save_pin(self, name: str) -> None:
        path = GAME.save_state(self.env, name)
        provenance = {
            "rom_sha1": ROM_SHA1, "start_state": self.start_state, "frames": self.frame,
            "state_loads": self.state_loads, "ram_writes": 0,
            "state_sha256": hashlib.sha256(read_state_bytes(path)).hexdigest(),
            "snapshot": self.state.record(),
        }
        path.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    def run(self, rows: list[dict]) -> bool:
        try:
            for row in rows:
                result = self.play_row(row)
                self.rows.append({"id": row["id"], "passed": result.passed, "reason": result.reason,
                                  "frames": result.frames, "end_frame": self.frame,
                                  "snapshot": result.state.record()})
                s = result.state
                print(f"{row['id']:22} {'ok ' if result.passed else 'FAIL'} {result.reason:16} {result.frames:6} "
                      f"f={self.frame:7} loc={s.location:03X} story={s.story:02X} party={list(s.party)} "
                      f"hp={s.hp}/{s.max_hp} mp={s.mp}/{s.max_mp} gold={s.gold}", flush=True)
                if result.passed:
                    if row.get("pin", True):
                        self.save_pin(row["id"])
                else:
                    self.save_pin("Frontier")
                    return False
            return True
        finally:
            self.atlas.sample(self.state, self.ram, self.frame)
            self.atlas.write(self.output)
            report = {
                "rom_sha1": ROM_SHA1, "start_state": self.start_state,
                "continuous_from_power_on": self.start_state is None,
                "state_loads": self.state_loads, "mid_run_state_loads": 0, "ram_writes": 0,
                "frames": self.frame, "wall_seconds": round(time.monotonic() - self.started, 2),
                "route_passed": len(self.rows) == len(rows) and all(r["passed"] for r in self.rows),
                "game_complete": False, "rows": self.rows, "final": self.state.record(),
            }
            (self.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            (self.output / "inputs.json").write_text(json.dumps(self.tape, indent=2) + "\n")
            (self.output / "battles.json").write_text(json.dumps(self.battle_events, indent=2) + "\n")
            import pygame
            observation = self.env.render() if self.env.render_mode != "human" else self.env.get_screen()
            pygame.image.save(pygame.surfarray.make_surface(observation.swapaxes(0, 1)), str(self.output / "final.png"))
            self.env.close()


def load_rows(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("route/tape must be a nonempty JSON array")
    ids = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("each route row must be an object")
        # A recorded tape is also a route input, through the same Composer.
        row.setdefault("id", f"Input{i:04d}")
        row.setdefault("kind", "press")
        row.setdefault("pin", False)
        kind = row["kind"]
        if kind in {"exit", "go", "talk", "shop", "outfit"}:
            continue  # routed rows fail at run time with a named reason
        if kind not in {"walk", "touch", "interact", "confirm", "press", "wait", "battle"}:
            raise ValueError(f"unknown row kind: {kind}")
        if row["id"] in ids or not row["id"].replace("_", "").isalnum():
            raise ValueError("row IDs must be unique alphanumeric pin names (underscores allowed)")
        ids.add(row["id"])
        if kind in {"wait", "press"}:
            if not isinstance(row.get("frames"), int) or row["frames"] < 1:
                raise ValueError("press/wait rows require positive integer frames")
            row.setdefault("timeout", row["frames"] + 1)
        if not isinstance(row.get("timeout", 1800), int) or row.get("timeout", 1800) <= 0:
            raise ValueError("row timeout must be a positive integer")
        if kind in {"walk", "touch", "interact"} and row.get("location") not in range(512):
            raise ValueError("movement rows require a location ID in 0..511")
        if kind == "walk":
            target = row.get("target")
            if not isinstance(target, list) or len(target) != 2 or not all(
                isinstance(v, (int, float)) and math.isfinite(v) for v in target
            ):
                raise ValueError("walk rows require a finite [x, y] target")
        if kind in {"touch", "interact"} and row.get("object") not in range(64):
            raise ValueError("object ID must be in 0..63")
        if kind in {"confirm", "touch", "interact", "battle"} and not row.get("goal"):
            raise ValueError(f"{kind} rows require a goal")
        if "goal" in row and (not isinstance(row["goal"], dict) or not row["goal"] or any(
            k not in Snapshot.__dataclass_fields__ and k not in GOAL_KEYS
            for k in row["goal"]
        )):
            raise ValueError("goal must be a nonempty object of snapshot predicates")
        buttons(*row.get("buttons", []))  # Validate all button names before opening an emulator.
        buttons(*row.get("push", []))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("setup", help="normalize and link a legally supplied US v1.0 ROM")
    setup.add_argument("source", type=Path, nargs="?", default=GAME_DIR.parents[1] / "roms/Super Nintendo/Chrono Trigger.zip")
    maps = sub.add_parser("maps", help="extract local ROM location and exit graph")
    maps.add_argument("--path", nargs=2, type=lambda s: int(s, 0), metavar=("FROM", "TO"))
    run = sub.add_parser("run", help="play opening route or a JSON route/input tape")
    run.add_argument("--route", default="route", help="route, opening, or a route/tape JSON file")
    run.add_argument("--state", help="named practice checkpoint; omission starts from power-on")
    run.add_argument("--from", dest="first", metavar="ROW",
                     help="practice: start at this row from its predecessor's pin (or --state)")
    run.add_argument("--until", metavar="ROW", help="stop after this row")
    run.add_argument("--output", type=Path, default=GAME_DIR / "recordings" / "opening")
    run.add_argument("--headed", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "setup":
            setup_rom(args.source)
        elif args.command == "maps":
            graph = exit_graph(ROM.read_bytes())
            GENERATED.mkdir(parents=True, exist_ok=True)
            (GENERATED / "locations.json").write_text(json.dumps(graph, indent=2) + "\n")
            print(f"{len(graph['locations'])} locations; {sum(len(r['exits']) for r in graph['locations'])} exits -> {GENERATED}")
            if args.path:
                print("Geometry only (event prerequisites unverified):", geometry_path(graph, *args.path))
        else:
            rows = {"route": ROUTE, "opening": OPENING}.get(args.route) or load_rows(Path(args.route))
            ids = [row["id"] for row in rows]
            state = args.state
            if args.first:
                at = ids.index(args.first)
                state = state or ids[at - 1]
                rows = rows[at:]
            if args.until:
                rows = rows[:[row["id"] for row in rows].index(args.until) + 1]
            return 0 if Runner(state=state, output=args.output, headed=args.headed).run(rows) else 1
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"chrono_trigger: {error}\n")
    return 0
