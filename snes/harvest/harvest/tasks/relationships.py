"""Bounded social stops on an existing town errand, with measured receipts.

The wife keeps her own family routine. These stops grow the other relationships
gradually; an interaction is never inferred from proximity to a sprite.
"""
from dataclasses import dataclass, field
import os

from retro_harness import ActionResult, Task, TaskResult, TaskStatus

from harvest.core.npc_catalog import ROMANCE_NPCS, ROMANCE_MARRIAGE_BITS, game_objects
from harvest.core.ram_catalog import read_ram_value as value
from harvest.core.ram_catalog import read_ram_u16
from harvest.calendar_schedule import festival_on
from harvest.planner.tasks.inventory_exit import room_walk_action
from harvest.planner.tasks.navigation import NavTask
from harvest.tasks.nav import Point, get_pos_from_ram, make_action
from harvest.tasks.social import HEART_LO, HEART_HI, gift_delta


# Weekday town actors, as measured by PR-021. Do not use a +/-8 sprite match:
# Nina's first frame is immediately after Eve's last frame.
TOWN_SPRITES = {
    "maria": range(0x230, 0x23C),
    "ann": range(0x224, 0x230),
    "ellen": range(0x23C, 0x244),
    "eve": range(0x244, 0x24B),
    "nina": range(0x24B, 0x255),
}
FLOWER_PRICE = 300
SOCIAL_RESERVE = 5_000


def social_mode():
    """Explicit A/B switch; production defaults stay off until the year bench."""
    mode = os.environ.get("HARVEST_SOCIAL", "off").lower()
    if mode not in {"off", "talk", "gifts"}:
        raise ValueError("HARVEST_SOCIAL must be off, talk or gifts")
    return mode


def flower_affordable(ram):
    """Optional flowers spend only surplus, after the seed receipt has posted."""
    reserve = 99_840 if value(ram, "year") >= 2 else SOCIAL_RESERVE
    return (social_mode() == "gifts" and value(ram, "money") >= reserve + FLOWER_PRICE
            and value(ram, "season") == 0 and not read_ram_u16(ram, 0x196, live_offset=False) & 0xA
            and not (value(ram, "year") == 0 and value(ram, "day") <= 2)
            and festival_on(value(ram, "year"), value(ram, "season"), value(ram, "day")) is None
            and value(ram, "nina_hearts") < HEART_LO
            and not value(ram, "marriage_flags") & ROMANCE_MARRIAGE_BITS["nina"]
            and value(ram, "held_item") == 0 and value(ram, "hour") < 14
            and value(ram, "weekday") in range(1, 6))


def nearby_candidates(ram, *, visited=(), max_detour=240):
    """Only live, non-wife weekday actors within a short walk of this errand.

    The PR-021 schedule is a Spring schedule. Other seasons and weekend actors
    need their own ROM receipts before enabling their automatic stops.
    """
    tilemap = value(ram, "tilemap")
    if (value(ram, "season") != 0 or value(ram, "weekday") not in range(1, 6)
            or value(ram, "hour") >= 16 or value(ram, "game_state") & 0x8000):
        return []
    if festival_on(value(ram, "year"), value(ram, "season"), value(ram, "day")) is not None:
        return []
    pos = get_pos_from_ram(ram)
    married = value(ram, "marriage_flags")
    rows = []
    for girl in ROMANCE_NPCS:
        if girl in visited or married & ROMANCE_MARRIAGE_BITS[girl]:
            continue
        hearts = value(ram, f"{girl}_hearts")
        if hearts >= HEART_LO:
            continue
        held = value(ram, "held_item")
        # Ordinary Spring Nina/Ann lines pay zero. Do not buy that detour
        # with crop time; a held gift makes the stop productive.
        if not held and girl not in {"maria", "ellen"}:
            continue
        if held and (gift_delta(girl, held) <= 0 or hearts + gift_delta(girl, held) > HEART_HI):
            continue
        if tilemap != 4 and not (tilemap == 0x22 and girl == "ann"):
            continue
        for obj in game_objects(ram):
            if not obj.is_npc_candidate or obj.sprite_table_idx not in TOWN_SPRITES[girl]:
                continue
            distance = abs(pos.x - obj.pixel[0]) + abs(pos.y - obj.pixel[1] - 27)
            if distance <= max_detour:
                rows.append((hearts, distance, girl, obj.slot))
    return sorted(rows)


def stand_walk_action(ram, target, face):
    """Walk to an indoor talk stand: BFS to the tile behind it, then a short straight step.

    The interior BFS takes only 0x00/0x01/0xA0/0xC3 cells; a stand can sit on an edge
    cell the ROM still lets the farmer onto (church Maria, (206,134) on 0xA2).
    """
    pos = get_pos_from_ram(ram)
    dx, dy = target[0] - pos.x, target[1] - pos.y
    if abs(dx) <= 24 and abs(dy) <= 24:
        if abs(dy) > 1:
            return make_action(down=dy > 0, up=dy < 0)
        return make_action(right=dx > 0, left=dx < 0)
    back = {"left": (16, 0), "right": (-16, 0), "up": (0, 16), "down": (0, -16)}[face]
    return room_walk_action(ram, (target[0] + back[0], target[1] + back[1]))


@dataclass
class RelationshipTask(Task):
    """Approach one live actor and finish one talk or held-gift interaction.

    No shopping or map changes. The caller owns the trip and the daily visit
    ledger. Gifts require a positive decoded delta that fits below 512.
    """
    girl: str = "nina"
    slot: int = -1
    name: str = "relationship"
    timeout: int = 1_200
    # Venue overrides: sprite ranges by girl, and a fixed ((x, y), face) stand
    # (indoor venues, where the recorded stand is beside the actor, not below).
    sprites: dict | None = None
    stand: tuple | None = None
    cap: int = HEART_LO
    receipt: dict = field(default_factory=dict, init=False)

    def reset(self, world):
        if self.girl not in ROMANCE_NPCS:
            raise ValueError(self.girl)
        self._steps = self._press = self._closed = 0
        self._saw_dialog = False
        self._phase = "walk"
        self._map = value(world.ram, "tilemap")
        self._before = {g: value(world.ram, f"{g}_hearts") for g in ROMANCE_NPCS}
        self._item = value(world.ram, "held_item")
        self._text = 0
        self._nav = None
        self.receipt = {}

    def can_start(self, world):
        return True

    def _result(self, status, reason="", **buttons):
        return TaskResult(status, action=ActionResult(make_action(**buttons)), reason=reason)

    def _finish(self, ram):
        after = {g: value(ram, f"{g}_hearts") for g in ROMANCE_NPCS}
        delta = after[self.girl] - self._before[self.girl]
        self.receipt = {"girl": self.girl, "item": self._item, "text": self._text,
                        "before": self._before[self.girl], "after": after[self.girl],
                        "delta": delta, "frames": self._steps,
                        "conversation": self._saw_dialog}
        wrong = [g for g in ROMANCE_NPCS if g != self.girl and after[g] != self._before[g]]
        ok = self._saw_dialog and not wrong and delta >= 0 and after[self.girl] <= HEART_HI
        if self._item:
            ok = ok and delta > 0 and value(ram, "held_item") == 0
        return self._result(TaskStatus.SUCCESS if ok else TaskStatus.FAILURE,
                            f"social {self.receipt} other_hearts_changed={wrong}")

    def step(self, world):
        self._steps += 1
        ram = world.ram
        if value(ram, "tilemap") != self._map:
            return self._result(TaskStatus.FAILURE, "social map changed")
        lock = value(ram, "input_lock")
        if self._steps > self.timeout and lock == 1:
            return self._result(TaskStatus.FAILURE, "social stop timed out")
        if self._steps > self.timeout + 600:
            return self._result(TaskStatus.FAILURE, "social dialogue did not close")
        if self._phase == "dialog":
            if lock != 1:
                self._saw_dialog = True
                self._text = value(ram, "dialog_text_id")
                self._closed = 0
            elif self._saw_dialog:
                self._closed += 1
                if self._closed >= 8:
                    return self._finish(ram)
                return self._result(TaskStatus.RUNNING)
            elif self._press > 150:
                return self._result(TaskStatus.FAILURE, "social press missed")
            self._press += 1
            # PR-018: Maria's God question and Ellen's animal question reward
            # cursor zero. Other ordinary lines have no choice on these actors.
            if value(ram, "player_action") == 9 and value(ram, "dialog_menu_cursor") != 0:
                return self._result(TaskStatus.RUNNING, up=True)
            return self._result(TaskStatus.RUNNING, a=self._press % 12 < 2)
        if self._before[self.girl] >= self.cap:
            return self._result(TaskStatus.SUCCESS, "social target already reached")
        if self._item:
            gain = gift_delta(self.girl, self._item)
            if gain <= 0 or self._before[self.girl] + gain > HEART_HI:
                return self._result(TaskStatus.FAILURE, "gift unsafe for score window")
            if value(ram, "held_item") != self._item:
                return self._result(TaskStatus.FAILURE, "gift lost before reaching recipient")
        if lock != 1 or value(ram, "game_state") & 0x8000:
            return self._result(TaskStatus.RUNNING)
        sprites = self.sprites or TOWN_SPRITES
        obj = next((o for o in game_objects(ram) if o.slot == self.slot and o.is_npc_candidate
                    and o.sprite_table_idx in sprites[self.girl]), None)
        if obj is None:
            return self._result(TaskStatus.FAILURE, "social actor absent")
        if self._phase == "face":
            self._press += 1
            if self._press <= 6:
                return self._result(TaskStatus.RUNNING, **{self._face: True})
            self._phase, self._press = "dialog", 0
            return self._result(TaskStatus.RUNNING, a=True)
        pos = get_pos_from_ram(ram)
        if self.stand is not None:
            target, self._face = self.stand
        else:
            self._face = "left" if self._map == 0x1E and self.girl == "eve" else "up"
            target = ((obj.pixel[0] + 22, obj.pixel[1] - 2) if self._face == "left" else
                      (obj.pixel[0], obj.pixel[1] + 27))
        if max(abs(pos.x - target[0]), abs(pos.y - target[1])) <= 2:
            self._phase, self._press = "face", 0
            return self._result(TaskStatus.RUNNING)
        if self._map == 4:
            if self._nav is None or self._nav.target_px != Point(*target):
                self._nav = NavTask(target_px=Point(*target), radius=2, timeout=self.timeout)
                self._nav.reset(world)
            result = self._nav.step(world)
            if result.status == TaskStatus.SUCCESS:
                self._nav = None
                return self._result(TaskStatus.RUNNING)
            return result
        if self.stand is not None:
            return TaskResult(TaskStatus.RUNNING, action=ActionResult(stand_walk_action(ram, target, self._face)))
        return TaskResult(TaskStatus.RUNNING, action=ActionResult(room_walk_action(ram, target)))


@dataclass
class ErrandRelationshipsTask(Task):
    """At most two nearby stops, then return to the original errand waypoint."""
    name: str = "errand_relationships"
    visited: set = field(default_factory=set)
    max_stops: int = 2
    recipient: str | None = None
    max_detour: int = 160
    receipts: list = field(default_factory=list, init=False)

    def reset(self, world):
        pos = get_pos_from_ram(world.ram)
        self._origin = (pos.x, pos.y)
        self._map = value(world.ram, "tilemap")
        self._child = None
        self._attempted = set()
        self._steps = 0
        self.receipts = []
        self._returning = False
        self._nav = None

    def can_start(self, world):
        return True

    def step(self, world):
        self._steps += 1
        ram = world.ram
        if self._child is not None:
            result = self._child.step(world)
            if result.status == TaskStatus.RUNNING:
                return result
            receipt = dict(self._child.receipt, status=result.status.name, reason=result.reason)
            self.receipts.append(receipt)
            if receipt.get("conversation"):
                self.visited.add(self._child.girl)
            self._child = None
        if not self._returning and len(self._attempted) < self.max_stops and self._steps < 1_800:
            candidates = nearby_candidates(ram, visited=self.visited | self._attempted,
                                           max_detour=self.max_detour)
            if self.recipient:
                candidates = [row for row in candidates if row[2] == self.recipient]
            if candidates:
                _, _, girl, slot = candidates[0]
                self._attempted.add(girl)
                self._child = RelationshipTask(girl=girl, slot=slot)
                self._child.reset(world)
                return self._child.step(world)
        self._returning = True
        pos = get_pos_from_ram(ram)
        if max(abs(pos.x - self._origin[0]), abs(pos.y - self._origin[1])) <= 3:
            return TaskResult(TaskStatus.SUCCESS, reason=f"social stops={self.receipts}")
        if self._steps > 3_000 or value(ram, "tilemap") != self._map:
            return TaskResult(TaskStatus.FAILURE, reason=f"social return failed stops={self.receipts}")
        if self._map == 4:
            if self._nav is None:
                self._nav = NavTask(target_px=Point(*self._origin), radius=3, timeout=1_200)
                self._nav.reset(world)
            result = self._nav.step(world)
            if result.status == TaskStatus.SUCCESS:
                self._nav = None
                return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action()))
            return result
        return TaskResult(TaskStatus.RUNNING, action=ActionResult(room_walk_action(ram, self._origin)))


@dataclass
class BuyFlowerTask(Task):
    """Shelf $BC at (224,336): pay 300G, keep item $19, close its box.

    The unbought shelf preview is item $49; holding it is not a receipt.
    """
    name: str = "buy_flower"

    def reset(self, world):
        self._money = value(world.ram, "money")
        self._steps = self._press = 0
        self._phase = "walk"
        self.receipt = {}

    def can_start(self, world):
        return value(world.ram, "tilemap") == 0x1C

    def step(self, world):
        self._steps += 1
        ram = world.ram
        lock = value(ram, "input_lock")
        paid = self._money - value(ram, "money")
        if paid == FLOWER_PRICE and value(ram, "held_item") == 0x19 and lock == 1:
            self.receipt = {"flower_cost": paid, "item": 0x19, "frames": self._steps}
            return TaskResult(TaskStatus.SUCCESS, reason=f"flower {self.receipt}")
        if self._steps > 1_400 and lock == 1:
            return TaskResult(TaskStatus.FAILURE, reason=f"flower unverified paid={paid}")
        pos = get_pos_from_ram(ram)
        if self._phase == "walk":
            # Same east-wall stand as the general-store shelves. Back off the
            # wall before changing rows; seed purchases finish at y=342.
            if abs(pos.y - 344) > 1:
                action = (make_action(left=True) if pos.x >= 216 else
                          make_action(down=pos.y < 344, up=pos.y > 344))
            elif pos.x < 217:
                action = make_action(right=True)
            else:
                self._phase = "press"
                action = make_action(right=True)
            return TaskResult(TaskStatus.RUNNING, action=ActionResult(action))
        self._press += 1
        return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action(a=self._press % 12 < 2)))


@dataclass
class ErrandSocial:
    """Shopping hook; holds the suspended errand's optional stop and receipts."""
    visited: set
    receipts: list = field(default_factory=list)
    gold_spent: int = 0
    pending_gift: bool = False
    _child: Task | None = None
    _checked: set = field(default_factory=set)
    _flower_checked: bool = False
    _drop_frames: int = 0

    def step(self, shop, world):
        from harvest.tasks.buy_seeds import town_coords_settled

        ram = world.ram
        tilemap = value(ram, "tilemap")
        if self._child is not None:
            result = self._child.step(world)
            if result.status == TaskStatus.RUNNING:
                return result
            if isinstance(self._child, BuyFlowerTask):
                self.pending_gift = result.status == TaskStatus.SUCCESS
                self.gold_spent = self._child._money - value(ram, "money")
                self.receipts.append(dict(self._child.receipt, status=result.status.name))
            else:
                self.receipts.extend(self._child.receipts)
                self.pending_gift = False
            self._child = None
            shop._child = None
            shop._buy_queue.clear()
            shop._arm_nav(world)
        if social_mode() == "off":
            return None
        free = value(ram, "input_lock") == 1 and not value(ram, "game_state") & 0x8000
        if (not self._flower_checked and shop._bought and tilemap == 0x1C and free):
            self._flower_checked = True
            if flower_affordable(ram) and "nina" not in self.visited:
                shop._buy_queue.clear()
                self._child = BuyFlowerTask()
        if (tilemap == 4 and free and town_coords_settled(ram) and shop._child is not None
                and (not shop._bought or shop._town_ready(ram))):
            # The church-side square is one short westward detour from the
            # seed-shop return. Other errands stay within their own road.
            detour = 400 if shop._bought and shop.shop.tilemap == 0x1C else 160
            candidates = nearby_candidates(ram, visited=self.visited | self._checked,
                                           max_detour=400 if self.pending_gift else detour)
            if self.pending_gift:
                candidates = [row for row in candidates if row[2] == "nina"]
            if candidates:
                self._checked.add(candidates[0][2])
                self._child = ErrandRelationshipsTask(visited=self.visited, max_stops=1,
                                                      recipient=candidates[0][2],
                                                      max_detour=400 if self.pending_gift else detour)
            elif value(ram, "held_item") == 0x19:
                # A missing/failed recipient cannot carry a flower into the
                # crop tools. Drop outside, away from the shop's shelf menus.
                self._drop_frames += 1
                self.pending_gift = False
                if self._drop_frames < 120:
                    return TaskResult(TaskStatus.RUNNING, action=ActionResult(
                        make_action(a=self._drop_frames % 12 < 2)))
        if self._child is not None:
            self._child.reset(world)
            return self._child.step(world)
        return None


# ── Sunday church (group 3 event 1) ────────────────────────────────────────
# Loaded by CODE_83D689 on a clear Sunday once $7F1F68 bit 0 (the first church
# visit) is set; rain, snow, the hurricane eve and $0196 & 0x200 load other
# scripts. Maria (unless wed or $7F1F6A bit 12) and Ann ($7F1F6A bit 14
# changes her line) ask one question per church entry, with no season, heart
# or hour test: choice 0 +4, choice 1 +1. The script then repeats a line that
# pays nothing until the map reloads, so each +4 costs one exit and re-entry.
# Ellen's Sunday talk pays nothing; her gift ladder (+5 crops and forage, +8
# eggs and milk) returns to her main loop and repeats. Anchors and sprite
# bases measured on PR_f1ctl_s1_Y1_Fall_D3 (07:10, 1,550 frames from the
# house); the stands are the recorded Sunday tape's.
CHURCH_TILEMAP = 0x1B
CHURCH_SPRITES = {"maria": range(0x224, 0x228), "ann": range(0x228, 0x22C), "ellen": range(0x23E, 0x242)}
CHURCH_STANDS = {"ann": ((203, 409), "right"), "ellen": ((85, 409), "left"), "maria": ((206, 134), "left")}
CHURCH_TALK_GAIN = {"maria": 4, "ann": 4, "ellen": 0}
# The spawn tile (8,28) is 0xFF, which the interior BFS refuses: walk to the row above, then hold down.
CHURCH_DOOR_PX = (128, 440)
# The church door refuses from 17:00 (calendar_schedule DOOR_FLAGS church 0x02).
CHURCH_LAST_ENTRY_HOUR = 16


def church_talk_wanted(ram, targets, *, dry=()):
    """Girls in the church worth a stop this entry, in walking order from the door."""
    married = value(ram, "marriage_flags")
    held = value(ram, "held_item")
    rows = []
    for girl in CHURCH_STANDS:
        cap = targets.get(girl)
        if cap is None or girl in dry or married & ROMANCE_MARRIAGE_BITS[girl]:
            continue
        hearts = value(ram, f"{girl}_hearts")
        gain = gift_delta(girl, held) if held else CHURCH_TALK_GAIN[girl]
        # Stop before a gain would pass the cap: cap 199 is the pre-wedding guard, and 196 + 4 = 200 unmarried
        # played both girls' 200-heart events (f4all Fall D14/D17; Ann paid nothing the next Sunday).
        if gain <= 0 or hearts + gain > cap or hearts + gain > HEART_HI:
            continue
        rows.append(girl)
    return rows


@dataclass
class ChurchCourtTask(Task):
    """Walk to the Sunday church, talk to each girl below her cap, re-enter, repeat; walk home.

    ``targets`` maps girl to the heart count she stops at. Every stop is a measured
    receipt (before/after hearts, text); a girl whose talk pays nothing is dropped
    for the rest of the visit, so a flag or rain script cannot loop for nothing.
    """
    name: str = "church_court"
    targets: dict = field(default_factory=dict)
    max_entries: int = 8
    timeout: int = 60_000
    receipts: list = field(default_factory=list, init=False)

    def reset(self, world):
        self._steps = self._wait = self._entries = 0
        self._phase = "settle" if value(world.ram, "tilemap") == CHURCH_TILEMAP else "travel"
        self._child = None
        self._queue = None
        self._dry = set()
        self._gained = {girl: 0 for girl in CHURCH_STANDS}
        self.receipts = []

    def can_start(self, world):
        return True

    def _summary(self):
        return f"entries={self._entries} gained={self._gained} frames={self._steps}"

    def _route(self, world, name):
        from harvest.maps.map_routes import ROUTES
        from harvest.planner.tasks.multi_nav import MultiMapNavTask
        self._child = MultiMapNavTask(name=f"{self.name}_{name}", waypoints=list(ROUTES[name]),
                                      timeout=12_000, initial_settle_frames=60, auto_slice=True)
        self._child.reset(world)
        return self._child.step(world)

    def _idle(self, **buttons):
        return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action(**buttons)))

    def step(self, world):
        from harvest.planner.tasks.transitions import screen_faded_in

        self._steps += 1
        ram = world.ram
        if self._steps > self.timeout:
            return TaskResult(TaskStatus.FAILURE, reason=f"church court timed out {self._summary()}")
        if self._child is not None:
            result = self._child.step(world)
            if result.status == TaskStatus.RUNNING:
                return result
            child, self._child = self._child, None
            if isinstance(child, RelationshipTask):
                receipt = dict(child.receipt, status=result.status.name, entry=self._entries)
                self.receipts.append(receipt)
                gain = receipt.get("delta", 0) if result.status == TaskStatus.SUCCESS else 0
                self._gained[child.girl] += max(0, gain)
                if gain <= 0:
                    self._dry.add(child.girl)
                print(f"[CHURCH] entry {self._entries} {value(ram, 'hour'):02d}:{value(ram, 'minute'):02d} "
                      f"{child.girl}: {receipt}", flush=True)
            elif self._phase == "travel":
                if result.status != TaskStatus.SUCCESS and value(ram, "tilemap") != CHURCH_TILEMAP:
                    return TaskResult(TaskStatus.FAILURE, reason=f"church walk failed: {result.reason}")
                self._phase, self._wait = "settle", 0
            elif self._phase == "home":
                status = TaskStatus.SUCCESS if result.status == TaskStatus.SUCCESS else TaskStatus.FAILURE
                return TaskResult(status, reason=f"church court {self._summary()} home={result.reason}")
        tilemap = value(ram, "tilemap")
        lock = value(ram, "input_lock")
        if self._phase == "travel":
            if tilemap == CHURCH_TILEMAP:
                self._phase, self._wait = "settle", 0
                return self._idle()
            return self._route(world, "farm_to_church")
        if self._phase == "settle":
            if tilemap != CHURCH_TILEMAP:
                return self._idle()
            self._wait = self._wait + 1 if lock == 1 and screen_faded_in(ram) else 0
            if self._wait < 20:
                return self._idle()
            self._phase, self._queue = "court", None
        if self._phase == "court":
            if self._queue is None:
                self._queue = church_talk_wanted(ram, self.targets, dry=self._dry)
            while self._queue:
                girl = self._queue.pop(0)
                slot = next((o.slot for o in game_objects(ram) if o.is_npc_candidate
                             and o.sprite_table_idx in CHURCH_SPRITES[girl]), None)
                if slot is None:
                    self._dry.add(girl)
                    continue
                self._child = RelationshipTask(girl=girl, slot=slot, sprites=CHURCH_SPRITES,
                                               stand=CHURCH_STANDS[girl], cap=self.targets[girl],
                                               timeout=1_800)
                self._child.reset(world)
                return self._child.step(world)
            self._entries += 1
            again = church_talk_wanted(ram, self.targets, dry=self._dry)
            more = bool(again and self._entries < self.max_entries
                        and value(ram, "hour") < CHURCH_LAST_ENTRY_HOUR)
            self._phase, self._wait = ("door" if more else "leave"), 0
        if self._phase == "door":
            pos = get_pos_from_ram(ram)
            if tilemap != CHURCH_TILEMAP:
                self._phase, self._wait = "outside", 0
                return self._idle()
            # Latch the exit: holding down leaves the stand, and a re-walk to it would undo the step.
            if not self._wait and max(abs(pos.x - CHURCH_DOOR_PX[0]), abs(pos.y - CHURCH_DOOR_PX[1])) > 3:
                return TaskResult(TaskStatus.RUNNING, action=ActionResult(room_walk_action(ram, CHURCH_DOOR_PX)))
            self._wait += 1
            if self._wait > 240:
                return TaskResult(TaskStatus.FAILURE, reason=f"church exit stuck at {pos} {self._summary()}")
            return self._idle(down=True)
        if self._phase == "outside":
            if tilemap == CHURCH_TILEMAP:
                self._phase, self._wait = "settle", 0
                return self._idle()
            self._wait = self._wait + 1 if lock == 1 and screen_faded_in(ram) else 0
            if self._wait < 8:
                return self._idle()
            return self._idle(up=True)
        if self._phase == "leave":
            self._phase = "home"
            return self._route(world, "church_to_farm")
        return self._idle()


# ── Weekday town questions (group 1 event 0) ───────────────────────────────
# Ellen (648,904) asks 0x00CB in Spring and 0x00C7 in Fall, Nina (520,264) 0x00AF in Summer and 0x00B3 in Winter,
# each on one RNG roll in three; a line that pays nothing returns to her main loop, so talking again re-rolls. Choice
# 0 is +8, then she repeats a done line (Ellen 0x00DB) until the town reloads. Entering the shop beside her and
# walking out reloads it. Measured on PR_f1ctl_s1_Y1_Fall_D4: Ellen 8 -> 40 in four loops, 1-4 talks a loop, the
# shop round trip ~490 frames, ~16-20 game minutes a loop.
TOWN_COURT = {
    # girl: (route in, door (x, y), shop tilemap, route home)
    "ellen": ("farm_to_ellen_weekday", (601, 874), 0x24, "animal_shop_town_to_farm"),
    "nina": ("farm_to_nina_weekday", (600, 262), 0x1C, "nina_weekday_to_farm"),
}
TOWN_COURT_GAIN = 8
TOWN_COURT_TALKS = 8          # conversations per town load before she counts as having no question today
TOWN_COURT_LAST_HOUR = 16     # shops and the girls' daytime script end at 17:00


@dataclass
class TownCourtTask(Task):
    """Walk to a girl's weekday spot, talk until her question pays, reload the town through the shop door, repeat."""
    girl: str = "ellen"
    cap: int = HEART_LO
    loops: int = 4
    name: str = "town_court"
    timeout: int = 60_000
    receipts: list = field(default_factory=list, init=False)

    def reset(self, world):
        if self.girl not in TOWN_COURT:
            raise ValueError(self.girl)
        self._route_in, self._door, self._shop, self._route_home = TOWN_COURT[self.girl]
        self._steps = self._wait = self._done = self._talks = self._gained = 0
        self._phase = "travel" if value(world.ram, "tilemap") != 4 else "settle_town"
        self._child = None
        self._navs = 0
        self.receipts = []

    def can_start(self, world):
        return True

    def _summary(self):
        return f"{self.girl} loops={self._done} gained={self._gained} talks={len(self.receipts)} frames={self._steps}"

    def _route(self, world, name):
        from harvest.maps.map_routes import ROUTES
        from harvest.planner.tasks.multi_nav import MultiMapNavTask
        self._child = MultiMapNavTask(name=f"{self.name}_{name}", waypoints=list(ROUTES[name]),
                                      timeout=12_000, initial_settle_frames=60, auto_slice=True)
        self._child.reset(world)
        return self._child.step(world)

    def _idle(self, **buttons):
        return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action(**buttons)))

    def _settled(self, ram, tilemap):
        from harvest.planner.tasks.transitions import screen_faded_in
        ok = value(ram, "tilemap") == tilemap and value(ram, "input_lock") == 1 and screen_faded_in(ram)
        self._wait = self._wait + 1 if ok else 0
        return self._wait >= 20

    def step(self, world):
        self._steps += 1
        ram = world.ram
        if self._steps > self.timeout:
            return TaskResult(TaskStatus.FAILURE, reason=f"town court timed out {self._summary()}")
        if self._child is not None:
            result = self._child.step(world)
            if result.status == TaskStatus.RUNNING:
                return result
            child, self._child = self._child, None
            if isinstance(child, RelationshipTask):
                receipt = dict(child.receipt, status=result.status.name, loop=self._done)
                self.receipts.append(receipt)
                self._talks += 1
                gain = receipt.get("delta", 0) if result.status == TaskStatus.SUCCESS else 0
                if gain > 0:
                    self._gained += gain
                    self._done += 1
                    print(f"[TOWN_COURT] {self.girl} loop {self._done} +{gain} -> {receipt.get('after')} "
                          f"at {value(ram, 'hour'):02d}:{value(ram, 'minute'):02d}", flush=True)
                    self._phase, self._wait = "door", 0
                elif self._talks >= TOWN_COURT_TALKS or result.status != TaskStatus.SUCCESS:
                    self._phase = "home"   # no question in this load (season, flag) or the walk failed
            elif self._phase == "travel":
                if value(ram, "tilemap") != 4:
                    return TaskResult(TaskStatus.FAILURE, reason=f"town court walk failed: {result.reason}")
                self._phase, self._wait = "settle_town", 0
            elif self._phase == "going_home":
                status = TaskStatus.SUCCESS if result.status == TaskStatus.SUCCESS else TaskStatus.FAILURE
                return TaskResult(status, reason=f"town court {self._summary()} home={result.reason}")
        if self._phase == "travel":
            return self._route(world, self._route_in)
        if self._phase == "talk":
            hearts = value(ram, f"{self.girl}_hearts")
            if (hearts + TOWN_COURT_GAIN > self.cap or self._done >= self.loops
                    or value(ram, "hour") >= TOWN_COURT_LAST_HOUR):
                self._phase = "home"
            else:
                slot = next((o.slot for o in game_objects(ram) if o.is_npc_candidate
                             and o.sprite_table_idx in TOWN_SPRITES[self.girl]), None)
                if slot is None:
                    self._phase = "home"
                else:
                    self._child = RelationshipTask(girl=self.girl, slot=slot, cap=self.cap, timeout=1_500)
                    self._child.reset(world)
                    return self._child.step(world)
        if self._phase == "door":
            from harvest.planner.tasks.navigation import NavTask
            if value(ram, "tilemap") == self._shop:
                self._phase, self._wait = "inside", 0
                return self._idle()
            pos = get_pos_from_ram(ram)
            below = (self._door[0], self._door[1] + 16)
            if abs(pos.x - below[0]) <= 3 and abs(pos.y - below[1]) <= 3 or self._wait:
                self._wait += 1
                if self._wait > 300:
                    return TaskResult(TaskStatus.FAILURE, reason=f"town court door refused {self._summary()}")
                return self._idle(up=True)
            self._navs += 1
            if self._navs > 4 * (self._done + 1):
                return TaskResult(TaskStatus.FAILURE, reason=f"town court cannot reach the door {self._summary()}")
            self._child = NavTask(target_px=Point(*below), radius=3, timeout=1_500)
            self._child.reset(world)
            return self._child.step(world)
        if self._phase == "inside":
            if not self._settled(ram, self._shop):
                return self._idle()
            self._phase, self._wait = "out", 0
        if self._phase == "out":
            if value(ram, "tilemap") != 4:
                self._wait += 1
                if self._wait > 300:
                    return TaskResult(TaskStatus.FAILURE, reason=f"town court shop exit stuck {self._summary()}")
                return self._idle(down=True)
            self._phase, self._wait, self._talks = "settle_town", 0, 0
        if self._phase == "settle_town":
            # Step off the door tile before anything walks: the shop door holds input at (600,888) for a few frames.
            if not self._settled(ram, 4):
                return self._idle()
            self._phase, self._talks = "talk", 0
            return self._idle(down=self._done > 0)
        if self._phase == "home":
            self._phase = "going_home"
            return self._route(world, self._route_home)
        return self._idle()
