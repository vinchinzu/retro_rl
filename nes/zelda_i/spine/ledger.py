"""Run-wide ledger: where the frames, the drops and the flutters went.

Wraps ``env.step`` once (like :class:`zelda_i.runner.VideoTap`), so every
stage loop is measured without any controller knowing. Observation only.

Five books:

- **rooms**: one row per consecutive visit to a ``level:screen``. The slowest
  visits are the stall list; a room's frame total is what a reroute saves.
- **drops**: every enemy floor drop (``0x60`` with an item ObjState) and how
  it ended: ``picked`` (Link on it, or its counter moved), ``expired`` (gone
  while Link stayed), ``left`` (Link changed room first). A heart or fairy
  that is not picked while Link is hurt is a *missed heart*.
- **flutter**: Link's position reversing on one axis within
  ``FLUTTER_WINDOW`` frames (a left-right or up-down twitch on the tape).
  Knockback frames (i-frames) are not counted.
- **room items**: every dungeon room that holds an item (``$00AB`` not
  ``NO_ROOM_ITEM``) and whether its world-flag item bit was set by the time
  Link left. An item left behind is a *missed item* (keys the Survival
  top-up then pays for, maps, heart containers).
- **inventory**: every inventory rise, with the room and whether it came
  from ``play`` (inside a frame) or a ``write`` between frames (assist,
  Survival top-up, shop poke). A load that moves Link is not a write.
- **spends**: every bomb and key decrease in play, so assist grants can be
  compared with actual wall, fight and door costs.
- **forced drop windows**: the first frame with nine uninterrupted kills;
  a bomb finishing the next eligible enemy can force a four-bomb drop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from zelda_i.combat import is_floor_drop
from zelda_i.dungeon.ids import (
    NO_ROOM_ITEM,
    item_code_name,
    BOMB_DROP_STATE,
    CLOCK_DROP_STATE,
    FAIRY_DROP_STATE,
    FIVE_RUPEE_DROP_STATE,
    HEART_DROP_STATE,
    RUPEE_DROP_STATE,
)
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOMBS,
    ADDR_BOOK,
    ADDR_BOOMERANG,
    ADDR_BOW,
    ADDR_BRACELET,
    ADDR_CANDLE,
    ADDR_COMPASS,
    ADDR_FOOD,
    ADDR_HEALTH,
    ADDR_KEYS,
    ADDR_LADDER,
    ADDR_LETTER,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MAGIC_BOOMERANG,
    ADDR_MAGIC_KEY,
    ADDR_MAGIC_SHIELD,
    ADDR_MAP,
    ADDR_POTION,
    ADDR_RAFT,
    ADDR_RING,
    ADDR_ROD,
    ADDR_RUPEES,
    ADDR_RUPEES_TO_ADD,
    ADDR_SWORD,
    ADDR_TRIFORCE,
    ADDR_WHISTLE,
    PLAY_MODE,
    ZeldaSnapshot,
    hearts_held,
    read_snapshot,
    room_item_taken,
)

DROP_KINDS = {
    RUPEE_DROP_STATE: "rupee",
    FIVE_RUPEE_DROP_STATE: "rupee5",
    HEART_DROP_STATE: "heart",
    FAIRY_DROP_STATE: "fairy",
    BOMB_DROP_STATE: "bomb",
    CLOCK_DROP_STATE: "clock",
}
HEALING_KINDS = frozenset({"heart", "fairy"})
# Link's 16x16 box touching an 8x16 item, with a pixel of slack.
PICKUP_DX = 13
PICKUP_DY = 14
FLUTTER_WINDOW = 8
# Inventory bytes the inventory book watches. Containers are ``$066F``'s
# high nibble; health itself is the damage book's.
INVENTORY_ADDRS: dict[str, int] = {
    "sword": ADDR_SWORD, "bombs": ADDR_BOMBS, "arrows": ADDR_ARROWS,
    "bow": ADDR_BOW, "candle": ADDR_CANDLE, "whistle": ADDR_WHISTLE,
    "food": ADDR_FOOD, "potion": ADDR_POTION, "rod": ADDR_ROD,
    "raft": ADDR_RAFT, "book": ADDR_BOOK, "ring": ADDR_RING,
    "ladder": ADDR_LADDER, "magic_key": ADDR_MAGIC_KEY,
    "bracelet": ADDR_BRACELET, "letter": ADDR_LETTER, "compass": ADDR_COMPASS,
    "map": ADDR_MAP, "rupees": ADDR_RUPEES, "keys": ADDR_KEYS,
    "triforce": ADDR_TRIFORCE, "boomerang": ADDR_BOOMERANG,
    "magic_boomerang": ADDR_MAGIC_BOOMERANG, "magic_shield": ADDR_MAGIC_SHIELD,
}
# A rise continues the last row while it keeps coming this close (the HUD
# counts rupees up one a frame).
GAIN_MERGE_FRAMES = 2
_BUTTONS = ("B", "", "s", "S", "U", "D", "L", "R", "A")
TOP_VISITS = 15


def room_key(snap: ZeldaSnapshot) -> str:
    return f"{int(snap.level)}:{int(snap.screen):02x}"


@dataclass
class Visit:
    room: str
    enter_frame: int
    frames: int = 0
    play_frames: int = 0
    damage: float = 0.0
    flutters: int = 0

    def row(self) -> dict[str, Any]:
        return {
            "room": self.room,
            "enter": self.enter_frame,
            "frames": self.frames,
            "play": self.play_frames,
            "damage": round(self.damage, 2),
            "flutters": self.flutters,
        }


@dataclass
class Drop:
    room: str
    slot: int
    kind: str
    frame: int
    x: int
    y: int
    hurt: bool  # Link below full health when it appeared
    bombs_at_spawn: int | None = None
    bomb_capacity: int | None = None
    outcome: str = "open"
    end_frame: int = 0

    def row(self) -> dict[str, Any]:
        row = {
            "room": self.room,
            "kind": self.kind,
            "frame": self.frame,
            "life": self.end_frame - self.frame,
            "outcome": self.outcome,
            "hurt": self.hurt,
            "xy": [self.x, self.y],
        }
        if self.kind == "bomb":
            row["bombs_at_spawn"] = self.bombs_at_spawn
            row["bomb_capacity"] = self.bomb_capacity
            row["bankable"] = (
                self.bomb_capacity is not None
                and self.bombs_at_spawn is not None
                and self.bombs_at_spawn < self.bomb_capacity
            )
        return row


@dataclass
class RoomItem:
    room: str
    item: int
    visits: int = 0
    taken: bool = False

    def row(self) -> dict[str, Any]:
        return {
            "room": self.room,
            "item": item_code_name(self.item),
            "visits": self.visits,
            "taken": self.taken,
        }


@dataclass
class Gain:
    field: str
    room: str
    frame: int
    before: int
    after: int
    source: str  # "play" | "write"
    end_frame: int = 0

    def row(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "room": self.room,
            "frame": self.frame,
            "from": self.before,
            "to": self.after,
            "source": self.source,
        }


def _inventory(ram: Any) -> dict[str, int]:
    held = {name: int(ram[addr]) for name, addr in INVENTORY_ADDRS.items()}
    held["containers"] = (int(ram[ADDR_HEALTH]) >> 4) + 1
    return held


@dataclass
class _Axis:
    sign: int = 0
    frame: int = -1_000


@dataclass
class RunLedger:
    frame: int = 0
    visits: list[Visit] = field(default_factory=list)
    drops: list[Drop] = field(default_factory=list)
    _open: dict[int, Drop] = field(default_factory=dict)
    _prev: ZeldaSnapshot | None = None
    _hearts: float | None = None
    _to_add: int = 0
    _axes: tuple[_Axis, _Axis] = field(default_factory=lambda: (_Axis(), _Axis()))
    room_items: dict[str, RoomItem] = field(default_factory=dict)
    gains: list[Gain] = field(default_factory=list)
    spends: list[dict[str, Any]] = field(default_factory=list)
    forced_drop_windows: list[dict[str, Any]] = field(default_factory=list)
    _last_help_count: int | None = None
    _held: dict[str, int] | None = None
    _in_game: bool = False
    _link: tuple[int, int] | None = None
    # Opt-in per-frame tape: (frame, room, x, y, mode, buttons). ``--trace``.
    trace: list[tuple] | None = None

    # ----------------------------------------------------------------- wiring
    def attach(self, env: Any) -> None:
        """Observe every frame ``env.step`` plays from now on."""
        inner = env.step

        def step(action, *args, **kwargs):
            self.observe_writes(env.get_ram())
            out = inner(action, *args, **kwargs)
            ram = env.get_ram()
            snap = read_snapshot(ram)
            self.observe(snap, to_add=int(ram[ADDR_RUPEES_TO_ADD]), ram=ram)
            if self.trace is not None:
                pressed = "".join(n for n, b in zip(_BUTTONS, action) if b)
                self.trace.append(
                    (self.frame, room_key(snap), int(snap.link_x), int(snap.link_y), int(snap.mode), pressed)
                )
            return out

        env.step = step

    # ------------------------------------------------------------ observation
    def observe_writes(self, ram: Any) -> None:
        """Inventory rises between two frames: assist, top-up, shop poke.

        A save-state load also lands between frames; it moves Link, a poke
        does not, so a load re-bases the book instead of booking gains.
        """
        if self._held is None or not self.visits:
            return
        link = (int(ram[ADDR_LINK_X]), int(ram[ADDR_LINK_Y]))
        if link != self._link:
            self._held = _inventory(ram)
            return
        self._book_gains(_inventory(ram), self.visits[-1].room, "write")

    def observe(
        self, snap: ZeldaSnapshot, *, to_add: int = 0, ram: Any = None
    ) -> None:
        self.frame += 1
        room = room_key(snap)
        if not self.visits or self.visits[-1].room != room:
            self._close_drops("left")
            if self.visits and ram is not None:
                self._close_room_item(self.visits[-1].room, ram)
            self.visits.append(Visit(room=room, enter_frame=self.frame))
            self._prev = None
        visit = self.visits[-1]
        visit.frames += 1
        hearts = hearts_held(snap)
        play = int(snap.mode) == PLAY_MODE
        # Power-on RAM settles through the title and file select; the first
        # play frame is where hearts start meaning health (0:00 read 16h).
        self._in_game = self._in_game or play
        if self._in_game and self._hearts is not None and hearts < self._hearts:
            visit.damage += self._hearts - hearts
        if play:
            visit.play_frames += 1
            self._observe_drop_streak(snap, room)
            self._observe_flutter(snap, visit)
            self._observe_drops(snap, room, hearts, to_add)
            if visit.play_frames == 1:
                self._open_room_item(snap, room)
        if ram is not None:
            self._book_gains(_inventory(ram), room, "play")
            self._link = (int(snap.link_x), int(snap.link_y))
        self._hearts = hearts
        self._to_add = to_add
        self._prev = snap if play else None

    def _observe_drop_streak(self, snap: ZeldaSnapshot, room: str) -> None:
        count = int(snap.help_drop_count)
        if count == 9 and self._last_help_count != 9:
            world = int(snap.world_kill_count)
            self.forced_drop_windows.append({
                "frame": self.frame,
                "room": room,
                "world_kills": world,
                "fairy_preempts_next_kill": world == 15,
            })
        self._last_help_count = count

    def _observe_flutter(self, snap: ZeldaSnapshot, visit: Visit) -> None:
        prev = self._prev
        if prev is None or int(getattr(snap, "link_iframes", 0)) > 0:
            return
        for axis, delta in zip(
            self._axes,
            (int(snap.link_x) - int(prev.link_x), int(snap.link_y) - int(prev.link_y)),
        ):
            if not delta or abs(delta) > 4:  # 0 = standing; >4 = a warp/scroll
                continue
            sign = 1 if delta > 0 else -1
            if axis.sign == -sign and self.frame - axis.frame <= FLUTTER_WINDOW:
                visit.flutters += 1
            axis.sign, axis.frame = sign, self.frame

    def _observe_drops(
        self, snap: ZeldaSnapshot, room: str, hearts: float, to_add: int
    ) -> None:
        live = {
            int(o.slot): o
            for o in snap.objects
            if is_floor_drop(o) and int(o.state) in DROP_KINDS
        }
        for slot, drop in list(self._open.items()):
            obj = live.get(slot)
            if obj is not None and DROP_KINDS[int(obj.state)] == drop.kind:
                drop.x, drop.y = int(obj.x), int(obj.y)
                continue
            gained = (
                hearts > (self._hearts or 0.0)
                if drop.kind in HEALING_KINDS
                else to_add > self._to_add and drop.kind.startswith("rupee")
            )
            near = (
                abs(int(snap.link_x) - drop.x) <= PICKUP_DX
                and abs(int(snap.link_y) - drop.y) <= PICKUP_DY
            )
            self._close(slot, "picked" if near or gained else "expired")
        full = bool(snap.health_is_full)
        for slot, obj in live.items():
            if slot not in self._open:
                drop = Drop(
                    room=room,
                    slot=slot,
                    kind=DROP_KINDS[int(obj.state)],
                    frame=self.frame,
                    x=int(obj.x),
                    y=int(obj.y),
                    hurt=not full,
                    bombs_at_spawn=int(snap.bombs) if DROP_KINDS[int(obj.state)] == "bomb" else None,
                    bomb_capacity=int(snap.max_bombs) if DROP_KINDS[int(obj.state)] == "bomb" else None,
                )
                self._open[slot] = drop
                self.drops.append(drop)

    def _open_room_item(self, snap: ZeldaSnapshot, room: str) -> None:
        if int(snap.level) == 0 or int(snap.room_item_id) == NO_ROOM_ITEM:
            return
        entry = self.room_items.setdefault(
            room, RoomItem(room=room, item=int(snap.room_item_id))
        )
        entry.visits += 1

    def _close_room_item(self, room: str, ram: Any) -> None:
        entry = self.room_items.get(room)
        if entry is None or entry.taken:
            return
        level, _, screen = room.partition(":")
        entry.taken = room_item_taken(ram, int(level), int(screen, 16))

    def _book_gains(self, held: dict[str, int], room: str, source: str) -> None:
        before = self._held
        self._held = held
        if before is None:
            return
        for name, value in held.items():
            old = before.get(name, value)
            if value < old and name in ("bombs", "keys"):
                self.spends.append({
                    "field": name, "room": room, "frame": self.frame,
                    "from": old, "to": value, "source": source,
                })
            if value <= old:
                continue
            last = next((g for g in reversed(self.gains) if g.field == name), None)
            if (
                last is not None
                and last.room == room
                and last.source == source
                and self.frame - last.end_frame <= GAIN_MERGE_FRAMES
                and last.after == old
            ):
                last.after, last.end_frame = value, self.frame
                continue
            self.gains.append(
                Gain(name, room, self.frame, old, value, source, end_frame=self.frame)
            )

    def _close(self, slot: int, outcome: str) -> None:
        drop = self._open.pop(slot)
        drop.outcome = outcome
        drop.end_frame = self.frame

    def _close_drops(self, outcome: str) -> None:
        for slot in list(self._open):
            self._close(slot, outcome)

    # ----------------------------------------------------------------- report
    def missed(self) -> list[Drop]:
        return [d for d in self.drops if d.outcome in ("expired", "left")]

    def report(self, ram: Any = None) -> dict[str, Any]:
        """The books. Pass the live ``ram`` so the room Link is still in
        gets its item graded too."""
        self._close_drops("left")
        if ram is not None and self.visits:
            self._close_room_item(self.visits[-1].room, ram)
        rooms: dict[str, dict[str, Any]] = {}
        for v in self.visits:
            row = rooms.setdefault(
                v.room,
                {"visits": 0, "frames": 0, "max_visit": 0, "flutters": 0, "damage": 0.0},
            )
            row["visits"] += 1
            row["frames"] += v.frames
            row["max_visit"] = max(row["max_visit"], v.frames)
            row["flutters"] += v.flutters
            row["damage"] = round(row["damage"] + v.damage, 2)
        by_kind: dict[str, dict[str, int]] = {}
        for d in self.drops:
            tally = by_kind.setdefault(d.kind, {})
            tally[d.outcome] = tally.get(d.outcome, 0) + 1
        # Under the Survival refill Link is nearly always full, so ``hurt``
        # undercounts: every missed heart is one the refill paid for instead.
        missed_heal = [d for d in self.missed() if d.kind in HEALING_KINDS]
        missed_hearts = [d for d in missed_heal if d.hurt]
        slow = sorted(self.visits, key=lambda v: v.frames, reverse=True)
        items = sorted(self.room_items.values(), key=lambda r: r.room)
        return {
            "frames": self.frame,
            "visits": len(self.visits),
            "flutters": sum(v.flutters for v in self.visits),
            "damage": round(sum(v.damage for v in self.visits), 2),
            "room_items": {
                "rooms": len(items),
                "taken": sum(r.taken for r in items),
                "missed": [r.row() for r in items if not r.taken],
            },
            "gains": [g.row() for g in self.gains],
            "spends": self.spends,
            "forced_drop_windows": self.forced_drop_windows,
            "slowest_visits": [v.row() for v in slow[:TOP_VISITS]],
            "rooms": rooms,
            "drops": {
                "total": len(self.drops),
                "by_kind": by_kind,
                "missed": len(self.missed()),
                "missed_hearts": len(missed_heal),
                "missed_hearts_while_hurt": len(missed_hearts),
                "missed_rows": [d.row() for d in self.missed()],
            },
        }

    def summary_lines(self, ram: Any = None) -> list[str]:
        rep = self.report(ram)
        drops = rep["drops"]
        lines = [
            f"ledger: {rep['frames']}f {rep['visits']} visits "
            f"flutters={rep['flutters']} drops={drops['total']} "
            f"missed={drops['missed']} missed_hearts={drops['missed_hearts']} "
            f"(hurt {drops['missed_hearts_while_hurt']})",
            "  drops by kind: "
            + " ".join(
                f"{k}={'/'.join(f'{o}:{n}' for o, n in sorted(v.items()))}"
                for k, v in sorted(drops["by_kind"].items())
            ),
            "  slowest visits: "
            + " ".join(f"{r['room']}@{r['enter']}={r['frames']}f" for r in rep["slowest_visits"][:10]),
        ]
        flutter_rooms = sorted(
            rep["rooms"].items(), key=lambda kv: kv[1]["flutters"], reverse=True
        )[:10]
        lines.append(
            "  flutter rooms: "
            + " ".join(f"{room}={row['flutters']}" for room, row in flutter_rooms)
        )
        hurt_rooms = sorted(
            rep["rooms"].items(), key=lambda kv: kv[1]["damage"], reverse=True
        )[:10]
        lines.append(
            f"  damage {rep['damage']}h, worst rooms: "
            + " ".join(f"{room}={row['damage']}" for room, row in hurt_rooms if row["damage"])
        )
        items = rep["room_items"]
        lines.append(
            f"  room items: {items['taken']}/{items['rooms']} taken; missed: "
            + (" ".join(f"{r['room']}={r['item']}" for r in items["missed"]) or "-")
        )
        writes = [g for g in rep["gains"] if g["source"] == "write"]
        lines.append(
            f"  inventory: {len(rep['gains'])} rises, {len(writes)} written between frames: "
            + " ".join(f"{g['field']}@{g['room']}" for g in writes[:12])
        )
        return lines
