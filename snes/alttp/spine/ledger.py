"""Run ledger. Observation only: wraps ``env.step`` and never writes RAM.

A hit is an HP drop on ``$F36D``. Rooms re-arm ``$031F`` with no damage;
those frames are not hits. The cause is the nearest active sprite or
Link ancilla at the drop (manhattan), sprite on a tie.

Room visits count settled play frames (module play, submodule 0) under
the room-timer location key. Flutter is an xy sign reversal on one axis
within 8 frames. ``$0046`` knockback is not a flutter, and a step bigger
than a tug is not one either (walk speed is still unmeasured).

Inventory rises and falls are ``play`` (inside a step) or ``write``
(between steps). A link move between steps is a load, not a write. HP
stays on the hit book. Spends are drops in keys, bombs, arrows, magic,
rupees, and the four bottle slots. ``route_rooms`` are the route's chest rooms; on dungeon
exit a clear ``$F000`` chest bit (``0x0010``) is still unopened.

Sprite census counts every active sprite and ancilla on settled frames,
including ``hp == 0``. A type hurts Link when it is the hit cause.
Damage is HP in, HP out, hits by cause, and damage by room, per stage.
The open stage is ``run`` until ``begin_stage``. Closing an attached
env prints that table.

A ``FrameAction`` reason, ``note_reason``, or the bound controller's
``last_reason`` is totaled in ``reason_by_room``. ``watch_stall`` is
opt-in: motion under ``stall_pixels`` across ``stall_frames`` settled
frames writes ``recordings/<tag>_stall.json`` and raises ``Stall``.
"""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from retro_harness.controls import SNES_BUTTON_NAMES, pressed_snes_buttons
from retro_harness.input_script import FrameAction

from alttp.paths import RECORDINGS_DIR
from alttp.primitives import active_ancillae, active_sprites
from alttp.ram import (
    INVENTORY_FIELDS,
    ROOM_FLAG_CHEST,
    read_inventory,
    read_snapshot,
    room_flags,
)
from alttp.room_timer import PLAY_MODULES, location_key
from alttp.spine.tape import load_tape
from alttp.sprite_ids import SPRITE_NAMES, sprite_name

# Last samples kept on each hit, including the drop frame.
HIT_TRAIL = 24
FLUTTER_WINDOW = 8
# Above a 1-2px tug, below a screen warp. AP-047 has not measured walk speed.
FLUTTER_MAX_STEP = 16
TOP_VISITS = 15
# Last samples kept when a stall fires. The task asks for this many.
STALL_DUMP_FRAMES = 240
# Default window: under 4px across 240 settled frames. Opt-in via watch_stall.
STALL_FRAMES = 240
STALL_PIXELS = 4
# Castle soldier block verified in sprite_ids (0x41 blue soldier .. 0x4B knight).
CASTLE_SOLDIER_LO = 0x41
CASTLE_SOLDIER_HI = 0x4B
# HP drops are hits. The rest of the save block is the inventory book.
_SKIP_INVENTORY = frozenset({"hp"})
_SPEND_FIELDS = frozenset(
    {
        "keys",
        "bombs",
        "arrows",
        "magic",
        "rupees",
        "bottle_1",
        "bottle_2",
        "bottle_3",
        "bottle_4",
    }
)
_PIN_SPRITE_RE = re.compile(
    r"^slot\s+(\d+)\s+(.*?)\s+state=0x([0-9A-Fa-f]+)\s+hp=(-?\d+)"
    r"\s+\((-?\d+),(-?\d+)\)\s*$"
)
_NAME_TO_TYPE = {name: type_id for type_id, name in SPRITE_NAMES.items()}
SOLDIER_NAMES = frozenset(
    sprite_name(type_id)
    for type_id in range(CASTLE_SOLDIER_LO, CASTLE_SOLDIER_HI + 1)
)

# Names for the ancilla types ``active_ancillae`` keeps (AP-011, $0C4A).
_ANCILLA_NAMES = {
    0x01: "somaria blast",
    0x02: "fire rod shot",
    0x04: "beam hit",
    0x05: "boomerang",
    0x07: "bomb",
    0x09: "arrow",
    0x0A: "halted arrow",
    0x0B: "ice rod shot",
    0x0C: "sword beam",
    0x11: "ice shot spread",
    0x13: "ice shot sparkle",
    0x1F: "hookshot",
    0x3A: "super bomb explosion",
}

# One trail sample: ((x, y), pressed button names).
TrailSample = tuple[tuple[int, int], tuple[str, ...]]


def _pressed(action: Any) -> tuple[str, ...]:
    if action is None:
        return ()
    try:
        raw = list(action)
    except TypeError:
        return ()
    values: list[int] = []
    for item in raw[: len(SNES_BUTTON_NAMES)]:
        try:
            values.append(int(item))
        except (TypeError, ValueError):
            values.append(0)
    return tuple(pressed_snes_buttons(values))


def _ancilla_name(ancilla_type: int) -> str:
    return _ANCILLA_NAMES.get(int(ancilla_type), f"ancilla_{int(ancilla_type):02X}")


def _nearest_cause(env: Any, snap: Any) -> tuple[str, str, int | None]:
    """Return ``(name, kind, distance)``. Kind is sprite, ancilla, or none."""
    best: tuple[int, int, str, str] | None = None
    for sprite in active_sprites(env):
        cand = (sprite.distance_to(snap), 0, "sprite", sprite_name(sprite.sprite_type))
        if best is None or cand < best:
            best = cand
    for ancilla in active_ancillae(env):
        cand = (
            ancilla.distance_to(snap),
            1,
            "ancilla",
            _ancilla_name(ancilla.ancilla_type),
        )
        if best is None or cand < best:
            best = cand
    if best is None:
        return "", "none", None
    distance, _rank, kind, name = best
    return name, kind, distance


def _inventory(ram: Any) -> dict[str, int]:
    inv = read_inventory(np.asarray(ram))
    return {
        item.name: int(getattr(inv, item.name))
        for item in INVENTORY_FIELDS
        if item.name not in _SKIP_INVENTORY
    }


def _dungeon_id(snap: Any) -> int | None:
    """Indoors palace index, or None on the overworld and in houses (``0xFF``)."""
    if not bool(snap.indoors):
        return None
    dung = int(snap.dungeon_id) & 0xFF
    if dung == 0xFF:
        return None
    return dung


def _settled(snap: Any) -> bool:
    return int(snap.game_mode) in PLAY_MODULES and int(snap.submodule) == 0


def _visit_key(snap: Any) -> tuple[Any, ...]:
    return location_key(
        indoors=bool(snap.indoors),
        room=int(snap.room_id),
        dark_world=bool(snap.dark_world),
        screen=int(snap.screen_id),
    )


def _room_label(key: tuple[Any, ...]) -> str:
    if key[0] == "in":
        return f"in:{int(key[1]):04X}"
    world = 1 if key[1] else 0
    return f"ow:{world}:{int(key[2]):02X}"


class Stall(RuntimeError):
    """Stage failed because Link did not move. The argument is ``stall``."""


def _split_action(action: Any) -> tuple[Any, str]:
    """Return ``(buttons, reason)``. A bare button vector has no reason."""
    if isinstance(action, FrameAction):
        return action.action, str(action.reason or "")
    return action, ""


def _is_soldier(kind: str, type_id: int, name: str) -> bool:
    if kind != "sprite":
        return False
    if CASTLE_SOLDIER_LO <= int(type_id) <= CASTLE_SOLDIER_HI:
        return True
    return name in SOLDIER_NAMES or "soldier" in name


def soldiers_from_rooms(rooms: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Soldier-type names per room, in type order. Empty rooms stay listed."""
    found: dict[str, list[str]] = {}
    for room, book in rooms.items():
        names: list[str] = []
        seen: set[str] = set()
        for row in book.get("types", ()):
            name = str(row.get("name", ""))
            kind = str(row.get("kind", "sprite"))
            type_id = int(row.get("type", -1))
            if name in seen or not _is_soldier(kind, type_id, name):
                continue
            seen.add(name)
            names.append(name)
        found[room] = names
    return found


def _type_for_name(name: str) -> int:
    return int(_NAME_TO_TYPE.get(name, -1))


def _parse_pin_sprites(text: str) -> list[dict[str, Any]]:
    """Sprite lines from a ``pin_probe`` text. ``hp == 0`` rows are kept."""
    found: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = _PIN_SPRITE_RE.match(line.strip())
        if match is None:
            continue
        name = match.group(2).strip()
        found.append(
            {
                "slot": int(match.group(1)),
                "name": name,
                "state": int(match.group(3), 16),
                "hp": int(match.group(4)),
                "x": int(match.group(5)),
                "y": int(match.group(6)),
                "type": _type_for_name(name),
                "kind": "sprite",
            }
        )
    return found


def census_verified_tip(
    tape_path: Path | str, index_path: Path | str
) -> dict[str, Any]:
    """Sprite census of a scanned tape. Does not boot the ROM.

    ``tape_path`` is the button tape (identity and frame count).
    ``index_path`` is ``recordings/scan/index.json``. Sprite rows come
    from each pin's ``pin_probe`` text beside that index. Ancillae are
    not in that text; the live ledger records them.
    """
    tape = load_tape(tape_path)
    index_file = Path(index_path)
    index = json.loads(index_file.read_text(encoding="utf-8"))
    out = Path(str(index.get("out") or index_file.parent))
    rooms: dict[str, dict[str, Any]] = {}
    for pin in index.get("pins") or []:
        label = _room_label(tuple(pin["location"]))
        text_path = out / f"{pin['pin']}.txt"
        sprites = (
            _parse_pin_sprites(text_path.read_text(encoding="utf-8"))
            if text_path.is_file()
            else []
        )
        book = rooms.setdefault(
            label, {"max_alive": 0, "types": [], "hp0": 0, "ancillae": []}
        )
        book["max_alive"] = max(int(book["max_alive"]), len(sprites))
        grouped: dict[str, dict[str, Any]] = {
            str(row["name"]): row for row in book["types"]
        }
        counts: dict[str, int] = {}
        for sprite in sprites:
            counts[sprite["name"]] = counts.get(sprite["name"], 0) + 1
        for name, count in counts.items():
            sample = next(row for row in sprites if row["name"] == name)
            row = grouped.get(name)
            if row is None:
                row = {
                    "name": name,
                    "kind": "sprite",
                    "type": int(sample["type"]),
                    "frames": 0,
                    "max_alive": 0,
                    "hp0": 0,
                    "hurt": None,
                }
                grouped[name] = row
            row["frames"] = int(row["frames"]) + 1
            row["max_alive"] = max(int(row["max_alive"]), count)
            row["hp0"] = int(row["hp0"]) + sum(
                1 for item in sprites if item["name"] == name and int(item["hp"]) == 0
            )
        types = sorted(grouped.values(), key=lambda item: (int(item["type"]), item["name"]))
        book["types"] = types
        book["hp0"] = sum(int(row["hp0"]) for row in types)
    for visit in index.get("visited") or []:
        label = _room_label(tuple(visit["location"]))
        rooms.setdefault(label, {"max_alive": 0, "types": [], "hp0": 0, "ancillae": []})
    tape_frames = int(tape.buttons.shape[0])
    scan_frames = int(index.get("frames") or 0)
    return {
        "tape": Path(tape_path).name,
        "tape_frames": tape_frames,
        "boot_state": tape.boot_state,
        "scan_frames": scan_frames,
        "desynced": int(index.get("desynced") or 0),
        "writes": int(index.get("writes") or 0),
        "loads": int(index.get("loads") or 0),
        "frames_match": tape_frames == scan_frames,
        "rooms": rooms,
        "soldiers_by_room": soldiers_from_rooms(rooms),
        "ancillae": "scan pin text has sprites only; the live ledger records ancillae",
    }


def format_soldier_census(report: dict[str, Any]) -> str:
    """One line per room that the verified-tip scan actually sampled."""
    lines = ["soldiers:"]
    rooms = report.get("rooms") or {}
    for room, names in (report.get("soldiers_by_room") or {}).items():
        if not names and not (rooms.get(room) or {}).get("types"):
            continue
        listed = ", ".join(names) if names else "(none)"
        lines.append(f"  {room}: {listed}")
    return "\n".join(lines)


@dataclass
class TypeSeen:
    name: str
    kind: str
    type_id: int
    frames: int = 0
    max_alive: int = 0
    hp0: int = 0
    hurt: int = 0

    def row(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "type": self.type_id,
            "frames": self.frames,
            "max_alive": self.max_alive,
            "hp0": self.hp0,
            "hurt": self.hurt,
        }


@dataclass
class RoomSprites:
    max_alive: int = 0
    types: dict[tuple[str, int], TypeSeen] = field(default_factory=dict)

    def row(self) -> dict[str, Any]:
        types = [row.row() for row in self.types.values()]
        types.sort(key=lambda item: (item["kind"], item["type"], item["name"]))
        return {"max_alive": self.max_alive, "types": types}


@dataclass
class StageBook:
    name: str
    hp_in: int | None = None
    hp_out: int | None = None
    damage: int = 0
    start_frame: int = 0
    end_frame: int = 0
    hits_by_cause: dict[str, int] = field(default_factory=dict)
    damage_by_room: dict[str, int] = field(default_factory=dict)

    def row(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hp_in": self.hp_in,
            "hp_out": self.hp_out,
            "damage": self.damage,
            "start": self.start_frame,
            "end": self.end_frame,
            "hits_by_cause": dict(self.hits_by_cause),
            "damage_by_room": dict(self.damage_by_room),
        }


@dataclass
class Visit:
    location: tuple[Any, ...]
    enter_frame: int
    frames: int = 0
    flutters: int = 0

    def row(self) -> dict[str, Any]:
        return {
            "room": _room_label(self.location),
            "location": list(self.location),
            "enter": self.enter_frame,
            "frames": self.frames,
            "flutters": self.flutters,
        }


@dataclass
class _Axis:
    sign: int = 0
    frame: int = -1_000


@dataclass(frozen=True)
class InventoryChange:
    field: str
    room: str
    frame: int
    before: int
    after: int
    source: str

    def row(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "room": self.room,
            "frame": self.frame,
            "from": self.before,
            "to": self.after,
            "source": self.source,
        }


@dataclass(frozen=True)
class Hit:
    frame: int
    room: int
    damage: int
    cause: str
    kind: str
    distance: int | None
    link_state: int
    trail: tuple[TrailSample, ...]

    def row(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "room": self.room,
            "damage": self.damage,
            "cause": self.cause,
            "kind": self.kind,
            "distance": self.distance,
            "link_state": self.link_state,
            "trail": [
                {"xy": [x, y], "buttons": list(buttons)}
                for (x, y), buttons in self.trail
            ],
        }


@dataclass
class RunLedger:
    frame: int = 0
    hits: list[Hit] = field(default_factory=list)
    visits: list[Visit] = field(default_factory=list)
    _hp: int | None = None
    _trail: deque[TrailSample] = field(
        default_factory=lambda: deque(maxlen=HIT_TRAIL)
    )
    _axes: tuple[_Axis, _Axis] = field(default_factory=lambda: (_Axis(), _Axis()))
    _prev_xy: tuple[int, int] | None = None
    # Chest rooms the route wants graded. Empty means nothing to list.
    route_rooms: tuple[int, ...] = ()
    changes: list[InventoryChange] = field(default_factory=list)
    spends: list[dict[str, Any]] = field(default_factory=list)
    missed_chests: list[dict[str, Any]] = field(default_factory=list)
    _held: dict[str, int] | None = None
    _link: tuple[int, int] | None = None
    _dungeon: int | None = None
    tag: str | None = None
    watch_stall: bool = False
    stall_frames: int = STALL_FRAMES
    stall_pixels: int = STALL_PIXELS
    print_on_close: bool = True
    recordings: Path | None = None
    blocker: str = ""
    stall_path: str = ""
    _pending_reason: str = ""
    _controller: Any = None
    _sprites: dict[str, RoomSprites] = field(default_factory=dict)
    _reasons: dict[str, dict[str, int]] = field(default_factory=dict)
    _stages: list[StageBook] = field(default_factory=list)
    _open: StageBook | None = None
    _dump: deque[tuple[Any, ...]] = field(default_factory=deque)
    _play: deque[tuple[int, int]] = field(default_factory=deque)
    _stall_room: str | None = None
    _stalled: bool = False
    _close_wrapped: bool = False

    def __post_init__(self) -> None:
        if self.stall_frames < 1:
            raise ValueError(f"stall_frames must be >= 1, got {self.stall_frames}")
        if self.stall_pixels < 1:
            raise ValueError(f"stall_pixels must be >= 1, got {self.stall_pixels}")
        if self.watch_stall and not self.tag:
            raise ValueError("watch_stall needs a tag")
        if self.tag is not None and (
            not self.tag or "/" in self.tag or "\\" in self.tag or self.tag in {".", ".."}
        ):
            raise ValueError(f"stall tag must be a filename, got {self.tag!r}")
        self._dump = deque(maxlen=STALL_DUMP_FRAMES)
        self._play = deque(maxlen=self.stall_frames)

    def note_reason(self, reason: str) -> None:
        """Reason for the next frame when the action has none."""
        self._pending_reason = str(reason)

    def bind_controller(self, controller: Any) -> None:
        """Use ``last_reason`` or ``_last_reason`` when the action has none."""
        self._controller = controller

    def begin_stage(self, name: str) -> None:
        """Close the open stage and start ``name`` at the current HP."""
        if self._open is not None:
            self._open.end_frame = self.frame
            self._stages.append(self._open)
        self._open = StageBook(
            name=str(name),
            hp_in=self._hp,
            hp_out=self._hp,
            start_frame=self.frame,
            end_frame=self.frame,
        )

    def attach(self, env: Any) -> None:
        """Observe every frame ``env.step`` plays from now on."""
        inner = env.step

        def step(action, *args, **kwargs):
            buttons, reason = _split_action(action)
            if reason:
                self._pending_reason = reason
            self.observe_writes(env.get_ram())
            out = inner(buttons, *args, **kwargs)
            self.observe(env, buttons)
            return out

        env.step = step
        if self.print_on_close and not self._close_wrapped:
            close = getattr(env, "close", None)
            if callable(close):
                def wrapped_close(*args, **kwargs):
                    try:
                        self.print_damage()
                    finally:
                        return close(*args, **kwargs)

                env.close = wrapped_close
                self._close_wrapped = True

    def observe(self, env: Any, action: Any = None) -> None:
        """One post-step sample. A hit is recorded only when HP falls."""
        self.frame += 1
        snap = read_snapshot(env.get_ram())
        xy = (int(snap.link_x), int(snap.link_y))
        buttons_action, reason_from_action = _split_action(action)
        if reason_from_action:
            self._pending_reason = reason_from_action
        buttons = _pressed(buttons_action)
        reason = self._take_reason()
        self._trail.append((xy, buttons))
        self._dump.append((self.frame, xy[0], xy[1], buttons, reason))
        room = _room_label(_visit_key(snap))
        settled = _settled(snap)
        self._count_reason(room, reason)
        if settled:
            self._observe_census(env, room)
        hp = int(snap.hp)
        stage = self._ensure_stage()
        if stage.hp_in is None:
            stage.hp_in = self._hp if self._hp is not None else hp
        if self._hp is not None and hp < self._hp:
            damage = self._hp - hp
            cause, kind, distance = _nearest_cause(env, snap)
            self.hits.append(
                Hit(
                    frame=self.frame,
                    room=int(snap.room_id),
                    damage=damage,
                    cause=cause,
                    kind=kind,
                    distance=distance,
                    link_state=int(snap.link_state),
                    trail=tuple(self._trail),
                )
            )
            self._mark_hurt(room, cause)
            stage.damage += damage
            stage.hits_by_cause[cause] = stage.hits_by_cause.get(cause, 0) + 1
            stage.damage_by_room[room] = stage.damage_by_room.get(room, 0) + damage
        stage.hp_out = hp
        stage.end_frame = self.frame
        self._hp = hp
        self._observe_visit(snap)
        self._observe_inventory(snap, env.get_ram())
        self._observe_dungeon(snap, env.get_ram())
        self._observe_stall(settled, xy, room)

    def observe_writes(self, ram: Any) -> None:
        """Inventory edits between steps. A link move is a load, not a write."""
        if self._held is None or not self.visits:
            return
        snap = read_snapshot(np.asarray(ram))
        link = (int(snap.link_x), int(snap.link_y))
        if link != self._link:
            self._held = _inventory(ram)
            self._link = link
            return
        self._book_changes(
            _inventory(ram), _room_label(self.visits[-1].location), "write"
        )

    def _observe_inventory(self, snap: Any, ram: Any) -> None:
        if self.visits:
            room = _room_label(self.visits[-1].location)
        else:
            room = _room_label(_visit_key(snap))
        self._book_changes(_inventory(ram), room, "play")
        self._link = (int(snap.link_x), int(snap.link_y))

    def _book_changes(self, held: dict[str, int], room: str, source: str) -> None:
        before = self._held
        self._held = held
        if before is None:
            return
        for name, value in held.items():
            old = int(before.get(name, value))
            if value == old:
                continue
            change = InventoryChange(name, room, self.frame, old, int(value), source)
            self.changes.append(change)
            if int(value) < old and name in _SPEND_FIELDS:
                self.spends.append(change.row())

    def _observe_dungeon(self, snap: Any, ram: Any) -> None:
        dung = _dungeon_id(snap)
        if self._dungeon is not None and dung != self._dungeon:
            self._close_dungeon(ram, self._dungeon)
        self._dungeon = dung

    def _close_dungeon(self, ram: Any, dungeon_id: int) -> None:
        image = np.asarray(ram)
        for room in self.route_rooms:
            if int(room_flags(image, int(room))) & ROOM_FLAG_CHEST:
                continue
            self.missed_chests.append(
                {"dungeon": dungeon_id, "room": int(room), "frame": self.frame}
            )

    def _observe_visit(self, snap: Any) -> None:
        xy = (int(snap.link_x), int(snap.link_y))
        prev = self._prev_xy
        self._prev_xy = xy
        if not _settled(snap):
            return
        key = _visit_key(snap)
        if not self.visits or self.visits[-1].location != key:
            self.visits.append(Visit(location=key, enter_frame=self.frame))
            self._axes = (_Axis(), _Axis())
        visit = self.visits[-1]
        visit.frames += 1
        # Knockback shoves the same axis a tug would. Drop the sample.
        if int(snap.knockback) > 0 or prev is None:
            return
        self._observe_flutter(visit, xy, prev)

    def _observe_flutter(
        self, visit: Visit, xy: tuple[int, int], prev: tuple[int, int]
    ) -> None:
        for axis, delta in zip(self._axes, (xy[0] - prev[0], xy[1] - prev[1])):
            if not delta or abs(delta) > FLUTTER_MAX_STEP:
                continue
            sign = 1 if delta > 0 else -1
            if axis.sign == -sign and self.frame - axis.frame <= FLUTTER_WINDOW:
                visit.flutters += 1
            axis.sign = sign
            axis.frame = self.frame

    def _take_reason(self) -> str:
        reason = self._pending_reason
        self._pending_reason = ""
        if reason:
            return reason
        ctrl = self._controller
        if ctrl is None:
            return ""
        for attr in ("last_reason", "_last_reason"):
            value = getattr(ctrl, attr, None)
            if isinstance(value, str) and value:
                return value
        return ""

    def _ensure_stage(self) -> StageBook:
        if self._open is None:
            self._open = StageBook(name="run", start_frame=self.frame)
        return self._open

    def _count_reason(self, room: str, reason: str) -> None:
        bucket = self._reasons.setdefault(room, {})
        bucket[reason] = bucket.get(reason, 0) + 1

    def _observe_census(self, env: Any, room: str) -> None:
        sprites = active_sprites(env)
        ancillae = active_ancillae(env)
        book = self._sprites.setdefault(room, RoomSprites())
        alive = len(sprites) + len(ancillae)
        if alive > book.max_alive:
            book.max_alive = alive
        counts: dict[tuple[str, int], int] = {}
        hp0: dict[tuple[str, int], int] = {}
        for sprite in sprites:
            key = ("sprite", int(sprite.sprite_type))
            counts[key] = counts.get(key, 0) + 1
            if int(sprite.hp) == 0:
                hp0[key] = hp0.get(key, 0) + 1
        for ancilla in ancillae:
            key = ("ancilla", int(ancilla.ancilla_type))
            counts[key] = counts.get(key, 0) + 1
        for key, count in counts.items():
            row = book.types.get(key)
            if row is None:
                kind, type_id = key
                name = (
                    sprite_name(type_id) if kind == "sprite" else _ancilla_name(type_id)
                )
                row = TypeSeen(name=name, kind=kind, type_id=type_id)
                book.types[key] = row
            row.frames += 1
            row.max_alive = max(row.max_alive, count)
            row.hp0 += hp0.get(key, 0)

    def _mark_hurt(self, room: str, cause: str) -> None:
        book = self._sprites.get(room)
        if book is None or not cause:
            return
        for row in book.types.values():
            if row.name == cause:
                row.hurt += 1

    def _observe_stall(
        self, settled: bool, xy: tuple[int, int], room: str
    ) -> None:
        if not self.watch_stall or self._stalled:
            return
        if not settled:
            self._play.clear()
            self._stall_room = None
            return
        if room != self._stall_room:
            self._play.clear()
            self._stall_room = room
        self._play.append(xy)
        if len(self._play) < self.stall_frames:
            return
        xs = [point[0] for point in self._play]
        ys = [point[1] for point in self._play]
        if max(xs) - min(xs) >= self.stall_pixels:
            return
        if max(ys) - min(ys) >= self.stall_pixels:
            return
        self._fail_stall()

    def _fail_stall(self) -> None:
        self._stalled = True
        self.blocker = "stall"
        if self._open is not None:
            self._open.end_frame = self.frame
        directory = RECORDINGS_DIR if self.recordings is None else Path(self.recordings)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.tag}_stall.json"
        payload = {
            "tag": self.tag,
            "blocker": "stall",
            "frames": [
                {
                    "frame": frame,
                    "xy": [x, y],
                    "buttons": list(buttons),
                    "reason": reason,
                }
                for frame, x, y, buttons, reason in self._dump
            ],
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self.stall_path = str(path)
        raise Stall("stall")

    def _stage_rows(self) -> list[StageBook]:
        rows = list(self._stages)
        if self._open is not None:
            rows.append(self._open)
        return rows

    def damage_report(self) -> dict[str, Any]:
        stages = [stage.row() for stage in self._stage_rows()]
        hits_by_cause: dict[str, int] = {}
        damage_by_room: dict[str, int] = {}
        for stage in stages:
            for name, count in stage["hits_by_cause"].items():
                hits_by_cause[name] = hits_by_cause.get(name, 0) + int(count)
            for room, amount in stage["damage_by_room"].items():
                damage_by_room[room] = damage_by_room.get(room, 0) + int(amount)
        return {
            "hp_in": stages[0]["hp_in"] if stages else None,
            "hp_out": stages[-1]["hp_out"] if stages else None,
            "damage": sum(int(stage["damage"]) for stage in stages),
            "hits_by_cause": hits_by_cause,
            "damage_by_room": damage_by_room,
            "stages": stages,
        }

    def print_damage(self) -> None:
        """HP in and out, hits by cause, and damage by room for each stage."""
        damage = self.damage_report()
        print("damage:")
        for stage in damage["stages"]:
            print(
                f"  stage={stage['name']} hp {stage['hp_in']}->{stage['hp_out']} "
                f"damage={stage['damage']}"
            )
            causes = stage["hits_by_cause"]
            text = " ".join(f"{name}={count}" for name, count in causes.items()) or "-"
            print(f"    hits_by_cause: {text}")
            for room, amount in stage["damage_by_room"].items():
                print(f"    room {room} damage={amount}")
        if self.blocker:
            print(f"blocker={self.blocker}")

    def report(self) -> dict[str, Any]:
        rooms: dict[str, dict[str, Any]] = {}
        for visit in self.visits:
            row = rooms.setdefault(
                _room_label(visit.location),
                {"visits": 0, "frames": 0, "max_visit": 0, "flutters": 0},
            )
            row["visits"] += 1
            row["frames"] += visit.frames
            row["max_visit"] = max(row["max_visit"], visit.frames)
            row["flutters"] += visit.flutters
        slowest = sorted(self.visits, key=lambda visit: visit.frames, reverse=True)
        sprites = {room: book.row() for room, book in self._sprites.items()}
        payload = {
            "frames": self.frame,
            "hits": [hit.row() for hit in self.hits],
            "flutters": sum(visit.flutters for visit in self.visits),
            "visits": [visit.row() for visit in self.visits],
            "slowest_visits": [visit.row() for visit in slowest[:TOP_VISITS]],
            "rooms": rooms,
            "changes": [change.row() for change in self.changes],
            "writes": [change.row() for change in self.changes if change.source == "write"],
            "spends": self.spends,
            "missed_chests": self.missed_chests,
            "sprites": sprites,
            "soldiers_by_room": soldiers_from_rooms(sprites),
            "damage": self.damage_report(),
            "reason_by_room": {
                room: dict(counts) for room, counts in self._reasons.items()
            },
            "blocker": self.blocker,
        }
        if self._stalled:
            payload["stall"] = {"blocker": "stall", "path": self.stall_path}
        return payload
