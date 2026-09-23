"""Run-wide ledger: where the frames, the drops and the flutters went.

Wraps ``env.step`` once (like :class:`zelda_i.runner.VideoTap`), so every
stage loop is measured without any controller knowing. Observation only.

Three books:

- **rooms**: one row per consecutive visit to a ``level:screen``. The slowest
  visits are the stall list; a room's frame total is what a reroute saves.
- **drops**: every enemy floor drop (``0x60`` with an item ObjState) and how
  it ended: ``picked`` (Link on it, or its counter moved), ``expired`` (gone
  while Link stayed), ``left`` (Link changed room first). A heart or fairy
  that is not picked while Link is hurt is a *missed heart*.
- **flutter**: Link's position reversing on one axis within
  ``FLUTTER_WINDOW`` frames (a left-right or up-down twitch on the tape).
  Knockback frames (i-frames) are not counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from zelda_i.combat import is_floor_drop
from zelda_i.dungeon.ids import (
    BOMB_DROP_STATE,
    CLOCK_DROP_STATE,
    FAIRY_DROP_STATE,
    FIVE_RUPEE_DROP_STATE,
    HEART_DROP_STATE,
    RUPEE_DROP_STATE,
)
from zelda_i.ram import (
    ADDR_RUPEES_TO_ADD,
    PLAY_MODE,
    ZeldaSnapshot,
    hearts_held,
    read_snapshot,
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
    outcome: str = "open"
    end_frame: int = 0

    def row(self) -> dict[str, Any]:
        return {
            "room": self.room,
            "kind": self.kind,
            "frame": self.frame,
            "life": self.end_frame - self.frame,
            "outcome": self.outcome,
            "hurt": self.hurt,
            "xy": [self.x, self.y],
        }


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

    # ----------------------------------------------------------------- wiring
    def attach(self, env: Any) -> None:
        """Observe every frame ``env.step`` plays from now on."""
        inner = env.step

        def step(action, *args, **kwargs):
            out = inner(action, *args, **kwargs)
            ram = env.get_ram()
            self.observe(read_snapshot(ram), to_add=int(ram[ADDR_RUPEES_TO_ADD]))
            return out

        env.step = step

    # ------------------------------------------------------------ observation
    def observe(self, snap: ZeldaSnapshot, *, to_add: int = 0) -> None:
        self.frame += 1
        room = room_key(snap)
        if not self.visits or self.visits[-1].room != room:
            self._close_drops("left")
            self.visits.append(Visit(room=room, enter_frame=self.frame))
            self._prev = None
        visit = self.visits[-1]
        visit.frames += 1
        hearts = hearts_held(snap)
        if self._hearts is not None and hearts < self._hearts:
            visit.damage += self._hearts - hearts
        play = int(snap.mode) == PLAY_MODE
        if play:
            visit.play_frames += 1
            self._observe_flutter(snap, visit)
            self._observe_drops(snap, room, hearts, to_add)
        self._hearts = hearts
        self._to_add = to_add
        self._prev = snap if play else None

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
                )
                self._open[slot] = drop
                self.drops.append(drop)

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

    def report(self) -> dict[str, Any]:
        self._close_drops("left")
        rooms: dict[str, dict[str, Any]] = {}
        for v in self.visits:
            row = rooms.setdefault(
                v.room, {"visits": 0, "frames": 0, "max_visit": 0, "flutters": 0}
            )
            row["visits"] += 1
            row["frames"] += v.frames
            row["max_visit"] = max(row["max_visit"], v.frames)
            row["flutters"] += v.flutters
        by_kind: dict[str, dict[str, int]] = {}
        for d in self.drops:
            tally = by_kind.setdefault(d.kind, {})
            tally[d.outcome] = tally.get(d.outcome, 0) + 1
        missed_hearts = [
            d for d in self.missed() if d.kind in HEALING_KINDS and d.hurt
        ]
        slow = sorted(self.visits, key=lambda v: v.frames, reverse=True)
        return {
            "frames": self.frame,
            "visits": len(self.visits),
            "flutters": sum(v.flutters for v in self.visits),
            "slowest_visits": [v.row() for v in slow[:TOP_VISITS]],
            "rooms": rooms,
            "drops": {
                "total": len(self.drops),
                "by_kind": by_kind,
                "missed": len(self.missed()),
                "missed_hearts_while_hurt": len(missed_hearts),
                "missed_rows": [d.row() for d in self.missed()],
            },
        }

    def summary_lines(self) -> list[str]:
        rep = self.report()
        drops = rep["drops"]
        lines = [
            f"ledger: {rep['frames']}f {rep['visits']} visits "
            f"flutters={rep['flutters']} drops={drops['total']} "
            f"missed={drops['missed']} missed_hearts_hurt={drops['missed_hearts_while_hurt']}",
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
        return lines
