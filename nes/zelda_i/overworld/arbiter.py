"""Per-frame decision order as data, so it can be read and asserted.

``OverworldPathController.step``, ``OverworldPathController._do_hop`` and
``ScreenHunter.step`` each decide a frame with a chain of
``if act is not None: return act``. The order of those rungs is real policy —
it decides which behaviour owns a contested frame — but it is written as
source-line order, which means it cannot be named, reordered, or asserted on
without driving a whole ``step()`` and matching ``FrameAction.reason``
strings. Two live failures came out of that:

* **The beam under a stall escape.** ``AGENTS.md`` Traps: "travelling frames
  must offer ``ScreenHunter.take_beam`` above stall-escape (600f commits used
  to zero the weapon)". The fix was to call ``take_beam`` from
  ``path._do_hop`` directly, out of band with the rest of the hunt, purely to
  get the shot high enough in the chain. The hop now reaches into the
  hunter's blade because there was no other way to express "this rung goes
  above that one".
* **The 0x79 inversion.** ``overworld.shop_p7._extra_hop_action`` opens with
  ``if self.hunter is not None and 0x79 not in self.hunter.done: return
  None`` — an override hook declining the frame, twice, so the hunt rung
  below it can have it. That is a precedence edit written as a read of
  another module's bookkeeping, because the hook's position in the chain is
  fixed by where it is called from.

This module is the ladder itself: a :class:`Rung` is a named, ordered
callable over a :class:`~zelda_i.ram.ZeldaSnapshot`, and an :class:`Arbiter`
runs them in priority order and hands back the first frame anyone claims.
Ordering becomes a number a test can set, precedence becomes a rung nobody
else's private state has to be consulted for, and the winner is recorded, so
the per-behaviour counters ``ScreenHunter`` keeps by hand (``hunt_frames``,
``guard_frames``, ``peel_frames``, ``transit_frames``, ``collect_frames``,
``off_line_frames``) fall out of the arbitration instead of being incremented
at six separate sites that can drift from the branch they claim to count.

Pure, like ``overworld.prey``: the input is a snapshot, the output is a
``FrameAction`` a rung already built. No emulator, no RAM array, no I/O, and
no rungs constructed here — they are handed in, which is what lets one
arbiter hold ``path``'s ladder and ``hunt``'s ladder at once.

Nothing calls this yet; wiring the live controllers on to it is a separate
change.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

from retro_harness.input_script import FrameAction
from zelda_i.ram import ZeldaSnapshot

__all__ = [
    "STAMP_SEPARATOR",
    "Arbiter",
    "Rung",
    "RungFn",
]

RungFn = Callable[[ZeldaSnapshot], "FrameAction | None"]

# ``reason`` strings are matched exactly by a large number of existing tests
# and read by eye in run logs, so a stamp appends rather than replaces, and
# the separator is a character no current reason uses (they are ``a_b_c``
# snake case, plus hex screen ids).
STAMP_SEPARATOR = "|"


@dataclass(frozen=True)
class Rung:
    """One step of a decision ladder: a name, a place, and a claim.

    ``fn`` returns a ``FrameAction`` to claim the frame or ``None`` to
    decline it, which is exactly the shape the existing ``if act is not None:
    return act`` chains already have — a bound method of the controller that
    owns the behaviour drops in unchanged.

    ``priority`` counts *down* the ladder: 0 is the top, and the lowest
    number that claims the frame wins. Equal priorities are ordered by name,
    never by registration order, so the ladder is fully determined by the
    data and a caller cannot change policy by moving a line.
    """

    name: str
    priority: int
    fn: RungFn


@dataclass
class Arbiter:
    """An ordered ladder of rungs, plus the accounting of who won.

    Configuration is ``rungs`` and ``stamp``; everything else on this object
    is accounting and is safe to :meth:`reset`. Sorting happens once at
    construction, so :meth:`decide` walks a fixed tuple.

    ``stamp`` is off by default. When on, the winning rung's name is appended
    to ``FrameAction.reason`` behind :data:`STAMP_SEPARATOR` (via
    ``dataclasses.replace`` — ``FrameAction`` is frozen and the rung's own
    object is never mutated), which keeps the existing reason text intact for
    the tests and log greps that match it. It is opt-in rather than always-on
    because ``last_winner`` and :meth:`census` already recover the name
    without touching the action at all; the stamp is only for run logs, where
    the reason string is the one thing that gets written down.
    """

    rungs: tuple[Rung, ...]
    stamp: bool = False
    last_winner: str | None = field(default=None, init=False)
    wins: dict[str, int] = field(default_factory=dict, init=False)
    frames: int = field(default=0, init=False)
    idle_frames: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        ordered = tuple(self.rungs)
        names = [r.name for r in ordered]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            # A silent duplicate makes the census lie (two behaviours sharing
            # one counter) and hides whichever copy sorts second forever.
            raise ValueError(f"duplicate rung names: {', '.join(duplicates)}")
        self.rungs = tuple(sorted(ordered, key=lambda r: (r.priority, r.name)))
        self.stamp = bool(self.stamp)
        self.reset()

    def decide(self, snap: ZeldaSnapshot) -> FrameAction | None:
        """The frame the highest rung claims, or ``None`` if nobody wants it.

        ``None`` from every rung is a real answer, not a failure: it is the
        fall-through the callers end with (``align_and_push`` on the hop,
        ``ScreenHunter.step`` handing the frame back to the path).
        """
        self.frames += 1
        for rung in self.rungs:
            act = rung.fn(snap)
            if act is None:
                continue
            self.last_winner = rung.name
            self.wins[rung.name] = self.wins.get(rung.name, 0) + 1
            return self._stamped(act, rung.name)
        self.last_winner = None
        self.idle_frames += 1
        return None

    def census(self) -> dict[str, int]:
        """Frames won per rung, in ladder order. Declines are not counted.

        This is the replacement for the hand-kept ``*_frames`` counters: a
        rung is credited exactly when its action is the one returned, so the
        count cannot drift from the branch it names.
        """
        return {rung.name: self.wins.get(rung.name, 0) for rung in self.rungs}

    def reset(self) -> None:
        """Clear the accounting, keep the ladder (a new screen, a new hop)."""
        self.last_winner = None
        self.wins = {r.name: 0 for r in self.rungs}
        self.frames = 0
        self.idle_frames = 0

    def _stamped(self, act: FrameAction, name: str) -> FrameAction:
        if not self.stamp:
            return act
        reason = f"{act.reason}{STAMP_SEPARATOR}{name}" if act.reason else name
        return replace(act, reason=reason)
