"""Event detectors for Super Metroid TAS movie traces.

Pure functions over consecutive :class:`~super_metroid.ram.SuperMetroidState`
(plus optional button names). Used by :mod:`super_metroid.tas.trace` during
emulator replay; no I/O.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from super_metroid.ram import GameplayPhase, SuperMetroidState, probe_pin

# Item / beam bit labels (collected masks).
_ITEM_BITS: tuple[tuple[int, str], ...] = (
    (0x0001, "varia"),
    (0x0002, "spring_ball"),
    (0x0004, "morph"),
    (0x0008, "screw_attack"),
    (0x0020, "gravity"),
    (0x0100, "hi_jump"),
    (0x0200, "space_jump"),
    (0x1000, "bombs"),
    (0x2000, "speed_booster"),
    (0x4000, "grapple"),
    (0x8000, "xray"),
)
_BEAM_BITS: tuple[tuple[int, str], ...] = (
    (0x0001, "wave"),
    (0x0002, "ice"),
    (0x0004, "spazer"),
    (0x0008, "plasma"),
    (0x1000, "charge"),
)

# High-value pose clusters (skip noisy spin/walljump for default annotate).
_POSE_CLUSTERS: dict[str, frozenset[int]] = {
    "morph": frozenset(
        {0x1D, 0x1E, 0x1F, 0x31, 0x32, 0x41, 0x79, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E}
    ),
    "shinespark": frozenset({0xC9, 0xCA, 0xCB, 0xCC, 0xCD, 0xCE}),
    "knockback": frozenset({0x53, 0x54}),
    # Optional tech clusters (enable via Annotator.extra_pose_clusters).
    "walljump": frozenset({0x13, 0x14, 0x15, 0x16, 0x51, 0x52}),
    "spinjump": frozenset({0x19, 0x1A, 0x1B, 0x1C}),
}
_DEFAULT_POSE_CLUSTERS = frozenset({"morph", "shinespark", "knockback"})

_ORDINARY_GS = 8


@dataclass(frozen=True)
class TraceEvent:
    """One named milestone or anomaly in a TAS replay."""

    frame: int
    kind: str
    detail: str = ""
    room_id: int = 0
    pose: int = 0
    x: int = 0
    y: int = 0
    pin: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["room_id_hex"] = f"0x{self.room_id:04X}"
        return d


def _pin(state: SuperMetroidState) -> dict[str, Any]:
    return dict(probe_pin(state))


def _event(
    state: SuperMetroidState,
    kind: str,
    detail: str = "",
) -> TraceEvent:
    return TraceEvent(
        frame=int(state.frame),
        kind=kind,
        detail=detail,
        room_id=int(state.room_id),
        pose=int(state.pose),
        x=int(state.samus_x),
        y=int(state.samus_y),
        pin=_pin(state),
    )


def _bits_gained(before: int, after: int, table: Sequence[tuple[int, str]]) -> list[str]:
    gained = (after & ~before) & 0xFFFF
    if not gained:
        return []
    names = [name for mask, name in table if gained & mask]
    # Unknown bits still reported as hex masks.
    known = 0
    for mask, _ in table:
        known |= mask
    unknown = gained & ~known
    if unknown:
        names.append(f"bits_0x{unknown:04X}")
    return names


def pose_cluster(
    pose: int,
    *,
    enabled: frozenset[str] | None = None,
) -> str | None:
    """Return the first matching pose cluster name, or None.

    ``enabled`` defaults to morph/shinespark/knockback (skips noisy spin/WJ).
    """
    allow = enabled if enabled is not None else _DEFAULT_POSE_CLUSTERS
    p = int(pose)
    for name, members in _POSE_CLUSTERS.items():
        if name not in allow:
            continue
        if p in members:
            return name
    return None


def is_settled_control(state: SuperMetroidState) -> bool:
    """Ordinary controllable room (same settle rule as RoomTimer)."""
    return (
        state.phase is GameplayPhase.ORDINARY_GAMEPLAY
        and int(state.game_state) == _ORDINARY_GS
        and int(state.door_transition) == 0
        and int(state.room_id) != 0
    )


@dataclass
class Annotator:
    """Incremental event detector. Call :meth:`observe` once per frame after step."""

    stall_frames: int = 90
    pose_clusters: frozenset[str] = field(default_factory=lambda: _DEFAULT_POSE_CLUSTERS)
    events: list[TraceEvent] = field(default_factory=list)
    _prev: SuperMetroidState | None = field(default=None, repr=False)
    _saw_control: bool = False
    _prev_speed_counter: int = 0
    _prev_shine: int = 0
    _prev_pose_cluster: str | None = None
    _last_pose_cluster_frame: int = -10_000
    _pose_cluster_debounce: int = 45
    _stall_start: int | None = None
    _stall_pose: int = -1
    _stall_x: int = -1
    _stall_y: int = -1
    _stall_buttons_active: bool = False
    _last_room_settled: int | None = None
    _in_transition: bool = False

    def observe(
        self,
        state: SuperMetroidState,
        *,
        buttons: Sequence[str] | None = None,
    ) -> list[TraceEvent]:
        """Ingest one post-step state; return newly emitted events this frame."""
        new: list[TraceEvent] = []
        btns = tuple(buttons or ())
        active = bool(btns)

        # First ordinary control after boot/menu.
        if not self._saw_control and is_settled_control(state):
            self._saw_control = True
            new.append(
                _event(
                    state,
                    "control",
                    f"first_control room=0x{state.room_id:04X}",
                )
            )

        prev = self._prev
        settled = is_settled_control(state)
        # Ignore boot/menu WRAM noise for progress + kinematics events.
        gameplay_ok = state.phase not in (
            GameplayPhase.BOOT_OR_MENU,
            GameplayPhase.UNKNOWN,
        ) and int(state.room_id) != 0

        if prev is not None:
            prev_settled = is_settled_control(prev)

            if settled and not prev_settled:
                # Entered ordinary gameplay (possibly new room).
                if self._last_room_settled is None or self._last_room_settled != int(
                    state.room_id
                ):
                    detail = f"enter 0x{state.room_id:04X}"
                    if self._last_room_settled is not None:
                        detail = (
                            f"0x{self._last_room_settled:04X} -> 0x{state.room_id:04X}"
                        )
                    new.append(_event(state, "room_enter", detail))
                    self._last_room_settled = int(state.room_id)
                self._in_transition = False
            elif prev_settled and not settled:
                if prev.phase is not GameplayPhase.DEATH_OR_GAME_OVER:
                    new.append(
                        _event(
                            prev,
                            "room_leave",
                            f"leave 0x{prev.room_id:04X} gs={state.game_state} "
                            f"door={state.door_transition}",
                        )
                    )
                self._in_transition = True

            # Same-room re-settle after brief blip: still emit if room changed mid-transition.
            if (
                settled
                and prev_settled
                and int(state.room_id) != int(prev.room_id)
                and int(state.room_id) != 0
            ):
                new.append(
                    _event(
                        state,
                        "room_enter",
                        f"0x{prev.room_id:04X} -> 0x{state.room_id:04X}",
                    )
                )
                self._last_room_settled = int(state.room_id)

            # Item / beam / capacity — only after we are in a real room.
            if gameplay_ok:
                for name in _bits_gained(
                    prev.collected_items, state.collected_items, _ITEM_BITS
                ):
                    new.append(_event(state, "item_gain", name))
                for name in _bits_gained(
                    prev.collected_beams, state.collected_beams, _BEAM_BITS
                ):
                    new.append(_event(state, "beam_gain", name))

                if state.max_missiles > prev.max_missiles:
                    new.append(
                        _event(
                            state,
                            "capacity_gain",
                            f"missiles {prev.max_missiles}->{state.max_missiles}",
                        )
                    )
                if state.max_super_missiles > prev.max_super_missiles:
                    new.append(
                        _event(
                            state,
                            "capacity_gain",
                            f"supers {prev.max_super_missiles}->{state.max_super_missiles}",
                        )
                    )
                if state.max_power_bombs > prev.max_power_bombs:
                    new.append(
                        _event(
                            state,
                            "capacity_gain",
                            f"pbs {prev.max_power_bombs}->{state.max_power_bombs}",
                        )
                    )
                if state.max_health > prev.max_health:
                    new.append(
                        _event(
                            state,
                            "capacity_gain",
                            f"energy {prev.max_health}->{state.max_health}",
                        )
                    )

            # Phase transitions of interest.
            if (
                state.phase is GameplayPhase.DEATH_OR_GAME_OVER
                and prev.phase is not GameplayPhase.DEATH_OR_GAME_OVER
            ):
                new.append(_event(state, "death", f"gs={state.game_state}"))
            if (
                state.phase is GameplayPhase.ENDING_OR_CREDITS
                and prev.phase is not GameplayPhase.ENDING_OR_CREDITS
            ):
                new.append(_event(state, "ending", f"gs={state.game_state}"))
            if (
                state.phase is GameplayPhase.PAUSE_OR_INVENTORY
                and prev.phase is not GameplayPhase.PAUSE_OR_INVENTORY
            ):
                new.append(_event(state, "pause", f"gs={state.game_state}"))
            if (
                prev.phase is GameplayPhase.PAUSE_OR_INVENTORY
                and state.phase is not GameplayPhase.PAUSE_OR_INVENTORY
            ):
                new.append(_event(state, "unpause", f"gs={state.game_state}"))

        # Speed / shine / pose — only in real rooms (boot WRAM is garbage).
        if gameplay_ok or settled:
            sc = int(state.speed_counter)
            if sc != self._prev_speed_counter:
                if sc > self._prev_speed_counter and sc >= 1:
                    new.append(
                        _event(
                            state,
                            "speed_echo",
                            f"{self._prev_speed_counter}->{sc}"
                            + (" boost" if sc >= 4 else ""),
                        )
                    )
                elif sc < self._prev_speed_counter and self._prev_speed_counter >= 1:
                    new.append(
                        _event(
                            state,
                            "speed_echo_drop",
                            f"{self._prev_speed_counter}->{sc}",
                        )
                    )
                self._prev_speed_counter = sc

            shine = int(state.shinespark_timer)
            if shine > 0 and self._prev_shine == 0:
                new.append(_event(state, "shine_arm", f"timer={shine}"))
            elif shine == 0 and self._prev_shine > 0:
                new.append(_event(state, "shine_clear", f"was={self._prev_shine}"))
            self._prev_shine = shine

            cluster = pose_cluster(state.pose, enabled=self.pose_clusters)
            if (
                cluster is not None
                and cluster != self._prev_pose_cluster
                and int(state.frame) - self._last_pose_cluster_frame
                >= self._pose_cluster_debounce
            ):
                new.append(_event(state, "pose_cluster", cluster))
                self._last_pose_cluster_frame = int(state.frame)
            if cluster is not None:
                self._prev_pose_cluster = cluster
            elif self._prev_pose_cluster is not None:
                # Left cluster; allow re-entry after debounce window.
                self._prev_pose_cluster = None
        else:
            # Keep counters latched so first gameplay frame doesn't false-edge.
            self._prev_speed_counter = int(state.speed_counter)
            self._prev_shine = int(state.shinespark_timer)
            self._prev_pose_cluster = pose_cluster(
                state.pose, enabled=self.pose_clusters
            )

        # Desync / stall: frozen pose+xy while buttons active in ordinary play.
        if is_settled_control(state) and active:
            same = (
                int(state.pose) == self._stall_pose
                and int(state.samus_x) == self._stall_x
                and int(state.samus_y) == self._stall_y
                and self._stall_buttons_active
            )
            if same:
                if self._stall_start is None:
                    self._stall_start = int(state.frame)
                held = int(state.frame) - int(self._stall_start)
                if held == self.stall_frames:
                    new.append(
                        _event(
                            state,
                            "desync_suspect",
                            f"frozen {self.stall_frames}f pose={state.pose} "
                            f"xy=({state.samus_x},{state.samus_y}) "
                            f"btns={'+'.join(btns) or 'idle'}",
                        )
                    )
            else:
                self._stall_start = int(state.frame)
                self._stall_pose = int(state.pose)
                self._stall_x = int(state.samus_x)
                self._stall_y = int(state.samus_y)
                self._stall_buttons_active = True
        else:
            self._stall_start = None
            self._stall_buttons_active = False
            self._stall_pose = -1

        self._prev = state
        self.events.extend(new)
        return new

    def summary(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        for e in self.events:
            by_kind[e.kind] = by_kind.get(e.kind, 0) + 1
        rooms = [
            e.to_dict()
            for e in self.events
            if e.kind in ("room_enter", "control")
        ]
        return {
            "event_count": len(self.events),
            "by_kind": by_kind,
            "first_control_frame": next(
                (e.frame for e in self.events if e.kind == "control"), None
            ),
            "room_milestones": rooms[:200],
            "item_gains": [
                e.to_dict() for e in self.events if e.kind in ("item_gain", "beam_gain", "capacity_gain")
            ],
            "desync_suspects": [
                e.to_dict() for e in self.events if e.kind == "desync_suspect"
            ],
        }


__all__ = [
    "Annotator",
    "TraceEvent",
    "is_settled_control",
    "pose_cluster",
]
