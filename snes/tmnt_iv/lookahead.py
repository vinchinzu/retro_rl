"""ROM lookahead guard: the emulator is the enemy model.

Why this exists (measured 2026-11-14, see ``docs/STATUS.md``): the reactive
tactic stack takes 4-24 HP chip hits it cannot see coming, and every per-stage
retune moved deaths somewhere else. From power-on the 2026-11 policy died in
Big Apple with every turtle. Heuristic dodging cannot be made reliable; asking
the ROM can.

The ROM is cheap to ask and exact. On this machine ``em.step`` is ~700 us
(~23x real time), ``get_state`` ~30 us, ``set_state`` ~80 us, and a restore
plus 300 replayed frames is bit-exact (PRNG and all). A rollout is not an
estimate of the future; it *is* the future.

How it plays -- a rolling verified frontier:

1. The base policy (``Stage1Policy``) is simulated ``horizon`` frames ahead
   of the live frame. Live frames are committed from the front of that
   verified plan, so every committed frame already has a hit-free future
   ``horizon`` frames long.
2. Each live frame extends the frontier (one restore to the frontier, a few
   policy frames, one restore back to live). Cost is ~1x emulation.
3. When an extension shows an HP drop or a life loss, the guard branches from
   snapshot nodes inside the uncommitted window -- latest first, then earlier
   ones for more freedom -- trying every :data:`MACROS` prefix followed by
   the base policy. The best hit-free branch is spliced in. If none exists
   the least-damage branch is taken and counted as ``forced``.
4. Every planned frame stores the CRC of the RAM it expects to start from.
   A live mismatch drops the plan (``desyncs``); determinism makes that rare.

Contract: every restore goes to the live frame or to a descendant of it,
never to an ancestor. The committed tape therefore contains no rewinds, and
replaying it from power-on with no lookahead reproduces the run
(``tmnt_iv.tape``). ``restores`` counts these restores so the trial audit
can tell them from a mid-run state load. ``_live_unchanged`` proves each
guard call returned the live RAM bit-for-bit.
"""

from __future__ import annotations

import copy
import zlib
from collections import Counter, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from retro_harness.actions import buttons, idle_action
from retro_harness.bot_runner import NodeStatus, TickResult
from retro_harness.controls import SNES_A, SNES_START
from retro_harness.input_script import FrameAction
from retro_harness.ram_state import GameMode, GameState
from tmnt_iv.observe import living_hp, policy_input
from tmnt_iv.policy import Stage1Policy
from tmnt_iv.ram import is_stage_clear_zero, parse_game_state

__all__ = [
    "MACROS",
    "Branch",
    "GuardedPolicy",
    "LookaheadConfig",
    "control_action",
    "ram_crc",
]

Action = tuple[int, ...]


def _vec(*names: str) -> Action:
    return tuple(int(v) for v in (buttons(*names) if names else idle_action()))


def _hold(frames: int, *names: str) -> tuple[Action, ...]:
    return (_vec(*names),) * frames


# Evasive and offensive prefixes. Each is played literally, then the base
# policy resumes. Y attacks, B jumps; A (special) drains HP and is never
# used. Diagonals matter: depth (UP/DOWN) is the main dodge axis in a
# belt-scroller.
_DIRS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("u", ("UP",)),
    ("d", ("DOWN",)),
    ("l", ("LEFT",)),
    ("r", ("RIGHT",)),
    ("ul", ("UP", "LEFT")),
    ("ur", ("UP", "RIGHT")),
    ("dl", ("DOWN", "LEFT")),
    ("dr", ("DOWN", "RIGHT")),
)


def _build_macros() -> tuple[tuple[str, tuple[Action, ...]], ...]:
    rows: list[tuple[str, tuple[Action, ...]]] = [
        ("idle8", _hold(8)),
        ("idle24", _hold(24)),
        ("idle48", _hold(48)),
    ]
    for tag, dirs in _DIRS:
        rows.append((f"walk_{tag}8", _hold(8, *dirs)))
        rows.append((f"walk_{tag}16", _hold(16, *dirs)))
        # Long holds: the base policy walks straight back into a threat
        # after a short dodge, so some hits need a real retreat.
        rows.append((f"walk_{tag}32", _hold(32, *dirs)))
    rows.append(("jump", _hold(4, "B") + _hold(14)))
    for tag, dirs in (("l", ("LEFT",)), ("r", ("RIGHT",)), ("u", ("UP",)), ("d", ("DOWN",))):
        rows.append((f"jump_{tag}", _hold(4, "B", *dirs) + _hold(14, *dirs)))
    rows.append(("jump_kick", _hold(4, "B") + _hold(4) + _hold(2, "Y") + _hold(8)))
    for tag, dirs in (("l", ("LEFT",)), ("r", ("RIGHT",))):
        rows.append(
            (
                f"jump_kick_{tag}",
                _hold(4, "B", *dirs) + _hold(4, *dirs) + _hold(2, "Y", *dirs) + _hold(8, *dirs),
            )
        )
    # Grab escapes. A Starbase Foot (0x66) holds the turtle in place and
    # chips 4-10 HP every ~33 frames until dead; held directions, jumps and
    # pokes never break it. Measured from the held state (2026-11-14):
    # UP/DOWN alternating every frame frees him in 7 frames with 0 HP lost,
    # an L/U/R/D cycle in 8; LEFT/RIGHT frees in 7 but eats one hit.
    rows.append(("mash_ud", tuple(_vec("UP" if i % 2 == 0 else "DOWN") for i in range(16))))
    rows.append(("mash_dirs", tuple(_vec(("LEFT", "UP", "RIGHT", "DOWN")[i % 4]) for i in range(16))))
    rows.append(("mash_lr", tuple(_vec("LEFT" if i % 2 == 0 else "RIGHT") for i in range(16))))
    rows.append(("mash_y", tuple(_vec("Y") if i % 2 == 0 else _vec() for i in range(16))))
    for tag, dirs in (("", ()), ("_l", ("LEFT",)), ("_r", ("RIGHT",))):
        rows.append((f"poke{tag}", _hold(2, "Y", *dirs) + _hold(6)))
        rows.append((f"poke2{tag}", (_hold(2, "Y", *dirs) + _hold(3)) * 2 + _hold(4)))
    return tuple(rows)


MACROS: tuple[tuple[str, tuple[Action, ...]], ...] = _build_macros()


@dataclass(frozen=True)
class LookaheadConfig:
    """Search knobs. Frames are emulator frames at 60 Hz."""

    horizon: int = 64
    chunk: int = 8
    # A branch must stay hit-free this far past the threatened frame.
    margin: int = 24
    # Branch points, as frames of lead before the threatened frame.
    leads: tuple[int, ...] = (16, 32, 48, 64)
    # When no single macro is hit-free, try this many best first macros
    # followed by every macro (depth 2) before accepting a forced hit.
    pair_seeds: int = 6
    # A lead stops after this many hit-free branches (macros are tried in
    # order of past rescues), and pairs are skipped within this many frames
    # of the last forced hit: inside a sustained engagement they cost ~15 s
    # each and rarely find anything.
    enough_safe: int = 3
    pair_cooldown: int = 240
    # Clean forbids lost lives, not damage. When nothing has progressed
    # (no enemy HP lost, no stage change) for ``stall_frames``, the guard lets
    # the base policy take a predicted hit that leaves HP >= ``hp_floor`` of
    # the bar: Technodrome's tank throw needs a charge the strict guard vetoes
    # forever. 600 fired inside a winnable Baxter fight, pushed the run off
    # its 0-damage path, and into a releft loop; real deadlocks last 20k+.
    stall_frames: int = 1800
    hp_floor: float = 0.5
    # Deadlock breaker: while stalled, every ``search_every`` frames try
    # every macro from the live point for ``search_frames`` and splice the
    # branch that damages an enemy (life kept, HP over the floor). The base
    # policy deadlocked on enemies just off the walkable band (Technodrome
    # Foot at y=144 over a y=160 floor, a Prehistoric dino at y=128 over
    # y=136); the ROM finds the approach the tactic cannot.
    search_frames: int = 120
    search_every: int = 300
    # HP lost is worth this many points; a life loss is never chosen if
    # any branch keeps the life.
    hp_weight: float = 1000.0
    enemy_weight: float = 4.0
    pizza_weight: float = 400.0
    progress_weight: float = 0.25


def ram_crc(ram: Any) -> int:
    """CRC32 of the full RAM image (``env.get_ram()``)."""
    return zlib.crc32(np.asarray(ram, dtype=np.uint8).tobytes()) & 0xFFFFFFFF


def _strip(action: Any) -> Action:
    vec = [int(v) for v in action]
    vec[SNES_START] = 0
    vec[SNES_A] = 0
    return tuple(vec)


def engaged(state: GameState) -> bool:
    """True when the turtle is controllable and worth guarding."""
    return (
        state.mode is GameMode.PLAYING
        and state.player_x > 0
        and living_hp(state.health)
    )


def control_action(policy: Stage1Policy, state: GameState) -> tuple[Action, str]:
    """The base policy's frame, as a rollout continuation plays it.

    No START (pause) and no A (HP-draining special) inside a rollout. Out of
    control (fades, loads) the continuation idles.
    """
    if not engaged(state):
        return _vec(), "la_idle"
    action, reason = policy_input(policy, state)
    return _strip(action), reason


def _enemy_hp(state: GameState) -> int:
    """HP in every occupied enemy slot, on-screen or not.

    The Technodrome tank (slot 0, char 0xFE, HP 90, -10 per screen throw)
    sits off-screen, so ``living_enemies`` drops it; counting only living
    enemies made a 300,000-frame tank loop look like progress.
    """
    return sum(int(e.health) for e in state.enemies if e.kind != 0)


@dataclass
class _Step:
    action: Action
    reason: str
    crc: int  # RAM CRC before this action is applied


@dataclass(eq=False)
class _Node:
    """Emulator state and policy copy at a plan offset (0 = live frame).

    Offsets shift down by one per committed frame. A node at a negative
    offset is in the past and is never branched from (that would rewind).
    """

    offset: int
    frame: int  # trial frame number the policy sees at this node
    state: bytes
    policy: Stage1Policy


@dataclass
class Branch:
    """One rolled future, read off the ROM."""

    label: str
    steps: list[_Step]
    nodes: list[_Node]
    end: _Node | None
    hp_lost: int = 0
    start_hp: int = 0
    life_lost: bool = False
    first_hit: int | None = None
    enemy_dmg: int = 0
    pizza: int = 0
    progress: int = 0
    stage_advanced: bool = False

    @property
    def safe(self) -> bool:
        return self.hp_lost == 0 and not self.life_lost

    def score(self, cfg: LookaheadConfig) -> float:
        s = 0.0
        if self.life_lost:
            s -= 1e9
        s -= cfg.hp_weight * self.hp_lost
        s += cfg.enemy_weight * self.enemy_dmg
        s += cfg.pizza_weight * self.pizza
        s += cfg.progress_weight * self.progress
        if self.stage_advanced:
            s += 1e5
        if self.first_hit is not None:
            s += self.first_hit  # later hits are better than early ones
        return s


@dataclass
class GuardStats:
    rollout_frames: int = 0
    restores: int = 0
    threats: int = 0
    rescued: int = 0
    pairs: int = 0
    forced: int = 0
    forced_hp: int = 0
    desyncs: int = 0
    accepted: int = 0
    accepted_hp: int = 0
    searches: int = 0
    unstuck: int = 0
    branches: int = 0
    macros: Counter[str] = field(default_factory=Counter)
    leads: Counter[int] = field(default_factory=Counter)

    def report(self) -> dict[str, Any]:
        return {
            "rollout_frames": self.rollout_frames,
            "restores": self.restores,
            "threats": self.threats,
            "rescued": self.rescued,
            "pairs": self.pairs,
            "forced": self.forced,
            "forced_hp": self.forced_hp,
            "desyncs": self.desyncs,
            "accepted": self.accepted,
            "accepted_hp": self.accepted_hp,
            "searches": self.searches,
            "unstuck": self.unstuck,
            "branches": self.branches,
            "top_macros": self.macros.most_common(12),
            "rescue_leads": dict(sorted(self.leads.items())),
        }


class GuardedPolicy:
    """``Stage1Policy`` with a ROM-verified future. Drop-in for ``run_trial``.

    ``bind`` hands it the raw emulator (not the trial's audited wrapper) so
    its restores are counted here as lookahead, not as state loads.
    """

    def __init__(
        self,
        base: Stage1Policy | None = None,
        config: LookaheadConfig | None = None,
        *,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.base = base or Stage1Policy()
        self.cfg = config or LookaheadConfig()
        self.stats = GuardStats()
        self._em: Any = None
        self._get_ram: Callable[[], Any] | None = None
        self._plan: deque[_Step] = deque()
        self._nodes: deque[_Node] = deque()
        self._frontier: _Node | None = None
        self._log = log
        self._last_forced_frame = -10**9
        self._progress_frame = 0
        self._enemy_hp = 0
        self._stage = -1
        self._bar = 0
        self._last_search_frame = -10**9

    # ----------------------------------------------------------------- wiring
    def bind(self, em: Any, get_ram: Callable[[], Any]) -> None:
        self._em = em
        self._get_ram = get_ram

    def reset(self) -> None:
        self.base.reset()
        self._drop()

    def report(self) -> dict[str, Any]:
        return self.stats.report()

    # ------------------------------------------------------------------- tick
    def tick(self, state: GameState) -> TickResult:
        self._note_progress(state)
        if self._em is None or not engaged(state):
            self._drop()
            return self.base.tick(state)
        ram = self._get_ram()
        crc = ram_crc(ram)
        if self._plan and self._plan[0].crc != crc:
            self.stats.desyncs += 1
            self._drop()
        live = bytes(self._em.get_state())
        try:
            if not self._plan:
                self._seed(live, state.frame)
            self._maybe_unstick(state.frame)
            while len(self._plan) < self.cfg.horizon:
                if not self._extend():
                    break
        finally:
            self._em.set_state(live)
            self.stats.restores += 1
        if ram_crc(self._get_ram()) != crc:
            raise RuntimeError("lookahead left the live RAM changed")
        step = self._plan.popleft()
        self._advance()
        return TickResult(
            status=NodeStatus.RUNNING,
            action=FrameAction(action=list(step.action), reason=step.reason),
            reason=step.reason,
        )

    # ------------------------------------------------------------------ plan
    def _drop(self) -> None:
        if self._nodes:
            # Keep the freshest policy copy at or before live as the base.
            self.base = self._nodes[0].policy
        self._plan.clear()
        self._nodes.clear()
        self._frontier = None

    def _seed(self, live: bytes, frame: int) -> None:
        node = _Node(0, frame, live, copy.deepcopy(self.base))
        self._nodes = deque([node])
        self._frontier = node

    def _advance(self) -> None:
        """Shift offsets after one committed frame; forget the far past."""
        for node in self._nodes:
            node.offset -= 1
        while len(self._nodes) > 1 and self._nodes[1].offset <= 0:
            self._nodes.popleft()

    def _extend(self) -> bool:
        """Grow the verified plan by one chunk; branch on a threat.

        False when the plan now ends in a forced life loss: nothing more to
        verify past it this tick.
        """
        front = self._frontier
        assert front is not None
        branch = self._roll(front, prefix=(), label="policy", frames=self.cfg.chunk)
        if branch.safe:
            self._splice(front.offset, branch)
            return True
        if self._affordable(branch):
            self.stats.accepted += 1
            self.stats.accepted_hp += branch.hp_lost
            self._splice(front.offset, branch)
            return True
        self.stats.threats += 1
        hit_at = front.offset + (branch.first_hit or 0)
        return self._rescue(hit_at, branch)

    def _note_progress(self, state: GameState) -> None:
        """Live progress clock: enemy HP lost, a stage change, or an empty
        screen (walking on is progress; a stall is a fight going nowhere)."""
        enemy = _enemy_hp(state)
        if state.stage != self._stage or enemy < self._enemy_hp or enemy == 0:
            self._progress_frame = state.frame
        self._stage = state.stage
        self._enemy_hp = enemy
        if living_hp(state.health):
            self._bar = max(self._bar, int(state.health))

    def _affordable(self, branch: Branch) -> bool:
        """A stalled guard may let the policy trade HP above the floor."""
        if branch.life_lost or not self._nodes:
            return False
        head = self._nodes[0]
        now = head.frame - head.offset
        if now - self._progress_frame < self.cfg.stall_frames:
            return False
        return branch.start_hp - branch.hp_lost >= self.cfg.hp_floor * max(self._bar, 1)

    def _maybe_unstick(self, frame: int) -> None:
        """While stalled, splice the best enemy-damaging macro branch."""
        cfg = self.cfg
        if frame - self._progress_frame < cfg.stall_frames:
            return
        if frame - self._last_search_frame < cfg.search_every:
            return
        self._last_search_frame = frame
        node = self._node_at_or_before(0)
        if node is None:
            return
        self.stats.searches += 1
        floor = cfg.hp_floor * max(self._bar, 1)
        best: tuple[float, Branch] | None = None
        for label, prefix in MACROS:
            branch = self._roll(
                node,
                prefix=prefix,
                label=f"unstick:{label}",
                frames=max(cfg.search_frames, len(prefix)),
            )
            if branch.life_lost or branch.enemy_dmg <= 0:
                continue
            if branch.start_hp - branch.hp_lost < floor:
                continue
            score = branch.score(cfg)
            if best is None or score > best[0]:
                best = (score, branch)
        if best is not None:
            self.stats.unstuck += 1
            self.stats.macros[best[1].label] += 1
            self._splice(node.offset, best[1])

    def _rescue(self, hit_at: int, failed: Branch) -> bool:
        """Splice the best hit-free branch, else the least-bad one.

        Keeps only the branches it may splice (each holds a 430 KB snapshot
        every chunk). Returns False when the splice still loses a life, so
        ``tick`` stops extending a lost future.
        """
        cfg = self.cfg
        forced: tuple[float, int, Branch] = (
            failed.score(cfg),
            self._frontier.offset if self._frontier else 0,
            failed,
        )
        scores: dict[int, list[tuple[float, str]]] = {}
        earliest: _Node | None = None
        seen: set[int] = set()
        for lead in cfg.leads:
            node = self._node_at_or_before(hit_at - lead)
            if node is None or id(node) in seen:
                continue
            seen.add(id(node))
            if earliest is None or node.offset < earliest.offset:
                earliest = node
            length = max(hit_at + cfg.margin - node.offset, cfg.chunk)
            best: tuple[float, Branch] | None = None
            safe = 0
            for label, prefix in self._macro_order():
                branch = self._roll(node, prefix=prefix, label=label, frames=length)
                score = branch.score(cfg)
                scores.setdefault(id(node), []).append((score, label))
                if branch.safe:
                    safe += 1
                    if best is None or score > best[0]:
                        best = (score, branch)
                    if safe >= cfg.enough_safe:
                        break
                elif score > forced[0]:
                    forced = (score, node.offset, branch)
            if best is not None:
                self.stats.rescued += 1
                self.stats.leads[lead] += 1
                self.stats.macros[best[1].label] += 1
                self._splice(node.offset, best[1])
                return True
        # No single macro keeps HP: two in a row from the earliest node tried
        # (most freedom), seeded by the best first moves seen there.
        frame_now = self._nodes[0].frame - self._nodes[0].offset if self._nodes else 0
        cooled = frame_now - self._last_forced_frame > cfg.pair_cooldown
        if earliest is not None and cooled:
            pair = self._rescue_pairs(hit_at, earliest, scores.get(id(earliest), []))
            if pair is not None:
                self.stats.rescued += 1
                self.stats.pairs += 1
                self.stats.macros[pair.label] += 1
                self._splice(earliest.offset, pair)
                return True
        _score, offset, best_forced = forced
        self._last_forced_frame = frame_now
        self.stats.forced += 1
        self.stats.forced_hp += best_forced.hp_lost
        self.stats.macros[f"forced:{best_forced.label}"] += 1
        if self._log is not None:
            nodes = [n.offset for n in self._nodes if n.offset >= 0]
            head = self._nodes[0] if self._nodes else None
            self._log(
                f"lookahead forced: hp_lost={best_forced.hp_lost} "
                f"life_lost={best_forced.life_lost} via {best_forced.label} at +{offset}; "
                f"hit_at=+{hit_at} plan={len(self._plan)} nodes={nodes[:12]} "
                f"frame={head.frame - head.offset if head else '?'}"
            )
        self._splice(offset, best_forced)
        return not best_forced.life_lost

    def _macro_order(self) -> list[tuple[str, tuple[Action, ...]]]:
        """``MACROS`` ordered by past rescues (stable for ties)."""
        wins = self.stats.macros
        return sorted(MACROS, key=lambda row: -wins.get(row[0], 0))

    def _rescue_pairs(
        self, hit_at: int, node: _Node, firsts: list[tuple[float, str]]
    ) -> Branch | None:
        cfg = self.cfg
        if cfg.pair_seeds <= 0 or not firsts:
            return None
        prefixes = dict(MACROS)
        length = max(hit_at + cfg.margin - node.offset, cfg.chunk)
        best: tuple[float, Branch] | None = None
        for _score, first in sorted(firsts, reverse=True)[: cfg.pair_seeds]:
            head = prefixes[first]
            for label, tail in MACROS:
                branch = self._roll(
                    node,
                    prefix=head + tail,
                    label=f"{first}+{label}",
                    frames=max(length, len(head) + len(tail)),
                )
                if branch.safe:
                    score = branch.score(cfg)
                    if best is None or score > best[0]:
                        best = (score, branch)
        return None if best is None else best[1]

    def _node_at_or_before(self, offset: int) -> _Node | None:
        """Latest non-past node at or before ``offset`` (else the earliest)."""
        future = [node for node in self._nodes if node.offset >= 0]
        if not future:
            return None
        pick = future[0]
        for node in future:
            if node.offset <= offset:
                pick = node
        return pick

    def _splice(self, offset: int, branch: Branch) -> None:
        """Replace the plan after ``offset`` with ``branch``."""
        keep = max(offset, 0)
        while len(self._plan) > keep:
            self._plan.pop()
        while self._nodes and self._nodes[-1].offset > keep:
            self._nodes.pop()
        if not self._nodes or self._nodes[-1].offset != keep:
            raise RuntimeError("splice point is not a snapshot node")
        self._plan.extend(branch.steps)
        self._nodes.extend(branch.nodes)
        self._frontier = branch.end

    # --------------------------------------------------------------- rollout
    def _roll(
        self,
        node: _Node,
        *,
        prefix: Sequence[Action],
        label: str,
        frames: int,
    ) -> Branch:
        em = self._em
        get_ram = self._get_ram
        cfg = self.cfg
        em.set_state(node.state)
        self.stats.restores += 1
        self.stats.branches += 1
        policy = copy.deepcopy(node.policy)
        ram = get_ram()
        state = parse_game_state(ram, frame=node.frame)
        start_stage = state.stage
        start_progress = int(state.extras.get("progress_x", state.camera_x))
        prev_hp = state.health
        prev_lives = state.lives
        prev_enemy = _enemy_hp(state)
        branch = Branch(label=label, steps=[], nodes=[], end=None, start_hp=int(state.health))
        total = max(frames, len(prefix))
        mask = np.zeros(12, dtype=np.uint8)
        for i in range(total):
            crc = ram_crc(ram)
            if i < len(prefix):
                action, reason = prefix[i], f"la:{label}"
            else:
                action, reason = control_action(policy, state)
            branch.steps.append(_Step(action, reason, crc))
            mask[:] = action
            em.set_button_mask(mask, 0)
            em.step()
            self.stats.rollout_frames += 1
            ram = get_ram()
            state = parse_game_state(ram, frame=node.frame + i + 1)
            hp = state.health
            if state.lives < prev_lives or state.player_dead:
                branch.life_lost = True
                branch.first_hit = branch.first_hit or (i + 1)
                break
            # HP 0 is a KO only with the turtle on screen outside a fade:
            # a stage clear zeroes the player entity (HP and X) on its own.
            ko = hp == 0 and not is_stage_clear_zero(state)
            if living_hp(prev_hp) and (living_hp(hp) or ko) and hp < prev_hp:
                branch.hp_lost += prev_hp - hp
                if branch.first_hit is None:
                    branch.first_hit = i + 1
                if hp == 0:
                    branch.life_lost = True
                    break
            elif living_hp(prev_hp) and living_hp(hp) and hp > prev_hp:
                branch.pizza += hp - prev_hp
            if living_hp(hp):
                prev_hp = hp
            enemy = _enemy_hp(state)
            if enemy < prev_enemy:
                branch.enemy_dmg += prev_enemy - enemy
            prev_enemy = enemy
            prev_lives = state.lives
            if state.stage != start_stage:
                branch.stage_advanced = True
            if (i + 1) % cfg.chunk == 0 and i + 1 < total:
                branch.nodes.append(self._snap(node, i + 1, policy))
            if branch.stage_advanced and i + 1 >= len(prefix):
                break
        # The end is always a node, so the next extension can splice there.
        branch.end = self._snap(node, len(branch.steps), policy)
        branch.nodes.append(branch.end)
        branch.progress = int(state.extras.get("progress_x", state.camera_x)) - start_progress
        return branch

    def _snap(self, node: _Node, steps: int, policy: Stage1Policy) -> _Node:
        return _Node(
            node.offset + steps,
            node.frame + steps,
            bytes(self._em.get_state()),
            copy.deepcopy(policy),
        )
