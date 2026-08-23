"""Headless Liu Kang tournament: power-on → fights with per-round model swap.

Bronze/Clean: read-only RAM for boot, screen class, and (when available) v3
policies. Pixel CNN specialists remain valid fallbacks until RAM models exist.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from retro_harness.actions import snes_action
from retro_harness.env import make_env, reset_obs
from mortal_kombat.boot import BootController, Phase, action_from_buttons
from mortal_kombat.paths import GAME_DIR, GAME_ID, MODEL_DIR
from mortal_kombat.ram import (
    FightSnapshot,
    Screen,
    parse_ram,
    rounds_settled,
)
from mortal_kombat.roster import (
    KIND_RAM_V3,
    KIND_SCRIPT,
    SCRIPT_NAME,
    STAGES,
    StageSlot,
    backup_on_round_loss,
    build_slots,
    slot_for_match,
)

PolicyLoader = Callable[[Path, str], object]


def _force_scripted_slots(slots: list[StageSlot]) -> list[StageSlot]:
    """Use the zip-less scripted policy for every fight, even with no models."""
    if slots:
        return [replace(slot, model=SCRIPT_NAME, kind=KIND_SCRIPT) for slot in slots]
    return [
        StageSlot(
            prefix=prefix,
            display=display,
            match_id=match_id,
            model=SCRIPT_NAME,
            kind=KIND_SCRIPT,
        )
        for prefix, display, match_id in STAGES
    ]


# Post-match: KO → FINISH HIM → victory pose → VS. START too early drops
# out of tournament mode (see cheat_extractor.wait_for_health_reset).
_MENU_QUIET_FRAMES = 900


@dataclass
class RamEvent:
    """One RAM-visible screen / fight / round change."""

    frame: int
    screen: str
    match: int
    p2: int
    p1_rounds: int
    p2_rounds: int
    hp: tuple[int, int]
    timer: int
    swap: str | None = None


@dataclass
class TournamentResult:
    """One power-on attempt."""

    cleared: bool
    furthest: str
    wins: int
    losses: int
    frames: int
    credits: bool
    swaps: list[str] = field(default_factory=list)
    events: list[RamEvent] = field(default_factory=list)


class TournamentRunner:
    """Drive one continuous attempt with roster specialists."""

    def __init__(
        self,
        model_dir: Path | None = None,
        *,
        deterministic: bool = False,
        policy_loader: PolicyLoader | None = None,
        on_frame: Callable | None = None,
        menu_quiet_frames: int = _MENU_QUIET_FRAMES,
        force_scripted: bool = False,
        ladder_model: str | None = None,
    ):
        self.model_dir = model_dir or MODEL_DIR
        self.deterministic = deterministic
        self._policy_loader = policy_loader
        self.on_frame = on_frame
        self.menu_quiet_frames = menu_quiet_frames
        self.slots = build_slots(self.model_dir)
        if force_scripted:
            self.slots = _force_scripted_slots(self.slots)
        elif ladder_model:
            self.slots = [
                replace(slot, model=ladder_model, kind=KIND_RAM_V3)
                if slot.match_id <= 6
                else slot
                for slot in self.slots
            ]
        self._policies: dict[str, object] = {}
        self._active: object | None = None
        self._active_slot: StageSlot | None = None
        self.swaps: list[str] = []
        self.events: list[RamEvent] = []
        self._fight_key: tuple[int, int] | None = None
        self._menu_quiet_until = 0
        self._seen_p1 = 0
        self._seen_p2 = 0
        self._score_match = -1

    def _load_policy(self, path: Path, kind: str):
        if self._policy_loader is not None:
            return self._policy_loader(path, kind)
        from mortal_kombat.compat import install_fighters_common_alias
        from mortal_kombat.policy import load_policy

        install_fighters_common_alias()
        return load_policy(path, kind)

    def _policy_for(self, slot: StageSlot):
        key = slot.model
        if key not in self._policies:
            path = self.model_dir / slot.model
            self._policies[key] = self._load_policy(path, slot.kind)
        return self._policies[key]

    def _swap(self, slot: StageSlot, reason: str) -> str:
        policy = self._policy_for(slot)
        if policy is self._active:
            self._active_slot = slot
            return ""
        policy.reset()
        self._active = policy
        self._active_slot = slot
        label = f"{reason}:{slot.prefix}:{slot.model}"
        self.swaps.append(label)
        return label

    def _slot_for(self, snap: FightSnapshot) -> StageSlot | None:
        if snap.p2_character > 8 or not self.slots:
            return None
        return slot_for_match(snap.match_counter, snap.p2_character, self.slots)

    def _maybe_swap_fight(self, snap: FightSnapshot) -> tuple[str, StageSlot | None]:
        slot = self._slot_for(snap)
        if slot is None:
            return "", None
        key = (snap.match_counter, snap.p2_character)
        if key == self._fight_key:
            return "", slot
        self._fight_key = key
        return self._swap(slot, "fight"), slot

    def _maybe_swap_round(self, prev: FightSnapshot, snap: FightSnapshot) -> str:
        if self._active_slot is None:
            return ""
        lost_round = snap.p2_rounds > prev.p2_rounds
        if not lost_round:
            return ""
        backup = backup_on_round_loss(self._active_slot, self.model_dir)
        if backup is None:
            return ""
        alt = StageSlot(
            prefix=self._active_slot.prefix,
            display=self._active_slot.display,
            match_id=self._active_slot.match_id,
            model=backup,
            kind=KIND_SCRIPT if backup == SCRIPT_NAME else "pixel",
            backups=[],
        )
        return self._swap(alt, "round_loss")

    def _record(
        self,
        frame: int,
        snap: FightSnapshot,
        prev: FightSnapshot | None,
        swap: str,
    ) -> None:
        changed = prev is None or swap or (
            snap.screen is not prev.screen
            or snap.match_counter != prev.match_counter
            or snap.p2_character != prev.p2_character
            or snap.p1_rounds != prev.p1_rounds
            or snap.p2_rounds != prev.p2_rounds
        )
        if not changed:
            return
        self.events.append(
            RamEvent(
                frame=frame,
                screen=snap.screen.name,
                match=snap.match_counter,
                p2=snap.p2_character,
                p1_rounds=snap.p1_rounds,
                p2_rounds=snap.p2_rounds,
                hp=(snap.p1_health, snap.p2_health),
                timer=snap.timer,
                swap=swap or None,
            )
        )

    def run(self, *, max_frames: int = 200_000) -> TournamentResult:
        env = make_env(GAME_ID, "NONE", GAME_DIR, render_mode="rgb_array")
        try:
            reset_obs(env)
            return self._loop(env, max_frames=max_frames)
        finally:
            env.close()

    def run_on(self, env, *, max_frames: int = 200_000) -> TournamentResult:
        """Drive an already-reset env (tests, round probe, save-state starts)."""
        return self._loop(env, max_frames=max_frames)

    def _attempt_result(
        self,
        *,
        cleared: bool,
        furthest: str,
        wins: int,
        losses: int,
        frames: int,
        credits: bool,
    ) -> TournamentResult:
        return TournamentResult(
            cleared, furthest, wins, losses, frames, credits, self.swaps, self.events
        )

    def _terminal(
        self,
        snap: FightSnapshot,
        *,
        frame: int,
        furthest: str,
        wins: int,
        losses: int,
    ) -> TournamentResult | None:
        if snap.screen is Screen.CREDITS:
            return self._attempt_result(
                cleared=True,
                furthest="credits",
                wins=wins,
                losses=losses,
                frames=frame,
                credits=True,
            )
        if snap.screen is Screen.CONTINUE:
            # Clean attempt ends at Continue. HUD can jump 0→2 without a +1 tick.
            already_lost = self._seen_p2 >= 2 and self._seen_p2 > self._seen_p1
            if not already_lost:
                losses += 1
            return self._attempt_result(
                cleared=False,
                furthest=furthest,
                wins=wins,
                losses=losses,
                frames=frame,
                credits=False,
            )
        return None

    def _loop(self, env, *, max_frames: int) -> TournamentResult:
        boot = BootController(allow_continue=False)
        prev = parse_ram(env.unwrapped.get_ram())
        wins = 0
        losses = 0
        furthest = "boot"
        furthest_match = -1
        credits = False
        self.swaps = []
        self.events = []
        self._fight_key = None
        self._menu_quiet_until = 0
        self._seen_p1 = 0
        self._seen_p2 = 0
        self._score_match = -1
        last_swap = ""

        for frame in range(max_frames):
            ram = env.unwrapped.get_ram()
            snap = parse_ram(ram)
            phase, menu_buttons = boot.decide(snap, frame)

            ended = self._terminal(
                snap, frame=frame, furthest=furthest, wins=wins, losses=losses
            )
            if ended is not None:
                self._record(frame, snap, prev, last_swap)
                return ended

            fight_swap, slot = self._maybe_swap_fight(snap)
            # Furthest is the roster slot's match_id, not whether the policy object changed.
            if slot is not None and slot.match_id > furthest_match:
                furthest_match = slot.match_id
                furthest = slot.display

            if snap.match_counter != self._score_match:
                self._score_match = snap.match_counter
                self._seen_p1 = 0
                self._seen_p2 = 0
            if snap.screen is Screen.FIGHT and snap.timer > 50:
                self._seen_p1 = max(self._seen_p1, snap.p1_rounds)
                self._seen_p2 = max(self._seen_p2, snap.p2_rounds)
            round_swap = ""
            if rounds_settled(snap):
                if snap.p1_rounds == self._seen_p1 + 1 and snap.p1_rounds <= 2:
                    self._seen_p1 = snap.p1_rounds
                    if self._seen_p1 >= 2 and self._seen_p1 > self._seen_p2:
                        wins += 1
                        self._menu_quiet_until = frame + self.menu_quiet_frames
                if snap.p2_rounds == self._seen_p2 + 1 and snap.p2_rounds <= 2:
                    self._seen_p2 = snap.p2_rounds
                    round_swap = self._maybe_swap_round(prev, snap)
                    if self._seen_p2 >= 2 and self._seen_p2 > self._seen_p1:
                        losses += 1
            last_swap = fight_swap or round_swap
            self._record(frame, snap, prev, last_swap)

            entered_fight = (
                snap.screen is Screen.FIGHT
                and prev.screen is not Screen.FIGHT
                and self._active is not None
            )
            if entered_fight:
                self._active.reset()

            stop = False
            if self.on_frame is not None:
                stop = self.on_frame(env, frame, snap, prev) is True
            if stop:
                return self._attempt_result(
                    cleared=False,
                    furthest=furthest,
                    wins=wins,
                    losses=losses,
                    frames=frame,
                    credits=credits,
                )

            if phase is Phase.FIGHT and self._active is not None:
                rgb = None
                if getattr(self._active, "kind", "") == "pixel":
                    rgb = env.render()
                buttons = self._active.act(ram, rgb, deterministic=self.deterministic)
            elif (
                menu_buttons
                and snap.screen is Screen.MENU
                and frame < self._menu_quiet_until
            ):
                buttons = snes_action(dtype=np.int8)
            elif menu_buttons:
                buttons = action_from_buttons(menu_buttons)
            else:
                buttons = snes_action(dtype=np.int8)

            env.step(buttons)
            prev = snap

        return self._attempt_result(
            cleared=False,
            furthest=furthest,
            wins=wins,
            losses=losses,
            frames=max_frames,
            credits=credits,
        )
