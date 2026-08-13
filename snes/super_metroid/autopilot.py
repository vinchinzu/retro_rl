"""Human-hot-swappable Super Metroid room autopilot."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from super_metroid.paths import GAME_DIR
from super_metroid.ram import GameplayPhase, SuperMetroidState, parse_env_state
from super_metroid.reactive_policy import (
    POLICY_KIND,
    ReactivePolicyRunner,
    ReactiveRoomPolicy,
)
from super_metroid.room_adapter import AdapterSearchConfig, search_live_adapter

DEFAULT_REACTIVE_POLICY_DIR = GAME_DIR / "policies" / "reactive_rooms"


@dataclass(frozen=True)
class AutopilotStatus:
    active: bool
    room_id: int
    policy_id: str | None
    variant_id: str | None
    mode: str
    detail: str = ""

    def summary(self) -> str:
        room = f"0x{self.room_id:04X}" if self.room_id else "room=?"
        policy = self.policy_id or "no-policy"
        variant = f"/{self.variant_id}" if self.variant_id else ""
        suffix = f" {self.detail}" if self.detail else ""
        return f"AP {self.mode} {room} {policy}{variant}{suffix}"


class ReactivePolicyRegistry:
    """Room/inventory/from-room lookup over compiled policy JSON files."""

    def __init__(
        self,
        policies: tuple[ReactiveRoomPolicy, ...] = (),
        *,
        allow_candidates: bool = False,
    ) -> None:
        self.policies = policies
        self.allow_candidates = allow_candidates

    @classmethod
    def load_dir(
        cls,
        path: Path | str = DEFAULT_REACTIVE_POLICY_DIR,
        *,
        allow_candidates: bool = False,
    ) -> ReactivePolicyRegistry:
        root = Path(path)
        policies: list[ReactiveRoomPolicy] = []
        if root.is_dir():
            for candidate in sorted(root.glob("*.json")):
                try:
                    raw = json.loads(candidate.read_text(encoding="utf-8"))
                    if raw.get("kind") != POLICY_KIND:
                        continue
                    policies.append(ReactiveRoomPolicy.from_dict(raw))
                except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    print(f"[AUTOPILOT] skip {candidate}: {exc}")
        return cls(tuple(policies), allow_candidates=allow_candidates)

    def select(
        self,
        state: SuperMetroidState,
        *,
        from_room_id: int | None = None,
        route_id: str | None = None,
    ) -> tuple[ReactiveRoomPolicy, Any] | None:
        matches: list[
            tuple[tuple[int, float, int, str], ReactiveRoomPolicy, Any]
        ] = []
        for policy in self.policies:
            if policy.room_id != int(state.room_id):
                continue
            if route_id is not None and policy.route_id != route_id:
                continue
            if not self.allow_candidates and policy.status != "verified_live_anchor":
                continue
            variant = policy.select_variant(int(state.collected_items))
            if variant is None:
                continue
            from_match = int(
                from_room_id is not None and policy.from_room_id == int(from_room_id)
            )
            from_generic = int(policy.from_room_id is None)
            # A first mid-room handoff may not know which door the human used.
            # Rank those ambiguous policies by live kinematic fit, once, here.
            projection = ReactivePolicyRunner(variant).project_global(state)
            matches.append(
                (
                    (
                        from_match,
                        -float(projection.score),
                        from_generic,
                        policy.policy_id,
                    ),
                    policy,
                    variant,
                )
            )
        if not matches:
            return None
        _, policy, variant = max(matches, key=lambda row: row[0])
        return policy, variant


class RoomAutopilot:
    """Callable PlaySession bot that can attach at any in-room frame.

    When no compiled room skill matches, status detail may cite a
    **practice repertoire** recovery pin (route session + hop_key) for
    thrash reseed — see ``practice_repertoire.recover_session``.

    Hard rooms (e.g. Red Tower Ice climb) stay human-only until a verified
    reactive policy exists; no route-specific checkpoint trees live here.
    """

    def __init__(
        self,
        env: Any,
        *,
        registry: ReactivePolicyRegistry | None = None,
        policy_dir: Path | str = DEFAULT_REACTIVE_POLICY_DIR,
        route_id: str = "kpdr",
        allow_candidates: bool = False,
        use_adapter: bool = True,
        adapter_config: AdapterSearchConfig = AdapterSearchConfig(),
    ) -> None:
        self.env = env
        self.registry = registry or ReactivePolicyRegistry.load_dir(
            policy_dir,
            allow_candidates=allow_candidates,
        )
        self.route_id = route_id
        self.use_adapter = use_adapter
        self.adapter_config = adapter_config
        self.policy: ReactiveRoomPolicy | None = None
        self.runner: ReactivePolicyRunner | None = None
        self._adapter_frames: deque[tuple[int, ...]] = deque()
        self._resume_after_adapter = False
        self._last_room = 0
        self._from_room: int | None = None
        self._active = False
        self._mode = "human"
        self._detail = ""

    def mission_status(self) -> AutopilotStatus:
        variant = self.runner.variant.variant_id if self.runner is not None else None
        return AutopilotStatus(
            active=self._active,
            room_id=self._last_room,
            policy_id=self.policy.policy_id if self.policy is not None else None,
            variant_id=variant,
            mode=self._mode,
            detail=self._detail,
        )

    def _observe_room(self, state: SuperMetroidState) -> None:
        room = int(state.room_id)
        if room and self._last_room and room != self._last_room:
            self._from_room = self._last_room
            self.policy = None
            self.runner = None
            self._adapter_frames.clear()
        if room:
            self._last_room = room

    def _recovery_detail(self, state: SuperMetroidState) -> str:
        try:
            from super_metroid.practice_repertoire import recovery_hint_for_state

            hint = recovery_hint_for_state(state)
        except (ImportError, OSError, KeyError, TypeError, ValueError):
            hint = None
        if hint is not None:
            return (
                f"no skill; recover→{hint.session_id} "
                f"grade={hint.grade} hop={hint.hop_key or '?'}"
            )
        return "waiting for compiled room skill"

    def _attach(self, state: SuperMetroidState) -> bool:
        self._observe_room(state)
        selected = self.registry.select(
            state,
            from_room_id=self._from_room,
            route_id=self.route_id,
        )
        if selected is None:
            self.policy = None
            self.runner = None
            self._mode = "human-fallback"
            self._detail = self._recovery_detail(state)
            return False
        policy, variant = selected
        self.policy = policy
        self.runner = ReactivePolicyRunner(variant)
        target = self.runner.resume(state)
        self._mode = "tracking"
        self._detail = f"join={target.sample_index} d={target.score:.0f}"

        if self.use_adapter and target.score > variant.rejoin_threshold:
            plan = search_live_adapter(
                self.env,
                self.runner,
                config=self.adapter_config,
            )
            self._adapter_frames = deque(plan.frames)
            self._resume_after_adapter = bool(plan.frames)
            if plan.frames:
                self._mode = "adapting"
                self._detail = (
                    f"{plan.frame_count}f {plan.score_before:.0f}→{plan.score_after:.0f}"
                )
        return True

    def on_human_takeover(self) -> None:
        self._active = False
        self._mode = "human"
        self._detail = ""
        self._adapter_frames.clear()
        self._resume_after_adapter = False

    def on_autopilot_resume(self) -> None:
        self._active = True
        state = parse_env_state(self.env, mode="nav")
        self._attach(state)

    def recovery_hint(self, state: SuperMetroidState | None = None) -> dict[str, Any] | None:
        """Nearest practice-repertoire pin for thrash reseed.

        Used when no reactive room skill is compiled yet, or after AP falls
        back to human. Does not load the state — callers reseed PlaySession.
        """
        live = state or parse_env_state(self.env, mode="nav")
        try:
            from super_metroid.practice_repertoire import recovery_hint_for_state

            hint = recovery_hint_for_state(live)
        except (ImportError, OSError, KeyError, TypeError, ValueError):
            return None
        return hint.to_dict() if hint is not None else None

    def __call__(self, _obs: Any, _info: Any) -> np.ndarray | None:
        if not self._active:
            return None
        if self._adapter_frames:
            raw = self._adapter_frames.popleft()
            return np.asarray(raw, dtype=np.int8)

        # Span playback needs no RAM parse or nearest-neighbor pass. Bounded
        # spans ensure a room transition / drift is noticed within <=8 frames.
        if (
            self.runner is not None
            and not self._resume_after_adapter
            and self.runner.has_held_action
        ):
            return self.runner.continue_action()

        state = parse_env_state(self.env, mode="nav")
        self._observe_room(state)
        if self._resume_after_adapter:
            self._resume_after_adapter = False
            if self.runner is not None:
                self.runner.resume(state)

        if self.policy is None or self.runner is None or state.room_id != self.policy.room_id:
            if not self._attach(state):
                return None
        assert self.policy is not None and self.runner is not None

        if int(state.room_id) == self.policy.exit_room_id:
            self.policy = None
            self.runner = None
            self._mode = "room-complete"
            return None
        if state.phase is not GameplayPhase.ORDINARY_GAMEPLAY:
            # Keep the last planned door input only when the trajectory says so;
            # otherwise wait for settled ordinary gameplay before new lookup.
            if self.runner.has_held_action:
                return self.runner.continue_action()
        action = self.runner.action(state)
        status = self.runner.status()
        self._mode = "tracking"
        self._detail = (
            f"{status['cursor']}/{status['samples']} d={status['projection_score']:.0f}"
        )
        return action


__all__ = [
    "AutopilotStatus",
    "DEFAULT_REACTIVE_POLICY_DIR",
    "ReactivePolicyRegistry",
    "RoomAutopilot",
]
