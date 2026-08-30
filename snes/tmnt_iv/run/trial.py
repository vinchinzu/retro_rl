"""One emulator trial for TMNT IV probes and the continuous Hard run."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from retro_harness.actions import buttons, idle_action
from retro_harness.controls import SNES_START
from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.ram_state import GameMode, GameState
from retro_harness.segment_runner import SegmentOutcome, configure_headless
from tmnt_iv.assist import (
    EMERGENCY_HP_RESTORE,
    apply_emergency_hp,
    apply_form2_iframe_hold,
)
from tmnt_iv.menus import RAPH_HARD_BOOT_LAST, raph_hard_boot_action
from tmnt_iv.observe import HpDelta, living_hp, policy_input
from tmnt_iv.paths import GAME, GAME_DIR, INTEGRATION_DIR
from tmnt_iv.policy import Stage1Policy
from tmnt_iv.ram import (
    ADDR_DIFFICULTY,
    ADDR_LIVES,
    ADDR_STAGE,
    PLAYER_BASE,
    OFF_HP,
    OFF_IFRAMES,
    parse_game_state,
)
from tmnt_iv.run.freeze import (
    FREEZE_ABORT_FRAMES,
    FlightRecorder,
    dump_abort_snapshot,
)
from tmnt_iv.run.metrics import (
    HARD_VALUE,
    METRIC_HOLD_FRAMES,
    STAGE_NAMES,
    CreditsTracker,
    RunMetrics,
    StageSplit,
    format_duration,
)

EntryKind = Literal["power_on", "state"]
ObjectiveKind = Literal["stage_advance", "boss_fade", "credits", "wave_chain"]
ContractName = Literal["clean", "assisted", "dev"]
OnFrame = Callable[["TrialFrame"], None]
LivePred = Callable[[GameState], bool]

_SUCCESS = frozenset({"stage_advance", "boss_down", "cleared", "credits"})
_BOSS_FADE_EVENTS = frozenset({0x0B, 0x19, 0x04})
_WRITE_ADDR = {
    "player_hp": PLAYER_BASE + OFF_HP,
    "player_iframes": PLAYER_BASE + OFF_IFRAMES,
    "stage": ADDR_STAGE,
    "lives": ADDR_LIVES,
    "player_lives": ADDR_LIVES,
}
_LIVES_KEYS = frozenset({"lives", "player_lives"})
_CRAFTED_NAME = re.compile(r"_hp\d+", re.I)
_UNDERFOOT = (14, 18)
_DEV_HEAL_BELOW = 28


@dataclass(frozen=True)
class ProgressSpec:
    """Stage-aware stall watchdog. Freeze abort stays 12_000 enemyless X."""

    stall_frames: int = 2400
    freeze_abort_frames: int = FREEZE_ABORT_FRAMES


@dataclass(frozen=True)
class TrialEntry:
    """Power-on or save-state launch, with optional live / identity checks."""

    kind: EntryKind
    state_name: str
    is_live: LivePred | None = None
    boot_actions: Sequence[Any] | None = None
    expected_stage: int | None = None
    expected_character: int | None = None
    expected_difficulty: int | None = None
    entry_state_prefix: str | None = None


@dataclass(frozen=True)
class TrialObjective:
    """What the trial must prove before it may succeed."""

    kind: ObjectiveKind
    stop_stage_gt: int | None = None
    strict_advance: bool = False
    tracker: Any | None = None
    metric_hold_frames: int = METRIC_HOLD_FRAMES


@dataclass(frozen=True)
class TrialContract:
    """Clean fails closed: empty writes, no continue after KO."""

    name: ContractName
    emergency_hp: bool
    iframe_hold: bool
    fail_on_life_loss: bool = True
    allow_continue: bool = False
    allowed_write_keys: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.name == "clean" and self.allow_continue:
            raise ValueError("Clean contract must not allow continue")


@dataclass(frozen=True)
class TrialLimits:
    """Frame cap plus freeze / stall watchdogs."""

    max_frames: int
    freeze_abort_frames: int = FREEZE_ABORT_FRAMES
    progress: ProgressSpec | None = None


@dataclass(frozen=True)
class TrialFrame:
    """Per-frame hook context. Tactics stay outside this object."""

    frame: int
    state: GameState
    action: Any
    reason: str
    env: Any
    obs: Any
    live: bool


@dataclass(frozen=True)
class TrialResult:
    """Truthful trial outcome. ``success`` only for objective completions."""

    outcome: str
    success: bool
    frames: int
    total_frames: int
    start_hp: int | None
    end_hp: int
    min_hp: int | None
    start_lives: int | None
    end_lives: int
    life_losses: int
    lives: str
    damage_taken: int
    wave_damage: int
    boss_damage: int
    max_hit: int
    pizza_heals: list[dict[str, Any]]
    pizza_heal_count: int
    entry_stage: int | None
    entry_event: int | None
    entry_character: int | None
    entry_difficulty: int | None
    end_stage: int
    end_event: int
    end_character: int | None
    emergency_hp_writes: int
    iframe_writes: int
    ram_writes: list[dict[str, Any]]
    state_loads_after_launch: int
    a_special_uses: int
    hits: list[dict[str, Any]]
    action_reasons: dict[str, int]
    top_reasons: list[tuple[str, int]]
    contract_violations: list[dict[str, Any]]
    assist: str
    heal_mode: str
    integrity: dict[str, bool]
    failure: dict[str, Any] | None = None
    boss_entry_hp: int | None = None
    post_boot_start_presses: int = 0
    credits_start_frame: int | None = None
    credits_complete_frame: int | None = None
    hard_credits_event_seen: bool = False
    hard_confirmed: bool = False
    stage_splits: list[StageSplit] = field(default_factory=list)
    damage_by_stage: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly snapshot."""
        payload = asdict(self)
        payload["top_reasons"] = [list(row) for row in self.top_reasons]
        payload["damage_by_stage"] = {
            str(k): v for k, v in self.damage_by_stage.items()
        }
        return payload

    def to_metrics(self) -> RunMetrics:
        """Fill the full-run ``RunMetrics`` bag from this result."""
        metrics = RunMetrics(
            total_damage_taken=self.damage_taken,
            max_single_frame_damage=self.max_hit,
            health_guard_interventions=self.emergency_hp_writes,
            final_boss_iframe_guard_frames=self.iframe_writes,
            life_losses=self.life_losses,
            lives_start=self.start_lives,
            lives_peak=self.start_lives,
            lives_end=self.end_lives,
            min_health_seen=self.min_hp,
            credits_start_frame=self.credits_start_frame,
            credits_complete_frame=self.credits_complete_frame,
            hard_credits_event_seen=self.hard_credits_event_seen,
            stage_splits=list(self.stage_splits),
            damage_by_stage=dict(self.damage_by_stage),
        )
        if self.start_lives is not None and self.end_lives is not None:
            metrics.lives_peak = max(self.start_lives, self.end_lives)
        metrics.action_reasons.update(self.action_reasons)
        return metrics


CLEAN_CONTRACT = TrialContract(
    name="clean",
    emergency_hp=False,
    iframe_hold=False,
    allowed_write_keys=frozenset(),
)
ASSISTED_CONTRACT = TrialContract(
    name="assisted",
    emergency_hp=True,
    iframe_hold=True,
    allowed_write_keys=frozenset({"player_hp", "player_iframes"}),
)
DEV_CONTRACT = TrialContract(
    name="dev",
    emergency_hp=False,
    iframe_hold=False,
    allowed_write_keys=frozenset({"player_hp"}),
)


def catalog_state(
    name: str,
    state: GameState,
    *,
    path: Path | None = None,
) -> dict[str, Any]:
    """Provenance row for one pin. Does not scan the state directory."""
    pin = path if path is not None else (INTEGRATION_DIR / f"{name}.state")
    digest = None
    if pin.is_file():
        digest = hashlib.sha256(pin.read_bytes()).hexdigest()
    last_life = state.lives == 0
    crafted = bool(_CRAFTED_NAME.search(name))
    return {
        "name": name,
        "sha256": digest,
        "character": int(state.extras.get("char_id", -1)),
        "difficulty": int(state.extras.get("difficulty", -1)),
        "stage": state.stage,
        "event": int(state.extras.get("event", -1)),
        "hp": state.health,
        "lives": state.lives,
        "last_life": last_life,
        "hp_crafted": crafted,
        "clean_gate_eligible": not last_life and not crafted,
    }


class _AuditEm:
    def __init__(self, em: Any, owner: "_AuditEnv") -> None:
        self._em = em
        self._owner = owner

    def set_state(self, state: bytes) -> Any:
        if self._owner.live:
            self._owner.loads += 1
        return self._em.set_state(state)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._em, name)


class _AuditEnv:
    """Intercept ``env.set_value`` and ``em.set_state`` (AuditedEnv is data-only)."""

    def __init__(self, env: Any) -> None:
        self._env = env
        self.writes: list[dict[str, Any]] = []
        self.loads = 0
        self.frame = 0
        self.live = False
        em = getattr(env, "em", None)
        self._em = _AuditEm(em, self) if em is not None else None

    @property
    def em(self) -> Any:
        if self._em is not None:
            return self._em
        return getattr(self._env, "em")

    def set_value(self, name: str, value: Any) -> Any:
        self.writes.append(
            {"frame": int(self.frame), "key": str(name), "value": value}
        )
        setter = getattr(self._env, "set_value", None)
        if callable(setter):
            return setter(name, value)
        data = getattr(self._env, "data", None)
        if data is not None and hasattr(data, "set_value"):
            return data.set_value(name, value)
        raise AttributeError("environment has no set_value")

    def close(self) -> None:
        closer = getattr(self._env, "close", None)
        if callable(closer):
            closer()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


def _pizza_boxes(state: GameState) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    for item in state.extras.get("pickups") or ():
        if len(item) >= 3 and int(item[2]) == 0x30:
            rows.append((int(item[0]), int(item[1])))
    return rows


def _pizza_transition(prev: GameState, state: GameState) -> bool:
    prev_box = _pizza_boxes(prev)
    if not prev_box:
        return False
    if not _pizza_boxes(state):
        return True
    adx, ady = _UNDERFOOT
    return any(
        abs(x - prev.player_x) <= adx and abs(y - prev.player_y) <= ady
        for x, y in prev_box
    )


def _pressed(action: Any, index: int) -> bool:
    try:
        return bool(action[index])
    except Exception:
        return False


def _ko_state(state: GameState) -> bool:
    if state.player_x <= 0:
        return False
    if state.health == 0:
        return True
    return state.mode is GameMode.CONTINUE and state.health > 0x60


def _is_live(entry: TrialEntry, state: GameState) -> bool:
    if entry.is_live is not None:
        return bool(entry.is_live(state))
    return (
        state.mode is GameMode.PLAYING
        and living_hp(state.health)
        and state.player_x > 0
        and int(state.extras.get("event", 0)) >= 0x0A
    )


def _heal_mode(contract: TrialContract) -> str:
    if contract.emergency_hp:
        return "emergency"
    if contract.name == "dev":
        return "dev"
    return "none"


def _assist_label(pizza_count: int, contract: TrialContract, emergency: int, iframe: int) -> str:
    if pizza_count:
        return "pizza_only"
    if emergency or iframe or contract.name == "assisted":
        return "assisted"
    return "none"


def _violation(frame: int, kind: str, detail: str) -> dict[str, Any]:
    return {"frame": frame, "kind": kind, "detail": detail}


def _write_detail(write: dict[str, Any]) -> str:
    key = str(write.get("key"))
    addr = _WRITE_ADDR.get(key)
    if addr is None:
        return f"{key} (unknown)"
    return f"{key} @ {addr:#06x}"


def _identity_error(entry: TrialEntry, state: GameState, ram: Any) -> str | None:
    char = int(state.extras.get("char_id", -1))
    difficulty = int(state.extras.get("difficulty", -1))
    if difficulty < 0 and ram is not None:
        difficulty = int(ram[ADDR_DIFFICULTY])
    if entry.expected_character is not None and char != entry.expected_character:
        return f"character {char} != {entry.expected_character}"
    if (
        entry.expected_difficulty is not None
        and difficulty != entry.expected_difficulty
    ):
        return f"difficulty {difficulty} != {entry.expected_difficulty}"
    if entry.expected_stage is not None and state.stage != entry.expected_stage:
        return f"stage {state.stage} != {entry.expected_stage}"
    return None


def _fps(env: Any) -> float:
    getter = getattr(getattr(env, "em", None), "get_screen_rate", None)
    return float(getter()) if callable(getter) else 60.0


def _fail(recorder: FlightRecorder, reason: str) -> dict[str, Any]:
    return {
        "last_progress": recorder.last_progress,
        "first_cycle": recorder.first_cycle,
        "window_summary": recorder.window(),
        "reason": reason,
    }


def run_trial(
    entry: TrialEntry,
    objective: TrialObjective,
    contract: TrialContract,
    limits: TrialLimits,
    *,
    env: Any = None,
    policy: Stage1Policy | None = None,
    on_frame: OnFrame | None = None,
) -> TrialResult:
    """Observe, act, and classify one attempt. Inject ``env`` in tests."""
    if contract.name == "clean":
        contract = replace(contract, allow_continue=False)
    owned = env is None
    obs: Any = getattr(env, "obs", None) if env is not None else None
    if owned:
        configure_headless()
        env = make_env(GAME, entry.state_name, GAME_DIR, render_mode="rgb_array")
        obs, _info = reset_obs(env)
    env = _AuditEnv(env)
    policy = policy or Stage1Policy()
    spec = limits.progress or ProgressSpec(
        stall_frames=2400,
        freeze_abort_frames=limits.freeze_abort_frames,
    )
    recorder = FlightRecorder(
        freeze_abort_frames=limits.freeze_abort_frames,
        stall_frames=spec.stall_frames,
    )
    boot = list(entry.boot_actions or ())
    boot_i = 0
    live = False
    play0 = 0
    prev_state: GameState | None = None
    hp = HpDelta(count_zero=objective.kind == "credits")
    reasons: Counter[str] = Counter()
    pizza_heals: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    splits: list[StageSplit] = []
    split_stages: set[int] = set()
    damage_by_stage: dict[int, int] = {}
    credits = CreditsTracker()
    metrics = RunMetrics()
    outcome = "timeout"
    saw_boss = False
    saw_boss_fade = False
    boss_entry_hp: int | None = None
    start_hp: int | None = None
    start_lives: int | None = None
    entry_stage: int | None = None
    entry_event: int | None = None
    entry_character: int | None = None
    entry_difficulty: int | None = None
    prev_lives: int | None = None
    last_stage = -1
    emergency = 0
    iframe = 0
    a_special = 0
    life_losses = 0
    hard_confirmed = False
    failure: dict[str, Any] | None = None
    tracker = objective.tracker
    fps = _fps(env)
    final = parse_game_state(env.get_ram(), frame=0)
    seen_loads = 0

    def consume_new_writes(frame: int) -> str | None:
        nonlocal seen_loads
        for write in env.writes:
            if write.get("_seen"):
                continue
            write["_seen"] = True
            if str(write["key"]) not in contract.allowed_write_keys:
                violations.append(
                    _violation(frame, "ram_write", _write_detail(write))
                )
                return "contract_violation"
        if env.loads > seen_loads:
            seen_loads = env.loads
            violations.append(
                _violation(frame, "state_load", "em.set_state after live")
            )
            return "contract_violation"
        return None

    try:
        for frame in range(0, limits.max_frames + 1):
            env.frame = frame
            state = parse_game_state(env.get_ram(), frame=frame)
            final = state
            event = int(state.extras.get("event", -1))
            became_live = (not live) and _is_live(entry, state)
            if became_live:
                live = True
                env.live = True
                play0 = frame
                policy.reset()
                hp = HpDelta.start(
                    state.health, count_zero=objective.kind == "credits"
                )
                start_hp = state.health
                start_lives = state.lives
                prev_lives = state.lives
                entry_stage = state.stage
                entry_event = event
                entry_character = int(state.extras.get("char_id", -1))
                entry_difficulty = int(state.extras.get("difficulty", -1))
                ident = _identity_error(entry, state, env.get_ram())
                if ident is not None:
                    outcome, failure = "error", _fail(recorder, ident)
                    break
                if tracker is not None:
                    tracker.begin(state)

            if (
                live
                and objective.kind == "wave_chain"
                and entry_stage is not None
                and state.stage > entry_stage
            ):
                state = replace(state, level_complete=True)
                final = state

            if live and prev_state is not None and living_hp(state.health):
                if hp.prev is not None and state.health > hp.prev:
                    fade_hp = (
                        event in _BOSS_FADE_EVENTS
                        or int(prev_state.extras.get("event", -1))
                        in _BOSS_FADE_EVENTS
                    )
                    if _pizza_transition(prev_state, state):
                        pizza_heals.append(
                            {
                                "frame": frame - play0,
                                "from_hp": hp.prev,
                                "to_hp": state.health,
                                "player_x": state.player_x,
                                "pickup_seen": True,
                            }
                        )
                        hp.prev = state.health
                    elif fade_hp:
                        # Post-kill 0x0B/0x19 HP restore is the game, not pizza.
                        hp.prev = state.health
                    else:
                        violations.append(
                            _violation(
                                frame,
                                "unlabeled_hp_gain",
                                f"hp {hp.prev}->{state.health} without pizza",
                            )
                        )
                        outcome = "contract_violation"
                        failure = _fail(recorder, "unlabeled_hp_gain")
                        break

            if live:
                hit = hp.note(state.health)
                if hit:
                    hits.append(
                        {
                            "frame": frame - play0,
                            "hit": hit,
                            "hp": state.health,
                            "player_x": state.player_x,
                            "boss": state.boss_active,
                        }
                    )
                    damage_by_stage[state.stage] = (
                        damage_by_stage.get(state.stage, 0) + hit
                    )
                if (
                    state.boss_active
                    and boss_entry_hp is None
                    and living_hp(state.health)
                ):
                    boss_entry_hp = state.health
                if state.boss_active:
                    saw_boss = True
                if contract.emergency_hp and "player_hp" in contract.allowed_write_keys:
                    if apply_emergency_hp(env, state.health):
                        emergency += 1
                        state = parse_game_state(env.get_ram(), frame=frame)
                        final = state
                        hp.prev = state.health
                elif (
                    contract.name == "dev"
                    and "player_hp" in contract.allowed_write_keys
                    and 0 < state.health < _DEV_HEAL_BELOW
                    and state.health <= 0x60
                ):
                    env.set_value("player_hp", EMERGENCY_HP_RESTORE)
                    state = parse_game_state(env.get_ram(), frame=frame)
                    final = state
                    hp.prev = state.health
                if contract.iframe_hold and "player_iframes" in contract.allowed_write_keys:
                    if apply_form2_iframe_hold(
                        env, stage=state.stage, event=event
                    ):
                        iframe += 1
                bad = consume_new_writes(frame)
                if bad:
                    outcome = bad
                    failure = _fail(
                        recorder,
                        violations[-1]["detail"] if violations else bad,
                    )
                    break

            stop: str | None = None
            if live:
                if state.mode is GameMode.TITLE:
                    stop = "title"
                elif state.player_dead:
                    stop = "player_dead"
                elif state.mode is GameMode.GAME_OVER:
                    stop = "game_over"
                elif (
                    prev_lives is not None
                    and state.lives < prev_lives
                    and contract.fail_on_life_loss
                ):
                    life_losses += prev_lives - state.lives
                    stop = "life_loss"
                elif (not contract.allow_continue) and _ko_state(state):
                    stop = "ko"
                if prev_lives is None or state.lives >= (prev_lives or 0):
                    if stop != "life_loss":
                        prev_lives = state.lives
                if stop is None and objective.kind == "stage_advance":
                    limit = (
                        objective.stop_stage_gt
                        if objective.stop_stage_gt is not None
                        else (entry_stage if entry_stage is not None else -1)
                    )
                    if state.stage > limit:
                        if (not objective.strict_advance) or (
                            state.mode is GameMode.PLAYING and living_hp(state.health)
                        ):
                            stop = "stage_advance"
                if stop is None and saw_boss and not state.boss_active:
                    if event in _BOSS_FADE_EVENTS:
                        saw_boss_fade = True
                        if objective.kind == "boss_fade":
                            if event in {0x19, 0x04} or (
                                entry_stage is not None and state.stage > entry_stage
                            ):
                                stop = "cleared"
                            else:
                                stop = "boss_down"
                if stop is None and objective.kind == "credits":
                    difficulty = int(state.extras.get("difficulty", -1))
                    if difficulty == HARD_VALUE:
                        hard_confirmed = True
                    if not hard_confirmed and frame > 2500:
                        stop = "error"
                        failure = _fail(
                            recorder, f"difficulty changed from Hard: {difficulty}"
                        )
                    elif last_stage >= 0 and state.stage < last_stage:
                        stop = "error"
                        failure = _fail(
                            recorder,
                            f"stage regressed {last_stage}->{state.stage}",
                        )
                    else:
                        last_stage = max(last_stage, state.stage)
                        if state.stage not in split_stages and state.player_x > 0:
                            split_stages.add(state.stage)
                            policy.reset()
                            hp.prev = (
                                state.health if living_hp(state.health) else None
                            )
                            name = STAGE_NAMES.get(state.stage, "UNKNOWN")
                            splits.append(
                                StageSplit(
                                    stage=state.stage,
                                    name=name,
                                    frame=frame,
                                    elapsed_seconds=frame / fps,
                                )
                            )
                            if entry.entry_state_prefix:
                                save_state(
                                    env,
                                    GAME_DIR,
                                    GAME,
                                    f"{entry.entry_state_prefix}Stage{state.stage + 1}",
                                )
                            print(
                                f"stage {state.stage + 1:02d} {name} "
                                f"at {format_duration(frame / fps)} "
                                f"dmg={hp.damage}",
                                flush=True,
                            )
                    credits.update(state, frame=frame, metrics=metrics)
                    complete = metrics.credits_complete_frame
                    if (
                        complete is not None
                        and frame >= complete + objective.metric_hold_frames
                    ):
                        stop = "credits"
                if (
                    stop is None
                    and objective.kind == "wave_chain"
                    and tracker is not None
                    and not became_live
                ):
                    tracked = tracker.update(state)
                    if tracked is SegmentOutcome.SUCCESS:
                        stop = "cleared"
                    elif tracked is SegmentOutcome.DEATH:
                        stop = "ko"
                    elif tracked is SegmentOutcome.TIMEOUT:
                        stop = "timeout"

            action: Any
            reason: str
            if not live:
                if (
                    objective.kind == "credits"
                    and entry.kind == "power_on"
                    and frame <= RAPH_HARD_BOOT_LAST
                ):
                    action = raph_hard_boot_action(frame)
                    reason = "boot_menu" if any(action) else "boot_idle"
                elif boot_i < len(boot):
                    action = boot[boot_i]
                    boot_i += 1
                    reason = "boot_menu" if any(action) else "boot_idle"
                elif entry.kind == "power_on" and objective.kind != "credits":
                    action = (
                        buttons("START") if frame % 40 == 0 else idle_action()
                    )
                    reason = "boot_wait"
                else:
                    action, reason = policy_input(policy, state)
            elif stop is not None:
                action, reason = idle_action(), stop
            else:
                if objective.kind == "credits" and metrics.credits_start_frame is not None:
                    action, reason = idle_action(), "credits_idle"
                elif objective.kind == "credits" and (
                    state.player_x == 0
                    or state.mode in {GameMode.CUTSCENE, GameMode.CONTINUE}
                ):
                    action, reason = idle_action(), "transition_idle"
                else:
                    action, reason = policy_input(policy, state)
                if (
                    objective.kind == "credits"
                    and frame > RAPH_HARD_BOOT_LAST
                    and _pressed(action, SNES_START)
                ):
                    action, reason = idle_action(), "suppressed_start"
                if _pressed(action, 8):
                    a_special += 1
                    stop = "forbidden_action"
                    reason = "forbidden_action"

            reasons[reason] += 1
            if tracker is not None and live and stop is None:
                tracker.note_reason(reason)

            freeze_armed = (
                live
                and state.mode is GameMode.PLAYING
                and state.player_x > 0
                and not state.living_enemies
                and metrics.credits_start_frame is None
            )
            abort = recorder.observe(
                freeze_armed=freeze_armed,
                live=live,
                state=state,
                frame=frame,
                reason=reason,
                damage=hp.damage,
            )
            if abort is not None and stop is None:
                dump_abort_snapshot(env, obs, state, frame)
                stop = abort.kind
                failure = {
                    "last_progress": abort.last_progress,
                    "first_cycle": abort.first_cycle,
                    "window_summary": abort.window_summary,
                    "reason": abort.reason,
                }

            if on_frame is not None:
                on_frame(
                    TrialFrame(
                        frame=frame,
                        state=state,
                        action=action,
                        reason=reason,
                        env=env,
                        obs=obs,
                        live=live,
                    )
                )
            hooked = consume_new_writes(frame)
            if hooked and stop is None:
                stop = hooked
                failure = _fail(
                    recorder,
                    violations[-1]["detail"] if violations else hooked,
                )

            if live and objective.kind == "credits" and frame and frame % 10_000 == 0:
                print(
                    f"frame {frame}  stage={state.stage} event={event:#04x} "
                    f"damage={hp.damage} lives={state.lives} "
                    f"p=({state.player_x},{state.player_y}) hp={state.health} "
                    f"reason={reason}",
                    flush=True,
                )

            if stop is not None:
                outcome = stop
                break
            if frame >= limits.max_frames:
                if saw_boss_fade and objective.kind in {"boss_fade", "stage_advance"}:
                    outcome = "boss_down"
                    failure = _fail(recorder, "boss faded without stage advance")
                else:
                    outcome = "timeout"
                    failure = _fail(recorder, f"timeout at {limits.max_frames}")
                break
            stepped = env.step(action)
            if isinstance(stepped, tuple) and stepped:
                obs = stepped[0]
            prev_state = state
        else:
            outcome = "boss_down" if saw_boss_fade else "timeout"
            failure = _fail(recorder, f"timeout at {limits.max_frames}")
    except Exception as exc:  # noqa: BLE001 — classified as error
        outcome = "error"
        failure = _fail(recorder, str(exc))
    finally:
        if owned:
            env.close()

    if outcome in _SUCCESS:
        failure = None
    success = outcome in _SUCCESS and (
        (objective.kind == "stage_advance" and outcome == "stage_advance")
        or (objective.kind == "boss_fade" and outcome in {"boss_down", "cleared"})
        or (objective.kind == "credits" and outcome == "credits")
        or (objective.kind == "wave_chain" and outcome == "cleared")
    )
    ram_writes = [
        {k: v for k, v in row.items() if k != "_seen"} for row in env.writes
    ]
    stage_writes = sum(1 for w in ram_writes if w["key"] == "stage")
    lives_writes = sum(1 for w in ram_writes if w["key"] in _LIVES_KEYS)
    integrity = {
        "emergency_hp_zero": emergency == 0,
        "iframe_guard_zero": iframe == 0,
        "life_losses_zero": life_losses == 0,
        "state_loads_zero": env.loads == 0,
        "stage_writes_zero": stage_writes == 0,
        "lives_writes_zero": lives_writes == 0,
    }
    end_lives = final.lives
    lives_s = (
        f"{start_lives}->{end_lives}"
        if start_lives is not None
        else f"{end_lives}->{end_lives}"
    )
    top = sorted(reasons.items(), key=lambda kv: -kv[1])[:16]
    play_frames = (
        (final.frame - play0) if live else final.frame
    )
    if outcome == "stall" or outcome == "freeze":
        success = False
    return TrialResult(
        outcome=outcome,
        success=success,
        frames=play_frames,
        total_frames=final.frame,
        start_hp=start_hp,
        end_hp=final.health,
        min_hp=hp.min_hp,
        start_lives=start_lives,
        end_lives=end_lives,
        life_losses=life_losses,
        lives=lives_s,
        damage_taken=hp.damage,
        wave_damage=sum(h["hit"] for h in hits if not h["boss"]),
        boss_damage=sum(h["hit"] for h in hits if h["boss"]),
        max_hit=hp.max_hit,
        pizza_heals=pizza_heals,
        pizza_heal_count=len(pizza_heals),
        entry_stage=entry_stage,
        entry_event=entry_event,
        entry_character=entry_character,
        entry_difficulty=entry_difficulty,
        end_stage=final.stage,
        end_event=int(final.extras.get("event", -1)),
        end_character=int(final.extras.get("char_id", -1)),
        emergency_hp_writes=emergency,
        iframe_writes=iframe,
        ram_writes=ram_writes,
        state_loads_after_launch=env.loads,
        a_special_uses=a_special,
        hits=hits,
        action_reasons=dict(reasons),
        top_reasons=top,
        contract_violations=violations,
        assist=_assist_label(len(pizza_heals), contract, emergency, iframe),
        heal_mode=_heal_mode(contract),
        integrity=integrity,
        failure=failure,
        boss_entry_hp=boss_entry_hp,
        post_boot_start_presses=0,
        credits_start_frame=metrics.credits_start_frame,
        credits_complete_frame=metrics.credits_complete_frame,
        hard_credits_event_seen=metrics.hard_credits_event_seen,
        hard_confirmed=hard_confirmed
        or int(final.extras.get("difficulty", -1)) == HARD_VALUE,
        stage_splits=splits,
        damage_by_stage=damage_by_stage,
    )
