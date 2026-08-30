"""ROM-free trial loop: finish / time / damage / result truth."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from retro_harness.actions import buttons, idle_action
from retro_harness.bot_runner import NodeStatus, TickResult
from retro_harness.controls import SNES_START
from retro_harness.input_script import FrameAction
from tmnt_iv.assist import assist_integrity
from tmnt_iv.observe import living_hp
from tmnt_iv.ram import (
    ADDR_EVENT,
    ADDR_LIVES,
    ADDR_MENU,
    ADDR_STAGE,
    PLAYER_BASE,
    MenuId,
    OFF_CHAR,
    OFF_HP,
    OFF_IFRAMES,
    OFF_X,
    OFF_Y,
    parse_game_state,
    write_u16le,
)
from tmnt_iv.run.clean_suite import STAGE1_CLEAN, STAGE2_CLEAN, _result_to_clean_report, run_suite
from tmnt_iv.run.freeze import FREEZE_ABORT_FRAMES
from tmnt_iv.run.metrics import HARD_VALUE, RunMetrics
from tmnt_iv.run.trial import (
    ProgressSpec,
    TrialContract,
    TrialEntry,
    TrialLimits,
    TrialObjective,
    catalog_state,
    run_trial,
)
from tmnt_iv.tests._state import ram, write_enemy

ADDR_DIFFICULTY = 0x1FEE
A = 8


def play_ram(
    *,
    hp: int = 80,
    lives: int = 2,
    x: int = 80,
    y: int = 160,
    stage: int = 0,
    event: int = 0x0A,
    char: int = 8,
    difficulty: int = HARD_VALUE,
    menu: int | None = None,
    title: bool = False,
    pizza: bool = False,
    boss: bool = False,
    enemy: bool = False,
) -> np.ndarray:
    """One WRAM snapshot for ScriptedEnv."""
    buf = ram()
    buf[ADDR_MENU] = MenuId.TITLE if title else (MenuId.PLAYING if menu is None else menu)
    buf[ADDR_EVENT] = event
    buf[ADDR_STAGE] = stage
    buf[ADDR_LIVES] = lives
    buf[PLAYER_BASE + OFF_HP] = hp
    buf[PLAYER_BASE + OFF_CHAR] = char
    buf[ADDR_DIFFICULTY] = difficulty
    write_u16le(buf, PLAYER_BASE + OFF_X, 0 if title else x)
    write_u16le(buf, PLAYER_BASE + OFF_Y, y)
    if pizza:
        write_enemy(buf, 0, x=x, y=y, health=0, char_id=0x30)
    if boss:
        write_enemy(buf, 1, x=180, y=160, health=96, char_id=0x44)
    if enemy:
        write_enemy(buf, 2, x=140, y=160, health=16, char_id=0x60)
    return buf


class ScriptedEnv:
    """Minimal Gym-shaped env: scripted WRAM, recorded writes/loads/actions."""

    def __init__(self, frames: list[np.ndarray]):
        self._frames = frames
        self._i = 0
        self.actions: list[list[int]] = []
        self.writes: list[tuple[str, int]] = []
        self.loads: list[bytes] = []
        self.closed = False
        self.obs = np.zeros((8, 8, 3), dtype=np.uint8)
        self.em = _ScriptedEm(self)

    def get_ram(self) -> np.ndarray:
        return self._frames[min(self._i, len(self._frames) - 1)]

    def step(self, action) -> tuple:
        self.actions.append([int(v) for v in action])
        if self._i + 1 < len(self._frames):
            self._i += 1
        return self.obs, 0.0, False, False, {}

    def set_value(self, name: str, value: int) -> None:
        self.writes.append((name, int(value)))
        buf = self.get_ram()
        if name == "player_hp":
            buf[PLAYER_BASE + OFF_HP] = int(value) & 0xFF
        elif name == "player_iframes":
            buf[PLAYER_BASE + OFF_IFRAMES] = int(value) & 0xFF
        elif name == "stage":
            buf[ADDR_STAGE] = int(value) & 0xFF
        elif name in {"lives", "player_lives"}:
            buf[ADDR_LIVES] = int(value) & 0xFF

    def close(self) -> None:
        self.closed = True


class _ScriptedEm:
    def __init__(self, owner: ScriptedEnv) -> None:
        self._owner = owner

    def set_state(self, state: bytes) -> None:
        self._owner.loads.append(state)


class _FixedPolicy:
    def __init__(self, action, reason: str = "script") -> None:
        self._action = action
        self._reason = reason

    def reset(self) -> None:
        return None

    def tick(self, state) -> TickResult:
        fa = FrameAction(action=self._action, reason=self._reason)
        return TickResult(status=NodeStatus.RUNNING, action=fa, reason=self._reason)


CLEAN = TrialContract(
    name="clean",
    emergency_hp=False,
    iframe_hold=False,
    fail_on_life_loss=True,
    allow_continue=False,
    allowed_write_keys=frozenset(),
)


def _live(state) -> bool:
    return (
        state.mode.name == "PLAYING"
        and living_hp(state.health)
        and 0 < state.player_x < 400
        and int(state.extras.get("event", 0)) >= 0x0A
    )


def _run(buffers, *, objective=None, entry=None, limits=None, policy=None, on_frame=None, contract=None):
    env = ScriptedEnv(buffers)
    result = run_trial(
        entry
        or TrialEntry(kind="state", state_name="Stage1", is_live=_live),
        objective
        or TrialObjective(kind="stage_advance", stop_stage_gt=0, strict_advance=True),
        contract or CLEAN,
        limits or TrialLimits(max_frames=20, progress=ProgressSpec(stall_frames=10_000)),
        env=env,
        policy=policy,
        on_frame=on_frame,
    )
    return result, env


def test_waiting_entry_captures_live_start_hp_not_eighty() -> None:
    title = play_ram(title=True, hp=80, lives=2, x=0)
    live = play_ram(hp=64, lives=3, stage=0)
    nxt = play_ram(hp=64, lives=3, stage=1)
    result, env = _run(
        [title, title, live, nxt],
        entry=TrialEntry(kind="power_on", state_name="NONE", is_live=_live),
    )
    assert env.closed is False
    assert result.start_hp == 64
    assert result.start_hp != 80
    assert result.start_lives == 3
    assert result.lives == "3->3"
    assert result.life_losses == 0
    assert result.success is True
    assert result.outcome == "stage_advance"


def test_lives_fields_are_numeric_and_use_live_start() -> None:
    result, _env = _run(
        [play_ram(hp=70, lives=4, stage=0), play_ram(hp=70, lives=4, stage=1)]
    )
    payload = result.to_dict()
    assert result.start_lives == 4
    assert result.end_lives == 4
    assert result.life_losses == 0
    assert isinstance(result.life_losses, int)
    assert result.lives == "4->4"
    assert payload["life_losses"] == 0
    assert payload["start_lives"] == 4


def test_pizza_heal_requires_pickup_transition() -> None:
    pizza, _env = _run(
        [
            play_ram(hp=40, pizza=True, stage=0),
            play_ram(hp=80, pizza=False, stage=0),
            play_ram(hp=80, stage=1),
        ]
    )
    assert pizza.pizza_heal_count == 1
    assert pizza.assist == "pizza_only"
    assert pizza.pizza_heals[0]["from_hp"] == 40
    assert pizza.pizza_heals[0]["to_hp"] == 80
    assert pizza.pizza_heals[0]["pickup_seen"] is True

    unlabeled, _env = _run(
        [
            play_ram(hp=40, pizza=False, stage=0),
            play_ram(hp=80, pizza=False, stage=0),
            play_ram(hp=80, stage=1),
        ]
    )
    assert unlabeled.pizza_heal_count == 0
    assert unlabeled.assist != "pizza_only"
    assert unlabeled.success is False
    assert unlabeled.outcome == "contract_violation"


def test_boss_fade_hp_restore_is_not_unlabeled_pizza() -> None:
    result, _env = _run(
        [
            play_ram(hp=37, event=0x0A, boss=True, stage=1),
            play_ram(hp=37, event=0x0B, boss=False, stage=1),
            play_ram(hp=80, event=0x0B, boss=False, stage=1),
            play_ram(hp=80, event=0x19, stage=2),
        ],
        objective=TrialObjective(
            kind="stage_advance", stop_stage_gt=1, strict_advance=False
        ),
    )
    assert result.outcome == "stage_advance"
    assert result.success is True
    assert result.pizza_heal_count == 0
    assert result.contract_violations == []
    assert result.failure is None


def test_clean_row_includes_failure_and_violations() -> None:
    unlabeled, _env = _run(
        [
            play_ram(hp=40, pizza=False, stage=0),
            play_ram(hp=80, pizza=False, stage=0),
            play_ram(hp=80, stage=1),
        ]
    )
    assert unlabeled.outcome == "contract_violation"
    row = _result_to_clean_report(STAGE2_CLEAN, unlabeled, label="Boss2")
    assert "failure" in row
    assert row["failure"] is not None
    assert row["failure"]["reason"] == "unlabeled_hp_gain"
    assert row["contract_violations"]
    assert row["contract_violations"][0]["kind"] == "unlabeled_hp_gain"


def test_clean_hp_write_is_contract_violation() -> None:
    def poke(ctx) -> None:
        if ctx.live:
            ctx.env.set_value("player_hp", 80)

    result, _env = _run(
        [play_ram(hp=40), play_ram(hp=40), play_ram(hp=40, stage=1)],
        on_frame=poke,
    )
    assert result.success is False
    assert result.outcome == "contract_violation"
    assert any(w["key"] == "player_hp" for w in result.ram_writes)
    assert result.ram_writes[0]["frame"] >= 0


def test_clean_iframe_write_is_contract_violation() -> None:
    def poke(ctx) -> None:
        if ctx.live:
            ctx.env.set_value("player_iframes", 1)

    result, _env = _run(
        [play_ram(), play_ram(), play_ram(stage=1)],
        on_frame=poke,
    )
    assert result.success is False
    assert result.outcome == "contract_violation"
    assert any(w["key"] == "player_iframes" for w in result.ram_writes)


def test_a_button_is_forbidden_action() -> None:
    pressed = list(idle_action())
    pressed[A] = 1
    result, _env = _run(
        [play_ram(), play_ram(), play_ram(stage=1)],
        policy=_FixedPolicy(pressed, "special"),
    )
    assert result.success is False
    assert result.outcome == "forbidden_action"
    assert result.a_special_uses >= 1


def test_state_load_after_live_is_contract_violation() -> None:
    def reload(ctx) -> None:
        if ctx.live:
            ctx.env.em.set_state(b"pin")

    result, env = _run(
        [play_ram(), play_ram(), play_ram(stage=1)],
        on_frame=reload,
    )
    assert result.success is False
    assert result.outcome == "contract_violation"
    assert result.state_loads_after_launch >= 1
    assert env.loads


def test_ko_does_not_press_continue_start() -> None:
    result, env = _run(
        [
            play_ram(hp=16, lives=2),
            play_ram(hp=0, lives=2),
            play_ram(hp=0, lives=2),
        ]
    )
    assert result.success is False
    assert result.outcome in {"ko", "player_dead"}
    after = env.actions
    assert all(not action[SNES_START] for action in after)


def test_title_and_game_over_are_not_timeout() -> None:
    title, _env = _run(
        [play_ram(), play_ram(title=True, x=0, hp=80)],
        limits=TrialLimits(max_frames=30, progress=ProgressSpec(stall_frames=10_000)),
    )
    assert title.outcome == "title"
    assert title.success is False
    assert title.outcome != "timeout"

    over, _env = _run(
        [play_ram(hp=8, lives=0), play_ram(hp=0, lives=0)],
        limits=TrialLimits(max_frames=30, progress=ProgressSpec(stall_frames=10_000)),
    )
    assert over.outcome in {"game_over", "player_dead"}
    assert over.success is False
    assert over.outcome != "timeout"


def test_lives_decrement_is_life_loss() -> None:
    result, _env = _run(
        [play_ram(hp=40, lives=2), play_ram(hp=40, lives=1)]
    )
    assert result.outcome == "life_loss"
    assert result.success is False
    assert result.life_losses >= 1
    assert result.start_lives == 2
    assert result.end_lives == 1
    assert result.lives == "2->1"


def test_stage_advance_with_living_hp_succeeds() -> None:
    result, _env = _run(
        [play_ram(hp=72, stage=0), play_ram(hp=72, stage=1)]
    )
    assert result.success is True
    assert result.outcome == "stage_advance"
    assert result.entry_stage == 0
    assert result.end_stage == 1


def test_boss_fade_is_boss_down() -> None:
    fade, _env = _run(
        [
            play_ram(hp=80, event=0x0A, boss=True),
            play_ram(hp=80, event=0x0B, boss=False),
        ],
        objective=TrialObjective(kind="boss_fade"),
    )
    assert fade.outcome == "boss_down"
    assert fade.success is True

    ignored, _env = _run(
        [
            play_ram(hp=80, event=0x0A, boss=True, stage=0),
            play_ram(hp=80, event=0x0B, boss=False, stage=0),
            play_ram(hp=80, event=0x0B, stage=0),
        ],
        objective=TrialObjective(kind="stage_advance", stop_stage_gt=0, strict_advance=True),
        limits=TrialLimits(max_frames=4, progress=ProgressSpec(stall_frames=10_000)),
    )
    assert ignored.outcome == "boss_down"
    assert ignored.success is False


def test_timeout_at_max_frames() -> None:
    result, _env = _run(
        [play_ram(stage=0) for _ in range(8)],
        limits=TrialLimits(max_frames=3, progress=ProgressSpec(stall_frames=10_000)),
    )
    assert result.outcome == "timeout"
    assert result.success is False
    assert result.frames <= 3 or result.total_frames <= 3


def test_progress_stall_with_living_enemies_fires() -> None:
    stuck = play_ram(hp=80, enemy=True, x=54)
    result, _env = _run(
        [stuck for _ in range(20)],
        limits=TrialLimits(
            max_frames=30,
            freeze_abort_frames=FREEZE_ABORT_FRAMES,
            progress=ProgressSpec(stall_frames=6),
        ),
        policy=_FixedPolicy(idle_action(), "combat_stall_escape"),
    )
    assert result.outcome in {"stall", "freeze"}
    assert result.success is False
    assert result.failure is not None
    assert "last_progress" in result.failure


def test_suite_keeps_both_rows_when_second_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_probe(spec, **kwargs):
        if kwargs.get(spec.extra_entry):
            raise RuntimeError("extra failed")
        name = kwargs.get("state_name") or spec.default_state
        return {
            "state": name,
            "outcome": "stage_advance",
            "success": True,
            "emergency_hp_writes": 0,
            "iframe_writes": 0,
        }

    monkeypatch.setattr("tmnt_iv.run.clean_suite.run_clean_probe", fake_probe)
    report = run_suite(STAGE1_CLEAN, max_frames=1)
    assert report["suite_size"] == len(STAGE1_CLEAN.suite_states) + 1
    assert len(report["results"]) == report["suite_size"]
    extra = report["results"][-1]
    assert extra["outcome"] == "error"
    assert extra["success"] is False
    assert extra.get("emergency_hp_writes", 0) == 0
    assert extra.get("iframe_writes", 0) == 0


def test_integrity_flags_false_when_writes_or_loads_happened() -> None:
    def poke(ctx) -> None:
        if ctx.live:
            ctx.env.set_value("player_hp", 80)
            ctx.env.em.set_state(b"x")

    result, _env = _run([play_ram(), play_ram()], on_frame=poke)
    flags = result.integrity
    assert flags["state_loads_zero"] is False
    assert flags.get("emergency_hp_zero") is False or result.ram_writes
    audited = assist_integrity(
        RunMetrics(health_guard_interventions=0, life_losses=0),
        state_loads=result.state_loads_after_launch,
        stage_writes=sum(1 for w in result.ram_writes if w["key"] == "stage"),
        lives_writes=sum(
            1 for w in result.ram_writes if w["key"] in {"lives", "player_lives"}
        ),
    )
    assert audited["state_loads_zero"] is False

    empty = assist_integrity(RunMetrics())
    assert empty["state_loads_zero"] is not True
    assert empty["stage_writes_zero"] is not True
    assert empty["lives_writes_zero"] is not True


def test_catalog_quarantines_last_life_and_refuses_wrong_character() -> None:
    live = parse_game_state(play_ram(hp=80, lives=0, char=2))
    row = catalog_state("Boss3_hp0", live)
    assert row["name"] == "Boss3_hp0"
    assert row["lives"] == 0
    assert row["clean_gate_eligible"] is False

    result, _env = _run(
        [play_ram(char=2, hp=80), play_ram(char=2, hp=80, stage=1)],
        entry=TrialEntry(
            kind="state",
            state_name="Stage1",
            is_live=_live,
            expected_character=8,
            expected_difficulty=HARD_VALUE,
        ),
    )
    assert result.success is False
    assert result.outcome in {"error", "contract_violation"}
