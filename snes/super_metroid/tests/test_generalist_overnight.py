"""Overnight keep/discard rules and process management (no emulator)."""

from __future__ import annotations

import json
import signal
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from super_metroid.generalist.evaluate import act, heuristic_action
from super_metroid.generalist.obs import GeneralistObs
from super_metroid.generalist import farm, overnight
from super_metroid.generalist.overnight import decide_keep, parse_until, should_promote_join
from retro_harness.platformer.neuro.net import GRID_SIZE


class _ClockDateTime:
    current = datetime(2026, 8, 26, 20, 0)

    @classmethod
    def now(cls) -> datetime:
        return cls.current


def _value_after(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def _training_result(
    out_dir: Path,
    cmd: list[str],
    *,
    ppo_join: float,
    heuristic_join: float,
    checkpoint_bytes: bytes,
) -> None:
    tag = _value_after(cmd, "--tag")
    seed = int(_value_after(cmd, "--seed"))
    (out_dir / f"ppo_{tag}_s{seed}.zip").write_bytes(checkpoint_bytes)
    (out_dir / f"train_{tag}_s{seed}.json").write_text(
        json.dumps(
            {
                "heuristic": {"join_rate": heuristic_join, "stall_rate": 0.5},
                "ppo": {"join_rate": ppo_join, "stall_rate": 0.2},
            }
        ),
        encoding="utf-8",
    )


def test_parse_until_rolls_to_tomorrow_when_past() -> None:
    now = datetime(2026, 8, 26, 20, 30)
    target = parse_until("08:00", now=now)
    assert target == datetime(2026, 8, 27, 8, 0)


def test_decide_keep_discards_clear_loss_promotes_clear_win() -> None:
    assert decide_keep(
        ppo_join=0.1, heuristic_join=0.5, ppo_stall=0.8, heuristic_stall=0.4
    ) == "discard"
    assert decide_keep(
        ppo_join=0.75, heuristic_join=0.5, ppo_stall=0.2, heuristic_stall=0.4
    ) == "promote"
    assert decide_keep(
        ppo_join=0.5, heuristic_join=0.5, ppo_stall=0.4, heuristic_stall=0.4
    ) == "keep"
    assert decide_keep(
        ppo_join=0.0, heuristic_join=0.0, ppo_stall=1.0, heuristic_stall=1.0
    ) == "keep"
    assert should_promote_join(1.0, 1.0) is True
    assert decide_keep(
        ppo_join=1.0, heuristic_join=1.0, ppo_stall=0.0, heuristic_stall=0.0
    ) == "promote"
    assert decide_keep(
        ppo_join=0.99, heuristic_join=0.99, ppo_stall=0.0, heuristic_stall=0.0
    ) == "keep"


def test_widening_rebaselines_and_uses_a_phase_local_best(
    tmp_path: Path, monkeypatch: Any
) -> None:
    until = datetime(2026, 8, 26, 21, 0)
    _ClockDateTime.current = datetime(2026, 8, 26, 20, 0)
    commands: list[list[str]] = []
    waits = 0

    class CompletedProcess:
        def wait(self, timeout: float | None = None) -> int:
            del timeout
            nonlocal waits
            waits += 1
            if waits == 4:
                _ClockDateTime.current = until
            return 0

    def popen(cmd: list[str], **_kwargs: Any) -> CompletedProcess:
        commands.append(cmd)
        tag = _value_after(cmd, "--tag")
        seed = int(_value_after(cmd, "--seed"))
        if tag == "same_room":
            _training_result(
                tmp_path,
                cmd,
                ppo_join=0.8 if seed == 0 else 0.6,
                heuristic_join=0.5,
                checkpoint_bytes=f"same-room-{seed}".encode(),
            )
        else:
            _training_result(
                tmp_path,
                cmd,
                ppo_join=0.4 if seed == 0 else 0.1,
                heuristic_join=0.3,
                checkpoint_bytes=f"crateria-{seed}".encode(),
            )
        return CompletedProcess()

    monkeypatch.setattr(overnight, "datetime", _ClockDateTime)
    monkeypatch.setattr(farm.subprocess, "Popen", popen)

    result = overnight.run_overnight(
        until=until,
        out_dir=tmp_path,
        n_jobs=2,
        n_envs=1,
        cycle_timesteps=10,
        eval_episodes=1,
        python="python",
    )

    mix_commands = commands[2:]
    assert len(mix_commands) == 2
    assert all("--same-room" not in cmd for cmd in mix_commands)
    assert all("--skip-baselines" not in cmd for cmd in mix_commands)
    assert _value_after(mix_commands[0], "--checkpoint") == str(
        tmp_path / "ppo_same_room_s0.zip"
    )
    assert "--checkpoint" not in mix_commands[1]
    assert result["best"]["tag"] == "crateria"
    assert result["best"]["join_rate"] == 0.4
    assert (tmp_path / "best.zip").read_bytes() == b"crateria-0"
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert [row["cycle"] for row in status["history"]][:2] == [1, 2]


def test_ceiling_join_widens_to_mix(tmp_path: Path, monkeypatch: Any) -> None:
    until = datetime(2026, 8, 26, 21, 0)
    _ClockDateTime.current = datetime(2026, 8, 26, 20, 0)
    commands: list[list[str]] = []
    waits = 0

    class CompletedProcess:
        def wait(self, timeout: float | None = None) -> int:
            del timeout
            nonlocal waits
            waits += 1
            if waits == 4:
                _ClockDateTime.current = until
            return 0

    def popen(cmd: list[str], **_kwargs: Any) -> CompletedProcess:
        commands.append(cmd)
        tag = _value_after(cmd, "--tag")
        seed = int(_value_after(cmd, "--seed"))
        if tag == "same_room":
            _training_result(
                tmp_path,
                cmd,
                ppo_join=1.0,
                heuristic_join=1.0,
                checkpoint_bytes=f"same-room-{seed}".encode(),
            )
        else:
            _training_result(
                tmp_path,
                cmd,
                ppo_join=0.2,
                heuristic_join=0.1,
                checkpoint_bytes=f"crateria-{seed}".encode(),
            )
        return CompletedProcess()

    monkeypatch.setattr(overnight, "datetime", _ClockDateTime)
    monkeypatch.setattr(farm.subprocess, "Popen", popen)

    overnight.run_overnight(
        until=until,
        out_dir=tmp_path,
        n_jobs=2,
        n_envs=1,
        cycle_timesteps=10,
        eval_episodes=1,
        python="python",
    )

    mix_commands = commands[2:]
    assert len(mix_commands) == 2
    assert all("--same-room" not in cmd for cmd in mix_commands)
    # Equal 1.0 Join: later worker wins the phase-local best.
    assert _value_after(mix_commands[0], "--checkpoint") == str(
        tmp_path / "ppo_same_room_s1.zip"
    )


def test_resume_status_promotes_ceiling_to_mix(tmp_path: Path, monkeypatch: Any) -> None:
    until = datetime(2026, 8, 26, 21, 0)
    _ClockDateTime.current = datetime(2026, 8, 26, 20, 0)
    same_zip = tmp_path / "ppo_same_room_s1.zip"
    same_zip.write_bytes(b"same-room-1")
    (tmp_path / "best.zip").write_bytes(b"same-room-1")
    (tmp_path / "status.json").write_text(
        json.dumps(
            {
                "cycle": 3,
                "same_room": True,
                "heuristic_join": 1.0,
                "heuristic_stall": 0.0,
                "best": {
                    "join_rate": 1.0,
                    "stall_rate": 0.0,
                    "checkpoint": str(same_zip),
                    "seed": 1,
                    "cycle": 3,
                    "tag": "same_room",
                },
                "workers": [
                    {"seed": 0, "checkpoint": str(tmp_path / "ppo_same_room_s0.zip")},
                    {"seed": 1, "checkpoint": str(same_zip)},
                ],
                "history": [{"cycle": 1}, {"cycle": 2}],
            }
        ),
        encoding="utf-8",
    )
    commands: list[list[str]] = []
    waits = 0

    class CompletedProcess:
        def wait(self, timeout: float | None = None) -> int:
            del timeout
            nonlocal waits
            waits += 1
            if waits == 2:
                _ClockDateTime.current = until
            return 0

    def popen(cmd: list[str], **_kwargs: Any) -> CompletedProcess:
        commands.append(cmd)
        _training_result(
            tmp_path,
            cmd,
            ppo_join=0.2,
            heuristic_join=0.1,
            checkpoint_bytes=b"crateria",
        )
        return CompletedProcess()

    monkeypatch.setattr(overnight, "datetime", _ClockDateTime)
    monkeypatch.setattr(farm.subprocess, "Popen", popen)

    overnight.run_overnight(
        until=until,
        out_dir=tmp_path,
        n_jobs=2,
        n_envs=1,
        cycle_timesteps=10,
        eval_episodes=1,
        python="python",
    )

    assert commands
    assert all("--same-room" not in cmd for cmd in commands)
    assert "--bc" not in commands[0]
    assert _value_after(commands[0], "--checkpoint") == str(same_zip)
    assert "--checkpoint" not in commands[1]


def test_deadline_terminates_worker_groups_and_ignores_stale_reports(
    tmp_path: Path, monkeypatch: Any
) -> None:
    until = datetime(2026, 8, 26, 20, 5)
    _ClockDateTime.current = datetime(2026, 8, 26, 20, 0)
    launches: list[dict[str, Any]] = []
    processes: dict[int, TimedOutProcess] = {}

    class TimedOutProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid
            self.returncode: int | None = None
            self.terminated = False

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            assert timeout is not None
            if self.returncode is not None:
                return self.returncode
            if not self.terminated:
                _ClockDateTime.current = until
            raise subprocess.TimeoutExpired("training", timeout)

    def popen(cmd: list[str], **kwargs: Any) -> TimedOutProcess:
        pid = 100 + len(processes)
        process = TimedOutProcess(pid)
        processes[pid] = process
        launches.append({"cmd": cmd, **kwargs})
        return process

    signals: list[tuple[int, signal.Signals]] = []

    def killpg(pid: int, sig: signal.Signals) -> None:
        signals.append((pid, sig))
        process = processes[pid]
        if sig == signal.SIGTERM:
            process.terminated = True
            if pid == 100:
                process.returncode = -signal.SIGTERM
        else:
            process.returncode = -signal.SIGKILL

    stale_cmd = overnight.train_command(
        python="python",
        out_dir=tmp_path,
        seed=0,
        timesteps=10,
        n_envs=1,
        same_room=True,
        eval_episodes=1,
        bc=True,
        checkpoint=None,
        skip_baselines=False,
        tag="same_room",
    )
    _training_result(
        tmp_path,
        stale_cmd,
        ppo_join=1.0,
        heuristic_join=0.1,
        checkpoint_bytes=b"stale",
    )
    monkeypatch.setattr(overnight, "datetime", _ClockDateTime)
    monkeypatch.setattr(farm.subprocess, "Popen", popen)
    monkeypatch.setattr(farm.os, "killpg", killpg)

    result = overnight.run_overnight(
        until=until,
        out_dir=tmp_path,
        n_jobs=2,
        n_envs=1,
        cycle_timesteps=10,
        eval_episodes=1,
        python="python",
    )

    assert all(launch["start_new_session"] is True for launch in launches)
    assert signals[:2] == [(100, signal.SIGTERM), (101, signal.SIGTERM)]
    assert (101, signal.SIGKILL) in signals
    assert result["deadline_reached"] is True
    assert result["best"] is None
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["best"] is None
    assert status["history"] == []


def _teacher_obs(
    *,
    dx: float = 0.0,
    dy: float = 0.0,
    ordinary: float = 1.0,
    door_transition: float = 0.0,
    same_room: bool = False,
    previous_action: int = 0,
    blocked_col: int | None = None,
) -> GeneralistObs:
    parts = GeneralistObs.blank()
    parts.samus[GeneralistObs.SAMUS_ORDINARY] = ordinary
    parts.samus[GeneralistObs.SAMUS_DOOR_TRANSITION] = door_transition
    parts.goal[GeneralistObs.GOAL_DX] = dx
    parts.goal[GeneralistObs.GOAL_DY] = dy
    parts.goal[GeneralistObs.GOAL_PREVIOUS_ACTION] = previous_action / 26.0
    parts.goal[GeneralistObs.GOAL_SAME_ROOM] = 1.0 if same_room else 0.0
    if blocked_col is not None:
        parts.grid[GRID_SIZE // 2, blocked_col] = 1.0
    return parts


def test_heuristic_jumps_when_occupancy_blocks_walk() -> None:
    obs = _teacher_obs(dx=-0.4, same_room=True, blocked_col=GRID_SIZE // 2 - 1)
    assert heuristic_action(obs) == 10
    clear = _teacher_obs(dx=-0.4, same_room=True)
    assert heuristic_action(clear) == 0
    assert act(None, None, clear) == 0


def test_heuristic_shoots_cross_room_door_and_idles_during_transition() -> None:
    obs = _teacher_obs(dx=0.4, same_room=False)
    assert heuristic_action(obs) == 3  # RIGHT + X
    idle = _teacher_obs(dx=0.4, ordinary=0.0, door_transition=1.0)
    assert heuristic_action(idle) == 25


def test_heuristic_pulses_jump_to_create_new_button_edges() -> None:
    obs = _teacher_obs(dx=0.4, same_room=True, blocked_col=GRID_SIZE // 2 + 1)
    assert heuristic_action(obs) == 11  # RIGHT + A
    released = _teacher_obs(
        dx=0.4, same_room=True, blocked_col=GRID_SIZE // 2 + 1, previous_action=11
    )
    assert heuristic_action(released) == 1  # release A for one decision
