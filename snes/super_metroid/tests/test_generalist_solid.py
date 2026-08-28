"""Editor collision occupancy + door potential (no emulator)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import super_metroid.generalist.env as env_module
import super_metroid.generalist.train as train_module
from super_metroid.generalist.goals import Goal
from super_metroid.generalist.obs import OBS_DIM
from super_metroid.generalist.solid import (
    CLIP_DOOR,
    CollisionDependencyError,
    load_room_solid,
    potential_xy,
    require_row_solids,
    room_solid_from_collision,
)


def test_air_and_door_are_walkable_solid_is_not() -> None:
    grid = [
        [8, 8, 8, 8],
        [8, 0, 9, 8],
        [8, 8, 8, 8],
    ]
    solid = room_solid_from_collision(0x91F8, grid)
    assert solid.is_solid(16 + 8, 16 + 8) is False
    assert solid.is_solid(32 + 8, 16 + 8) is False
    assert solid.is_solid(8, 8) is True
    assert solid.clip_at(32 + 8, 16 + 8) == CLIP_DOOR
    assert solid.nearest_door(16, 16) == (32 + 8, 16 + 8)
    assert solid.is_solid(-4, 8) is True


def test_potential_same_room_is_goal_else_door() -> None:
    grid = [
        [8, 8, 8, 8],
        [8, 0, 9, 8],
        [8, 8, 8, 8],
    ]
    solid = room_solid_from_collision(0x91F8, grid)
    goal = Goal("next", 0x91F8, 400, 400)
    here = SimpleNamespace(room_id=0x91F8, samus_x=24, samus_y=24)
    assert potential_xy(here, goal, solid) == (400, 400)
    other = SimpleNamespace(room_id=0x91F8, samus_x=24, samus_y=24)
    away = Goal("next", 0x92FD, 400, 400)
    assert potential_xy(other, away, solid) == (32 + 8, 16 + 8)


def test_cross_room_potential_rejects_missing_current_room_collision() -> None:
    state = SimpleNamespace(room_id=0x91F8, samus_x=24, samus_y=24)
    goal = Goal("next", 0x92FD, 400, 400)

    with pytest.raises(CollisionDependencyError, match="0x91F8"):
        potential_xy(state, goal, None)


def test_cross_room_potential_rejects_doorless_current_room_collision() -> None:
    state = SimpleNamespace(room_id=0x91F8, samus_x=24, samus_y=24)
    goal = Goal("next", 0x92FD, 400, 400)
    solid = room_solid_from_collision(0x91F8, [[0, 0], [0, 0]])

    with pytest.raises(CollisionDependencyError, match=r"0x91F8.*clip-9"):
        potential_xy(state, goal, solid)


def test_load_landing_site_if_editor_present() -> None:
    solid = load_room_solid(0x91F8)
    if solid is None:
        return
    assert solid.width == 144
    assert solid.height == 80
    # Ship pin ~ (1153, 1137) sits on the landing floor, not in a door.
    assert solid.is_solid(1153, 1137 + 16) is True
    # Parlor door on Landing is a clip-9 column around x=121.
    assert solid.clip_at(121, 1179) in {0, CLIP_DOOR} or not solid.is_solid(121, 1179)
    assert solid.nearest_door(121, 1179) is not None


def test_curriculum_collision_preflight_names_every_missing_room(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SUPER_METROID_EDITOR_NAV", str(tmp_path))
    rows = [SimpleNamespace(room_id=0x91F8, goal_room_id=0x92FD)]

    with pytest.raises(CollisionDependencyError) as caught:
        require_row_solids(rows)

    message = str(caught.value)
    assert "SUPER_METROID_EDITOR_NAV" in message
    assert "0x91F8" in message
    assert "0x92FD" in message


def test_curriculum_collision_preflight_rejects_ragged_grid(tmp_path) -> None:
    path = tmp_path / "room_91F8.json"
    path.write_text(
        json.dumps({"collision": [[0, CLIP_DOOR], [0]]}),
        encoding="utf-8",
    )
    rows = [SimpleNamespace(room_id=0x91F8, goal_room_id=0x91F8)]

    with pytest.raises(CollisionDependencyError, match=r"invalid 0x91F8"):
        require_row_solids(rows, root=tmp_path)


def test_cross_room_collision_preflight_rejects_doorless_start(tmp_path) -> None:
    for room_id in (0x91F8, 0x92FD):
        (tmp_path / f"room_{room_id:04X}.json").write_text(
            json.dumps({"collision": [[0, 0], [0, 0]]}),
            encoding="utf-8",
        )
    rows = [SimpleNamespace(room_id=0x91F8, goal_room_id=0x92FD)]

    with pytest.raises(CollisionDependencyError, match=r"0x91F8.*clip-9"):
        require_row_solids(rows, root=tmp_path)


def test_same_room_collision_preflight_allows_doorless_room(tmp_path) -> None:
    (tmp_path / "room_91F8.json").write_text(
        json.dumps({"collision": [[0, 0], [0, 0]]}),
        encoding="utf-8",
    )
    rows = [SimpleNamespace(room_id=0x91F8, goal_room_id=0x91F8)]

    solids = require_row_solids(rows, root=tmp_path)

    assert sorted(solids) == [0x91F8]


def test_training_aborts_for_missing_collision_before_emulator_creation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SUPER_METROID_EDITOR_NAV", str(tmp_path))
    goal = Goal("next", 0x92FD, 120, 180)
    row = SimpleNamespace(
        session_id="kpdr25/crateria/ship",
        room_id=0x91F8,
        goal_room_id=goal.room_id,
        state_path=str(tmp_path / "unused.state"),
        goal=lambda: goal,
    )
    monkeypatch.setattr(train_module, "resolve_rows", lambda **_kwargs: [row])
    monkeypatch.setattr(env_module, "assert_practice_rom", lambda *_args: "test")
    emulator_started = False

    def reject_emulator(*_args, **_kwargs):
        nonlocal emulator_started
        emulator_started = True
        raise AssertionError("emulator must not start before collision preflight")

    monkeypatch.setattr(env_module, "make_env", reject_emulator)

    with pytest.raises(CollisionDependencyError, match="0x91F8"):
        train_module.train(out_dir=tmp_path, timesteps=0, ppo=False)

    assert emulator_started is False


def test_env_aborts_for_missing_collision_before_practice_rom_check(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SUPER_METROID_EDITOR_NAV", str(tmp_path))
    goal = Goal("next", 0x92FD, 120, 180)
    row = SimpleNamespace(
        room_id=0x91F8,
        goal_room_id=goal.room_id,
        goal=lambda: goal,
    )
    practice_checked = False

    def record_practice_check(*_args):
        nonlocal practice_checked
        practice_checked = True
        return "test"

    monkeypatch.setattr(env_module, "assert_practice_rom", record_practice_check)

    with pytest.raises(CollisionDependencyError, match="0x92FD"):
        env_module.GeneralistEnv(rows=[row], area=None)

    assert practice_checked is False


def test_step_missing_active_room_truncates_as_unmapped_room(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = env_module.GeneralistEnv.__new__(env_module.GeneralistEnv)
    env.frame_skip = 1
    env.max_episode_frames = 1800
    env.stall_frames = 240
    env._frame = 0
    env._stall = 0
    env._last_opi = (0x91F8, 10, 10, 1)
    env._prev_distance = 40.0
    env._prev_action = 0
    env._goal = Goal("next", 0x91F8, 120, 180)
    env._row = SimpleNamespace(session_id="kpdr25/crateria/ship")
    env._assist = SimpleNamespace(
        apply=lambda *_args, **_kwargs: None,
        telemetry=SimpleNamespace(energy=SimpleNamespace(writes=0)),
    )
    env._last_obs = np.zeros(OBS_DIM, dtype=np.float32)
    env._last_rgb = None
    env._env = SimpleNamespace(
        step=lambda _vec: (None, 0.0, False, False, {}),
        data=None,
        get_ram=lambda: None,
    )
    state = SimpleNamespace(
        room_id=0x0001,
        samus_x=10,
        samus_y=10,
        pose=1,
        health=99,
        game_state=8,
    )
    monkeypatch.setattr(env_module, "parse_env_state", lambda *_args, **_kwargs: state)

    def boom(_state: object) -> None:
        raise CollisionDependencyError("editor collision missing for active room 0x0001")

    env._solid_for = boom  # type: ignore[method-assign]

    obs, reward, terminated, truncated, info = env.step(0)

    assert truncated is True
    assert terminated is False
    assert info["reason"] == "unmapped_room"
    assert info["stall"] == 0
    assert obs.shape == (OBS_DIM,)
    assert reward == pytest.approx(0.0)


def test_atomic_save_replaces_destination(tmp_path: Path) -> None:
    class Model:
        def save(self, path: str) -> None:
            Path(path).write_bytes(b"new-weights")

    dest = tmp_path / "ppo_same_room_s0.zip"
    dest.write_bytes(b"old-weights")
    schema = {"observation": "obs-v1", "reward": "reward-v2"}
    train_module._atomic_save(Model(), dest, schema=schema)
    assert dest.read_bytes() == b"new-weights"
    assert list(tmp_path.glob("*.saving.zip")) == []
    assert json.loads(
        train_module.checkpoint_schema_path(dest).read_text(encoding="utf-8")
    ) == schema


def test_training_resume_rejects_missing_or_mismatched_schema(tmp_path: Path) -> None:
    checkpoint = tmp_path / "old.zip"
    checkpoint.write_bytes(b"weights")
    current = {"observation": "obs-v1", "reward": "reward-v2"}

    with pytest.raises(RuntimeError, match="schema missing"):
        train_module.require_compatible_checkpoint(checkpoint, current)

    train_module.checkpoint_schema_path(checkpoint).write_text(
        json.dumps({"observation": "obs-v1", "reward": "reward-v1"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="reward"):
        train_module.require_compatible_checkpoint(checkpoint, current)

    train_module.checkpoint_schema_path(checkpoint).write_text(
        json.dumps(current), encoding="utf-8"
    )
    train_module.require_compatible_checkpoint(checkpoint, current)


def test_close_vec_kills_when_join_hangs() -> None:
    class Proc:
        def __init__(self) -> None:
            self.killed = False

        def kill(self) -> None:
            self.killed = True

        def join(self, timeout: float | None = None) -> None:
            del timeout

    class Remote:
        def close(self) -> None:
            return None

    class Vec:
        def __init__(self) -> None:
            self.processes = [Proc()]
            self.remotes = [Remote()]
            self.closed = False

        def close(self) -> None:
            time.sleep(5)

    vec = Vec()
    train_module._close_vec(vec, timeout=0.2)
    assert vec.processes[0].killed is True
    assert vec.closed is True
