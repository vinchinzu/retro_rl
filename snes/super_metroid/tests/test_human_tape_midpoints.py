"""Pure unit tests for lockstep scan + materialize (no emulator/ROM)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from super_metroid.human_tape import (
    fingerprint,
    lockstep_scan,
    materialize_lockstep_mid,
    resolve_hop_slice,
    write_gzip_state,
)
from super_metroid.human_tape.replay import (
    iter_hop_steps,
    resolve_assist,
    run_hop_replay,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeEm:
    def __init__(self, env: "_FakeEnv") -> None:
        self._env = env

    def get_state(self) -> bytes:
        # Blob for the last applied frame index (env.i after step).
        return f"blob-at-{self._env.i}".encode()


class _FakeEnv:
    """Minimal env: step advances index; parse_env_state is monkeypatched."""

    def __init__(self) -> None:
        self.i = -1  # before any step
        self.data = SimpleNamespace()
        self.em = _FakeEm(self)
        self.closed = False

    def step(self, _action: Any) -> None:
        self.i += 1

    def close(self) -> None:
        self.closed = True


def _nav_state(
    *,
    room_id: int,
    x: int,
    y: int,
    pose: int = 2,
    health: int = 100,
) -> SimpleNamespace:
    return SimpleNamespace(
        room_id=room_id,
        samus_x=x,
        samus_y=y,
        pose=pose,
        health=health,
        game_state=8,
        door_transition=0,
        collected_items=0,
        collected_beams=0,
        phase=SimpleNamespace(name="ORDINARY_GAMEPLAY", value="ordinary_gameplay"),
    )


def _match_parse(env: _FakeEnv, trace: list[dict[str, Any]], n_match: int):
    """Return parse_env_state that matches trace for indices < n_match."""

    def parse_env_state(e: Any, mode: str = "nav") -> SimpleNamespace:
        i = int(getattr(e, "i", -1))
        if i < 0:
            row = trace[0]
            return _nav_state(
                room_id=int(row["room"]),
                x=int(row["x"]),
                y=int(row["y"]),
                pose=int(row.get("pose", 2)),
            )
        if i < n_match and i < len(trace):
            row = trace[i]
            return _nav_state(
                room_id=int(row["room"]),
                x=int(row["x"]),
                y=int(row["y"]),
                pose=int(row.get("pose", 2)),
            )
        # Contiguous desync: wrong xy after n_match frames.
        room = int(trace[min(i, len(trace) - 1)]["room"]) if trace else 0
        return _nav_state(room_id=room, x=9999, y=9999, pose=99)

    return parse_env_state


def _full_match_parse(env: _FakeEnv, trace: list[dict[str, Any]]):
    def parse_env_state(e: Any, mode: str = "nav") -> SimpleNamespace:
        i = int(getattr(e, "i", -1))
        if i < 0:
            row = trace[0]
        else:
            row = trace[min(i, len(trace) - 1)]
        return _nav_state(
            room_id=int(row["room"]),
            x=int(row["x"]),
            y=int(row["y"]),
            pose=int(row.get("pose", 2)),
            health=int(row.get("energy") or 100),
        )

    return parse_env_state


def _synthetic_lockstep_task(
    tmp_path: Path,
    *,
    n: int = 12,
    room_a: int = 0xDE4D,
    room_b: int = 0xDE7A,
    leave_at: int = 8,
    enter_frame: int = 0,
) -> tuple[Path, Path, list[list[int]], list[dict[str, Any]]]:
    """Task with frames/trace + enter anchor gzip for materialize/resolve tests."""
    frames = [[0] * 12 for _ in range(n)]
    trace: list[dict[str, Any]] = []
    for i in range(n):
        room = room_a if i < leave_at else room_b
        x = 100 + i if room == room_a else 200
        y = 200 if room == room_a else 224
        trace.append(
            {
                "frame": i,
                "room": room,
                "room_hex": f"0x{room:04X}",
                "x": x,
                "y": y,
                "pose": 2,
                "energy": 100,
                "buttons": [],
            }
        )

    anchors_dir = tmp_path / "lock_anchors"
    anchors_dir.mkdir()
    enter_path = anchors_dir / f"f{enter_frame:06d}_enter_0x{room_a:04X}.state"
    write_gzip_state(enter_path, b"enter-blob")
    anchors = [
        fingerprint(
            frame=enter_frame,
            room_id=room_a,
            x=100,
            y=200,
            pose=2,
            kind="room_enter",
            path=str(enter_path),
        )
    ]
    idx = {
        "task": "lock",
        "anchors_dir": str(anchors_dir),
        "count": len(anchors),
        "anchors": anchors,
    }
    (tmp_path / "lock_anchors.json").write_text(
        json.dumps(idx, indent=2) + "\n", encoding="utf-8"
    )
    task = {
        "name": "lock",
        "frames": frames,
        "trace": trace,
        "frame_count": n,
        "start_state": "scratch/x.state",
        "recorded_at": "t",
        "metadata": {},
    }
    task_path = tmp_path / "lock.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    return task_path, enter_path, frames, trace


# ---------------------------------------------------------------------------
# 1. lockstep contiguous break
# ---------------------------------------------------------------------------


def test_lockstep_scan_stops_on_first_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    n_match = 5
    n_total = 12
    room = 0xDE4D
    frames = [[0] * 12 for _ in range(n_total)]
    trace = [
        {
            "frame": i,
            "room": room,
            "x": 100 + i,
            "y": 200,
            "pose": 2,
            "energy": 100,
        }
        for i in range(n_total)
    ]
    env = _FakeEnv()
    monkeypatch.setattr(
        "super_metroid.ram.parse_env_state",
        _match_parse(env, trace, n_match),
    )

    scan = lockstep_scan(
        env,
        frames,
        trace,
        0,
        n_total - 1,
        xy_tol=12,
        assist=False,
    )

    assert scan["last_ok_i"] == n_match - 1
    assert scan["last_match"] == n_match - 1
    assert scan["contiguous_last_match"] == n_match - 1
    assert scan["last_ok_blob"] == f"blob-at-{n_match - 1}".encode()
    assert scan["first_mismatch"] is not None
    assert scan["first_mismatch"]["index"] == n_match
    assert scan["first_mismatch"]["xy_ok"] is False
    # Contiguous: no sample after first mismatch index.
    ok_idxs = [s["index"] for s in scan["samples"] if s.get("ok")]
    bad_idxs = [s["index"] for s in scan["samples"] if not s.get("ok")]
    assert max(ok_idxs) == n_match - 1
    assert bad_idxs == [n_match]
    # Env stopped stepping after first mismatch (inclusive step to mismatch).
    assert env.i == n_match


# ---------------------------------------------------------------------------
# 2. materialize refuse past last match
# ---------------------------------------------------------------------------


def test_materialize_refuses_target_past_last_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_path, enter_path, _frames, trace = _synthetic_lockstep_task(tmp_path, n=12)
    n_match = 4
    env = _FakeEnv()
    monkeypatch.setattr(
        "super_metroid.ram.parse_env_state",
        _match_parse(env, trace, n_match),
    )
    monkeypatch.setattr(
        "super_metroid.dev.common.boot_from_state",
        lambda *_a, **_k: None,
    )

    result = materialize_lockstep_mid(
        task_path,
        from_frame=0,
        to_frame=11,
        target_index=8,  # past last_ok (3)
        anchor_path=enter_path,
        env=env,
        dual_verify=False,
        assist=False,
        update_index=False,
    )

    assert result["ok"] is False
    assert "past last_ok" in result["reason"]
    assert result["last_match"] == n_match - 1
    assert result["scan"]["first_mismatch"]["index"] == n_match


# ---------------------------------------------------------------------------
# 3. materialize index update + re-run replace
# ---------------------------------------------------------------------------


def test_materialize_writes_mid_lockstep_and_replaces_on_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_path, enter_path, _frames, trace = _synthetic_lockstep_task(tmp_path, n=12)
    target = 5
    env = _FakeEnv()
    monkeypatch.setattr(
        "super_metroid.ram.parse_env_state",
        _full_match_parse(env, trace),
    )
    monkeypatch.setattr(
        "super_metroid.dev.common.boot_from_state",
        lambda *_a, **_k: None,
    )

    out1 = materialize_lockstep_mid(
        task_path,
        from_frame=0,
        to_frame=11,
        target_index=target,
        anchor_path=enter_path,
        env=env,
        dual_verify=False,
        assist=False,
        update_index=True,
        label="mid",
    )
    assert out1["ok"] is True
    assert out1["dump_index"] == target
    mid = out1["mid"]
    assert mid["kind"] == "mid_lockstep"
    assert mid["frame"] == target
    state_path = Path(out1["state_path"])
    assert state_path.is_file()
    with gzip.open(state_path, "rb") as gz:
        assert gz.read() == f"blob-at-{target}".encode()

    idx_path = task_path.with_name(task_path.stem + "_anchors.json")
    idx1 = json.loads(idx_path.read_text(encoding="utf-8"))
    mid_rows = [r for r in idx1["anchors"] if r.get("kind") == "mid_lockstep"]
    assert len(mid_rows) == 1
    assert mid_rows[0]["frame"] == target
    assert Path(mid_rows[0]["path"]).name == state_path.name

    # Re-run materialize for same target — replace, do not duplicate.
    env2 = _FakeEnv()
    monkeypatch.setattr(
        "super_metroid.ram.parse_env_state",
        _full_match_parse(env2, trace),
    )
    out2 = materialize_lockstep_mid(
        task_path,
        from_frame=0,
        to_frame=11,
        target_index=target,
        anchor_path=enter_path,
        env=env2,
        dual_verify=False,
        assist=False,
        update_index=True,
        label="mid",
    )
    assert out2["ok"] is True
    idx2 = json.loads(idx_path.read_text(encoding="utf-8"))
    mid_rows2 = [r for r in idx2["anchors"] if r.get("kind") == "mid_lockstep"]
    assert len(mid_rows2) == 1
    assert mid_rows2[0]["frame"] == target
    assert Path(mid_rows2[0]["path"]).name == Path(out2["state_path"]).name
    # Original enter pin still present.
    assert any(r.get("kind") == "room_enter" for r in idx2["anchors"])


# ---------------------------------------------------------------------------
# 4. stepper parity / assist resolve
# ---------------------------------------------------------------------------


def test_resolve_assist_true_false_none_passthrough() -> None:
    assist = resolve_assist(True)
    assert assist is not None
    assert hasattr(assist, "apply")
    assert type(assist).__name__ == "UnlimitedResourcesAssist"

    assert resolve_assist(False) is None
    assert resolve_assist(None) is None

    sentinel = object()
    assert resolve_assist(sentinel) is sentinel


def test_iter_hop_steps_applies_frame_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = [[0] * 12 for _ in range(10)]
    env = _FakeEnv()
    calls: list[int] = []

    def parse_env_state(e: Any, mode: str = "nav") -> SimpleNamespace:
        calls.append(int(e.i))
        return _nav_state(room_id=1, x=0, y=0)

    monkeypatch.setattr("super_metroid.ram.parse_env_state", parse_env_state)

    steps = list(iter_hop_steps(env, frames, 2, 5, assist=False))
    assert [i for i, _ in steps] == [2, 3, 4, 5]
    # Inclusive [2,5] → 4 steps; fake env starts at i=-1 so final i=3.
    assert env.i == 3
    assert len(steps) == 4
    assert len(calls) == 4


def test_iter_hop_steps_empty_slice_raises() -> None:
    env = _FakeEnv()
    with pytest.raises(ValueError, match="empty slice"):
        list(iter_hop_steps(env, [[0] * 12] * 5, 4, 2, assist=False))


# ---------------------------------------------------------------------------
# 5. wrong-room anchor fails loud
# ---------------------------------------------------------------------------


def test_resolve_hop_slice_flags_wrong_room_anchor(tmp_path: Path) -> None:
    room_a = 0xDE4D
    room_b = 0xDE7A
    wrong_room = 0xABCD
    n = 30
    frames = [[0] * 12 for _ in range(n)]
    # Hop 0: room_a → room_b at index 20
    trace = []
    for i in range(n):
        room = room_a if i < 20 else room_b
        trace.append(
            {
                "frame": i,
                "room": room,
                "room_hex": f"0x{room:04X}",
                "x": 10 + i,
                "y": 100,
                "pose": 2,
                "energy": 100,
                "buttons": [],
            }
        )

    anchors_dir = tmp_path / "wrong_anchors"
    anchors_dir.mkdir()
    wrong_path = anchors_dir / "f000000_enter_0xABCD.state"
    write_gzip_state(wrong_path, b"wrong-room-blob")
    anchors = [
        fingerprint(
            frame=0,
            room_id=wrong_room,
            x=1,
            y=1,
            pose=2,
            kind="room_enter",
            path=str(wrong_path),
        )
    ]
    idx = {
        "task": "wrong",
        "anchors_dir": str(anchors_dir),
        "count": 1,
        "anchors": anchors,
    }
    (tmp_path / "wrong_anchors.json").write_text(
        json.dumps(idx, indent=2) + "\n", encoding="utf-8"
    )
    task_path = tmp_path / "wrong.json"
    task_path.write_text(
        json.dumps(
            {
                "name": "wrong",
                "frames": frames,
                "trace": trace,
                "frame_count": n,
                "start_state": "x",
                "recorded_at": "t",
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )

    info = resolve_hop_slice(task_path, hop_index=0, leave_extra=1)
    assert info["start_room"] == room_a
    assert info["anchor_room_mismatch"] is True
    assert info["anchor_mismatch_risk"] is True
    assert info["anchor_warning"] is not None
    assert "0xABCD" in info["anchor_warning"]
    assert f"0x{room_a:04X}" in info["anchor_warning"]


def test_run_hop_replay_fails_loud_on_wrong_room_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    room_a = 0xDE4D
    room_b = 0xDE7A
    wrong_room = 0xABCD
    n = 30
    frames = [[0] * 12 for _ in range(n)]
    trace = []
    for i in range(n):
        room = room_a if i < 20 else room_b
        trace.append(
            {
                "frame": i,
                "room": room,
                "room_hex": f"0x{room:04X}",
                "x": 10 + i,
                "y": 100,
                "pose": 2,
                "energy": 100,
                "buttons": [],
            }
        )
    anchors_dir = tmp_path / "wrong_anchors"
    anchors_dir.mkdir()
    wrong_path = anchors_dir / "f000000_enter_0xABCD.state"
    write_gzip_state(wrong_path, b"wrong-room-blob")
    anchors = [
        fingerprint(
            frame=0,
            room_id=wrong_room,
            x=1,
            y=1,
            pose=2,
            kind="room_enter",
            path=str(wrong_path),
        )
    ]
    (tmp_path / "wrong_anchors.json").write_text(
        json.dumps(
            {
                "task": "wrong",
                "anchors_dir": str(anchors_dir),
                "count": 1,
                "anchors": anchors,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    task_path = tmp_path / "wrong.json"
    task_path.write_text(
        json.dumps(
            {
                "name": "wrong",
                "frames": frames,
                "trace": trace,
                "frame_count": n,
                "start_state": "x",
                "recorded_at": "t",
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )

    # Must not open a real env when mismatch is detected first.
    def _boom_env() -> None:
        raise AssertionError("make_dev_env should not be called on mismatch fail")

    monkeypatch.setattr("super_metroid.dev.common.make_dev_env", _boom_env)

    out = run_hop_replay(task_path, hop_index=0, leave_extra=1, assist=False)
    assert out["ok"] is False
    assert out["green"] is False
    assert out["runs"] == []
    reason = out.get("reason") or ""
    assert "anchor room" in reason or "does not match" in reason
    assert out["slice"]["anchor_room_mismatch"] is True

    # Explicit allow_anchor_mismatch would proceed to boot — still no ROM:
    # force fail earlier by providing a missing path via allow only tests the
    # gate: with allow + real path it would call make_dev_env. Cover override
    # via explicit anchor_path of same wrong file: mismatch check is skipped
    # when anchor_path is forced (API contract).
    monkeypatch.setattr(
        "super_metroid.dev.common.make_dev_env",
        lambda: (_ for _ in ()).throw(RuntimeError("env-not-needed")),
    )
    # allow_anchor_mismatch=True should pass the room gate and attempt boot.
    with pytest.raises(RuntimeError, match="env-not-needed"):
        run_hop_replay(
            task_path,
            hop_index=0,
            leave_extra=1,
            assist=False,
            allow_anchor_mismatch=True,
        )
