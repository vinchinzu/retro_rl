"""Forward replay must prove the route goal as well as tape synchronization."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from alttp import ram
from alttp.scripts import audit_power_on as audit


@pytest.mark.parametrize(
    "source_ok, progress, flags, follower, submodule, accepted",
    [
        (True, 1, 0, 1, 0, False),  # Sanctuary arrival before Zelda/priest dialogue.
        (True, 2, 4, 1, 0, False),  # Zelda has not been handed off.
        (True, 2, 0, 0, 0, False),  # Story completion event is absent.
        (True, 2, 4, 0, 1, False),  # Transition has not settled.
        (False, 2, 4, 0, 0, False),  # A synchronized failed controller is still red.
        (True, 2, 4, 0, 0, True),
    ],
)
def test_synchronized_sanctuary_replay_requires_completed_story(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    source_ok: bool, progress: int, flags: int, follower: int,
    submodule: int, accepted: bool,
) -> None:
    data = np.zeros(0x24000, dtype=np.uint8)
    for offset, value in {
        ram.MODULE: 7, ram.SUBMODULE: submodule, ram.INDOORS: 1,
        ram.ROOM_ID: ram.SANCTUARY_ROOM, ram.EQUIP_SWORD: 1,
        ram.LINK_ITEM_LAMP: 1, ram.LINK_HP: 24, ram.LINK_MAX_HP: 24,
        ram.SRAM_PROGRESS: progress, ram.SRAM_PROGRESS_FLAGS: flags,
        ram.FOLLOWER: follower,
    }.items():
        data[ram.wram_index(offset)] = value
    env = SimpleNamespace(
        closed=False, get_ram=lambda: data,
        reset=lambda: (None, {}),
        step=lambda _action: (None, 0.0, False, False, {}),
        em=SimpleNamespace(), data=SimpleNamespace(),
    )
    env.close = lambda: setattr(env, "closed", True)
    final = ram.snapshot_to_diag(ram.read_snapshot(data))
    story = audit.story_glance(env)
    tape = tmp_path / "sanctuary.tape.npz"
    np.savez_compressed(
        tape, buttons=np.zeros((1, 12), dtype=np.uint8),
        frame_crc=np.asarray([audit.ram_crc(data)], dtype=np.uint32),
        meta=json.dumps({
            "start": "NONE", "through": "sanctuary", "survival": False,
            "source_ok": source_ok, "source_final": final, "source_story": story,
        }),
    )
    monkeypatch.setattr(audit, "build_boot_env", lambda **_kwargs: env)
    monkeypatch.setattr(audit, "inject_sram_into_env", lambda *_args: None)
    monkeypatch.setattr(audit, "save_end", lambda *_args: None)
    report = audit.replay(SimpleNamespace(tape=tape, out=tmp_path / "replay.json"))

    assert env.closed
    assert report["synchronization_ok"] is True
    assert report["desynced"] == 0
    assert report["audit"]["set_state"] == report["audit"]["writes"] == 0
    assert report["route_goal_ok"] is accepted
    assert report["ok"] is accepted
