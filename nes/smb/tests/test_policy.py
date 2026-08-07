"""Unit tests for SMB 1-1 seed policy (no emulator)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from smb.policy import (
    CONTINUOUS_SETTLE_FRAMES,
    DEFAULT_1_1_SEED,
    DEFAULT_CONTINUOUS_SEED,
    DEFAULT_FAST_4_2_SEED,
    DEFAULT_WARP_SUFFIX_SEED,
    POWERON_BOOT_FRAMES,
    POWERON_SETTLE_FRAMES,
    Level11ReplayPolicy,
    compress_nes9_rle,
    expand_nes9_rle,
    load_nes9_rle_seed,
)
from smb.ram import (
    ADDR_LEVEL,
    ADDR_LIVES,
    ADDR_PLAYER_STATE,
    ADDR_PLAYER_X,
    ADDR_WORLD,
    ADDR_X_PAGE,
    player_x,
    segment_1_1_success,
)


def test_seed_file_exists_and_loads() -> None:
    assert DEFAULT_1_1_SEED.exists()
    data = load_nes9_rle_seed(DEFAULT_1_1_SEED)
    assert data["format"] == "nes9_rle"
    assert data["start_state"] == "Level1_1"
    frames = expand_nes9_rle(data)
    assert len(frames) == data["num_frames"]
    # DEFAULT_1_1_SEED is TAS-polished (leave ~1903f); keep a floor above pre-clear
    # garbage but do not require the old 2029f clear seed length.
    assert len(frames) >= 1800
    assert all(len(f) == 9 for f in frames[:10])


def test_policy_replays_then_idles() -> None:
    policy = Level11ReplayPolicy()
    first = policy.step()
    assert first.action.shape == (9,)
    assert first.action.dtype == np.int8
    # Exhaust quickly with a tiny synthetic seed
    tiny = {
        "format": "nes9_rle",
        "segments": [{"b": [0, 0, 0, 0, 0, 0, 0, 1, 0], "n": 3}],
        "num_frames": 3,
    }
    path = Path(DEFAULT_1_1_SEED).parent / "_tiny_test_seed.json"
    path.write_text(__import__("json").dumps(tiny))
    try:
        p = Level11ReplayPolicy(seed_path=path)
        for _ in range(3):
            a = p.step()
            assert int(a.action[7]) == 1  # RIGHT
        assert p.remaining == 0
        idle = p.step()
        assert p.exhausted
        assert int(np.asarray(idle.action).sum()) == 0
    finally:
        path.unlink(missing_ok=True)


def test_compress_nes9_rle_round_trip() -> None:
    frames = [
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 1, 0],
    ]
    data = {"segments": compress_nes9_rle(frames)}
    assert data["segments"][0]["n"] == 2
    assert expand_nes9_rle(data) == frames


def test_warp_suffix_seed_reaches_ending_contract() -> None:
    assert DEFAULT_WARP_SUFFIX_SEED.exists()
    data = load_nes9_rle_seed(DEFAULT_WARP_SUFFIX_SEED)
    assert data["start_state"] == "Level1_2_WarpMid"
    assert data["target"] == "world_8_4_ending"
    assert data["verified_completed"] is True
    assert data["num_frames"] == 19_963
    assert sum(int(row["n"]) for row in data["segments"]) == 19_963


def test_continuous_seed_contract() -> None:
    assert DEFAULT_CONTINUOUS_SEED.exists()
    data = load_nes9_rle_seed(DEFAULT_CONTINUOUS_SEED)
    assert data["start_state"] == "Level1_1"
    assert data["settle_frames"] == CONTINUOUS_SETTLE_FRAMES == 14
    assert data["target"] == "world_8_4_ending"
    assert data["verified_completed"] is True
    assert data["num_frames"] == 21_731
    assert data["optimization"]["baseline_frames"] == 22_005
    assert data["optimization"]["frames_saved"] == 274
    assert sum(int(row["n"]) for row in data["segments"]) == 21_731
    frames = expand_nes9_rle(data)
    assert len(frames) == 21_731
    # Regression: an accidental Start toggle paused 1-2 for nearly five seconds.
    assert not any(frame[3] for frame in frames)


def test_reactive_continuous_seed_contract() -> None:
    path = DEFAULT_CONTINUOUS_SEED.with_name(
        "smb_1_1_to_ending_reactive_83_84.json"
    )
    data = load_nes9_rle_seed(path)
    assert data["start_state"] == "Level1_1"
    assert data["settle_frames"] == CONTINUOUS_SETTLE_FRAMES == 14
    assert data["target"] == "world_8_4_ending"
    assert data["verified_completed"] is True
    assert data["num_frames"] == 21_643
    assert data["verification"]["mode"] == "poweron"
    assert data["verification"]["successes"] == data["verification"]["trials"] == 3
    assert sum(int(row["n"]) for row in data["segments"]) == 21_643


def test_fast_4_2_fold_fragment_contract() -> None:
    assert DEFAULT_FAST_4_2_SEED.exists()
    data = load_nes9_rle_seed(DEFAULT_FAST_4_2_SEED)
    assert data["start_state"] == "natural_predecessor_4_2_underground"
    assert data["target"] == "world_8_entry"
    assert data["verification"]["natural_entry_required"] is True
    assert data["num_frames"] == 2_375
    assert sum(int(row["n"]) for row in data["segments"]) == 2_375


def test_poweron_phase_constants() -> None:
    """Power-on Clean uses fixed boot+settle distinct from Level1_1 settle."""
    assert POWERON_BOOT_FRAMES == 350
    assert POWERON_SETTLE_FRAMES == 16
    assert POWERON_SETTLE_FRAMES != CONTINUOUS_SETTLE_FRAMES


def test_continuous_seed_regenerates_byte_for_byte() -> None:
    """fold_continuous_policy must reproduce the published seed."""
    from smb.paths import FULLGAME_RECORDINGS_DIR
    from smb.scripts.fold_continuous_policy import build_continuous_seed

    if not (FULLGAME_RECORDINGS_DIR / "20260429_214207" / "summary.json").exists():
        return  # practice session not linked in this checkout
    if not DEFAULT_WARP_SUFFIX_SEED.exists() or not DEFAULT_CONTINUOUS_SEED.exists():
        return
    rebuilt = build_continuous_seed()
    published = load_nes9_rle_seed(DEFAULT_CONTINUOUS_SEED)
    assert rebuilt == published


def test_segment_success_requires_progress_and_level_change() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_LIVES] = 2
    ram[ADDR_X_PAGE] = 12
    ram[ADDR_PLAYER_X] = 100  # x = 12*256+100 = 3172
    ram[ADDR_WORLD] = 0
    ram[ADDR_LEVEL] = 0
    assert player_x(ram) == 3172
    assert not segment_1_1_success(ram, start_lives=2, max_player_x=3172)

    ram[ADDR_LEVEL] = 1  # 1-2
    assert segment_1_1_success(ram, start_lives=2, max_player_x=3172)
    assert not segment_1_1_success(ram, start_lives=2, max_player_x=1000)
    ram[ADDR_LIVES] = 1
    assert not segment_1_1_success(ram, start_lives=2, max_player_x=3172)


def test_parse_game_state_uses_absolute_x() -> None:
    from smb.ram import parse_game_state

    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_X_PAGE] = 2
    ram[ADDR_PLAYER_X] = 40
    ram[ADDR_PLAYER_STATE] = 0x08
    ram[0x0770] = 1
    ram[0x07F8] = 4
    ram[ADDR_LIVES] = 2
    state = parse_game_state(ram, frame=10)
    assert state.player_x == 2 * 256 + 40
    assert state.extras["level_id"] == 0
    assert state.extras["dying"] is False
