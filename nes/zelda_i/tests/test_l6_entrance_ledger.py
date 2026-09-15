"""``run_level6_entrance_tf.HealthLedger``: same numbers, far less RAM work.

The per-stage health ledger is the primary tuning tool in this tree, so the
contract under test is that reducing its RAM reads changed nothing it
reports — ``health_in``/``health_out`` for every stage boundary, and the
death count the ``ok`` flag is gated on.
"""

from __future__ import annotations

import numpy as np

from zelda_i.ram import ADDR_HEALTH, ADDR_HEART_PARTIAL, ADDR_MODE
from zelda_i.scripts.run_level6_entrance_tf import DEATH_MODE, HealthLedger, _hearts


class _Env:
    """A stub emulator whose RAM the test drives, counting ``get_ram`` calls."""

    def __init__(self) -> None:
        self.ram = np.zeros(0x800, dtype=np.uint8)
        self.ram[ADDR_MODE] = 5
        self.calls = 0

    def set(self, *, health: int, partial: int = 0xFF, mode: int = 5) -> None:
        self.ram[ADDR_HEALTH] = health
        self.ram[ADDR_HEART_PARTIAL] = partial
        self.ram[ADDR_MODE] = mode

    def get_ram(self) -> np.ndarray:
        self.calls += 1
        return self.ram


def _dense(frames: list[tuple[int, int, int]]) -> dict[int, tuple[int, int]]:
    """What the old per-frame table would have held for ``frames``."""
    return {frame: (health, partial) for frame, health, partial in frames}


def _script() -> list[tuple[int, int, int]]:
    """(frame, $066F, $0670): full, a partial hit, a whole heart, a refill."""
    rows = []
    for frame in range(1, 41):
        if frame < 10:
            rows.append((frame, 0x22, 0xFF))
        elif frame < 20:
            rows.append((frame, 0x22, 0x80))
        elif frame < 30:
            rows.append((frame, 0x21, 0xFF))
        else:
            rows.append((frame, 0x22, 0xFF))
    return rows


def test_at_matches_a_dense_per_frame_table_on_every_frame() -> None:
    """The change log answers ``at()`` exactly as one sample per frame did."""
    env = _Env()
    ledger = HealthLedger()
    env.set(health=0x22)
    ledger.seed(env)
    dense = {0: (0x22, 0xFF)}
    for frame, health, partial in _script():
        env.set(health=health, partial=partial)
        ledger.on_frame(env, None, None, frame)
        dense[frame] = (health, partial)

    for frame in range(0, 41):
        assert ledger.at(frame) == dense[frame], frame
    # Stage boundaries are just frames; in/out rows are byte-identical.
    for boundary in (0, 9, 10, 19, 20, 29, 30, 40):
        assert _hearts(ledger.at(boundary)) == _hearts(dense[boundary])
    assert ledger.at(-1) is None
    # A frame past the end holds the last value, as the backward walk did.
    assert ledger.at(999) == dense[40]


def test_change_log_stores_one_row_per_change_not_per_frame() -> None:
    env = _Env()
    ledger = HealthLedger()
    env.set(health=0x22)
    ledger.seed(env)
    for frame, health, partial in _script():
        env.set(health=health, partial=partial)
        ledger.on_frame(env, None, None, frame)
    assert ledger.reads == 41
    # seed + the three transitions in _script(); not 41 rows.
    assert ledger.samples == {
        0: (0x22, 0xFF),
        10: (0x22, 0x80),
        20: (0x21, 0xFF),
        30: (0x22, 0xFF),
    }


def test_observe_ram_shares_the_step_fetch_so_a_frame_reads_ram_once() -> None:
    """``on_frame`` reuses the array the death-mode step wrapper fetched."""
    env = _Env()
    ledger = HealthLedger()
    env.set(health=0x22)
    for frame in range(1, 11):
        ledger.observe_ram(env.get_ram())  # what the env.step wrapper does
        ledger.on_frame(env, None, None, frame)
    assert env.calls == 10
    assert ledger.at(10) == (0x22, 0xFF)
    # Without a step (the seed call), it still fetches for itself.
    ledger.on_frame(env, None, None, 11)
    assert env.calls == 11


def test_deaths_count_one_per_death_spiral() -> None:
    """Same edge rule the per-frame ``read_snapshot`` wrapper used."""
    env = _Env()
    ledger = HealthLedger()
    modes = [5, 5, DEATH_MODE, DEATH_MODE, DEATH_MODE, 5, 5, DEATH_MODE, 5]
    for mode in modes:
        env.set(health=0x20, mode=mode)
        ledger.observe_ram(env.get_ram())
    assert ledger.deaths == 2


def test_hearts_row_reads_the_066f_nibbles() -> None:
    assert _hearts(None) is None
    row = _hearts((0x21, 0x80))
    assert row["health_hex"] == "0x21"
    assert row["hearts"] == 1
    assert row["containers"] == 3
    assert row["partial"] == "0x80"
