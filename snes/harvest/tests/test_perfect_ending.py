"""Every parameter of the Perfect Rancher (999) ending, poked in RAM and judged by the ROM.

``perfect_ending_lab`` documents the method. In short: the point values are literal
(MASTER_PLAN section 2), the ROM's own calculator is the referee, and every poke is a
RAM write, so none of this is Clean evidence (ADR 0006).

* **Offline** (default run): the literal table sums to 999, every case agrees with the
  Python oracle in ``harvest.core.ranch_master``, and each case moves only its own
  ledger row.
* **``rom``**: the same cases on the real calculator (``$83F26D``, ending index 0x20), plus
  the natural final night, where the oracle cannot stand in for the ROM: the wife's hour-18
  drift, the shipping credit, the farm development count, cows that must still read 200.

Needs the local pin ``latest_y3_ending_ranch_master_999`` (gitignored, see ``harvest-pins-are-local``).
Run from the repo root::

    QT_QPA_PLATFORM=offscreen uv run python -m pytest snes/harvest/tests/test_perfect_ending.py -m rom -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from perfect_ending_lab import (  # noqa: E402
    B,
    C,
    CASES,
    CROPS,
    GIRLS,
    PERFECT_SCORE,
    PERFECT_VECTOR,
    PIN,
    S,
    TERM_MAX,
    WIFE_BITS,
    EndingLab,
    apply_pokes,
    NumpyPort,
    farm_map_with_counted,
    perfect_ram,
    scoring_vector,
)

from harvest.core.ram_catalog import read_ram_value  # noqa: E402
from harvest.core.ranch_master import (  # noqa: E402
    count_development_tiles,
    evaluate_ranch_mastery_score,
    morning_score_ledger,
    score_ledger,
)

PINS_DIR = Path(__file__).resolve().parents[1] / "custom_integrations" / "HarvestMoon-Snes"


def _case_ids(cases):
    return [case.id for case in cases]


def _oracle_total(ram: np.ndarray) -> int:
    return evaluate_ranch_mastery_score(ram).total_score


# ---------------------------------------------------------------------------------------
# Offline: the table, the oracle and the ledger agree before the ROM is asked.
# ---------------------------------------------------------------------------------------


def test_literal_term_maxima_sum_to_999() -> None:
    assert sum(TERM_MAX.values()) == PERFECT_SCORE
    assert len(TERM_MAX) == 21
    assert set(TERM_MAX) == {
        "money", "cows", "chickens", "power_berries", *GIRLS, *CROPS, "happiness", "house",
        "children", "marriage", "clock", "turtle", "development", "cow_happiness",
    }


def test_perfect_vector_is_999_with_every_term_at_its_literal_maximum() -> None:
    ram = perfect_ram()
    assert _oracle_total(ram) == PERFECT_SCORE
    ledger = score_ledger(ram)
    assert {name: row["value"] for name, row in ledger["terms"].items() if name != "total"} == TERM_MAX
    assert ledger["rating_text_id"] == 0x048B
    assert ledger["uncapped_total"] == PERFECT_SCORE


@pytest.mark.parametrize("case", CASES, ids=_case_ids(CASES))
def test_oracle_agrees_with_every_case(case) -> None:
    ram = perfect_ram()
    apply_pokes(NumpyPort(ram), case.pokes)
    assert _oracle_total(ram) == PERFECT_SCORE - case.loses


@pytest.mark.parametrize("case", [c for c in CASES if c.term != "composite"], ids=lambda c: c.id)
def test_a_single_term_case_moves_only_its_own_ledger_row(case) -> None:
    base = score_ledger(perfect_ram())["terms"]
    ram = perfect_ram()
    apply_pokes(NumpyPort(ram), case.pokes)
    ledger = score_ledger(ram)
    moved = {name for name, row in ledger["terms"].items() if name != "total" and row["value"] != base[name]["value"]}
    assert moved <= {case.term}, moved
    if case.cap:
        assert ledger["terms"][case.term]["value"] > TERM_MAX[case.term]
        assert ledger["uncapped_total"] > PERFECT_SCORE
        assert ledger["total_score"] == PERFECT_SCORE
    else:
        assert ledger["terms"][case.term]["gap"] == case.loses


# ---------------------------------------------------------------------------------------
# ROM: the real calculator at ending index 0x20.
# ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lab():
    if not (PINS_DIR / f"{PIN}.state").exists():
        pytest.skip(f"local pin {PIN} is not on this machine")
    lab = EndingLab()
    try:
        yield lab
    finally:
        lab.close()


@pytest.mark.rom
def test_rom_scores_the_999_pin_at_999(lab) -> None:
    result = lab.calc()
    assert result.rom_total == PERFECT_SCORE
    assert _oracle_total(result.ram) == PERFECT_SCORE
    assert score_ledger(result.ram)["rating_text_id"] == 0x048B


@pytest.mark.rom
@pytest.mark.parametrize("case", CASES, ids=_case_ids(CASES))
def test_rom_calculator_scores_every_case(lab, case) -> None:
    """The ROM total is 999 minus the literal loss, and the oracle read the same RAM to the same total."""
    result = lab.calc(case.pokes)
    assert result.rom_total == PERFECT_SCORE - case.loses
    assert _oracle_total(result.ram) == result.rom_total


@pytest.mark.rom
def test_scoring_fields_are_final_when_the_ending_starts(lab) -> None:
    """Nothing after the first ending scene touches a scoring field, so the night tests may read there."""
    night = lab.night()
    fork = scoring_vector(lab.calc_fork_ram())
    assert scoring_vector(night.after) == fork
    assert scoring_vector(lab.calc().ram) == fork, "the calculator itself must not move a scoring field"


# ---------------------------------------------------------------------------------------
# ROM: the natural final night (D29 17:00 -> first ending scene), the part a calculator
# fork cannot see. Each night is ~2.7k frames.
# ---------------------------------------------------------------------------------------


def _projection_total(before: np.ndarray) -> int:
    """The planner's own D30 forecast (``morning_score_ledger``) for the RAM it was handed."""
    return morning_score_ledger(before)["d30_projection"]["total_score"]


def _assert_night(night, expected_total: int) -> None:
    assert _oracle_total(night.after) == expected_total
    assert _projection_total(night.before) == expected_total, "planner's F0 projection disagrees with the ROM night"


@pytest.mark.rom
def test_the_base_night_changes_only_the_first_wife_by_minus_two(lab) -> None:
    night = lab.night()
    before, after = scoring_vector(night.before), scoring_vector(night.after)
    assert {k: after[k] - before[k] for k in before if after[k] != before[k]} == {"maria_hearts": -2}
    _assert_night(night, PERFECT_SCORE)


@pytest.mark.rom
def test_a_perfect_evening_vector_scores_999_after_the_night(lab) -> None:
    """The whole D29 evening target at once: windows, flags, herd, farm, Eve as wife at 500."""
    night = lab.night(PERFECT_VECTOR)
    assert read_ram_value(night.after, "eve_hearts", raw=True) == 498
    _assert_night(night, PERFECT_SCORE)


def _wife_pokes(wife: str, hearts: int):
    return (
        C("marriage_flags", 0x001F),
        B("marriage_flags", WIFE_BITS[wife]),
        *(S(f"{girl}_hearts", 500) for girl in GIRLS),
        S(f"{wife}_hearts", hearts),
    )


@pytest.mark.rom
@pytest.mark.parametrize("wife", GIRLS)
@pytest.mark.parametrize("start", [497, 498])
def test_the_wife_loses_two_on_the_final_night_and_nobody_else_does(lab, wife, start) -> None:
    """Entry window 498-511: 497 lands on 495 and costs a point, 498 lands on 496 and does not."""
    night = lab.night(_wife_pokes(wife, start))
    after = {girl: read_ram_value(night.after, f"{girl}_hearts", raw=True) for girl in GIRLS}
    assert after == {girl: (start - 2 if girl == wife else 500) for girl in GIRLS}
    _assert_night(night, PERFECT_SCORE - (1 if start == 497 else 0))


@pytest.mark.rom
@pytest.mark.parametrize(
    "money, shipping, loses",
    [(9984, 0, 0), (9980, 4, 0), (9970, 14, 0), (9980, 3, 1), (9970, 13, 1), (9983, 0, 1), (9984 - 128, 0, 1)],
)
def test_the_shipping_pile_is_credited_before_the_money_term(lab, money, shipping, loses) -> None:
    night = lab.night((S("money_raw", money), S("shipping_money_raw", shipping)))
    assert read_ram_value(night.after, "shipping_money_raw", raw=True) == 0
    assert read_ram_value(night.after, "money_raw", raw=True) == money + shipping
    _assert_night(night, PERFECT_SCORE - loses)


@pytest.mark.rom
@pytest.mark.parametrize(
    "tiles, byte, loses",
    [(2965, 100, 0), (2950, 100, 0), (2949, 99, 1), (2891, 98, 1), (2890, 97, 2), (2000, 67, 17)],
)
def test_the_farm_development_boundary_is_2950_counted_tiles(lab, tiles, byte, loses) -> None:
    """``CalculateFarmDevelopment`` ($82AA0C) then ``tiles * 10 / 295``, capped at 100; the term is ``>> 1``."""
    farm = farm_map_with_counted(lab.start_ram(), tiles)
    night = lab.night(farm=farm)
    assert count_development_tiles(night.after) == tiles
    assert read_ram_value(night.after, "ranch_development", raw=True) == byte
    _assert_night(night, PERFECT_SCORE - loses)


@pytest.mark.rom
@pytest.mark.parametrize("happiness, loses", [(200, 0), (255, 0), (199, 12)])
def test_twelve_fed_cows_keep_their_happiness_through_the_night(lab, happiness, loses) -> None:
    """A fed night adds and subtracts nothing: exactly 200 still scores 25 a cow, 199 scores 24."""
    night = lab.night(tuple(S(f"cow{slot}.happiness", happiness) for slot in range(12)))
    assert all(
        scoring_vector(night.after)[f"cow{slot}.happiness"] == happiness for slot in range(12)
    )
    _assert_night(night, PERFECT_SCORE - loses)


@pytest.mark.rom
@pytest.mark.parametrize("happiness, loses", [(999, 0), (992, 0), (991, 1)])
def test_player_happiness_window_survives_a_quiet_night(lab, happiness, loses) -> None:
    night = lab.night((S("happiness", happiness),))
    assert read_ram_value(night.after, "happiness", raw=True) == happiness
    _assert_night(night, PERFECT_SCORE - loses)
