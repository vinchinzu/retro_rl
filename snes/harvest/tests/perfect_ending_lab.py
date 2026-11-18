"""RAM-hack lab for the Perfect Rancher (999) ending.

Every poke here is a RAM write. That makes this research evidence about what the
ROM scores, never Clean evidence (ADR 0006): nothing in this file may feed a route,
a recording or STATUS.

The point values below are written out literally from ``docs/perfect_rancher/MASTER_PLAN.md``
section 2, not imported from ``harvest.core.ranch_master``. The ROM's own calculator
(bank_83 ``RanchMasteryCalculator`` at ``$83F26D``) is the referee: the oracle is checked
*against* these numbers, and so is every scoring decision the planner makes.

The base is ``latest_y3_ending_ranch_master_999`` (Y3 Summer D29 17:00, every term at its
maximum, twelve fed cows and twelve hens). Two ROM measurements hang off it:

* **calculator fork**: play the natural final night and the ending to the frame just before
  ending index ``0x20``, snapshot, then for each case restore, poke, and run the ~400 frames
  to index ``0x25``. ``$7F1F54`` there is the score the ROM actually computed.
* **natural night**: restore the pin, poke, run ``ReturnHomeTask`` + ``GoToSleepTask`` to the
  first ending scene. Every scoring field is final from there (tested), so the oracle can
  read the verdict. This is where the hour-18 wife drift, shipping credit and the farm
  development count are exercised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from harvest.core.ram_catalog import (
    animal_slot_address,
    encode_storage_value,
    field_spec,
    KIND_WIDTHS,
    ram_index,
    read_ram_value,
    read_storage_value,
)

PIN = "latest_y3_ending_ranch_master_999"
BUS_BASE = 0x7E0000
FARM_MAP_WRAM = 0xA4E6
FARM_MAP_TILES = 64 * 64

# ---- score anatomy, literal (MASTER_PLAN section 2) ---------------------------------

GIRLS = ("maria", "ann", "nina", "ellen", "eve")
WIFE_BITS = {"maria": 0x01, "ann": 0x02, "nina": 0x04, "ellen": 0x08, "eve": 0x10}
CROPS = ("tomatoes", "corn", "potatoes", "turnips")

TERM_MAX: dict[str, int] = {
    "money": 78,
    "cows": 36,
    "chickens": 36,
    "power_berries": 50,
    **{girl: 31 for girl in GIRLS},
    **{crop: 31 for crop in CROPS},
    "happiness": 31,
    "house": 32,
    "children": 32,
    "marriage": 32,
    "clock": 22,
    "turtle": 21,
    "development": 50,
    "cow_happiness": 300,
}
PERFECT_SCORE = 999

# ---- pokes -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Poke:
    """One RAM edit. ``field`` is a catalog key or ``cow<slot>.<field>``.

    ``set`` writes ``value`` (not clamped to the field's playable range: the ROM masks
    nine bits and wraps 16, so out-of-range raw values are the point). ``clear`` and
    ``setbits`` edit bits and keep the rest of the word.
    """

    field: str
    op: str
    value: int


def S(field: str, value: int) -> Poke:
    return Poke(field, "set", value)


def C(field: str, mask: int) -> Poke:
    return Poke(field, "clear", mask)


def B(field: str, mask: int) -> Poke:
    return Poke(field, "setbits", mask)


# What a perfect D29 evening holds (the pin is close to it). Windows from MASTER_PLAN section 2.
# The wife is the exception: she loses 2 on the hour-18 re-entry, so she starts at 498-511.
# Flag words keep their unrelated bits (children's ages, wedding bits): only scoring bits move.
PERFECT_VECTOR: tuple[Poke, ...] = (
    S("money_raw", 9984),            # 99,840G; the ROM reads only the low 16 bits (10G units)
    S("num_cows", 12),
    S("num_chickens", 12),
    S("max_stamina", 200),           # ten berries, +10 each
    *(S(f"{girl}_hearts", 500) for girl in GIRLS),
    *(S(f"shipped_{crop}", 500) for crop in CROPS),
    S("happiness", 999),
    B("upgrade_flags", 0x00C0),      # both house upgrades
    B("incubator_flags", 0x400C),    # two children + clock
    C("marriage_flags", 0x001F),
    B("marriage_flags", 0x0010),     # Eve is the owner's wife
    B("family_event_flags", 0x1000), # turtle shell
    S("ranch_development", 100),
    *(S(f"cow{slot}.status_raw", 0x09) for slot in range(12)),
    *(S(f"cow{slot}.happiness", 200) for slot in range(12)),
)


def locate(field: str) -> tuple[int, str]:
    """Return ``(wram_address, kind)`` for a catalog key or a cow-slot field."""
    if field.startswith("cow") and "." in field:
        slot, _, name = field[3:].partition(".")
        return animal_slot_address("cow", int(slot), name)
    spec = field_spec(field)
    return spec.address, spec.kind


class NumpyPort:
    """Reads and writes a RAM array the way the catalog addresses it (offline)."""

    def __init__(self, ram: np.ndarray) -> None:
        self.ram = ram

    def read(self, address: int, kind: str) -> int:
        return read_storage_value(self.ram, address, kind)

    def write(self, address: int, kind: str, value: int) -> None:
        index = ram_index(self.ram, address)
        for offset, byte in enumerate(encode_storage_value(kind, value & ((1 << (8 * KIND_WIDTHS[kind])) - 1))):
            self.ram[index + offset] = byte


class EnvPort:
    """Reads the emulator RAM and writes bus addresses (``0x7E0000 + WRAM``)."""

    def __init__(self, env) -> None:
        self.env = env

    def read(self, address: int, kind: str) -> int:
        return read_storage_value(np.asarray(self.env.get_ram(), dtype=np.uint8), address, kind)

    def write(self, address: int, kind: str, value: int) -> None:
        memory = self.env.unwrapped.data.memory
        for offset, byte in enumerate(encode_storage_value(kind, value & ((1 << (8 * KIND_WIDTHS[kind])) - 1))):
            memory.assign(BUS_BASE + address + offset, "|u1", int(byte))


def apply_pokes(port, pokes: Iterable[Poke]) -> None:
    for poke in pokes:
        address, kind = locate(poke.field)
        if poke.op == "set":
            value = poke.value
        elif poke.op == "clear":
            value = port.read(address, kind) & ~poke.value
        elif poke.op == "setbits":
            value = port.read(address, kind) | poke.value
        else:
            raise ValueError(f"unknown poke op {poke.op!r}")
        port.write(address, kind, value)


def perfect_ram(size: int = 0x20000) -> np.ndarray:
    """A blank RAM array holding exactly :data:`PERFECT_VECTOR` (offline oracle checks)."""
    ram = np.zeros(size, dtype=np.uint8)
    apply_pokes(NumpyPort(ram), PERFECT_VECTOR)
    return ram


# ---- the case table ---------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """Poke the perfect base; the ROM must report ``999 - loses``.

    ``cap`` marks pokes that push a term past its maximum: the ROM clamps the sum to 999,
    so the excess is invisible in the total (and only the oracle ledger can see it).
    ``term`` is the ledger row that must be the only one to move ("composite" mixes terms).
    """

    term: str
    label: str
    pokes: tuple[Poke, ...]
    loses: int
    cap: bool = False

    @property
    def id(self) -> str:
        return f"{self.term}:{self.label}"


def _build_cases() -> tuple[Case, ...]:
    cases: list[Case] = []

    def add(term: str, label: str, loses: int, *pokes: Poke, cap: bool = False) -> None:
        cases.append(Case(term, label, tuple(pokes), loses, cap))

    # Money: 10G units, LDA.L moneyL is a 16-bit load of the 24-bit word, min(78, low >> 7).
    for raw, lost in ((9984, 0), (65535, 0), (0x10000 + 9984, 0), (999_999, 0),
                      (9983, 1), (9856, 1), (128, 77), (0, 78), (0x10000, 78), (0x10000 + 9983, 1)):
        add("money", f"money_raw={raw}", lost, S("money_raw", raw))

    # Herds: 3 a head, twelve slots in the ROM loop.
    for term, field in (("cows", "num_cows"), ("chickens", "num_chickens")):
        for count, lost in ((12, 0), (11, 3), (6, 18), (0, 36)):
            add(term, f"{field}={count}", lost, S(field, count))
        add(term, f"{field}=13", 0, S(field, 13), cap=True)

    # Stamina: (max - 100) >> 1, ten berries is 200.
    for stamina, lost in ((200, 0), (199, 1), (198, 1), (190, 5), (100, 50)):
        add("power_berries", f"max_stamina={stamina}", lost, S("max_stamina", stamina))
    add("power_berries", "max_stamina=210", 0, S("max_stamina", 210), cap=True)

    # Hearts and shipped crops: (n & 0x1FF) >> 4, so the window is 496..511 mod 512.
    # 1008 = 512 + 496 and 999 & 0x1FF = 487: both are real values the game can hold.
    nine_bit = ((496, 0), (511, 0), (1008, 0), (495, 1), (999, 1), (0, 31), (512, 31))
    for girl in GIRLS:
        for hearts, lost in nine_bit:
            add(girl, f"{girl}_hearts={hearts}", lost, S(f"{girl}_hearts", hearts))
    for crop in CROPS:
        # The shipped counter is a 16-bit INC with no cap: 9999 & 0x1FF = 271, so 16 points.
        for shipped, lost in (*nine_bit, (9999, 15)):
            add(crop, f"shipped_{crop}={shipped}", lost, S(f"shipped_{crop}", shipped))

    # Happiness >> 5.
    for happiness, lost in ((999, 0), (992, 0), (991, 1), (960, 1), (959, 2), (0, 31)):
        add("happiness", f"happiness={happiness}", lost, S("happiness", happiness))

    # Flag terms.
    add("house", "bit 0x40 only", 16, C("upgrade_flags", 0x0080))
    add("house", "bit 0x80 only", 16, C("upgrade_flags", 0x0040))
    add("house", "neither bit", 32, C("upgrade_flags", 0x00C0))
    add("children", "first only", 16, C("incubator_flags", 0x0004))
    add("children", "second only", 16, C("incubator_flags", 0x0008))
    add("children", "none", 32, C("incubator_flags", 0x000C))
    add("clock", "clock bit clear", 22, C("incubator_flags", 0x4000))
    add("turtle", "turtle bit clear", 21, C("family_event_flags", 0x1000))
    for girl, bit in WIFE_BITS.items():
        add("marriage", f"wife={girl}", 0, S("marriage_flags", bit))
    add("marriage", "unmarried", 32, S("marriage_flags", 0))
    add("marriage", "only bits above 0x1F", 32, S("marriage_flags", 0x1E00))

    # Development byte: ``>> 1``.
    for byte, lost in ((100, 0), (101, 0), (99, 1), (98, 1), (97, 2), (0, 50)):
        add("development", f"ranch_development={byte}", lost, S("ranch_development", byte))

    # Cow happiness: per slot, existence is status bit 0, points min(25, happiness >> 3).
    for slot in range(12):
        add("cow_happiness", f"slot {slot} at 199", 1, S(f"cow{slot}.happiness", 199))
        add("cow_happiness", f"slot {slot} not present", 25, C(f"cow{slot}.status_raw", 0x01))
    add("cow_happiness", "all twelve at 200", 0, *(S(f"cow{slot}.happiness", 200) for slot in range(12)))
    add("cow_happiness", "all twelve at 199", 12, *(S(f"cow{slot}.happiness", 199) for slot in range(12)))
    add("cow_happiness", "slot 5 at 0", 25, S("cow5.happiness", 0))
    add("cow_happiness", "slot 11 at 100", 13, S("cow11.happiness", 100))

    # Composites: windows hold together, defects add, nothing compensates.
    add("composite", "every window at its low edge", 0,
        S("money_raw", 9984), S("max_stamina", 200), S("happiness", 992), S("ranch_development", 100),
        *(S(f"{girl}_hearts", 496) for girl in GIRLS), *(S(f"shipped_{crop}", 496) for crop in CROPS),
        *(S(f"cow{slot}.happiness", 200) for slot in range(12)))
    add("composite", "every window at its high edge", 0,
        S("money_raw", 65535), S("max_stamina", 200), S("happiness", 999), S("ranch_development", 100),
        *(S(f"{girl}_hearts", 511) for girl in GIRLS), *(S(f"shipped_{crop}", 511) for crop in CROPS),
        *(S(f"cow{slot}.happiness", 255) for slot in range(12)))
    add("composite", "defects add up", 3 + 3 + 5 + 1 + 1 + 16 + 25,
        S("num_cows", 11), S("num_chickens", 11), S("max_stamina", 190), S("maria_hearts", 495),
        S("shipped_corn", 495), C("upgrade_flags", 0x0040), C("cow3.status_raw", 0x01))
    add("composite", "everything empty", 999,
        S("money_raw", 0), S("num_cows", 0), S("num_chickens", 0), S("max_stamina", 100), S("happiness", 0),
        S("upgrade_flags", 0), S("incubator_flags", 0), S("marriage_flags", 0), S("family_event_flags", 0),
        S("ranch_development", 0),
        *(S(f"{girl}_hearts", 0) for girl in GIRLS), *(S(f"shipped_{crop}", 0) for crop in CROPS),
        *(S(f"cow{slot}.happiness", 0) for slot in range(12)))
    return tuple(cases)


CASES: tuple[Case, ...] = _build_cases()


# ---- scoring fields ------------------------------------------------------------------------

SCORING_FIELDS: tuple[str, ...] = (
    "money_raw", "num_cows", "num_chickens", "max_stamina", "happiness", "ranch_development",
    "upgrade_flags", "marriage_flags", "incubator_flags", "family_event_flags",
    *(f"{girl}_hearts" for girl in GIRLS),
    *(f"shipped_{crop}" for crop in CROPS),
)


def scoring_vector(ram: np.ndarray) -> dict[str, int]:
    vector = {field: read_ram_value(ram, field, raw=True) for field in SCORING_FIELDS}
    for slot in range(12):
        for name in ("status_raw", "happiness"):
            address, kind = animal_slot_address("cow", slot, name)
            vector[f"cow{slot}.{name}"] = read_storage_value(ram, address, kind)
    return vector


def farm_map_with_counted(ram: np.ndarray, counted: int, filler: int = 0x07) -> bytes:
    """The farm array with exactly ``counted`` development tiles (hoed soil fills the rest).

    ``CalculateFarmDevelopment`` ($82AA0C) counts ids below 0xA0 that are fence (5, 6) or
    at least 0x1D; ids 0x07/0x08 are bare soil and do not count.
    """
    base = ram_index(ram, FARM_MAP_WRAM)
    tiles = np.array(ram[base:base + FARM_MAP_TILES], dtype=np.uint8)
    hits = [i for i, v in enumerate(tiles.tolist()) if v < 0xA0 and (v in (5, 6) or v >= 0x1D)]
    if counted > len(hits):
        raise ValueError(f"base farm has only {len(hits)} counted tiles, asked for {counted}")
    for index in hits[counted:]:
        tiles[index] = filler
    return tiles.tobytes()


# ---- ROM driver ---------------------------------------------------------------------------

CALC_INDEX_LO, CALC_INDEX, TERMINAL_INDEX = 0x1B, 0x20, 0x25


@dataclass
class Night:
    frames: int
    before: np.ndarray   # RAM right after the pokes, before any frame ran
    after: np.ndarray    # RAM on the first frame of the ending


@dataclass
class Calc:
    frames: int
    rom_total: int       # $7F1F54 at ending index 0x25
    ram: np.ndarray


class EndingLab:
    """One emulator on the 999 pin, with the two measurements described in the module docstring."""

    def __init__(self, pin: str = PIN) -> None:
        from harvest.runtime.retro_setup import make_harvest_env

        self.env = make_harvest_env(pin, lean_step=True)
        self.obs, self.info = self.env.reset()
        self.start_state: bytes = self.env.em.get_state()
        self._calc_state: bytes | None = None
        self.port = EnvPort(self.env)

    def close(self) -> None:
        self.env.close()

    def ram(self) -> np.ndarray:
        return np.asarray(self.env.get_ram(), dtype=np.uint8)

    def index(self) -> int:
        return read_ram_value(self.ram(), "ending_scene_index", raw=True)

    def start_ram(self) -> np.ndarray:
        """RAM of the pin itself (D29 17:00), untouched."""
        self.env.em.set_state(self.start_state)
        return self.ram().copy()

    def _poke(self, pokes: Sequence[Poke], farm: bytes | None) -> None:
        apply_pokes(self.port, pokes)
        if farm is not None:
            memory = self.env.unwrapped.data.memory
            for offset, byte in enumerate(farm):
                memory.assign(BUS_BASE + FARM_MAP_WRAM + offset, "|u1", int(byte))

    def _drive_night(self, until, max_frames: int) -> int:
        """Return home, sleep, then press A through the ending until ``until(ram)``."""
        from retro_harness import TaskStatus, WorldState

        from harvest.planner.day_plan_tasks import GoToSleepTask, ReturnHomeTask
        from harvest.tasks.nav import make_action

        phase, task = "return_home", ReturnHomeTask()
        task.reset(WorldState(0, self.ram(), self.info, self.obs))
        for frame in range(1, max_frames + 1):
            ram = self.ram()
            if until(ram):
                return frame
            if phase == "credits":
                action = make_action(a=(frame % 3 != 0))
            else:
                world = WorldState(frame, ram, self.info, self.obs)
                result = task.step(world)
                action = result.action.action if result.action is not None else make_action()
                if result.status == TaskStatus.SUCCESS and phase == "return_home":
                    phase, task = "sleep", GoToSleepTask()
                    task.reset(world)
                elif result.status in (TaskStatus.SUCCESS, TaskStatus.FAILURE):
                    phase = "credits"
                    action = make_action(a=True)
            self.env.step(action)
        raise TimeoutError(f"no stop condition within {max_frames} frames, ending index 0x{self.index():02x}")

    def night(self, pokes: Sequence[Poke] = (), *, farm: bytes | None = None, max_frames: int = 6000) -> Night:
        """Poke the pin at D29 17:00 and play the natural night to the first ending scene."""
        self.env.em.set_state(self.start_state)
        self._poke(pokes, farm)
        before = self.ram().copy()
        frames = self._drive_night(lambda ram: read_ram_value(ram, "ending_scene_index", raw=True) != 0, max_frames)
        return Night(frames, before, self.ram().copy())

    def _fork(self) -> bytes:
        if self._calc_state is None:
            self.env.em.set_state(self.start_state)

            def at_fork(ram: np.ndarray) -> bool:
                return CALC_INDEX_LO <= read_ram_value(ram, "ending_scene_index", raw=True) < CALC_INDEX

            self._drive_night(at_fork, 60_000)
            self._calc_state = self.env.em.get_state()
        return self._calc_state

    def calc(self, pokes: Sequence[Poke] = ()) -> Calc:
        """Restore the calculator fork, poke, and read the ROM's verdict at index 0x25."""
        from harvest.tasks.nav import make_action

        blob = self._fork()
        self.env.em.set_state(blob)
        self._poke(pokes, None)
        for frame in range(1, 3001):
            if self.index() >= TERMINAL_INDEX:
                ram = self.ram().copy()
                return Calc(frame, read_ram_value(ram, "ranch_mastery", raw=True), ram)
            self.env.step(make_action(a=(frame % 3 != 0)))
        raise TimeoutError("ending never reached index 0x25 after the calculator fork")

    def calc_fork_ram(self) -> np.ndarray:
        self.env.em.set_state(self._fork())
        return self.ram().copy()
