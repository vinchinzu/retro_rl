"""Shared helpers and facts for Spring D1 town recon CLI."""

from __future__ import annotations

import gzip
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from harvest.paths import GAME_DIR, ensure_monorepo_on_path

ensure_monorepo_on_path()

from retro_harness import TaskStatus, WorldState

from harvest.core.ram_catalog import read_ram_value
from harvest.runtime.power_on import PowerOnStartTask
from harvest.tasks.nav import make_action

# ---------------------------------------------------------------------------
# Recon facts (docs/town_day1_recon.md)
# ---------------------------------------------------------------------------

DEFAULT_ENTRY_STATE = "Y1_Spring_D1_Town_Gate"
DEFAULT_RECORD_NAME = "town_day1_handoff"
TARGET_MASK = 0x3F
TOWN_TILEMAP = 0x04
GATE_PIXEL = (712, 424)
TRUCK_PIXEL = (728, 424)
GATE_TOLERANCE_PX = 24

# Bit → (person, working stand / notes) — stands from tasks/town_day1_rest.json
D1_TOWN_BITS: dict[int, tuple[str, str]] = {
    0x01: ("Ann", "town lower road; ~ (388,924) face left A"),
    0x02: ("Eve", "town lower-west; ~ (162,896) face up A"),
    0x04: ("Nina", "flower back 0x1D; ~ (101,102) face left A"),
    0x08: ("Flower shop owner", "flower shop 0x1C; ~ (34,347) face down A"),
    0x10: ("Livestock dealer", "animal shop 0x24; ~ (230,139) face down A"),
    0x20: ("Maria", "church 0x1B; ~ (103,405) face up A"),
}

VERIFIED_ROUTES = (
    "Town → flower shop: (688,280)→(600,280)→(600,262) walk up → 0x1C @(144,456)",
    "Flower shop → back room: left ~20f then up → 0x1D; Nina stand (101,102) face left",
    "Town → church: (688,280)→(600,280)→(500,280)→(376,280)→(376,200)→(375,139) up → Maria (103,405) face up",
    "Town → animal shop: lower road → 0x24; D1 stand (230,139) face down (not buy-cow 201,157)",
    "Truck ~ (728,424); leave after mask 0x3F cutscenes into farmhouse",
    "Shed free starters: grass (96,118) + watering can (96,168) face up A",
)

STILL_TO_RECORD = (
    "Green auto report: full mask from power-on/gate + shed pickups + D2 sleep",
    "Optional: re-record AnnEve fixture with house_size=0 (current is size2)",
)


@dataclass(frozen=True)
class TownSnapshot:
    frame: int
    tilemap: int
    x: int
    y: int
    day: int
    season: int
    hour: int
    minute: int
    mask: int
    input_lock: int

    @property
    def mask_hex(self) -> str:
        return f"0x{self.mask:02X}"

    @property
    def bits_set(self) -> list[str]:
        return decode_mask_bits(self.mask)

    @property
    def bits_missing(self) -> list[str]:
        return decode_mask_bits((~self.mask) & TARGET_MASK)

    def as_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["mask_hex"] = self.mask_hex
        row["tilemap_hex"] = f"0x{self.tilemap:02X}"
        row["bits_set"] = self.bits_set
        row["bits_missing"] = self.bits_missing
        row["mask_complete"] = self.mask == TARGET_MASK
        return row


def decode_mask_bits(mask: int) -> list[str]:
    """Return human labels for bits present in ``mask`` (low six only)."""
    labels: list[str] = []
    for bit, (person, _note) in D1_TOWN_BITS.items():
        if mask & bit:
            labels.append(f"{person}(0x{bit:02X})")
    return labels


def read_town_snapshot(ram, *, frame: int = 0) -> TownSnapshot:
    return TownSnapshot(
        frame=int(frame),
        tilemap=int(read_ram_value(ram, "tilemap")),
        x=int(read_ram_value(ram, "player_x")),
        y=int(read_ram_value(ram, "player_y")),
        day=int(read_ram_value(ram, "day")),
        season=int(read_ram_value(ram, "season")),
        hour=int(read_ram_value(ram, "hour")),
        minute=int(read_ram_value(ram, "minute")),
        mask=int(read_ram_value(ram, "d1_town_event_mask")),
        input_lock=int(read_ram_value(ram, "input_lock")),
    )


def is_town_gate_entry(snap: TownSnapshot) -> bool:
    if snap.tilemap != TOWN_TILEMAP:
        return False
    if snap.day != 1 or snap.season != 0:
        return False
    if snap.hour < 7:
        return False
    gx, gy = GATE_PIXEL
    return abs(snap.x - gx) <= GATE_TOLERANCE_PX and abs(snap.y - gy) <= GATE_TOLERANCE_PX


def world(env, frame: int) -> WorldState:
    return WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None)


def run_power_on(env, *, max_frames: int | None = None) -> tuple[dict[str, object], int]:
    task = PowerOnStartTask()
    if max_frames is not None:
        task.timeout = max_frames
    frame = 0
    task.reset(world(env, frame))
    print("[RECON] power-on bootstrap: attract → START → new diary → Spring D1 town", flush=True)
    while frame < task.timeout:
        w = world(env, frame)
        result = task.step(w)
        if result.status == TaskStatus.SUCCESS:
            summary = task.summary(w)
            print(
                f"[RECON] power-on ready after {frame} frames "
                f"tm=0x{int(read_ram_value(w.ram, 'tilemap')):02X} "
                f"pos=({int(read_ram_value(w.ram, 'player_x'))},"
                f"{int(read_ram_value(w.ram, 'player_y'))})",
                flush=True,
            )
            return summary, frame
        if result.status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
            summary = task.summary(w)
            summary["failure"] = result.reason or result.status.value
            return summary, frame
        action = result.action.action if result.action is not None else make_action()
        env.step(action)
        frame += 1
    w = world(env, frame)
    summary = task.summary(w)
    summary["failure"] = "power-on frame budget exhausted"
    return summary, frame


def save_state(env, name: str) -> Path:
    path = GAME_DIR / f"{name}.state"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        handle.write(env.em.get_state())
    return path


def end_state_paths(task_json: Path) -> list[Path]:
    task_state = task_json.with_name(task_json.stem + "_end.state")
    mirrored = GAME_DIR / task_state.name
    paths = [task_state]
    if mirrored.resolve() != task_state.resolve():
        paths.append(mirrored)
    return paths


def configure_headless() -> None:
    os.environ.setdefault("HEADLESS", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def unset_headless_for_interactive() -> None:
    for key in ("HEADLESS", "SDL_VIDEODRIVER", "SDL_AUDIODRIVER", "SDL_SOFTWARE_RENDERER"):
        os.environ.pop(key, None)
