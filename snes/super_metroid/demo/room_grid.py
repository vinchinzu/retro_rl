"""4x4 autobot room-grid demo: independent hops + ffmpeg mosaic.

Not continuous evidence. Default is sequential (``--workers 1``). Parallel
uses one spawned process per tile — not :class:`retro_harness.emulator_pool.EmulatorPool`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from super_metroid.demo.grid_mosaic import (
    CELL_H,
    CELL_W,
    DEFAULT_COLS,
    DEFAULT_ROWS,
    DEFAULT_SECONDS,
    NTSC_FPS,
    composite_grid,
    label_frame,
    xstack_filter,
)
from super_metroid.routes.kpdr.registry import KPDR_SEGMENTS
from super_metroid.source_states import get_source
# python + SNES core + RGB + ffmpeg pipe, generous.
WORKER_RSS_MIB = 256
# Refuse workers>1 when 1-min load / ncpus is at or above this (unless --force).
LOAD_FRACTION_LIMIT = 0.60
MEM_HEADROOM_MIB = 2048


@dataclass(frozen=True)
class GridTile:
    """One mosaic cell: KPDR segment + catalog pin + on-frame label."""

    segment: str
    source_id: str
    label: str


@dataclass(frozen=True)
class ParallelVerdict:
    """Host check for N simultaneous emulator processes."""

    ncpus: int
    load1: float
    mem_available_mib: int
    workers: int
    ok: bool
    reason: str
    recommended_workers: int

    def to_dict(self) -> dict[str, object]:
        return {
            "ncpus": self.ncpus,
            "load1": self.load1,
            "memAvailableMib": self.mem_available_mib,
            "workers": self.workers,
            "ok": self.ok,
            "reason": self.reason,
            "recommendedWorkers": self.recommended_workers,
        }


# Visually distinct start rooms across the living KPDR tip. Order is mosaic
# row-major (left→right, top→bottom).
DEFAULT_TILES: tuple[GridTile, ...] = (
    GridTile("ice_snake_to_ice", "post_ice_acid_to_snake_pure", "Ice Beam"),
    GridTile("ice_gate_to_acid", "post_business_to_ice_gate_wave_speed_pure", "Ice Gate"),
    GridTile("double_chamber_to_wave", "post_single_to_double_chamber_pure", "Wave Beam"),
    GridTile("speed_hall_to_speed", "post_bat_cave_to_speed_hall_pure", "Speed Booster"),
    GridTile("red_to_hellway", "post_ice_bat_to_red_pure", "Red Tower"),
    GridTile("moat_cross", "post_kihunter_to_moat_pure", "Moat"),
    GridTile("west_ocean_to_ws", "post_moat_poweron", "West Ocean"),
    GridTile("ws_entrance_to_main", "post_ws_poweron", "Wrecked Ship"),
    GridTile("phantoon_fight", "post_ws_basement_to_phantoon", "Phantoon"),
    GridTile("bubble_to_bat_cave", "post_bubble_entry_continuous", "Bubble Mountain"),
    GridTile("kraid_to_eye_return", "post_varia_continuous_to_kraid", "Kraid"),
    GridTile("warehouse_to_hijump", "post_warehouse_with_spazer_continuous", "Hi-Jump"),
    GridTile("spazer_collect", "post_spazer_entry_pure", "Spazer"),
    GridTile("below_to_bat", "post_ice_west_to_below_pure", "Bat Room"),
    GridTile("business_to_cathedral_entrance", "post_business_continuous", "Cathedral"),
    GridTile("bat_to_red", "post_ice_below_to_bat_pure", "Bat to Red"),
)


def tile_inventory(
    tiles: Sequence[GridTile] = DEFAULT_TILES,
) -> list[dict[str, object]]:
    """Resolve catalog rows + pin existence. Does not boot the emulator."""

    rows: list[dict[str, object]] = []
    for index, tile in enumerate(tiles):
        source = get_source(tile.source_id)
        play = KPDR_SEGMENTS.get(tile.segment)
        rows.append(
            {
                "index": index,
                "segment": tile.segment,
                "sourceId": tile.source_id,
                "label": tile.label,
                "roomIdHex": source.room_hex(),
                "path": str(source.path),
                "pinExists": source.path.is_file(),
                "segmentRegistered": play is not None,
            }
        )
    return rows


def mem_available_mib() -> int:
    """Linux MemAvailable in MiB; 0 if unreadable."""

    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return 0
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            kb = int(line.split()[1])
            return kb // 1024
    return 0


def probe_parallel(workers: int) -> ParallelVerdict:
    """Can this host run ``workers`` independent emulator processes now?"""

    ncpus = os.cpu_count() or 1
    load1 = float(os.getloadavg()[0])
    mem = mem_available_mib()
    recommended = max(1, min(16, max(1, ncpus - 2)))
    need_mib = workers * WORKER_RSS_MIB + MEM_HEADROOM_MIB
    load_frac = load1 / float(ncpus)
    reasons: list[str] = []
    if workers < 1:
        reasons.append("workers must be >= 1")
    if workers > 1 and load_frac >= LOAD_FRACTION_LIMIT:
        reasons.append(
            f"load1 {load1:.2f} / {ncpus} cpus = {load_frac:.2f} "
            f">= {LOAD_FRACTION_LIMIT:.2f} (pass --force if CPUs are free)"
        )
    if mem and mem < need_mib:
        reasons.append(
            f"MemAvailable {mem} MiB < {need_mib} MiB "
            f"({workers}×{WORKER_RSS_MIB} + {MEM_HEADROOM_MIB} headroom)"
        )
    ok = not reasons
    if ok:
        reason = (
            f"{workers} process lanes ok: load1 {load1:.2f}/{ncpus} cpus, "
            f"{mem} MiB available (est {need_mib} MiB)"
        )
    else:
        reason = "; ".join(reasons)
    return ParallelVerdict(
        ncpus=ncpus,
        load1=round(load1, 2),
        mem_available_mib=mem,
        workers=workers,
        ok=ok,
        reason=reason,
        recommended_workers=recommended,
    )


from super_metroid.demo.grid_record import (  # noqa: E402
    FrameBudget,
    record_play_flags,
    record_tile,
    record_tiles,
)


__all__ = [
    "CELL_H",
    "CELL_W",
    "DEFAULT_COLS",
    "DEFAULT_ROWS",
    "DEFAULT_SECONDS",
    "DEFAULT_TILES",
    "FrameBudget",
    "GridTile",
    "NTSC_FPS",
    "ParallelVerdict",
    "composite_grid",
    "label_frame",
    "mem_available_mib",
    "probe_parallel",
    "record_play_flags",
    "record_tile",
    "record_tiles",
    "tile_inventory",
    "xstack_filter",
]
