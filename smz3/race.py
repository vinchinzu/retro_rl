"""Race harness scaffold: two bots, one seed, video + room-timeout stop.

Not yet a live dual-emulator runner — this package records the contract so
later work plugs into a stable API without reshaping the tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smz3.paths import RECORDINGS_DIR, ROOM_TIMEOUT_MULTIPLIER, SEEDS_DIR
from smz3.room_timeout import RoomTimeoutWatchdog
from smz3.seed import SeedPackage, load_seed
from smz3.world import DualWorldSessionHooks


@dataclass
class BotRunResult:
    bot_id: str
    status: str  # finished | game_over_timeout | error | not_started
    frames: int = 0
    detail: str = ""
    video_path: str | None = None
    timeout_report: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bot_id": self.bot_id,
            "status": self.status,
            "frames": self.frames,
            "detail": self.detail,
            "video_path": self.video_path,
            "timeout_report": self.timeout_report,
        }


@dataclass
class RacePlan:
    """Describe a race without executing it."""

    seed: SeedPackage
    bot_count: int = 2
    room_timeout_multiplier: float = ROOM_TIMEOUT_MULTIPLIER
    record_video: bool = True
    output_dir: Path = field(default_factory=lambda: RECORDINGS_DIR / "races")

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_name": self.seed.name,
            "seed_number": self.seed.seed_number,
            "seed_url": self.seed.url,
            "hash_code": self.seed.hash_code,
            "rom_path": str(self.seed.rom_path) if self.seed.rom_path else None,
            "bot_count": self.bot_count,
            "room_timeout_multiplier": self.room_timeout_multiplier,
            "record_video": self.record_video,
            "output_dir": str(self.output_dir),
            "hooks": DualWorldSessionHooks(
                seed_name=self.seed.name,
                record_video=self.record_video,
                room_timeout_multiplier=self.room_timeout_multiplier,
                bots=self.bot_count,
            ).plan(),
        }


def plan_race(
    seed: str | Path | SeedPackage,
    *,
    bot_count: int = 2,
    record_video: bool = True,
) -> RacePlan:
    if isinstance(seed, SeedPackage):
        pkg = seed
    else:
        path = Path(seed)
        if not path.is_dir():
            path = SEEDS_DIR / str(seed)
        pkg = load_seed(path)
    return RacePlan(
        seed=pkg,
        bot_count=bot_count,
        record_video=record_video,
        output_dir=RECORDINGS_DIR / "races" / pkg.name,
    )


def make_bot_watchdog(
    *,
    baselines: dict[str, int] | None = None,
    multiplier: float = ROOM_TIMEOUT_MULTIPLIER,
) -> RoomTimeoutWatchdog:
    """Factory for per-bot room timeout watchdogs."""
    if baselines:
        return RoomTimeoutWatchdog.from_mapping(baselines, multiplier=multiplier)
    return RoomTimeoutWatchdog(multiplier=multiplier)
