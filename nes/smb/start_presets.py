"""Human-record start presets for ``./play smb``.

Short names resolve to power-on, an integration state, or a durable stage pin
written by a previous take (same ``--name``). Kept out of ``scripts/play.py``
so the recorder CLI stays small.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from smb.paths import INTEGRATION_V0_DIR, RECORDINGS_DIR
from smb.routes import ExitRoute, get_route

POWER_ON_STARTS = frozenset({"beginning", "full", "power-on", "poweron", "start"})
HUMAN_DIR = RECORDINGS_DIR / "human"
DEFAULT_TASK_NAME = "all_exits_v1"
DEFAULT_ROUTE_ID = "all_exits"

STAGE_IDS: tuple[str, ...] = tuple(
    f"{world}-{level}" for world in range(1, 9) for level in range(1, 5)
)


@dataclass(frozen=True)
class ResolvedStart:
    """Where a record session boots."""

    key: str
    kind: str  # power_on | state
    path: Path | None
    label: str
    blurb: str
    route_index: int = 0


def is_power_on(arg: str) -> bool:
    return arg.strip().lower() in POWER_ON_STARTS


def normalize_stage_id(arg: str) -> str | None:
    """Return ``W-L`` if *arg* names a main-game stage, else None."""
    raw = arg.strip().lower().replace("_", "-")
    if raw.startswith("smb-"):
        raw = raw[4:]
    if raw not in {s.lower() for s in STAGE_IDS}:
        return None
    world, level = (int(part) for part in raw.split("-", 1))
    return f"{world}-{level}"


def pins_dir(task_name: str, *, out_dir: Path | None = None) -> Path:
    root = Path(out_dir) if out_dir is not None else HUMAN_DIR
    return root / f"{task_name}_pins"


def pin_state_path(task_name: str, stage_id: str, *, out_dir: Path | None = None) -> Path:
    return pins_dir(task_name, out_dir=out_dir) / f"{stage_id}.state"


def pin_meta_path(task_name: str, stage_id: str, *, out_dir: Path | None = None) -> Path:
    return pins_dir(task_name, out_dir=out_dir) / f"{stage_id}.json"


def route_index_for(route: ExitRoute, stage_id: str) -> int:
    for idx, exit_seg in enumerate(route.exits):
        if exit_seg.exit_id == stage_id:
            return idx
    raise KeyError(f"{stage_id} is not on route {route.route_id}")


def _integration_state(name: str) -> Path:
    stem = name if name.endswith(".state") else f"{name}.state"
    return INTEGRATION_V0_DIR / stem


def resolve_start(
    arg: str,
    *,
    task_name: str = DEFAULT_TASK_NAME,
    route: ExitRoute | None = None,
    out_dir: Path | None = None,
) -> ResolvedStart:
    """Resolve a ``--from`` token to power-on or a loadable state file."""
    key = arg.strip()
    lowered = key.lower()
    route = route or get_route(DEFAULT_ROUTE_ID)
    human = Path(out_dir) if out_dir is not None else HUMAN_DIR

    if is_power_on(lowered):
        return ResolvedStart(
            key=key,
            kind="power_on",
            path=None,
            label="power_on",
            blurb="true power-on (title → 1-1 → all 32 exits)",
            route_index=0,
        )

    if lowered in {"resume", "end", "tip"}:
        candidates = [
            human / f"{task_name}_end.state",
            pins_dir(task_name, out_dir=human) / "resume.state",
        ]
        for path in candidates:
            if path.is_file():
                return ResolvedStart(
                    key=key,
                    kind="state",
                    path=path.resolve(),
                    label=str(path.relative_to(human)) if path.is_relative_to(human) else str(path),
                    blurb=f"resume last F5 / pin for {task_name}",
                    route_index=_resume_route_index(task_name, route, out_dir=human),
                )
        raise FileNotFoundError(
            f"No resume pin for {task_name!r}. Record from start first "
            f"(./play smb) or pass a stage id."
        )

    stage = normalize_stage_id(lowered)
    if stage is not None:
        pin = pin_state_path(task_name, stage, out_dir=human)
        fallback = _integration_state(f"Level{stage.replace('-', '_')}")
        # 1-1 also has the historical Level1_1 filename.
        if not fallback.is_file() and stage == "1-1":
            fallback = _integration_state("Level1_1")
        path = pin if pin.is_file() else fallback
        if not path.is_file():
            raise FileNotFoundError(
                f"No pin for {stage} under {pins_dir(task_name, out_dir=human)} "
                f"and no integration state {fallback.name}. "
                f"Play earlier stages first (./play smb) or start at power-on."
            )
        via = "pin" if path == pin else "integration"
        return ResolvedStart(
            key=key,
            kind="state",
            path=path.resolve(),
            label=f"{stage} ({via})",
            blurb=f"{stage} {via} — continue all-exits from this stage",
            route_index=route_index_for(route, stage),
        )

    raw = Path(key)
    candidates = [
        raw,
        human / key,
        human / f"{key}.state",
        _integration_state(key),
        INTEGRATION_V0_DIR / key,
    ]
    for path in candidates:
        if path.is_file():
            return ResolvedStart(
                key=key,
                kind="state",
                path=path.resolve(),
                label=path.name,
                blurb=f"explicit state {path}",
                route_index=0,
            )
    raise FileNotFoundError(f"Start state not found: {arg}")


def _resume_route_index(task_name: str, route: ExitRoute, *, out_dir: Path) -> int:
    """Best-effort: last written stage pin that is still on the route."""
    root = pins_dir(task_name, out_dir=out_dir)
    if not root.is_dir():
        return 0
    last = 0
    for idx, exit_seg in enumerate(route.exits):
        if (root / f"{exit_seg.exit_id}.state").is_file():
            last = idx
    return last


def list_start_presets(
    *,
    task_name: str = DEFAULT_TASK_NAME,
    route: ExitRoute | None = None,
    out_dir: Path | None = None,
) -> list[tuple[str, str, str]]:
    """Return ``(key, mark, blurb)`` rows for ``--list``."""
    route = route or get_route(DEFAULT_ROUTE_ID)
    human = Path(out_dir) if out_dir is not None else HUMAN_DIR
    rows: list[tuple[str, str, str]] = [
        (
            "start",
            "OK",
            "power-on (no savestate) — title → 1-1 → 32 exits",
        ),
    ]
    for stage in STAGE_IDS:
        pin = pin_state_path(task_name, stage, out_dir=human)
        fallback = _integration_state(f"Level{stage.replace('-', '_')}")
        if not fallback.is_file() and stage == "1-1":
            fallback = _integration_state("Level1_1")
        if pin.is_file():
            mark = "OK"
            extra = str(pin)
        elif fallback.is_file():
            mark = "OK"
            extra = f"{fallback.name} (integration)"
        else:
            mark = "MISSING"
            extra = str(pin)
        rows.append((stage, mark, extra))
    resume = human / f"{task_name}_end.state"
    rows.append(
        (
            "resume",
            "OK" if resume.is_file() else "MISSING",
            str(resume),
        )
    )
    return rows
