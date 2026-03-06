"""Bronze readiness and boot-probing helpers for Super Metroid.

These tools make the active route stack easier to reason about:

- `doctor` audits the current Super Metroid setup against the repo's Bronze
  expectations: key states, route segments, area maps, and SMEDIT export data.
- `boot-probe` replays deterministic button macros from a starting state
  (including `NONE`) and logs game_state / room transitions so the true-start
  bootstrap can be iterated on without ad hoc scripts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from platformer_common.auto_state import BUTTON_MAP, NavStep, parse_nav_string
from platformer_common.level_config import get_level_config
import platformer_common.levels.super_metroid  # noqa: F401
from retro_harness.env import get_available_states, make_env, save_state
from super_metroid_rl.navigation.map_data import DEFAULT_EXPORT_DIR, load_nav_graph
from super_metroid_rl.navigation.route import SPEEDRUN_ROUTE
from super_metroid_rl.navigation.trace_renderer import AREA_MAP_FILES

PROJECT_DIR = Path(__file__).resolve().parent
GAME = "SuperMetroid-Snes"
NOOP = np.zeros(12, dtype=np.int8)

EXPECTED_STATE_ROOMS: dict[str, int] = {
    "Start": 0xDF45,       # Ceres Elevator Room
    "ZebesStart": 0x91F8,  # Landing Site
    "BossTorizo": 0x9804,  # Bomb Torizo Room
}

OPTIONAL_MAP_FILES = (
    "bomb_torizo.png",
    "crateria_collision.png",
    "landing_site.png",
    "landing_site_all.png",
    "landing_site_bare.png",
    "west_ocean.png",
)

# Stock power-on bootstrap to controllable Ceres. The early title timing and
# file-menu confirmation both have a wide success window; these centered values
# are the published reproducible path.
NONE_TO_START_TITLE_WAIT = 2100
NONE_TO_START_FILE_MENU_WAIT = 120
NONE_TO_START_PROLOGUE_WAIT = 300
NONE_TO_START_PROLOGUE_SETTLE = 30
NONE_TO_START_MASH_REPEAT = 69
NONE_TO_START_MASH_STEP = NavStep(buttons=[BUTTON_MAP["A"]], hold_frames=10, wait_frames=110)

BOOT_MACRO_DESCRIPTIONS: dict[str, str] = {
    "none_to_start": "Power-on boot -> title/file menu -> controllable Ceres (`Start`).",
    "ceres_start_to_ridley_ground": (
        "Controllable Ceres (`Start`) -> fresh Ceres Ridley ground state "
        "(`0xE0B5`, 99 HP)."
    ),
    "ceres_ridley_ground_to_27hp_wait_state": (
        "Fresh Ceres Ridley ground state -> stable 27 HP ground checkpoint "
        "(`CeresRidleyGroundWait2321`)."
    ),
    "ceres_ridley_ground_27hp_to_elevator_room": (
        "27 HP Ceres Ridley ground checkpoint -> lower Ceres Elevator Room "
        "(`DF45`)."
    ),
    "ceres_pretrigger_to_elevator_room": (
        "Ceres Ridley pre-trigger -> live countdown escape -> lower Ceres Elevator "
        "Room (`DF45`)."
    ),
    "ceres_ridley_appeared_to_elevator_room": (
        "Active Ceres Ridley appearance -> countdown trigger -> lower Ceres Elevator "
        "Room (`DF45`)."
    ),
    "ceres_elevator_countdown_to_lowerledge": (
        "Lower Ceres Elevator countdown setup -> stable left lower ledge "
        "(`CeresEscapeElevatorLowerLedge`)."
    ),
    "ceres_lowerledge_to_landing_site": (
        "Published lower-ledge checkpoint -> elevator handoff -> Landing Site "
        "(`0x91F8`)."
    ),
}

BOOT_MACRO_EXPECTATIONS: dict[str, tuple[int | None, int | None]] = {
    "none_to_start": (EXPECTED_STATE_ROOMS["Start"], 8),
    "ceres_start_to_ridley_ground": (0xE0B5, 8),
    "ceres_ridley_ground_to_27hp_wait_state": (0xE0B5, 8),
    "ceres_ridley_ground_27hp_to_elevator_room": (0xDF45, 8),
    "ceres_pretrigger_to_elevator_room": (0xDF45, 8),
    "ceres_ridley_appeared_to_elevator_room": (0xDF45, 8),
    "ceres_elevator_countdown_to_lowerledge": (0xDF45, 8),
    "ceres_lowerledge_to_landing_site": (EXPECTED_STATE_ROOMS["ZebesStart"], 8),
}


@dataclass(frozen=True)
class StateAudit:
    state: str
    exists: bool
    room_id: int | None = None
    game_state: int | None = None
    health: int | None = None
    samus_x: int | None = None
    samus_y: int | None = None
    expected_room_id: int | None = None
    matches_expected_room: bool | None = None
    error: str | None = None


@dataclass(frozen=True)
class MapAudit:
    filename: str
    required: bool
    exists: bool
    width: int | None = None
    height: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class ExportAudit:
    path: str
    exists: bool
    layout: str
    has_nav_graph: bool
    has_rooms_dir: bool
    room_file_count: int
    node_count: int | None = None
    edge_count: int | None = None
    missing_route_rooms: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class SegmentAudit:
    segment_id: str
    start_state: str
    start_state_exists: bool
    entry_room_id: int
    exit_room_id: int


@dataclass(frozen=True)
class BronzeDoctorReport:
    available_state_count: int
    states: tuple[StateAudit, ...]
    maps: tuple[MapAudit, ...]
    export: ExportAudit
    segments: tuple[SegmentAudit, ...]
    bronze_runtime_ready: bool
    bronze_nav_ready: bool
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BootTransition:
    frame: int
    game_state: int | None
    room_id: int | None
    samus_x: int | None
    samus_y: int | None
    health: int | None


@dataclass(frozen=True)
class BootProbeResult:
    from_state: str
    total_frames: int
    transitions: tuple[BootTransition, ...]
    final_state: StateAudit
    saved_state_path: str | None
    expected_room_id: int | None
    expected_game_state: int | None
    success: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_state": self.from_state,
            "total_frames": self.total_frames,
            "transitions": [asdict(t) for t in self.transitions],
            "final_state": asdict(self.final_state),
            "saved_state_path": self.saved_state_path,
            "expected_room_id": self.expected_room_id,
            "expected_game_state": self.expected_game_state,
            "success": self.success,
        }


def _format_room(room_id: int | None) -> str:
    if room_id is None:
        return "?"
    return f"0x{room_id:04X}"


def _parse_room_id(text: str | None) -> int | None:
    if not text:
        return None
    text = text.strip()
    if text.startswith(("0x", "0X")):
        return int(text, 16)
    try:
        return int(text, 16)
    except ValueError:
        return int(text)


def _state_path(state_name: str) -> Path:
    return PROJECT_DIR / "custom_integrations" / GAME / f"{state_name}.state"


def inspect_state(state_name: str) -> StateAudit:
    """Load a published state and read its first live info frame."""
    path = _state_path(state_name)
    expected_room_id = EXPECTED_STATE_ROOMS.get(state_name)
    if not path.exists():
        return StateAudit(
            state=state_name,
            exists=False,
            expected_room_id=expected_room_id,
            error=f"missing state file: {path.name}",
        )

    try:
        env = make_env(GAME, state_name, PROJECT_DIR, render_mode="rgb_array")
        try:
            env.reset()
            _, _, _, _, info = env.step(NOOP)
        finally:
            env.close()
    except Exception as exc:  # pragma: no cover - exercised only with a broken runtime
        return StateAudit(
            state=state_name,
            exists=True,
            expected_room_id=expected_room_id,
            error=str(exc),
        )

    room_id = info.get("room_id")
    return StateAudit(
        state=state_name,
        exists=True,
        room_id=room_id,
        game_state=info.get("game_state"),
        health=info.get("health"),
        samus_x=info.get("samus_x"),
        samus_y=info.get("samus_y"),
        expected_room_id=expected_room_id,
        matches_expected_room=None if expected_room_id is None else room_id == expected_room_id,
    )


def inspect_maps(map_dir: str | Path | None = None) -> tuple[MapAudit, ...]:
    """Audit required area maps and optional local route maps."""
    from PIL import Image

    map_root = Path(map_dir) if map_dir is not None else PROJECT_DIR / "maps"
    entries: list[MapAudit] = []
    required_files = {filename for filename in AREA_MAP_FILES.values()}
    all_files = list(required_files) + [name for name in OPTIONAL_MAP_FILES if name not in required_files]

    for filename in all_files:
        path = map_root / filename
        required = filename in required_files
        if not path.exists():
            entries.append(MapAudit(filename=filename, required=required, exists=False))
            continue
        try:
            with Image.open(path) as img:
                width, height = img.size
            entries.append(
                MapAudit(
                    filename=filename,
                    required=required,
                    exists=True,
                    width=width,
                    height=height,
                )
            )
        except Exception as exc:  # pragma: no cover - requires corrupt image assets
            entries.append(
                MapAudit(
                    filename=filename,
                    required=required,
                    exists=True,
                    error=str(exc),
                )
            )
    return tuple(entries)


def inspect_export(export_dir: str | Path = DEFAULT_EXPORT_DIR) -> ExportAudit:
    """Inspect the directory used by SM navigation tooling."""
    export_root = Path(export_dir)
    nav_graph = export_root / "nav_graph.json"
    rooms_dir = export_root / "rooms"
    room_files = tuple(sorted(rooms_dir.glob("room_*.json"))) if rooms_dir.exists() else ()

    if not export_root.exists():
        return ExportAudit(
            path=str(export_root),
            exists=False,
            layout="missing",
            has_nav_graph=False,
            has_rooms_dir=False,
            room_file_count=0,
        )

    if nav_graph.exists() and rooms_dir.exists():
        try:
            nodes, edges = load_nav_graph(nav_graph)
            available_room_ids = {node.room_id for node in nodes}
            route_room_ids = {
                room_id
                for step in SPEEDRUN_ROUTE
                for room_id in (step.entry_room_id, step.exit_room_id)
                if room_id
            }
            missing_route_rooms = tuple(
                f"0x{room_id:04X}"
                for room_id in sorted(route_room_ids - available_room_ids)
            )
            return ExportAudit(
                path=str(export_root),
                exists=True,
                layout="smedit_export",
                has_nav_graph=True,
                has_rooms_dir=True,
                room_file_count=len(room_files),
                node_count=len(nodes),
                edge_count=len(edges),
                missing_route_rooms=missing_route_rooms,
            )
        except Exception as exc:  # pragma: no cover - requires broken export data
            return ExportAudit(
                path=str(export_root),
                exists=True,
                layout="smedit_export",
                has_nav_graph=True,
                has_rooms_dir=True,
                room_file_count=len(room_files),
                error=str(exc),
            )

    if (export_root / "region").exists() and (export_root / "connection").exists():
        return ExportAudit(
            path=str(export_root),
            exists=True,
            layout="refs_sm_json_data",
            has_nav_graph=False,
            has_rooms_dir=False,
            room_file_count=0,
            error=(
                "Looks like refs/sm-json-data, not an SMEDIT export. "
                "Navigation tools need nav_graph.json plus rooms/*.json."
            ),
        )

    return ExportAudit(
        path=str(export_root),
        exists=True,
        layout="unknown",
        has_nav_graph=nav_graph.exists(),
        has_rooms_dir=rooms_dir.exists(),
        room_file_count=len(room_files),
        error="Directory exists but does not match the expected SMEDIT export layout.",
    )


def inspect_segments() -> tuple[SegmentAudit, ...]:
    """Audit the published route segments and their start states."""
    audits: list[SegmentAudit] = []
    for step in SPEEDRUN_ROUTE:
        config = get_level_config(step.segment_id)
        audits.append(
            SegmentAudit(
                segment_id=step.segment_id,
                start_state=config.start_state,
                start_state_exists=_state_path(config.start_state).exists(),
                entry_room_id=step.entry_room_id,
                exit_room_id=step.exit_room_id,
            )
        )
    return tuple(audits)


def build_doctor_report(export_dir: str | Path = DEFAULT_EXPORT_DIR) -> BronzeDoctorReport:
    """Build a Super Metroid Bronze readiness report."""
    states = tuple(inspect_state(name) for name in EXPECTED_STATE_ROOMS)
    maps = inspect_maps()
    export = inspect_export(export_dir)
    segments = inspect_segments()

    state_ok = all(
        state.exists and state.error is None and state.matches_expected_room is not False
        for state in states
    )
    map_ok = all(entry.exists and entry.error is None for entry in maps if entry.required)
    segment_ok = all(segment.start_state_exists for segment in segments)
    nav_ok = (
        export.layout == "smedit_export"
        and export.error is None
        and not export.missing_route_rooms
    )

    notes: list[str] = []
    if export.layout == "refs_sm_json_data":
        notes.append(
            "refs/sm-json-data is reference data only. Export SMEDIT JSON with "
            "`nav_graph.json` and `rooms/*.json` for nav-path/nav-room tooling."
        )
    if not nav_ok:
        notes.append(
            "Navigation tests and nav-* commands need an SMEDIT export, usually at /tmp/sm_export."
        )
    notes.append(
        "Published stock bootstrap: `boot-probe --macro-name none_to_start` reaches controllable Ceres (`Start`) from `NONE`."
    )

    return BronzeDoctorReport(
        available_state_count=len(get_available_states(GAME, PROJECT_DIR)),
        states=states,
        maps=maps,
        export=export,
        segments=segments,
        bronze_runtime_ready=state_ok and map_ok and segment_ok,
        bronze_nav_ready=nav_ok,
        notes=tuple(notes),
    )


def format_doctor_report(report: BronzeDoctorReport) -> str:
    """Format a Bronze doctor report as readable plain text."""
    lines = [
        "Super Metroid Bronze Doctor",
        "",
        f"Published states: {report.available_state_count}",
        "",
        "Key states:",
    ]
    for state in report.states:
        status = "OK" if state.exists and state.error is None and state.matches_expected_room is not False else "FAIL"
        detail = f"room={_format_room(state.room_id)} game_state={state.game_state}"
        if state.error:
            detail = state.error
        elif state.expected_room_id is not None:
            detail += f" expected={_format_room(state.expected_room_id)}"
        lines.append(f"  {status:<4} {state.state:<11} {detail}")

    lines.extend(["", "Area maps:"])
    for entry in report.maps:
        status = "OK" if entry.exists and entry.error is None else "MISS"
        detail = f"{entry.width}x{entry.height}" if entry.width and entry.height else (entry.error or "")
        prefix = "req" if entry.required else "opt"
        lines.append(f"  {status:<4} [{prefix}] {entry.filename:<22} {detail}")

    export = report.export
    lines.extend(["", "Nav export:"])
    lines.append(f"  layout={export.layout} path={export.path}")
    lines.append(
        f"  nav_graph={export.has_nav_graph} rooms_dir={export.has_rooms_dir} room_files={export.room_file_count}"
    )
    if export.node_count is not None and export.edge_count is not None:
        lines.append(f"  nodes={export.node_count} edges={export.edge_count}")
    if export.missing_route_rooms:
        lines.append(f"  missing route rooms: {', '.join(export.missing_route_rooms)}")
    if export.error:
        lines.append(f"  note: {export.error}")

    lines.extend(["", "Route segments:"])
    for segment in report.segments:
        status = "OK" if segment.start_state_exists else "MISS"
        lines.append(
            f"  {status:<4} {segment.segment_id:<24} start={segment.start_state}"
        )

    lines.extend(
        [
            "",
            f"Bronze runtime ready: {report.bronze_runtime_ready}",
            f"Bronze nav ready:     {report.bronze_nav_ready}",
            "",
            "Notes:",
        ]
    )
    for note in report.notes:
        lines.append(f"  - {note}")
    return "\n".join(lines)


def _buttons_to_action(buttons: list[int]) -> np.ndarray:
    action = np.zeros(12, dtype=np.int8)
    for button in buttons:
        action[button] = 1
    return action


def _record_transition(
    transitions: list[BootTransition],
    frame: int,
    info: dict[str, Any],
    screenshot_dir: Path | None,
    obs: np.ndarray | None,
) -> None:
    room_id = info.get("room_id")
    transition = BootTransition(
        frame=frame,
        game_state=info.get("game_state"),
        room_id=room_id,
        samus_x=info.get("samus_x"),
        samus_y=info.get("samus_y"),
        health=info.get("health"),
    )
    transitions.append(transition)
    if screenshot_dir is not None and obs is not None:
        from PIL import Image

        screenshot_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"frame_{frame:05d}_gs_{transition.game_state}_room_{_format_room(room_id).replace('0x', '')}.png"
        )
        Image.fromarray(obs).save(screenshot_dir / filename)


def repeat_nav_steps(steps: list[NavStep], repeat: int) -> list[NavStep]:
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    return steps * repeat


def list_boot_macros() -> tuple[str, ...]:
    """Return the published named boot macros."""
    return tuple(sorted(BOOT_MACRO_DESCRIPTIONS))


def build_none_to_start_steps() -> list[NavStep]:
    """Build the published stock bootstrap from `NONE` to `Start`."""
    a_button = parse_button_combo("A")
    steps = [
        NavStep(buttons=[], hold_frames=0, wait_frames=NONE_TO_START_TITLE_WAIT),
        NavStep(buttons=a_button, hold_frames=10, wait_frames=NONE_TO_START_FILE_MENU_WAIT),
        NavStep(buttons=a_button, hold_frames=10, wait_frames=NONE_TO_START_PROLOGUE_WAIT),
        NavStep(buttons=a_button, hold_frames=10, wait_frames=NONE_TO_START_PROLOGUE_SETTLE),
    ]
    steps.extend(repeat_nav_steps([NONE_TO_START_MASH_STEP], NONE_TO_START_MASH_REPEAT))
    return steps


def build_ceres_start_to_ridley_ground_steps() -> list[NavStep]:
    """Build the solved `Start -> Ceres Ridley ground` route."""
    return parse_nav_string(
        "RIGHT+A:24:0 "
        "RIGHT:120:0 "
        "LEFT:120:0 "
        "RIGHT+B:240:60 "
        "RIGHT:24:0 "
        "RIGHT+B:24:0 "
        "RIGHT+B+A:24:0 "
        "RIGHT+A:24:0 "
        "RIGHT:24:0 "
        "RIGHT:24:0 "
        "RIGHT:24:0 "
        "RIGHT:24:0 "
        "RIGHT+B:24:12 "
        "RIGHT:24:0 "
        "WAIT:0:140 "
        "RIGHT:160:0 "
        "LEFT:120:0 "
        "RIGHT+B:96:0 "
        "WAIT:0:120 "
        "RIGHT+B:216:0 "
        "WAIT:0:150 "
        "RIGHT+B:240:0 "
        "WAIT:0:200"
    )


def build_ceres_ridley_ground_to_27hp_wait_state_steps() -> list[NavStep]:
    """Build the deterministic wait from fresh Ridley ground to the 27 HP checkpoint."""
    return parse_nav_string("WAIT:0:2321")


def build_ceres_ridley_ground_27hp_to_elevator_room_steps() -> list[NavStep]:
    """Build the solved 27 HP checkpoint -> lower `DF45` escape."""
    return parse_nav_string(
        "WAIT:0:540 "
        "LEFT+A:40:0 "
        "LEFT:1000:0 "
        "A:16:0 "
        "RIGHT+A:124:0 "
        "LEFT+A:60:0 "
        "LEFT:320:0 "
        "LEFT+A:40:0 "
        "LEFT:380:0"
    )


def build_ceres_pretrigger_to_elevator_room_steps() -> list[NavStep]:
    """Build the solved countdown escape from `CeresRidleyPreTrigger` to `DF45`."""
    return parse_nav_string(
        "LEFT+A:40:0 "
        "LEFT:1000:0 "
        "A:16:0 "
        "RIGHT+A:124:0 "
        "LEFT+A:60:0 "
        "LEFT:320:0 "
        "LEFT+A:40:0 "
        "LEFT:380:0"
    )


def build_ceres_ridley_appeared_to_elevator_room_steps() -> list[NavStep]:
    """Build the solved live Ridley->countdown->DF45 route from `CeresRidleyAppeared`."""
    return parse_nav_string(
        "WAIT:0:1888 "
        "LEFT+A:40:0 "
        "LEFT:1000:0 "
        "A:16:0 "
        "RIGHT+A:124:0 "
        "LEFT+A:60:0 "
        "LEFT:320:0 "
        "LEFT+A:40:0 "
        "LEFT:380:0"
    )


def build_ceres_elevator_countdown_to_lowerledge_steps() -> list[NavStep]:
    """Build the direct lower `DF45` countdown -> lower-ledge setup."""
    return parse_nav_string("LEFT+A:70:0")


def build_ceres_lowerledge_to_landing_site_steps() -> list[NavStep]:
    """Build the solved lower-ledge climb through the Ceres elevator handoff."""
    return parse_nav_string(
        "LEFT+A:94:0 "
        "RIGHT+A:80:0 "
        "LEFT+A:80:0 "
        "RIGHT+A:80:0 "
        "RIGHT+A:100:0 "
        "LEFT+A:70:0 "
        "RIGHT+A:90:0 "
        "LEFT+A:50:0 "
        "WAIT:0:2500"
    )


def get_boot_macro_steps(name: str) -> list[NavStep]:
    """Expand a named boot macro into concrete navigation steps."""
    if name == "none_to_start":
        return build_none_to_start_steps()
    if name == "ceres_start_to_ridley_ground":
        return build_ceres_start_to_ridley_ground_steps()
    if name == "ceres_ridley_ground_to_27hp_wait_state":
        return build_ceres_ridley_ground_to_27hp_wait_state_steps()
    if name == "ceres_ridley_ground_27hp_to_elevator_room":
        return build_ceres_ridley_ground_27hp_to_elevator_room_steps()
    if name == "ceres_pretrigger_to_elevator_room":
        return build_ceres_pretrigger_to_elevator_room_steps()
    if name == "ceres_ridley_appeared_to_elevator_room":
        return build_ceres_ridley_appeared_to_elevator_room_steps()
    if name == "ceres_elevator_countdown_to_lowerledge":
        return build_ceres_elevator_countdown_to_lowerledge_steps()
    if name == "ceres_lowerledge_to_landing_site":
        return build_ceres_lowerledge_to_landing_site_steps()
    available = ", ".join(list_boot_macros())
    raise KeyError(f"Unknown boot macro '{name}'. Available: {available}")


def get_boot_macro_expectation(name: str) -> tuple[int | None, int | None]:
    """Return the default expected room/game-state for a named macro."""
    if name not in BOOT_MACRO_EXPECTATIONS:
        available = ", ".join(list_boot_macros())
        raise KeyError(f"Unknown boot macro '{name}'. Available: {available}")
    return BOOT_MACRO_EXPECTATIONS[name]


def run_boot_probe(
    nav: str | None = None,
    *,
    steps: list[NavStep] | None = None,
    from_state: str = "NONE",
    repeat: int = 1,
    settle_frames: int = 0,
    screenshot_dir: str | Path | None = None,
    save_name: str | None = None,
    expected_room_id: int | None = None,
    expected_game_state: int | None = None,
) -> BootProbeResult:
    """Replay a macro from a starting state and log state/room transitions."""
    if nav is not None and steps is not None:
        raise ValueError("Provide either `nav` or `steps`, not both.")
    if steps is None:
        if not nav:
            raise ValueError("A boot probe needs either a nav string or named steps.")
        steps = repeat_nav_steps(parse_nav_string(nav), repeat)
    elif repeat != 1:
        raise ValueError("repeat applies only to nav strings; prebuilt step lists must be passed once.")

    screenshot_path = Path(screenshot_dir) if screenshot_dir is not None else None

    env = make_env(GAME, from_state, PROJECT_DIR, render_mode="rgb_array")
    saved_state_path: str | None = None
    transitions: list[BootTransition] = []
    total_frames = 0
    last_key: tuple[int | None, int | None] | None = None
    last_obs: np.ndarray | None = None
    final_info: dict[str, Any] = {}

    try:
        env.reset()

        def step_and_record(action: np.ndarray) -> None:
            nonlocal total_frames, last_key, last_obs, final_info
            obs, _, _, _, info = env.step(action)
            total_frames += 1
            key = (info.get("game_state"), info.get("room_id"))
            if key != last_key:
                _record_transition(transitions, total_frames, info, screenshot_path, obs)
                last_key = key
            last_obs = obs
            final_info = info

        for step in steps:
            action = _buttons_to_action(step.buttons)
            for _ in range(step.hold_frames):
                step_and_record(action)
            for _ in range(step.wait_frames):
                step_and_record(NOOP)

        for _ in range(settle_frames):
            step_and_record(NOOP)

        if screenshot_path is not None and last_obs is not None:
            from PIL import Image

            screenshot_path.mkdir(parents=True, exist_ok=True)
            final_room = _format_room(final_info.get("room_id")).replace("0x", "")
            final_name = f"final_gs_{final_info.get('game_state')}_room_{final_room}.png"
            Image.fromarray(last_obs).save(screenshot_path / final_name)

        if save_name:
            saved_state_path = str(save_state(env, PROJECT_DIR, GAME, save_name))
    finally:
        env.close()

    final_state = StateAudit(
        state=save_name or from_state,
        exists=True,
        room_id=final_info.get("room_id"),
        game_state=final_info.get("game_state"),
        health=final_info.get("health"),
        samus_x=final_info.get("samus_x"),
        samus_y=final_info.get("samus_y"),
        expected_room_id=expected_room_id,
        matches_expected_room=None if expected_room_id is None else final_info.get("room_id") == expected_room_id,
    )

    success: bool | None = None
    if expected_room_id is not None or expected_game_state is not None:
        success = True
        if expected_room_id is not None and final_state.room_id != expected_room_id:
            success = False
        if expected_game_state is not None and final_state.game_state != expected_game_state:
            success = False

    return BootProbeResult(
        from_state=from_state,
        total_frames=total_frames,
        transitions=tuple(transitions),
        final_state=final_state,
        saved_state_path=saved_state_path,
        expected_room_id=expected_room_id,
        expected_game_state=expected_game_state,
        success=success,
    )


def format_boot_probe(result: BootProbeResult) -> str:
    """Format a boot-probe result as readable plain text."""
    lines = [
        f"Boot probe from {result.from_state}",
        f"Total frames: {result.total_frames}",
        "",
        "Transitions:",
    ]
    for transition in result.transitions:
        lines.append(
            "  "
            f"f={transition.frame:<5d} "
            f"game_state={transition.game_state!s:<4} "
            f"room={_format_room(transition.room_id):<8} "
            f"samus=({transition.samus_x},{transition.samus_y}) "
            f"health={transition.health}"
        )
    lines.extend(
        [
            "",
            "Final:",
            "  "
            f"room={_format_room(result.final_state.room_id)} "
            f"game_state={result.final_state.game_state} "
            f"samus=({result.final_state.samus_x},{result.final_state.samus_y}) "
            f"health={result.final_state.health}",
        ]
    )
    if result.expected_room_id is not None or result.expected_game_state is not None:
        lines.append(
            "  "
            f"expected_room={_format_room(result.expected_room_id)} "
            f"expected_game_state={result.expected_game_state} "
            f"success={result.success}"
        )
    if result.saved_state_path:
        lines.append(f"  saved_state={result.saved_state_path}")
    return "\n".join(lines)


def doctor_cli(args: Any) -> int:
    report = build_doctor_report(args.export_dir)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_doctor_report(report))
    return 0 if report.bronze_runtime_ready else 1


def boot_probe_cli(args: Any) -> int:
    if bool(args.nav) == bool(args.macro_name):
        print("error: provide exactly one of --nav or --macro-name")
        return 2

    expected_room_id = _parse_room_id(args.expected_room)
    expected_game_state = args.expected_game_state
    nav = args.nav
    steps: list[NavStep] | None = None

    if args.macro_name:
        if args.repeat != 1:
            print("error: --repeat is only valid with --nav, not --macro-name")
            return 2
        steps = get_boot_macro_steps(args.macro_name)
        macro_room, macro_game_state = get_boot_macro_expectation(args.macro_name)
        if expected_room_id is None:
            expected_room_id = macro_room
        if expected_game_state is None:
            expected_game_state = macro_game_state
        nav = None

    result = run_boot_probe(
        nav,
        steps=steps,
        from_state=args.from_state,
        repeat=args.repeat,
        settle_frames=args.settle,
        screenshot_dir=args.screenshot_dir,
        save_name=args.save_name,
        expected_room_id=expected_room_id,
        expected_game_state=expected_game_state,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(format_boot_probe(result))
    if result.success is False:
        return 1
    return 0


def parse_button_combo(text: str) -> list[int]:
    """Parse a button combo like 'START+DOWN' into button indices."""
    buttons: list[int] = []
    for name in text.upper().split("+"):
        buttons.append(BUTTON_MAP[name])
    return buttons
