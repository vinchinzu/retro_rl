"""Save-state teleport and deterministic room-clear replay harness."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from retro_harness.actions import buttons, idle_action
from retro_harness.env import make_env, read_state_bytes, write_state_bytes
from super_metroid.paths import (
    GAME,
    GAME_DIR,
    ROOM_PROBLEMS_PATH,
)
from super_metroid.ram import GameplayPhase, parse_state
from super_metroid.rooms.room_graph import load_problem_catalog, problem_by_id


@dataclass(frozen=True)
class PolicySpan:
    buttons: tuple[str, ...]
    frames: int
    label: str


_AMMO_ITEM_FIELDS = {
    "missile": ("max_missiles", 5),
    "super missile": ("max_super_missiles", 5),
    "power bomb": ("max_power_bombs", 5),
}
_BEAM_ITEMS = {
    "charge beam",
    "ice beam",
    "plasma beam",
    "spazer",
    "wave beam",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expand_steps(
    steps: Sequence[Mapping[str, Any]],
    *,
    prefix: str = "",
) -> Iterator[PolicySpan]:
    for index, step in enumerate(steps):
        label = str(step.get("label", f"{prefix}step_{index:02d}"))
        if "steps" in step:
            repeat = int(step.get("repeat", 1))
            for repetition in range(repeat):
                yield from _expand_steps(
                    step["steps"],
                    prefix=f"{label}_{repetition:02d}_",
                )
            continue
        names = tuple(str(name).upper() for name in step.get("buttons", []))
        frames = int(step["frames"])
        if frames <= 0:
            raise ValueError(f"policy span frames must be positive: {step}")
        yield PolicySpan(names, frames, label)


def load_room_policy(path: Path) -> tuple[dict[str, object], tuple[PolicySpan, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("steps"), list):
        raise ValueError(f"room policy must contain a steps list: {path}")
    return payload, tuple(_expand_steps(payload["steps"]))


def scaffold_room_policy(
    problem_id: str,
    *,
    catalog_path: Path = ROOM_PROBLEMS_PATH,
    output_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Write an explicitly unverified, door-oriented starter policy."""
    catalog = load_problem_catalog(catalog_path)
    problem = problem_by_id(catalog, problem_id)
    _, default_policy_path, _ = _problem_paths(problem)
    path = output_path or default_policy_path
    if path.exists() and not overwrite:
        raise FileExistsError(f"policy already exists: {path}")

    exit_data = problem.get("exit")
    endpoint = exit_data.get("endpoint") if isinstance(exit_data, dict) else None
    orientation = (
        str(endpoint.get("orientation", "")).lower()
        if isinstance(endpoint, dict)
        else ""
    )
    direction = {
        "left": "LEFT",
        "right": "RIGHT",
        "up": "UP",
        "down": "DOWN",
    }.get(orientation)
    travel_buttons = [direction, "B"] if direction is not None else []
    door_buttons = [direction, "B", "X"] if direction is not None else []
    payload = {
        "schemaVersion": 1,
        "problemId": problem_id,
        "status": "generated_unverified",
        "description": (
            f"Starter policy for {problem['roomName']} "
            f"({problem['roomIdHex']}); tune against the captured entry state."
        ),
        "planning": {
            "objective": problem["objective"],
            "entry": problem["entry"],
            "exit": problem["exit"],
            "staticPlan": problem["staticPlan"],
        },
        "steps": [
            {"label": "entry_settle", "buttons": [], "frames": 30},
            {
                "label": "coarse_exit_approach",
                "buttons": travel_buttons,
                "frames": 180,
            },
            {
                "label": "open_and_enter_exit",
                "buttons": door_buttons,
                "frames": 120,
            },
        ],
        "acceptanceWarning": (
            "Generated scaffold only. It must cross and settle in emulator "
            "before its status can become verified_development_state."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {
        "problemId": problem_id,
        "policyPath": str(path.resolve()),
        "status": payload["status"],
        "orientationHint": orientation or None,
    }


def _problem_paths(
    problem: Mapping[str, object],
) -> tuple[Path, Path, Path]:
    practice = problem["practice"]
    return (
        GAME_DIR / str(practice["stateFile"]),
        GAME_DIR / str(practice["policyFile"]),
        GAME_DIR / str(practice["reportFile"]),
    )


def _objective_progress_failure(
    problem: Mapping[str, object],
    start: Any,
    final: Any,
) -> str | None:
    objective = str(problem.get("objective", ""))
    if not objective.startswith("collect"):
        return None
    item_names = [
        str(item["name"]).split(" (", 1)[0].lower()
        for item in problem.get("items", [])
        if isinstance(item, dict) and "name" in item
    ]
    counts = Counter(item_names)
    failures: list[str] = []
    for name, (field, increment) in _AMMO_ITEM_FIELDS.items():
        required = counts[name] * increment
        if required and getattr(final, field) - getattr(start, field) < required:
            failures.append(f"{field} did not increase by {required}")
    energy_required = counts["energy tank"] * 100
    if energy_required and final.max_health - start.max_health < energy_required:
        failures.append(f"max_health did not increase by {energy_required}")
    reserve_required = counts["reserve tank"] * 100
    if (
        reserve_required
        and final.max_reserve_health - start.max_reserve_health < reserve_required
    ):
        failures.append(f"max_reserve_health did not increase by {reserve_required}")
    beam_items = set(item_names) & _BEAM_ITEMS
    if beam_items and final.collected_beams == start.collected_beams:
        failures.append("collected_beams did not change")
    equipment_items = set(item_names) - {
        *_AMMO_ITEM_FIELDS,
        "energy tank",
        "reserve tank",
        *_BEAM_ITEMS,
    }
    if equipment_items and final.collected_items == start.collected_items:
        failures.append("collected_items did not change")
    if failures:
        return "; ".join(failures)
    return None


def capture_room_state(
    problem_id: str,
    source_state: Path,
    *,
    catalog_path: Path = ROOM_PROBLEMS_PATH,
) -> dict[str, object]:
    """Validate and import an existing emulator snapshot for one problem."""
    catalog = load_problem_catalog(catalog_path)
    problem = problem_by_id(catalog, problem_id)
    state_path, _, _ = _problem_paths(problem)
    source_state = source_state.expanduser().resolve()

    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        env.reset()
        env.em.set_state(read_state_bytes(source_state))
        state = parse_state(env.get_ram())
        if state.room_id != int(problem["roomId"]):
            raise ValueError(
                f"state room 0x{state.room_id:04X} does not match "
                f"{problem['roomIdHex']}"
            )
        if state.phase is not GameplayPhase.ORDINARY_GAMEPLAY:
            raise ValueError(f"state is not ordinary gameplay: {state.phase.value}")
        write_state_bytes(state_path, env.em.get_state())
    finally:
        env.close()
    provenance = {
        "schemaVersion": 1,
        "problemId": problem_id,
        "sourcePath": str(source_state),
        "sourceSha256": _sha256(source_state),
        "statePath": str(state_path.resolve()),
        "stateSha256": _sha256(state_path),
        "roomId": state.room_id,
        "roomIdHex": f"0x{state.room_id:04X}",
        "expectedEntrySourceRoomIdHex": (
            problem["entry"]["sourceRoomIdHex"]
            if problem["entry"] is not None
            else None
        ),
        "entryState": state.to_dict(),
        "entrySourceValidation": (
            "The settled state exposes room/phase/position/inventory but not "
            "the prior room; natural-entry promotion remains a separate step."
        ),
        "developmentOnly": True,
        "acceptanceWarning": (
            "This imported state is a teleport fixture for isolated room "
            "development, never continuous-run evidence."
        ),
        "capturedAt": datetime.now(timezone.utc).isoformat(),
    }
    provenance_path = state_path.with_suffix(".provenance.json")
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    return {**provenance, "provenancePath": str(provenance_path.resolve())}


def teleport_room_problem(
    problem_id: str,
    *,
    catalog_path: Path = ROOM_PROBLEMS_PATH,
    settle_frames: int = 1,
    screenshot_path: Path | None = None,
) -> dict[str, object]:
    """Load a problem snapshot and return its verified start state."""
    catalog = load_problem_catalog(catalog_path)
    problem = problem_by_id(catalog, problem_id)
    state_path, _, _ = _problem_paths(problem)
    if not state_path.is_file():
        raise FileNotFoundError(f"missing problem state: {state_path}")

    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        observation, _ = env.reset()
        env.em.set_state(read_state_bytes(state_path))
        for _ in range(max(1, settle_frames)):
            observation, *_ = env.step(idle_action())
        state = parse_state(env.get_ram(), frame=max(1, settle_frames))
        if state.room_id != int(problem["roomId"]):
            raise RuntimeError(
                f"teleport landed in 0x{state.room_id:04X}, "
                f"expected {problem['roomIdHex']}"
            )
        if screenshot_path is not None:
            import cv2

            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(screenshot_path),
                cv2.cvtColor(observation, cv2.COLOR_RGB2BGR),
            )
    finally:
        env.close()
    return {
        "problemId": problem_id,
        "statePath": str(state_path.resolve()),
        "stateSha256": _sha256(state_path),
        "state": state.to_dict(),
        "screenshotPath": (
            str(screenshot_path.resolve()) if screenshot_path is not None else None
        ),
    }


def run_room_problem(
    problem_id: str,
    *,
    catalog_path: Path = ROOM_PROBLEMS_PATH,
    report_path: Path | None = None,
    settle_timeout: int = 300,
) -> dict[str, object]:
    """Teleport, replay a compact policy, and verify a natural room exit."""
    catalog = load_problem_catalog(catalog_path)
    problem = problem_by_id(catalog, problem_id)
    state_path, policy_path, default_report_path = _problem_paths(problem)
    if not state_path.is_file():
        raise FileNotFoundError(f"missing problem state: {state_path}")
    if not policy_path.is_file():
        raise FileNotFoundError(f"missing problem policy: {policy_path}")
    if problem["exit"] is None:
        raise ValueError(f"problem has no target exit: {problem_id}")
    target_room_id = int(problem["exit"]["targetRoomId"])
    policy, spans = load_room_policy(policy_path)
    if policy.get("problemId") != problem_id:
        raise ValueError(
            f"policy problemId {policy.get('problemId')!r} != {problem_id!r}"
        )

    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    frame = 0
    crossing_frame: int | None = None
    settled_frame: int | None = None
    action_counts: dict[str, int] = {}
    failure: str | None = None
    start = None
    state = None
    try:
        env.reset()
        env.em.set_state(read_state_bytes(state_path))
        start = parse_state(env.get_ram(), frame=0)
        if start.room_id != int(problem["roomId"]):
            raise RuntimeError(
                f"problem state begins in 0x{start.room_id:04X}, "
                f"expected {problem['roomIdHex']}"
            )
        for span in spans:
            action = buttons(*span.buttons) if span.buttons else idle_action()
            for _ in range(span.frames):
                env.step(action)
                frame += 1
                action_counts[span.label] = action_counts.get(span.label, 0) + 1
                state = parse_state(env.get_ram(), frame=frame)
                if state.room_id == target_room_id:
                    crossing_frame = frame
                    break
            if crossing_frame is not None:
                break
        if crossing_frame is None:
            state = parse_state(env.get_ram(), frame=frame)
            failure = (
                f"policy ended in 0x{state.room_id:04X}; "
                f"expected 0x{target_room_id:04X}"
            )
        else:
            for _ in range(settle_timeout):
                state = parse_state(env.get_ram(), frame=frame)
                if (
                    state.room_id == target_room_id
                    and state.phase is GameplayPhase.ORDINARY_GAMEPLAY
                ):
                    settled_frame = frame
                    break
                env.step(idle_action())
                frame += 1
            else:
                failure = (
                    f"target 0x{target_room_id:04X} did not settle to "
                    f"ordinary gameplay in {settle_timeout} frames"
                )
            state = parse_state(env.get_ram(), frame=frame)
            if failure is None:
                objective_failure = _objective_progress_failure(
                    problem,
                    start,
                    state,
                )
                if objective_failure is not None:
                    failure = f"room objective incomplete: {objective_failure}"
    except Exception as exc:
        try:
            state = parse_state(env.get_ram(), frame=frame)
        except Exception:
            state = None
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()

    success = failure is None
    report = {
        "schemaVersion": 1,
        "problemId": problem_id,
        "success": success,
        "failure": failure,
        "startRoomId": int(problem["roomId"]),
        "startRoomIdHex": problem["roomIdHex"],
        "targetRoomId": target_room_id,
        "targetRoomIdHex": f"0x{target_room_id:04X}",
        "startState": start.to_dict() if start is not None else None,
        "crossingFrame": crossing_frame,
        "settledFrame": settled_frame,
        "totalFrames": frame,
        "finalState": state.to_dict() if state is not None else None,
        "objectiveVerification": {
            "objective": problem["objective"],
            "status": (
                "passed"
                if success
                else (
                    "failed"
                    if failure is not None
                    and failure.startswith("room objective incomplete:")
                    else "not_reached"
                )
            ),
        },
        "actionFrames": action_counts,
        "state": {
            "path": str(state_path.resolve()),
            "sha256": _sha256(state_path),
        },
        "policy": {
            "path": str(policy_path.resolve()),
            "sha256": _sha256(policy_path),
        },
        "developmentOnly": True,
        "acceptanceWarning": (
            "This room clear begins from a development save state. It proves "
            "the isolated policy/setup, not continuous full-run progression."
        ),
        "generatedAt": datetime.now(timezone.utc).isoformat(),
    }
    output = report_path or default_report_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def ready_problem_ids(
    *,
    catalog_path: Path = ROOM_PROBLEMS_PATH,
) -> list[str]:
    catalog = load_problem_catalog(catalog_path)
    ready = []
    for problem in catalog["problems"]:
        state_path, policy_path, _ = _problem_paths(problem)
        if not state_path.is_file() or not policy_path.is_file():
            continue
        policy, _ = load_room_policy(policy_path)
        if policy.get("status") == "verified_development_state":
            ready.append(str(problem["problemId"]))
    return ready
