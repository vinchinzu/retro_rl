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
from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.paths import (
    GAME,
    GAME_DIR,
    ROOM_PROBLEMS_PATH,
)
from super_metroid.ram import GameplayPhase, parse_state
from super_metroid.rooms.room_graph import load_problem_catalog, problem_by_id
from super_metroid.rooms.segment_contract import (
    EntryContract,
    opposite_direction,
    orientation_to_direction,
    segment_boundary_dict,
)


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


def _scaffold_frame_budget(problem: Mapping[str, object]) -> dict[str, int]:
    """Derive coarse policy frame budgets from static plan length.

    Avoids orientation-specific magic numbers; longer air paths get more
    approach time. Door-shell enter budget stays constant.
    """
    plan = problem.get("staticPlan") if isinstance(problem.get("staticPlan"), dict) else {}
    path_blocks = int(plan.get("pathBlocks") or 0) if plan else 0
    # ~6 frames per block for a short into-room push; clamp for stations.
    into_frames = max(30, min(90, 30 + path_blocks * 4))
    approach_frames = max(60, min(180, 50 + path_blocks * 8))
    enter_frames = 110  # door shell push (blue door open + cross)
    traverse_approach = max(80, min(220, 80 + path_blocks * 6))
    return {
        "into": into_frames,
        "approach": approach_frames,
        "enter": enter_frames,
        "traverse_approach": traverse_approach,
    }


def scaffold_room_policy(
    problem_id: str,
    *,
    catalog_path: Path = ROOM_PROBLEMS_PATH,
    output_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Write an explicitly unverified, door-oriented starter policy.

    Assumes a **doorway-natural** entry fixture (just inside the entry door).
    For same-door return rooms (save/map/refill), steps walk deeper into the
    room then reverse toward the entry/exit door. Frame budgets come from
    ``staticPlan.pathBlocks`` when available.
    """
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
    direction = orientation_to_direction(orientation)
    opposite = opposite_direction(direction)
    travel_buttons = [direction, "B"] if direction is not None else []
    door_buttons = [direction, "X"] if direction is not None else []
    approach_buttons = (
        [direction, "A", "B"] if direction in {"LEFT", "RIGHT"} else travel_buttons
    )
    into_buttons = [opposite, "B"] if opposite else []
    budget = _scaffold_frame_budget(problem)
    contract = EntryContract.from_problem(problem)

    objective = str(problem.get("objective", ""))
    is_collect = objective.startswith("collect")

    if contract.same_door_return and into_buttons and approach_buttons:
        # Collect rooms need extra into-room time + fanfare hold before reverse.
        into_frames = budget["into"] + (40 if is_collect else 0)
        steps: list[dict[str, object]] = [
            {"label": "entry_settle", "buttons": [], "frames": 20},
            {
                "label": "deeper_into_room",
                "buttons": into_buttons,
                "frames": into_frames,
            },
        ]
        if is_collect:
            steps.append(
                {"label": "item_fanfare_wait", "buttons": [], "frames": 360}
            )
        else:
            steps.append({"label": "turn_settle", "buttons": [], "frames": 10})
        steps.extend(
            [
                {
                    "label": "approach_exit_door",
                    "buttons": approach_buttons,
                    "frames": budget["approach"],
                },
                {
                    "label": "open_exit_door",
                    "buttons": door_buttons,
                    "frames": 4,
                },
                {"label": "door_open_wait", "buttons": [], "frames": 40},
                {
                    "label": "enter_exit_door",
                    "buttons": travel_buttons,
                    "frames": budget["enter"],
                },
            ]
        )
    else:
        steps = [
            {"label": "entry_settle", "buttons": [], "frames": 20},
            {
                "label": "coarse_exit_approach",
                "buttons": approach_buttons or travel_buttons,
                "frames": budget["traverse_approach"],
            },
            {
                "label": "open_exit_door",
                "buttons": door_buttons,
                "frames": 8,
            },
            {"label": "door_open_wait", "buttons": [], "frames": 45},
            {
                "label": "enter_exit_door",
                "buttons": approach_buttons or travel_buttons,
                "frames": budget["enter"] + 40,
            },
        ]

    payload = {
        "schemaVersion": 2,
        "problemId": problem_id,
        "status": "generated_unverified",
        "description": (
            f"Doorway-natural starter for {problem['roomName']} "
            f"({problem['roomIdHex']}); entry just inside door, exit via "
            f"{direction or 'unknown'}."
        ),
        "entryContract": contract.to_dict(),
        "planning": {
            "objective": problem["objective"],
            "entry": problem["entry"],
            "exit": problem["exit"],
            "staticPlan": problem["staticPlan"],
        },
        "steps": steps,
        "acceptanceWarning": (
            "Generated scaffold only. It must cross and settle in emulator "
            "before its status can become verified_development_state "
            "(use run --promote or the promote command)."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {
        "problemId": problem_id,
        "policyPath": str(path.resolve()),
        "status": payload["status"],
        "orientationHint": orientation or None,
        "sameDoorReturn": contract.same_door_return,
        "frameBudget": budget,
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
    """Return a failure string if collect objectives did not progress.

    Mid-game doorway boots often already own early-route ammo packs (PLMs are
    spent). When start capacity already covers the room's packs, skip that
    field — the PLM cannot re-grant. Real collection still required when the
    boot lacks the capacity.
    """
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
        if not required:
            continue
        gained = getattr(final, field) - getattr(start, field)
        if gained >= required:
            continue
        # Pre-collected in boot: capacity already present, PLM cannot re-fire.
        if getattr(start, field) >= required:
            continue
        failures.append(f"{field} did not increase by {required}")
    energy_required = counts["energy tank"] * 100
    if energy_required:
        gained = final.max_health - start.max_health
        if gained < energy_required and start.max_health < energy_required:
            failures.append(f"max_health did not increase by {energy_required}")
    reserve_required = counts["reserve tank"] * 100
    if reserve_required:
        gained = final.max_reserve_health - start.max_reserve_health
        if (
            gained < reserve_required
            and start.max_reserve_health < reserve_required
        ):
            failures.append(
                f"max_reserve_health did not increase by {reserve_required}"
            )
    beam_items = set(item_names) & _BEAM_ITEMS
    if beam_items and final.collected_beams == start.collected_beams:
        # Beams are bitflags; if already owned at start, PLM is spent.
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


def _load_entry_contract(
    state_path: Path,
    policy: Mapping[str, object],
    problem: Mapping[str, object],
) -> dict[str, object] | None:
    provenance_path = state_path.with_suffix(".provenance.json")
    if provenance_path.is_file():
        try:
            prov = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prov = None
        if isinstance(prov, dict):
            contract = EntryContract.from_dict(prov.get("entryContract"))
            if contract is not None:
                return contract.to_dict()
            # Legacy provenance without nested contract.
            if prov.get("doorPtrHex") or prov.get("method"):
                return {
                    "kind": "doorway_natural"
                    if "doorway" in str(prov.get("method", ""))
                    else "imported",
                    "method": prov.get("method"),
                    "doorPtrHex": prov.get("doorPtrHex"),
                    "samusX": prov.get("samusX"),
                    "samusY": prov.get("samusY"),
                }
    if isinstance(policy.get("entryContract"), dict):
        contract = EntryContract.from_dict(policy["entryContract"])  # type: ignore[arg-type]
        if contract is not None:
            return contract.to_dict()
    # Catalog-only fallback (scaffold before bootstrap).
    return EntryContract.from_problem(problem).to_dict()


def run_room_problem(
    problem_id: str,
    *,
    catalog_path: Path = ROOM_PROBLEMS_PATH,
    report_path: Path | None = None,
    settle_timeout: int = 300,
    promote: bool = False,
) -> dict[str, object]:
    """Teleport, replay a compact policy, and verify a natural room exit.

    When ``promote`` is true and the run succeeds, flips the policy status to
    ``verified_development_state`` after writing the report (sha-gated).
    """
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
    # Contract-allowed assist (energy + unlocked ammo) so heat/enemy rooms
    # can be practiced under the same attrition rules as continuous tips.
    assist = UnlimitedResourcesAssist(unlimited_energy=True, unlimited_ammo=True)
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
        assist.apply(env.data, start)
        for span in spans:
            action = buttons(*span.buttons) if span.buttons else idle_action()
            for _ in range(span.frames):
                env.step(action)
                frame += 1
                action_counts[span.label] = action_counts.get(span.label, 0) + 1
                state = parse_state(env.get_ram(), frame=frame)
                assist.apply(env.data, state)
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
                assist.apply(env.data, state)
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
    state_sha = _sha256(state_path)
    policy_sha = _sha256(policy_path)
    entry_contract = _load_entry_contract(state_path, policy, problem)

    report = {
        "schemaVersion": 2,
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
        "entryContract": entry_contract,
        "segmentBoundary": segment_boundary_dict(),
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
        "assist": assist.report(),
        "state": {
            "path": str(state_path.resolve()),
            "sha256": state_sha,
        },
        "policy": {
            "path": str(policy_path.resolve()),
            "sha256": policy_sha,
            "status": policy.get("status"),
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
    report["reportPath"] = str(output.resolve())

    if promote:
        if not success:
            report["promoted"] = False
            report["promoteError"] = "run failed; policy not promoted"
        else:
            promo = promote_room_policy(
                problem_id,
                catalog_path=catalog_path,
                report_path=output,
                require_matching_sha=True,
            )
            report["promoted"] = bool(promo.get("promoted"))
            report["policy"]["status"] = promo.get("policyStatus", policy.get("status"))
            if promo.get("error"):
                report["promoteError"] = promo["error"]
    return report


def promote_room_policy(
    problem_id: str,
    *,
    catalog_path: Path = ROOM_PROBLEMS_PATH,
    report_path: Path | None = None,
    require_matching_sha: bool = True,
) -> dict[str, object]:
    """Mark a policy verified only when a green report matches current artifacts.

    Gates:
    - report ``success`` is true
    - report problemId matches
    - optional: report state/policy sha256 match files on disk
    """
    catalog = load_problem_catalog(catalog_path)
    problem = problem_by_id(catalog, problem_id)
    state_path, policy_path, default_report_path = _problem_paths(problem)
    report_file = report_path or default_report_path
    if not report_file.is_file():
        return {
            "problemId": problem_id,
            "promoted": False,
            "error": f"missing report: {report_file}",
        }
    if not policy_path.is_file():
        return {
            "problemId": problem_id,
            "promoted": False,
            "error": f"missing policy: {policy_path}",
        }
    if not state_path.is_file():
        return {
            "problemId": problem_id,
            "promoted": False,
            "error": f"missing state: {state_path}",
        }

    try:
        report = json.loads(report_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "problemId": problem_id,
            "promoted": False,
            "error": f"unreadable report: {exc}",
        }
    if not isinstance(report, dict):
        return {
            "problemId": problem_id,
            "promoted": False,
            "error": "report is not an object",
        }
    if report.get("problemId") != problem_id:
        return {
            "problemId": problem_id,
            "promoted": False,
            "error": (
                f"report problemId {report.get('problemId')!r} != {problem_id!r}"
            ),
        }
    if not report.get("success"):
        return {
            "problemId": problem_id,
            "promoted": False,
            "error": f"report success is false: {report.get('failure')}",
        }

    state_sha = _sha256(state_path)
    policy_sha_before = _sha256(policy_path)
    if require_matching_sha:
        report_state = report.get("state") if isinstance(report.get("state"), dict) else {}
        report_policy = (
            report.get("policy") if isinstance(report.get("policy"), dict) else {}
        )
        if report_state.get("sha256") and report_state["sha256"] != state_sha:
            return {
                "problemId": problem_id,
                "promoted": False,
                "error": "report state sha256 does not match current .state file",
            }
        if (
            report_policy.get("sha256")
            and report_policy["sha256"] != policy_sha_before
        ):
            return {
                "problemId": problem_id,
                "promoted": False,
                "error": (
                    "report policy sha256 does not match current policy file "
                    "(re-run before promote after editing steps)"
                ),
            }

    policy, _ = load_room_policy(policy_path)
    if policy.get("status") == "verified_development_state":
        return {
            "problemId": problem_id,
            "promoted": True,
            "alreadyVerified": True,
            "policyPath": str(policy_path.resolve()),
            "policyStatus": "verified_development_state",
            "reportPath": str(report_file.resolve()),
        }

    policy["status"] = "verified_development_state"
    policy["promotedAt"] = datetime.now(timezone.utc).isoformat()
    policy["promotion"] = {
        "reportPath": str(report_file.resolve()),
        "reportSha256": _sha256(report_file),
        "stateSha256": state_sha,
        "policySha256Before": policy_sha_before,
    }
    policy_path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    return {
        "problemId": problem_id,
        "promoted": True,
        "alreadyVerified": False,
        "policyPath": str(policy_path.resolve()),
        "policyStatus": "verified_development_state",
        "policySha256After": _sha256(policy_path),
        "reportPath": str(report_file.resolve()),
    }


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

