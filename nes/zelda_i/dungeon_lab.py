"""Headless Zelda I dungeon policy laboratory.

Runs checkpoint-isolated policy sweeps, captures full/tail traces, annotates
RAM deltas, ranks configurations, probes known exits, and optionally promotes
the best result to a provenance-backed development checkpoint.
"""

from __future__ import annotations

import multiprocessing
import statistics
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from retro_harness.env import (
    make_env,
    state_path,
    write_state_bytes,
)
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.ram_state import snapshot as ram_snapshot
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.dungeon import (
    AliveRule,
    DoorRoute,
    DungeonPhase,
    GenericDungeonRoomController,
    RewardKind,
    ensure_default_specs,
    override_room_spec,
    spec_for_room,
)

# Populate room-spec registry (L1–L6) before lab lookups.
ensure_default_specs()
from zelda_i.dungeon_ids import object_name, room_item_name
from zelda_i.dungeon_trace import (
    TraceRecorder,
    compact_snapshot,
    first_trace_divergence,
    ram_delta_report,
    read_jsonl,
    write_state_provenance,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot


@dataclass(frozen=True)
class LabRequest:
    state: str
    room_id: int
    trials_per_config: int = 1
    jobs: int = 1
    attack_phases: tuple[int, ...] = (0,)
    engage_distances: tuple[int, ...] = ()
    enemy_types: tuple[int, ...] = ()
    alive_rule: AliveRule | None = None
    reward_mode: str = "spec"
    max_frames: int | None = None
    tail_frames: int = 120
    output_dir: str | None = None
    save_state: str | None = None
    probe_exits: bool = True

    def __post_init__(self) -> None:
        if self.trials_per_config <= 0:
            raise ValueError("trials_per_config must be positive")
        if self.jobs <= 0:
            raise ValueError("jobs must be positive")
        if not self.attack_phases:
            raise ValueError("at least one attack phase is required")
        if self.reward_mode not in {"spec", "auto", "clear"}:
            raise ValueError("reward_mode must be spec, auto, or clear")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["alive_rule"] = self.alive_rule.value if self.alive_rule else None
        payload["room_id_hex"] = f"0x{self.room_id:02X}"
        return payload


@dataclass(frozen=True)
class TrialRequest:
    trial_index: int
    state: str
    room_id: int
    attack_phase: int
    engage_distance: int
    enemy_types: tuple[int, ...]
    alive_rule: str | None
    reward_mode: str
    max_frames: int | None
    tail_frames: int
    output_dir: str


def _trial_spec(request: TrialRequest):
    spec = spec_for_room(request.room_id)
    reward_kind = None
    if request.reward_mode in {"auto", "clear"}:
        reward_kind = RewardKind.CLEAR_ONLY
    spec = override_room_spec(
        spec,
        enemy_types=request.enemy_types or None,
        alive_rule=AliveRule(request.alive_rule) if request.alive_rule else None,
        reward_kind=reward_kind,
        engage_distance=request.engage_distance,
        attack_phase=request.attack_phase,
    )
    if request.max_frames is not None:
        spec = replace(spec, max_frames=request.max_frames)
    return spec


def _clear_ready(controller: GenericDungeonRoomController, snap) -> bool:
    live = controller.spec.live_enemies(snap)
    return (
        snap.screen == controller.spec.room_id
        and not live
        and controller.max_live_enemies >= controller.spec.expected_enemy_count
        and snap.room_all_dead >= controller.spec.reward.settle_all_dead
    )


def run_lab_trial(request: TrialRequest) -> dict[str, Any]:
    """Worker-safe isolated trial. The raw state bytes are returned to parent."""
    configure_headless()
    spec = _trial_spec(request)
    trial_dir = Path(request.output_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)
    stem = f"trial_{request.trial_index:03d}"
    trace_path = trial_dir / f"{stem}.trace.jsonl"
    tail_path = trial_dir / f"{stem}.failure_tail.jsonl"
    screenshot_path = trial_dir / f"{stem}.final.png"

    env = make_env(GAME, request.state, GAME_DIR, render_mode="rgb_array")
    controller = GenericDungeonRoomController(spec)
    recorder = TraceRecorder(tail_frames=request.tail_frames)
    reason_counts: Counter[str] = Counter()
    entry_ram: np.ndarray | None = None
    clear_ram: np.ndarray | None = None
    state_bytes: bytes | None = None
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs, *_ = env.step(nes_idle_action())
        start_ram = ram_snapshot(env.get_ram())

        for frame in range(spec.max_frames):
            snap = read_snapshot(env.get_ram())
            if (
                entry_ram is None
                and snap.screen == spec.room_id
                and snap.mode == PLAY_MODE
            ):
                entry_ram = ram_snapshot(env.get_ram())
            if clear_ram is None and _clear_ready(controller, snap):
                clear_ram = ram_snapshot(env.get_ram())

            action = controller.step(snap)
            reason_counts[action.reason] += 1
            recorder.record(
                frame=frame,
                phase=controller.phase.name,
                reason=action.reason,
                action=action.action,
                snap=snap,
            )
            obs, *_ = env.step(action.action)
            if controller.success or controller.phase is DungeonPhase.FAILED:
                break

        final_ram = ram_snapshot(env.get_ram())
        final_snap = read_snapshot(final_ram)
        if clear_ram is None and controller.clear_signal_seen:
            clear_ram = final_ram.copy()
        state_bytes = env.em.get_state()

        recorder.write(trace_path)
        failure_tail = None
        if not controller.success:
            recorder.write(tail_path, tail_only=True)
            failure_tail = str(tail_path.resolve())
        save_rgb_png(obs, screenshot_path)

        entry = entry_ram if entry_ram is not None else start_ram
        cleared = clear_ram if clear_ram is not None else final_ram
        if controller.success:
            outcome = "success"
        elif "link_death" in controller.notes:
            outcome = "death"
        else:
            outcome = "timeout"

        start_to_entry = ram_delta_report(start_ram, entry)
        entry_to_clear = ram_delta_report(entry, cleared)
        clear_to_final = ram_delta_report(cleared, final_ram)
        start_to_final = ram_delta_report(start_ram, final_ram)
        inventory_symbols = {
            "sword",
            "bombs",
            "arrows",
            "bow",
            "candle",
            "whistle",
            "food",
            "potion",
            "rod",
            "raft",
            "book",
            "ring",
            "ladder",
            "magic_key",
            "bracelet",
            "letter",
            "compass",
            "map",
            "rupees",
            "keys",
            "health",
            "triforce",
        }
        inventory_changes = [
            row
            for row in start_to_final["known"]
            if row["symbol"] in inventory_symbols
        ]
        return {
            "trial_index": request.trial_index,
            "success": controller.success,
            "outcome": outcome,
            "frames": controller.frames,
            "controller": controller.report(),
            "policy": {
                "attack_phase": request.attack_phase,
                "engage_distance": request.engage_distance,
            },
            "entry": compact_snapshot(read_snapshot(entry)),
            "final": compact_snapshot(final_snap),
            "reason_counts": dict(sorted(reason_counts.items())),
            "ram_deltas": {
                "start_to_entry": start_to_entry,
                "entry_to_clear": entry_to_clear,
                "clear_to_final": clear_to_final,
                "start_to_final": start_to_final,
            },
            "reward_analysis": {
                "room_item_id": final_snap.room_item_id,
                "room_item_name": room_item_name(final_snap.room_item_id),
                "known_inventory_changes": inventory_changes,
                "classification": (
                    "known_inventory_change"
                    if inventory_changes
                    else "no_known_inventory_change"
                ),
            },
            "trace": str(trace_path.resolve()),
            "failure_tail": failure_tail,
            "screenshot": str(screenshot_path.resolve()),
            "_state_bytes": state_bytes,
        }
    finally:
        env.close()


def _drive_exit(
    state_data: bytes,
    *,
    spec_room: int,
    route: DoorRoute,
    screenshot_path: Path,
    max_frames: int = 900,
) -> dict[str, Any]:
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    waypoint_index = 0
    entered = False
    play_frames = 0
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        env.em.set_state(state_data)
        obs, *_ = env.step(nes_idle_action())
        for frame in range(max_frames):
            snap = read_snapshot(env.get_ram())
            if snap.screen != spec_room and snap.mode == PLAY_MODE:
                play_frames += 1
                if play_frames >= 40:
                    save_rgb_png(obs, screenshot_path)
                    object_counts = Counter(
                        obj.type_id
                        for obj in snap.objects
                        if 1 <= obj.slot <= 10 and obj.type_id
                    )
                    return {
                        "success": True,
                        "direction": route.direction,
                        "frames": frame,
                        "room": snap.screen,
                        "room_hex": f"0x{snap.screen:02X}",
                        "room_item_id": snap.room_item_id,
                        "room_item_name": room_item_name(snap.room_item_id),
                        "room_obj_count": snap.room_obj_count,
                        "objects": {
                            f"0x{type_id:02X}": {
                                "name": object_name(type_id),
                                "count": count,
                            }
                            for type_id, count in sorted(object_counts.items())
                        },
                        "screenshot": str(screenshot_path.resolve()),
                    }
                action = nes_idle_action()
            elif snap.screen == spec_room and snap.mode == PLAY_MODE and not entered:
                if waypoint_index < len(route.waypoints):
                    tx, ty = route.waypoints[waypoint_index]
                    dx = tx - snap.link_x
                    dy = ty - snap.link_y
                    if abs(dx) <= 2 and abs(dy) <= 2:
                        waypoint_index += 1
                        action = nes_idle_action()
                    elif abs(dx) > 2:
                        action = nes_action("RIGHT" if dx > 0 else "LEFT")
                    else:
                        action = nes_action("DOWN" if dy > 0 else "UP")
                else:
                    entered = True
                    action = nes_action(route.direction)
            elif snap.transitioning or entered:
                action = nes_action(route.direction)
            else:
                action = nes_idle_action()
            obs, *_ = env.step(action)
        save_rgb_png(obs, screenshot_path)
        snap = read_snapshot(env.get_ram())
        return {
            "success": False,
            "direction": route.direction,
            "frames": max_frames,
            "room": snap.screen,
            "room_hex": f"0x{snap.screen:02X}",
            "mode": snap.mode,
            "x": snap.link_x,
            "y": snap.link_y,
            "screenshot": str(screenshot_path.resolve()),
        }
    finally:
        env.close()


def probe_exits(
    state_data: bytes,
    *,
    room_id: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    spec = spec_for_room(room_id)
    results: list[dict[str, Any]] = []
    for route in spec.exit_routes:
        results.append(
            _drive_exit(
                state_data,
                spec_room=room_id,
                route=route,
                screenshot_path=output_dir
                / f"exit_{route.direction.lower()}.png",
            )
        )
    return results


def _rank_policies(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for trial in trials:
        policy = trial["policy"]
        key = (policy["attack_phase"], policy["engage_distance"])
        groups.setdefault(key, []).append(trial)

    ranking: list[dict[str, Any]] = []
    for (attack_phase, engage_distance), rows in groups.items():
        successes = [row for row in rows if row["success"]]
        ranking.append(
            {
                "attack_phase": attack_phase,
                "engage_distance": engage_distance,
                "trials": len(rows),
                "successes": len(successes),
                "success_rate": len(successes) / len(rows),
                "median_success_frames": (
                    statistics.median(row["frames"] for row in successes)
                    if successes
                    else None
                ),
                "outcomes": dict(Counter(row["outcome"] for row in rows)),
            }
        )
    ranking.sort(
        key=lambda row: (
            -row["success_rate"],
            row["median_success_frames"]
            if row["median_success_frames"] is not None
            else float("inf"),
            row["attack_phase"],
            row["engage_distance"],
        )
    )
    return ranking


def _trace_comparison(trials: list[dict[str, Any]]) -> dict[str, Any] | None:
    successes = sorted(
        (trial for trial in trials if trial["success"]),
        key=lambda trial: trial["frames"],
    )
    failures = [trial for trial in trials if not trial["success"]]
    if successes and failures:
        left, right = successes[0], failures[0]
    elif len(successes) >= 2:
        left, right = successes[0], successes[-1]
    elif len(trials) >= 2:
        left, right = trials[0], trials[-1]
    else:
        return None
    divergence = first_trace_divergence(
        read_jsonl(Path(left["trace"])),
        read_jsonl(Path(right["trace"])),
    )
    return {
        "left_trial": left["trial_index"],
        "right_trial": right["trial_index"],
        "divergence": divergence,
    }


def _default_output_dir(room_id: int) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return RECORDINGS_DIR / "dungeon_lab" / f"room_{room_id:02x}_{timestamp}"


def _write_generated_report(summary: dict[str, Any], path: Path) -> Path:
    """Write a compact Markdown handoff generated from lab evidence."""
    room = summary["room_spec"]
    ranking = summary["ranking"]
    best = ranking[0] if ranking else None
    reward_rows = [
        trial["reward_analysis"] for trial in summary["trials"] if trial["success"]
    ]
    reward = reward_rows[0] if reward_rows else None
    lines = [
        f"# Dungeon lab — room 0x{room['room_id']:02X}",
        "",
        f"- Successes: {summary['successes']}/{summary['trial_count']}",
        f"- Enemy types: {', '.join(f'0x{x:02X}' for x in room['enemy_types'])}",
        f"- Liveness rule: `{room['alive_rule']}`",
        f"- Room item: `0x{room['room_item_id']:02X}` "
        f"({room['room_item_name']})",
    ]
    if best:
        lines.extend(
            [
                f"- Best policy: attack phase {best['attack_phase']}, "
                f"engage distance {best['engage_distance']}",
                f"- Best median: {best['median_success_frames']} frames",
            ]
        )
    if reward:
        lines.append(f"- Reward analysis: {reward['classification']}")
    if summary["promoted_state"]:
        lines.append(f"- Promoted checkpoint: `{summary['promoted_state']}`")
    lines.extend(["", "## Exit probes", ""])
    if summary["exit_probes"]:
        for probe in summary["exit_probes"]:
            destination = (
                probe.get("room_hex", "unknown") if probe["success"] else "blocked"
            )
            lines.append(f"- {probe['direction']}: {destination}")
    else:
        lines.append("- Not requested")
    lines.extend(["", "## Policy ranking", ""])
    for index, row in enumerate(ranking, start=1):
        lines.append(
            f"{index}. phase={row['attack_phase']}, "
            f"engage={row['engage_distance']}, "
            f"success={row['successes']}/{row['trials']}, "
            f"median={row['median_success_frames']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_spec_suggestion(summary: dict[str, Any], path: Path) -> Path:
    """Emit the measured fields needed to promote a new room spec."""
    best = summary["ranking"][0] if summary["ranking"] else None
    suggestion = {
        "source": summary["summary_path"] if "summary_path" in summary else None,
        "room_spec": summary["room_spec"],
        "recommended_policy": (
            {
                "attack_phase": best["attack_phase"],
                "engage_distance": best["engage_distance"],
                "observed_success_rate": best["success_rate"],
                "median_success_frames": best["median_success_frames"],
            }
            if best
            else None
        ),
        "observed_exits": [
            {
                "direction": probe["direction"],
                "success": probe["success"],
                "room": probe.get("room"),
                "room_hex": probe.get("room_hex"),
            }
            for probe in summary["exit_probes"]
        ],
        "promotion_gate": (
            "Replay from the real predecessor and run natural-entry verification "
            "before marking the route edge ready."
        ),
    }
    return write_json_report(path, suggestion)


def run_lab(request: LabRequest) -> dict[str, Any]:
    """Execute and summarize a complete policy sweep."""
    configure_headless()
    base_spec = spec_for_room(request.room_id)
    distances = request.engage_distances or (base_spec.combat.engage_distance,)
    output_dir = (
        Path(request.output_dir).resolve()
        if request.output_dir
        else _default_output_dir(request.room_id).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_report(output_dir / "request.json", request.to_dict())

    trial_requests: list[TrialRequest] = []
    index = 0
    for attack_phase in request.attack_phases:
        for engage_distance in distances:
            for _ in range(request.trials_per_config):
                trial_requests.append(
                    TrialRequest(
                        trial_index=index,
                        state=request.state,
                        room_id=request.room_id,
                        attack_phase=attack_phase,
                        engage_distance=engage_distance,
                        enemy_types=request.enemy_types,
                        alive_rule=(
                            request.alive_rule.value if request.alive_rule else None
                        ),
                        reward_mode=request.reward_mode,
                        max_frames=request.max_frames,
                        tail_frames=request.tail_frames,
                        output_dir=str(output_dir),
                    )
                )
                index += 1

    if request.jobs == 1:
        raw_trials = [run_lab_trial(trial) for trial in trial_requests]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=request.jobs,
            mp_context=context,
        ) as executor:
            raw_trials = list(executor.map(run_lab_trial, trial_requests))
    raw_trials.sort(key=lambda trial: trial["trial_index"])

    successful = [trial for trial in raw_trials if trial["success"]]
    best = min(successful, key=lambda trial: trial["frames"]) if successful else None
    best_state_data = best["_state_bytes"] if best else None
    exit_results: list[dict[str, Any]] = []
    if request.probe_exits and best_state_data is not None:
        exit_results = probe_exits(
            best_state_data,
            room_id=request.room_id,
            output_dir=output_dir,
        )

    promoted_state = None
    provenance = None
    if request.save_state and best is not None and best_state_data is not None:
        promoted_path = state_path(GAME_DIR, GAME, request.save_state)
        write_state_bytes(promoted_path, best_state_data)
        source_path = state_path(GAME_DIR, GAME, request.state)
        public_best = {
            "trial_index": best["trial_index"],
            "success": best["success"],
            "outcome": best["outcome"],
            "frames": best["frames"],
            "policy": best["policy"],
            "controller": best["controller"],
            "final": best["final"],
            "reward_analysis": best["reward_analysis"],
            "trace": best["trace"],
            "screenshot": best["screenshot"],
        }
        provenance_path = write_state_provenance(
            promoted_path,
            source_state_path=source_path,
            request=request.to_dict(),
            selected_trial=public_best,
        )
        promoted_state = str(promoted_path.resolve())
        provenance = str(provenance_path.resolve())

    public_trials = [
        {key: value for key, value in trial.items() if key != "_state_bytes"}
        for trial in raw_trials
    ]
    summary = {
        "schema_version": 1,
        "request": request.to_dict(),
        "room_spec": {
            "spec_id": base_spec.spec_id,
            "source_room": base_spec.source_room,
            "room_id": base_spec.room_id,
            "enemy_types": list(base_spec.enemy_types),
            "expected_enemy_count": base_spec.expected_enemy_count,
            "alive_rule": base_spec.alive_rule.value,
            "room_item_id": base_spec.room_item_id,
            "room_item_name": (
                room_item_name(base_spec.room_item_id)
                if base_spec.room_item_id is not None
                else None
            ),
        },
        "trials": public_trials,
        "ranking": _rank_policies(public_trials),
        "successes": len(successful),
        "trial_count": len(public_trials),
        "trace_comparison": _trace_comparison(public_trials),
        "exit_probes": exit_results,
        "promoted_state": promoted_state,
        "provenance": provenance,
        "output_dir": str(output_dir),
    }
    summary_path = write_json_report(output_dir / "summary.json", summary)
    summary["summary_path"] = str(summary_path.resolve())
    report_path = _write_generated_report(summary, output_dir / "report.md")
    suggestion_path = _write_spec_suggestion(
        summary,
        output_dir / "room_spec_suggestion.json",
    )
    summary["generated_report"] = str(report_path.resolve())
    summary["spec_suggestion"] = str(suggestion_path.resolve())
    # Re-write so artifact paths are also discoverable from summary.json.
    write_json_report(summary_path, summary)
    return summary
