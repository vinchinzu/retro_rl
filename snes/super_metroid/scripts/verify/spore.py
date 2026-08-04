"""Independently verify the spore tip report and video."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SNES_IMPORT_ROOT = Path(__file__).resolve().parents[3]
for _p in (_REPO_ROOT, globals().get('_SNES_IMPORT_ROOT', _REPO_ROOT)):
    if _p is not None and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from super_metroid.paths import RECORDINGS_DIR, SHARED_ROM  # noqa: E402
from super_metroid.progression import SPORE_GRAPH  # noqa: E402
from super_metroid.ram import BOMBS_MASK, MORPH_BALL_MASK  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _video_frames(path: Path) -> int:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    return int(subprocess.check_output(command, text=True).strip())


def verify(report_path: Path, video_path: Path) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    checks["report_success"] = report["success"] is True
    checks["power_on_start"] = report["start_state"] == "power_on/retro.State.NONE"
    checks["rom_hash"] = report["rom_sha256"] == _sha256(SHARED_ROM)
    checks["state_loads_zero"] = report["state_loads"] == 0
    checks["progression_writes_zero"] = report["progression_writes"] == 0
    checks["capacity_writes_zero"] = report["assist"]["capacity_writes"] == 0
    checks["deaths_zero"] = report["assist"]["deaths"] == 0
    checks["ceres_energy_suspended"] = (
        report["assist"]["suspended_phase_frames"].get("energy:ceres", 0) > 0
    )

    final_state = report["final_state"]
    checks["post_spore_room"] = (
        final_state["room_id"] == 0x9B5B
        and final_state["phase"] == "ordinary_gameplay"
    )
    checks["natural_capacities"] = (
        final_state["max_health"] >= 199
        and final_state["max_missiles"] >= 10
        and final_state["max_super_missiles"] == 0
        and final_state["max_power_bombs"] == 0
    )
    checks["morph_and_bombs"] = (
        final_state["collected_items"] & (MORPH_BALL_MASK | BOMBS_MASK)
        == MORPH_BALL_MASK | BOMBS_MASK
    )

    boss = report["boss"]
    hp_history = [int(value) for value in boss["observed_hp"]]
    checks["spore_peak_hp"] = boss["peak_hp"] == 960
    checks["spore_hp_zero"] = 0 in hp_history
    checks["spore_hp_history_descends"] = (
        hp_history == sorted(set(hp_history), reverse=True)
        and hp_history[0] == 960
        and hp_history[-1] == 0
    )
    checks["spore_vulnerable_states"] = bool(boss["vulnerable_spritemaps"])
    checks["boss_frame_order"] = (
        boss["entry_frame"]
        <= boss["activation_frame"]
        < boss["defeat_frame"]
        < boss["exit_frame"]
    )

    expected_splits = (
        "first_ceres_control",
        "ridley_countdown",
        "zebes_landing",
        "morph_ball",
        "first_missiles",
        "blue_brinstar_missiles",
        "bombs",
        "bomb_torizo_defeated",
        "bomb_torizo_exit",
        "terminator_energy_tank",
        "green_brinstar_main_shaft",
        "spore_spawn_activated",
        "spore_spawn_defeated",
        "spore_spawn_exit",
    )
    split_frames = {
        split["split_id"]: int(split["frame"]) for split in report["splits"]
    }
    checks["required_splits"] = all(split in split_frames for split in expected_splits)
    checks["split_order"] = all(
        split_frames[left] < split_frames[right]
        for left, right in zip(expected_splits, expected_splits[1:])
    )

    transition_checks = []
    observed_pairs: list[tuple[int, int]] = []
    for observed in report["transitions"]:
        source = int(observed["source_room_id"])
        target = int(observed["target_room_id"])
        edge = SPORE_GRAPH.edge_for(source, target)
        transition_checks.append(
            edge is not None and edge.edge_id == observed["edge_id"]
        )
        observed_pairs.append((source, target))
    checks["typed_transitions"] = bool(transition_checks) and all(transition_checks)
    planned_pairs = [
        (int(left, 16), int(right, 16))
        for left, right in zip(
            report["route_plan"]["room_path"],
            report["route_plan"]["room_path"][1:],
        )
    ]
    checks["planned_post_torizo_path_observed"] = all(
        pair in observed_pairs for pair in planned_pairs
    )

    policy_checks = []
    source_checks = []
    for segment in report["segments"]:
        policy_path = Path(segment["policy_path"])
        policy_checks.append(_sha256(policy_path) == segment["policy_sha256"])
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
        source_path = Path(payload["metadata"]["source"])
        if source_path.is_file():
            source_checks.append(_sha256(source_path) == segment["source_sha256"])
    checks["prefix_policy_hashes"] = bool(policy_checks) and all(policy_checks)
    checks["available_source_hashes"] = bool(source_checks) and all(source_checks)

    for name, source in report["policy_sources"].items():
        source_path = Path(source["path"])
        checks[f"policy_source_{name}"] = (
            source_path.is_file() and _sha256(source_path) == source["sha256"]
        )

    plan = report["route_plan"]
    plan_path = Path(plan["path"])
    editor_path = Path(plan["editorNavPath"])
    reference_path = Path(plan["referenceRoutePath"])
    checks["plan_hash"] = _sha256(plan_path) == plan["sha256"]
    checks["editor_nav_hash"] = (
        _sha256(editor_path)
        == plan["editorNavPathSha256"]
        == plan["editorNavDeclaredSha256"]
    )
    checks["reference_route_hash"] = (
        _sha256(reference_path)
        == plan["referenceRoutePathSha256"]
        == plan["referenceRouteDeclaredSha256"]
    )
    checks["plan_not_mislabeled_continuous"] = (
        plan["status"] == "planned_not_continuous"
        and "not continuous-run evidence" in plan["acceptance_warning"]
    )

    video_hash = _sha256(video_path)
    frame_count = _video_frames(video_path)
    checks["video_hash"] = video_hash == report["video"]["sha256"]
    checks["video_frames"] = (
        frame_count
        == report["encoded_frames"]
        == report["total_frames"] + 1
    )

    result = {
        "schema_version": 1,
        "verified": all(checks.values()),
        "checks": checks,
        "report_path": str(report_path.resolve()),
        "report_sha256": _sha256(report_path),
        "video_path": str(video_path.resolve()),
        "video_sha256": video_hash,
        "video_frames": frame_count,
        "graph_id": SPORE_GRAPH.graph_id,
        "transition_count": len(transition_checks),
        "policy_count": len(policy_checks),
        "planned_post_torizo_edge_count": len(planned_pairs),
    }
    if not result["verified"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"verification failed: {failed}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=RECORDINGS_DIR / "spore.json",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=RECORDINGS_DIR / "spore.mp4",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RECORDINGS_DIR / "spore.verify.json",
    )
    args = parser.parse_args()
    result = verify(args.report, args.video)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
