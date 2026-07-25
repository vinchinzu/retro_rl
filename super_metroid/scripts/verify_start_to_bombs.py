"""Independently verify a start-to-Bomb-Torizo report and video artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from super_metroid.paths import RECORDINGS_DIR, SHARED_ROM  # noqa: E402
from super_metroid.progression import EARLY_GAME_GRAPH  # noqa: E402
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
    checks["two_missile_expansions"] = report["final_state"]["max_missiles"] >= 10
    checks["morph_and_bombs"] = (
        report["final_state"]["collected_items"]
        & (MORPH_BALL_MASK | BOMBS_MASK)
        == MORPH_BALL_MASK | BOMBS_MASK
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
    for observed in report["transitions"]:
        edge = EARLY_GAME_GRAPH.edge_for(
            int(observed["source_room_id"]),
            int(observed["target_room_id"]),
        )
        transition_checks.append(
            edge is not None and edge.edge_id == observed["edge_id"]
        )
    checks["typed_transitions"] = bool(transition_checks) and all(transition_checks)

    policy_checks = []
    source_checks = []
    for segment in report["segments"]:
        policy_path = Path(segment["policy_path"])
        policy_checks.append(_sha256(policy_path) == segment["policy_sha256"])
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
        source_path = Path(payload["metadata"]["source"])
        if source_path.is_file():
            source_checks.append(
                _sha256(source_path) == segment["source_sha256"]
            )
    checks["policy_hashes"] = bool(policy_checks) and all(policy_checks)
    checks["available_source_hashes"] = bool(source_checks) and all(source_checks)

    video_hash = _sha256(video_path)
    frame_count = _video_frames(video_path)
    checks["video_hash"] = video_hash == report["video"]["sha256"]
    checks["video_frames"] = frame_count == report["encoded_frames"]

    result = {
        "schema_version": 1,
        "verified": all(checks.values()),
        "checks": checks,
        "report_path": str(report_path.resolve()),
        "report_sha256": _sha256(report_path),
        "video_path": str(video_path.resolve()),
        "video_sha256": video_hash,
        "video_frames": frame_count,
        "graph_id": EARLY_GAME_GRAPH.graph_id,
        "transition_count": len(transition_checks),
        "policy_count": len(policy_checks),
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
        default=RECORDINGS_DIR / "start_to_bomb_torizo.json",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=RECORDINGS_DIR / "start_to_bomb_torizo.mp4",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RECORDINGS_DIR / "start_to_bomb_torizo.verify.json",
    )
    args = parser.parse_args()
    result = verify(args.report, args.video)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
