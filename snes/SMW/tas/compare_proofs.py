"""Compare independent SMW BizHawk oracle proofs for deterministic replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

_SEGMENT_FIELDS = (
    "index",
    "translevel",
    "entry_frame",
    "exit_frame",
    "max_player_x",
    "retry_count",
    "sublevel_count",
    "lives_drops",
    "completion_signal",
    "entry_ram",
    "exit_ram",
)


def _segment_fingerprint(segment: dict[str, Any]) -> dict[str, Any]:
    return {field: segment.get(field) for field in _SEGMENT_FIELDS}


def compare_proofs(paths: Sequence[Path | str]) -> dict[str, object]:
    """Require matching GREEN source, ROM, and per-segment RAM fingerprints."""

    if len(paths) < 2:
        raise ValueError("at least two proof paths are required")
    loaded = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    for path, proof in zip(paths, loaded, strict=True):
        if proof.get("status") != "GREEN":
            raise ValueError(f"proof is not GREEN: {path}")

    expected_source = loaded[0].get("source_sha256")
    expected_rom = loaded[0].get("rom_hash")
    expected_segments = [
        _segment_fingerprint(segment) for segment in loaded[0].get("segments", [])
    ]
    if not expected_segments:
        raise ValueError("proof contains no verified segments")
    for path, proof in zip(paths[1:], loaded[1:], strict=True):
        if proof.get("source_sha256") != expected_source:
            raise ValueError(f"source mismatch: {path}")
        if proof.get("rom_hash") != expected_rom:
            raise ValueError(f"ROM mismatch: {path}")
        segments = [
            _segment_fingerprint(segment) for segment in proof.get("segments", [])
        ]
        if segments != expected_segments:
            raise ValueError(f"segment fingerprint mismatch: {path}")

    return {
        "status": "GREEN",
        "independent_runs": len(paths),
        "source_sha256": expected_source,
        "rom_hash": expected_rom,
        "segments": expected_segments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proofs", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare_proofs(args.proofs), indent=2))


if __name__ == "__main__":
    main()
