"""Extract exact per-level input skills from verified SMW oracle segments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Protocol, Sequence

from SMW.tas.smv import word_to_buttons


class InputMovie(Protocol):
    """Minimum normalized-input interface needed for skill extraction."""

    p1_words: tuple[int, ...]

    @property
    def num_frames(self) -> int: ...


def rle_words(words: Iterable[int]) -> list[dict[str, object]]:
    """Run-length encode normalized SNES input words without losing buttons."""

    runs: list[dict[str, object]] = []
    for word in words:
        buttons = sorted(word_to_buttons(word))
        if runs and runs[-1]["word"] == word:
            runs[-1]["frames"] = int(runs[-1]["frames"]) + 1
        else:
            runs.append({"frames": 1, "word": word, "buttons": buttons})
    return runs


def extract_level_skills(
    movie: InputMovie,
    segments: Sequence[dict[str, object]],
    output_dir: Path | str,
) -> list[Path]:
    """Materialize one exact RLE input artifact per verified level segment."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for segment in segments:
        index = int(segment["index"])
        entry_frame = int(segment["entry_frame"])
        exit_frame = int(segment["exit_frame"])
        if not 0 <= entry_frame < exit_frame <= movie.num_frames:
            raise ValueError(
                f"invalid verified segment bounds {entry_frame}:{exit_frame}"
            )
        translevel = int(segment["translevel"])
        words = movie.p1_words[entry_frame:exit_frame]
        payload = {
            "schema_version": 1,
            "kind": "smw_exact_level_input_skill",
            "verification": "BizHawk RAM-backed oracle",
            "index": index,
            "translevel": translevel,
            "entry_frame": entry_frame,
            "exit_frame": exit_frame,
            "num_frames": len(words),
            "quality": (
                "clean_single_attempt"
                if int(segment.get("retry_count", 0)) == 0
                and int(segment.get("lives_drops", 0)) == 0
                else "replay_with_retries"
            ),
            "retry_count": int(segment.get("retry_count", 0)),
            "lives_drops": int(segment.get("lives_drops", 0)),
            "entry_ram": segment.get("entry_ram"),
            "exit_ram": segment.get("exit_ram"),
            "runs": rle_words(words),
        }
        path = output_dir / f"level_{index:02d}_trans_{translevel:02x}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return written
