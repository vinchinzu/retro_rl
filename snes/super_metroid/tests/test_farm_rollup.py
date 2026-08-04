"""Regression tests for evidence-based room-farm rollups."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
_SNES_IMPORT_ROOT = Path(__file__).resolve().parents[2]
ROLLUP_HELPER = ROOT / "snes" / "super_metroid" / "scripts" / "farm_rollup.sh"


def classify(card_id: str, log: Path, tasks_dir: Path) -> str:
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; farm_card_result "$2" "$3" "$4"',
            "bash",
            str(ROLLUP_HELPER),
            card_id,
            str(log),
            str(tasks_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def test_free_text_green_is_not_rollup_evidence(tmp_path: Path) -> None:
    log = tmp_path / "worker.log"
    log.write_text("The implementation is GREEN? and ready for review.\n")

    assert classify("SM-ROOM-SEG-99", log, tmp_path) == "NO_EVIDENCE"


def test_residual_result_is_authoritative_over_prior_runner_output(tmp_path: Path) -> None:
    card_id = "SM-ROOM-SEG-99"
    log = tmp_path / "worker.log"
    log.write_text('{"success": true, "phase": "initial-run"}\n')
    (tmp_path / f"{card_id}-residual.md").write_text(
        "### Result\n\nRED. Promotion recheck failed.\n"
    )

    assert classify(card_id, log, tmp_path) == "RED"


def test_residual_green_requires_schema_heading(tmp_path: Path) -> None:
    card_id = "SM-ROOM-SEG-99"
    log = tmp_path / "worker.log"
    log.write_text("No runner output.\n")
    (tmp_path / f"{card_id}-residual.md").write_text(
        "### Result\n\n**GREEN** — promoted after isolated run.\n"
    )

    assert classify(card_id, log, tmp_path) == "GREEN"


def test_json_success_is_green_when_no_residual_exists(tmp_path: Path) -> None:
    log = tmp_path / "worker.log"
    log.write_text('{"success":true,"targetRoomIdHex":"0xA011"}\n')

    assert classify("SM-ROOM-SEG-99", log, tmp_path) == "GREEN"


def test_shell_style_success_text_is_not_json_evidence(tmp_path: Path) -> None:
    log = tmp_path / "worker.log"
    log.write_text("success=true target=0xA011\n")

    assert classify("SM-ROOM-SEG-99", log, tmp_path) == "NO_EVIDENCE"
