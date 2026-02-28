"""CLI entry point for Super Mario Bros speedrun optimizer.

Usage:
    uv run python -m super_mario_bros.optimizer play
    uv run python -m super_mario_bros.optimizer selftest
    uv run python -m super_mario_bros.optimizer hillclimb --seed recording.json
    uv run python -m super_mario_bros.optimizer list-levels
    # SMB 8-3 raw-first wrapper (resume-only supported):
    #   ./super_mario_bros/run_8_3.sh train ...
"""

import platformer_common.levels.smb  # noqa: F401 — register SMB levels
from platformer_common.runner import main

main(default_level="smb_1_1")
