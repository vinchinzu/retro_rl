# SM-TIGHTEN-P2 Residual Note

Changed the Business Center setup sequence from `("RIGHT", "LEFT", "LEFT", "RIGHT")` to `("RIGHT", "LEFT", "LEFT")` in both the normal setup and the floor-recover re-climb.

The first RIGHT jump establishes the lateral setup, and the two LEFT jumps carry the climb toward the first platform. The final RIGHT jump was removed to target the speculative one-jump saving band (~115f); no settle durations, `runup_907`, or platform hop gates changed.

## Residual risk

- If three jumps miss `y=1339`, the fallback can still re-climb, but the retry costs more than the nominal one-jump saving.
- This is not pure-green evidence and is not a continuous claim. The planner must run `uv run python super_metroid/scripts/record/continuous.py --to kraid --no-video` at least once successfully, then perform the report-requested multi-run gate of at least three successful runs before any claim.
- The pure Business-floor probe result is MISSING; no suitable source-state probe was run in this task.
- No STATUS promotion was made, and no progression or capacity RAM was forged.
