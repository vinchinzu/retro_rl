# SM-TIGHTEN-01B — P1 settle trim residual

## Lines changed (8 settles, all 20→5f)

| Label | Line | Old | New |
|-------|------|-----|-----|
| `business_1339_settle` | 109 | 20 | 5 |
| `business_1227_settle` | 146 | 20 | 5 |
| `business_1147_settle` | 165 | 20 | 5 |
| `business_987_settle` | 196 | 20 | 5 |
| `business_907_settle` | 217 | 20 | 5 |
| `business_843_settle` | 239 | 20 | 5 |
| `business_779_settle` | 257 | 20 | 5 |
| `business_elevator_settle` | 287 | 20 | 5 |

## Not changed (out of scope)

- `business_1067_settle` (30f at line 176) — not a 20f platform settle per report
- P2 (setup jumps 4→3) — deferred
- P3 (runup_907 14→10) — deferred
- `business_elevator_center_settle` (8f at line 300) — centering, not platform settle
- `business_floor_recover_settle` (15f at line 331) — floor fallback, not platform settle
- `continuous.py`, `STATUS.md`, `progression.py`, tracker — no edits

## Speculative band

~160f (8×20f) if all `wait_standing_y` calls return in 0f as before. **Not claimed — requires re-record.**

## Planner verify

```bash
uv run python super_metroid/scripts/record/continuous.py --to kraid --no-video
uv run python super_metroid/scripts/export/split_dwell.py \
  super_metroid/recordings/start_to_kraid.json --top 20
# Also verify full chain:
uv run python super_metroid/scripts/record/continuous.py --to varia --no-video
```

If continuous fails after this patch, revert all 8 settles back to 20f.

## Risk

Low — each settle is followed by `_wait_standing_y` with a 30-90f timeout that polls every frame. The 5f idle is just a brief debounce before polling begins. Natural-entry variance on climb lips is unchanged (all standing gates, runup, and landing checks preserved).