# rr-iprz.5 residual — Starbase stall (next sitting)

**Status:** OPEN. Do not STATUS. Do not reopen Slash. Do not skip
dumpster on byte 8 at x=126 / 207. Do not tighten
`raph_starbase_close_gap` below `% 4`.

Shipped this sitting: **right-rail skip** (`starbase_rail_right`) when
stage 8, `player_x >= 220`, no living enemies. Wave dumpsters at x=126 /
207 still DOWN+JUMP. Dual-green is **not** met — Fast and Boss9 bars
moved; Diag time is the KEEP.

## Why this bead (whole-run clock)

`recordings/tmnt_iv_full_hard_dry_run_rr_iprz.json` vs on-disk
`tmnt_iv_full_hard_dry_run.json` (58:15, not the STATUS 57:19 file):
Starbase +5:11 / +18,672f. Continuous hole is the **Diag** entry.

## This sitting (2026-08-27) — rail predicate

Heal=emergency, never A. Stall traces: form-1 vanish is **x=229, dcam=+1,
event 0x0A, boss=0, n=0**. Wave dumpsters are x≈128 and x≈206–207, also
dcam=+1. `boss_active` does not distinguish them.

| Pin | Policy | Outcome | Frames | Dmg | Heals |
|-----|--------|---------|-------:|----:|------:|
| Fast / Diag / Boss9 | production dumpster 36f | stage_advance | **23072 / 33825 / 6300** | **863 / 1004 / 144** | 12 / 15 / 2 |
| Fast / Diag / Boss9 | **rail skip x≥220 RIGHT** | stage_advance | **23272 / 24645 / 6540** | **917 / 1076 / 64** | 13 / 16 / 1 |
| Fast / Diag / Boss9 | Y-steer rail to y=156 | Boss9 only | — / — / **8880** | — / — / 152 | — |
| Fast / Diag / Boss9 | 96f dumpster then RIGHT | Diag **timeout** | 23538 / 40000 / **6300** | 911 / 1532 / 144 | 13 / 23 / 2 |
| Fast / Diag / Boss9 | skip only after 0x52 seen | Diag **timeout** | **22302** / 40000 / 6540 | 927 / 2012 / 64 | 13 / 31 / 1 |

KEEP bars from the prior sitting: Fast **≤23,072f / 863**, Boss9
**6,300f / 144 / 2**. Diag must beat **33,825f / 1,004**.

Rail skip is the only Diag clear that cuts the 7k-frame dumpster loop
(−9,180f). Fast +200f / +54 dmg. Boss9 +240f / −80 dmg. No STATUS.
No full dry-run (Fast+Boss9 not dual-green).

## Exact next action

1. Claim `rr-iprz.5`. Read this residual. **Do not touch Slash.**
2. Keep rail skip (`starbase_rail_right` at x≥220). Do **not** gate
   dumpster off on x=126 / 207. Tests:
   `test_starbase_frozen_x_keeps_dumpster_unstick`,
   `test_starbase_mid_wave_freeze_still_dumpsters`,
   `test_starbase_form1_rail_skips_dumpster`,
   `test_starbase_spawn_delay_does_not_trigger_dumpster_escape`.
3. Do **not** set `_JUMP_PERIOD` below 4. Never hold B+Y.
4. Remaining hole is **Fast 23,272f / 917** vs 23,072 / 863 and **Boss9
   6,540f / 64** vs 6,300 / 144 — form-1 arrival Y after holding RIGHT
   at x=229. Recover those bars **without** restoring the Diag rail loop
   (24,645f). Do not 96f-budget or form-1-latch the skip.
5. Live probe (heal=emergency, never A):

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_boss_metrics RaphFastStage9 \
  --max-frames 40000 --stop-stage-gt 8 --heal emergency

SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_boss_metrics RaphDiagStage9 \
  --max-frames 40000 --stop-stage-gt 8 --heal emergency

SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_boss_metrics RaphFullHardBoss9 \
  --max-frames 20000 --heal emergency
```

   KEEP bars: Fast **≤23,072f / 863**, Boss9 **6,300f / 144 / 2**, Diag
   **≤24,645f / 1,076** (time already cut; do not give the 7k loop back).
   `--trace-stall` dumps x/y/cam/form-1 samples.
6. Dual-green: `RaphFastStage9` **and** `RaphDiagStage9`. No full dry-run
   until both are KEEP on frames **and** damage. No STATUS / BASELINE.
   Do not overwrite `tmnt_iv_full_hard_dry_run.json`.

## Burned (do not relearn)

| Item | Lesson |
|------|--------|
| Dumpster DOWN+JUMP on x≥220 rail | Diag 7k loop (stall_down 2,035 + cycle) |
| 96f dumpster then RIGHT on rail | Diag 40k timeout (15k rail_right, bad Y) |
| Skip rail only after 0x52 seen | Diag 40k timeout (21k rail_right) |
| Y-steer rail to y=156 | Boss9 6,300→8,880 |
| Sewer-like dumpster skip on byte 8 | 40k timeout, 29k walk_right, y=196 past form-1 |
| Climb-only instead of dumpster | 40k timeout, stuck x=126 y=139 |
| close_gap `%3` / `%2` | 40k timeout / jump-lock. `%4` is KEEP |
| Shorten stall_down to 8f on stage 8 | Diag win, Boss9 6,300→8,040. Empty form-1 pockets have `boss=0` |
| Slash rewrite sitting | 4 algorithms + 3 KEEP patches; all lost to 9,595/435 |
| `slash_spin_dodge_adx=40` | pin win, continuous +807 dmg |
| Global pizza seek | Skull soft-lock |
| A-special | HP drain |

## Pins / files

- Waves: `RaphFastStage9` (launch x=64), `RaphDiagStage9`
- Form-1: `RaphFullHardBoss9` (shipped 6,540f / 64 / 1 — old KEEP 6,300 / 144)
- Form-2 is **not** this bead (`rr-iprz.3`, Leo `Boss9_phase2` only)
- Code: `policy.py` `walk_action` launch-right `player_x <= 64`, rail-right
  `player_x >= 220`; `PlayerXStallWalk` dumpster **on** for x<220 on byte 8;
  `_JUMP_PERIOD = 4`
- No `RaphLiveStage9` state on disk
