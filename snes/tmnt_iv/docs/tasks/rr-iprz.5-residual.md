# rr-iprz.5 residual — Starbase stall (next sitting)

**Status:** OPEN. Do not STATUS. Do not reopen Slash. Do not skip
dumpster on byte 8 at x=126. Do not tighten
`raph_starbase_close_gap` below `% 4`. Do not encode video until a
power-on `--dry-run` on the sitting's tree has already reached credits.

Shipped: **right-rail skip** (`starbase_rail_right`, x≥220) and
**exhausted-207 RIGHT** (`starbase_unstick_right` after 3 dumpster
cycles in x=200–219). Wave dumpster at x=126 still DOWN+JUMP. Dual-green
speed bars are **not** met — Fast and Boss9 time still miss KEEP.
Continuous **does** finish on this tree.

## Why this bead (whole-run clock)

Encoder 90 idle + 300 at 1080p is **not** a route proof. The killed
1080p dump stalled at Stage 9 Starbase **x=207**, damage **5285** from
frame 180k–230k+, cycling `stall_right` → `stall_up` → `stall_up_right`.
Camera ticked +1/frame (auto-scroll). That is an infinite dumpster
unstick, not a finish. Last STATUS clear remains 00:57:19 on the old
`tmnt_iv_full_hard_dry_run.json`.

## This sitting (2026-08-27) — continuous 207 exhaust

Heal=emergency, never A. Fast / Diag pins still **stage_advance** with
dumpster at x=207 (recover in <600f). The encode/dry-run hole is
continuous-only: same x=207, Y sweeps 113–194 for 12k frames, `targets=[]`
while a Foot is visibly parked on the right edge.

| Pin / run | Policy | Outcome | Frames | Dmg | Heals |
|-----------|--------|---------|-------:|----:|------:|
| Fast / Diag / Boss9 | rail skip + dumpster | stage_advance | **23272 / 24645 / 6516** | **917 / 1076 / 64** | 13 / 16 / 1 |
| Power-on `--dry-run` HEAD (rail only) | dumpster at 207 | **frozen_x 12k** | 184186 abort | 5285 stuck | — |
| Power-on `--dry-run` + 207 exhaust RIGHT | 127f `starbase_unstick_right` | **credits** | **212829** | **5529** | **79** |

KEEP bars from the prior sitting: Fast **≤23,072f / 863**, Boss9
**6,300f / 144 / 2**. Diag time KEEP **≤24,645f / 1,076**. Fast +200f /
+54 dmg. Boss9 +216f / −80 dmg (6516/64). No STATUS. Published 00:57:19
file not overwritten.

Power-on proof (do not STATUS / BASELINE):
`recordings/tmnt_iv_full_hard_dry_run_rr_iprz5.json` — **00:59:01.318**,
5,529 dmg, 79 e-heals, 7,420 iframe, **0 lives lost**. Starbase split
00:43:13 → form-2 00:51:39. Freeze abort in the runner is 12,000
enemyless frozen-X frames (saves PNG + `ScratchFreeze_*` state).

## Exact next action

1. Claim `rr-iprz.5`. Read this residual. **Do not touch Slash.**
2. Keep rail skip (`starbase_rail_right` at x≥220). Keep 207 exhaust
   RIGHT after 3 dumpster cycles (`starbase_unstick_right`). Do **not**
   gate dumpster off on x=126. Tests:
   `test_starbase_frozen_x_keeps_dumpster_unstick`,
   `test_starbase_mid_wave_freeze_still_dumpsters`,
   `test_starbase_exhausted_207_holds_right`,
   `test_starbase_x126_dumpster_never_exhausts_to_right`,
   `test_starbase_form1_rail_skips_dumpster`,
   `test_starbase_spawn_delay_does_not_trigger_dumpster_escape`.
3. Do **not** set `_JUMP_PERIOD` below 4. Never hold B+Y.
4. Remaining speed hole is **Fast 23,272f / 917** vs 23,072 / 863 and
   **Boss9 6,516f / 64** vs 6,300 / 144 — form-1 arrival Y after holding
   RIGHT at x=229. Recover those bars **without** restoring the Diag
   rail loop (24,645f) or the 207 infinite dumpster. Do not 96f-budget
   or form-1-latch the skip.
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
  --max-frames 20000 --stop-stage-gt 8 --heal emergency
```

   KEEP bars: Fast **≤23,072f / 863**, Boss9 **6,300f / 144 / 2**, Diag
   **≤24,645f / 1,076**. `--trace-stall` dumps x/y/cam/form-1 samples.
6. Dual-green: `RaphFastStage9` **and** `RaphDiagStage9` on frames **and**
   damage before promoting BASELINE. Continuous finish on this tree is
   already proven (`rr_iprz5` dry-run). Re-dry-run after the Fast/Boss9
   recovery. No STATUS. Do not overwrite `tmnt_iv_full_hard_dry_run.json`.
   No video encode until that sitting's `--dry-run` reaches credits.

## Burned (do not relearn)

| Item | Lesson |
|------|--------|
| Encoder 90 idle + 300 at 1080p | Not a route proof. Native vs 1080p is irrelevant if Starbase loops |
| Dumpster DOWN+JUMP on x≥220 rail | Diag 7k loop (stall_down 2,035 + cycle) |
| 96f dumpster then RIGHT on rail | Diag 40k timeout (15k rail_right, bad Y) |
| Skip rail only after 0x52 seen | Diag 40k timeout (21k rail_right) |
| Y-steer rail to y=156 | Boss9 6,300→8,880 |
| Sewer-like dumpster skip on byte 8 | 40k timeout, 29k walk_right, y=196 past form-1 |
| Climb-only instead of dumpster | 40k timeout, stuck x=126 y=139 |
| Exhausted RIGHT on x=126 | same 40k class as sewer skip — 207 band only |
| close_gap `%3` / `%2` | 40k timeout / jump-lock. `%4` is KEEP |
| Shorten stall_down to 8f on stage 8 | Diag win, Boss9 6,300→8,040. Empty form-1 pockets have `boss=0` |
| Slash rewrite sitting | 4 algorithms + 3 KEEP patches; all lost to 9,595/435 |
| `slash_spin_dodge_adx=40` | pin win, continuous +807 dmg |
| Global pizza seek | Skull soft-lock |
| A-special | HP drain |
| Mid-run knob w/o full dry-run | Pin Fast/Diag green, power-on frozen at x=207 |

## Pins / files

- Waves: `RaphFastStage9` (launch x=64), `RaphDiagStage9`
- Form-1: `RaphFullHardBoss9` (shipped 6,516f / 64 / 1 — old KEEP 6,300 / 144)
- Form-2 is **not** this bead (`rr-iprz.3`, Leo `Boss9_phase2` only)
- Code: `policy.py` `walk_action` launch-right `player_x <= 64`, rail-right
  `player_x >= 220`; `PlayerXStallWalk` dumpster **on** for x<220; after 3
  cycles in x=200–219 emit `starbase_unstick_right`; `_JUMP_PERIOD = 4`
- Scratch live Starbase entry (`ScratchIprz5Stage9`) clears in 22,068f / 853
  — does **not** reproduce the 207 loop. Continuous power-on does.
- Freeze abort: `_FREEZE_ABORT_FRAMES = 12000` in `record_full_hard_run.py`
