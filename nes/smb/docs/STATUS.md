# Status

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M8 |
| Best verified result | Clean power-on to the 8-4 ending, 21,559 frames, 3/3 |
| Last verification | 2026-08-05 |
| Runtime class | Bronze |
| Intervention class | Clean |

`recordings/reactive_warp/natural_82_poweron_trials_report.json` is that run:
`policy_frames` 21559, successes 3, every outcome `ending`, boot 350, settle
16, mid-attempt state loads 0, lives 2, world 7, level 3, `oper_mode` 2.
Seed `models/smb_1_1_to_ending_natural_82.json`. Exit frames in the same
report: 1-1 1911, 1-2 3884, 4-1 6198, 4-2 8962, 8-1 12628, 8-2 15779,
8-3 17985, 8-4 21559.

Documented contract times for this 21,559-frame report (not recomputed
here): RTA any% 05:58.726 against HappyLee 04:54.032, and power-on
06:04.816 against HappyLee #1715 04:57.31.

## Capture

`recordings/warp_finish/warp_finish_poweron_m8_capture.json` describes a
preserved 22,005-frame MP4 (`policy_frames` 22005, seed
`smb_1_1_to_ending.json`). The note in that file calls the MP4 historical.
Its `current_best` is the older 21,731-frame controller (`verified_at`
2026-07-28), not the 21,559-frame report. The MP4 path named there is not
in this checkout. Do not treat a video as the 21,559-frame run.

`recordings/warp_finish/warp_finish_poweron_trials_report.json` is the
21,731-frame baseline (same seed, 3 successes).

## Not the gate

HappyLee slices, the hybrid showcase, a pure FCEUX replay of #1715M, and
the 32-exit track are not this Clean power-on result. Do not promote them
onto this gate. Rules: `docs/TAS_ADAPT.md`. Open work: `docs/plan.md`.

## Next

A faster Clean power-on seed, or the separate 32-exit route. The
21,559-frame report stays the gate until a newer Clean power-on report
replaces it.
