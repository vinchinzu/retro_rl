# Plan — Mike Tyson's Punch-Out!! (NES)

## Goal

Advance from M3 (isolated Glass Joe bout win) toward a verified continuous clear of Mike Tyson's Punch-Out!!.

## Next milestones

1. **M4 natural-entry** — same bout win from power-on / Level1 (real predecessor, not only Match1 load).
2. **M5+** — Von Kaiser and circuit chaining.

## Bottleneck (cleared for M3)

Post-KD1 survival was the M3 gate: continuous L/R spam desynced and Mac TKO'd before R2 taunt. Fixed with **timed dodge** on attack act change (wait ~32 frames, 5-frame LEFT/RIGHT pulse) plus strict `pattern_set == 150` taunt counters. Second knockdown count-outs Joe (KO); Mac stays full HP.

## Known working recipe (bout win from Match1)

1. Load `Match1` (or wait ~840f from `Level1` for clock).
2. Idle until `opp_pattern_set == 150` (Vive La France).
3. Left face jabs (A, 2 on / 3 off) → KD1.
4. On attack acts `{4,6,7,10,13,17,20,23}`: wait 32f idle, dodge 5f, optional short jab counter.
5. R2 taunt again → KD2; hold through long count → KO (verified 3/3).
6. Get-up if needed: A,A,idle,B,B,idle (2-frame presses).

## Notes

- Platform: NES (fceumm via stable-retro custom integration).
- Shared ROM root: `roms/Nintendo/NES/`.
- Verify: `run_glass_joe.py --goal win --trials 3 --record`.
