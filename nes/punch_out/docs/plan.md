# Plan — Mike Tyson's Punch-Out!! (NES)

## Goal

Advance from M2 (instrumented + first KD) toward a verified continuous clear of Mike Tyson's Punch-Out!!.

## Next milestones

1. **M3 isolated bout clear** — Glass Joe win from `Match1` with hard timeout (3 KDs, KO, or decision).
2. **M4 natural-entry** — same bout from power-on / real predecessor (not only Match1 load).
3. **M5+** — Von Kaiser and circuit chaining.

## Bottleneck

Post-KD2 Glass Joe (opp health ~48): regular punches still miss; dodge survival is not enough to reach R3 decision without Mac TKO. Need either:

- landable counters after his R2 hooks (pattern ids 4/6/8/18/20/21/23), or
- a third special knockdown window, or
- lower damage taken in R1 so Mac enters R2 with full/near-full HP.

## Known working recipe (KD1)

1. Load `Match1` (or wait ~840f from `Level1` for clock).
2. Idle until `opp_pattern_set == 150` (Vive La France).
3. Left face jabs (A, 2 on / 3 off) → opp health 0, knockdown.
4. Get-up if needed: A,A,idle,B,B,idle (2-frame presses).

## Notes

- Platform: NES (fceumm via stable-retro custom integration).
- Shared ROM root: `roms/Nintendo/NES/`.
- Taunt also appears ~0:32 of R2 for KD2.
