# ALttP route residual — 2026-10-02

C1 is complete: Clean power-on to completed Sanctuary, with no health refill
assistance, progression/capacity writes, mid-run loads, or deaths.

## Proof and segment breakdown

`recordings/clean_escape_20261002/controller.tape.npz` replays from `NONE`
(core reset, blank SRAM before frame1). `replay.json`: **36078f**, zero
full-RAM CRC desync, zero loads/writes, assist disabled. Final room0x12,
module/submodule0x07/0, (1272,568), HP10/24, sword/lamp1, follower0,
progress2, flags0x15, Sanctuary story acceptance true.

| Segment | Frames | End HP |
|---------|-------:|-------:|
| Boot and fresh profile | 4838 | 24 |
| Castle to uncle sword | 1947 | 24 |
| Secret-passage exit | 603 | 24 |
| Courtyard to main hall | 5693 | 20 |
| Main hall to room0x50 | 736 | 20 |
| Room0x50 east to0x01 | 424 | 16 |
| Stairs to0x72 | 431 | 16 |
| Small key and south to0x82 | 792 | 24 |
| Bridge west to0x81 | 686 | 20 |
| Required keys and Zelda rescue | 5655 | 16 |
| Escort, mantle, sewers and Sanctuary story | 14273 | 10 |

HP is never written. Two normal enemy heart drops restore16HP through play;
no refill assist runs. The replay takes30HP in11 hits and finishes alive.
This is a Clean completion, not a no-damage or no-heart-pickup claim.

## rr-vlor — complete

Read the big-key carrier marker `$0CBA == 2`, then forecast complete32-frame
approach, swing and recovery candidates on the live ROM, including flail
contact beyond the body. Commit eight frames and replan. Committing all32
could finish in an inescapable flail arc on the actual arrival. Unsafe
candidates and room/floor/layer changes are rejected; no candidate means red.
No copied approach coordinates or inventory writes.

Real no-assist predecessor: frame20698 of `controller.tape.npz`,
`DevClean_Ball`, HP16. Offsets0,7,11 each earn the key without damage and
pass same-pin forward replay, zero CRC desync/loads/writes. Evidence:
`rom_regressions.log` (14 ROM cases pass). The continuous composition rescues
Zelda at frame21805, not by loading a cell pin.

## rr-e8x7 — complete no-refill composition

The initial no-assist run caught the main-hall corridor lip and died before
room0x50. Mapped `natural_clear_north` plus one-frame corridor points with
role `axis` pass that leg with HP20 unchanged.

Escort forecasts preserve the exact existing mapped waypoint handoff.
Measured guards cover0x81,0x82,0x72,0x50,0x41,0x42,0x32. The48-frame
forecast includes facing, sword swing and recovery when a hostile blocks
the approach; pure evasion stalled the snake-room stair. The exact mantle
walker, stair settling and required carrier rat fight are preserved.

The real no-assist rescue pin `DevClean_Zelda` is captured at frame21805.
Whole suffix plus CRC forward replay passes offsets0 and7:

| Offset | Committed frames | End HP | Search restores | Replay loads/writes/desync |
|-------:|-----------------:|-------:|----------------:|---------------------------|
| 0 | 14273 | 10 | 3438 | 0/0/0 |
| 7 | 14148 | 12 | 5887 | 0/0/0 |

Evidence: `escort_offset_0.json`, `escort_offset_7.json`,
`escort_replay_tests.log`. Full controller discloses4148 search restores
(4141 rollout +7 bush candidate restores); the independent power-on replay
has none. These pins are development inputs, not independent continuous claims.

## Next

C2 Eastern Palace from the Clean Sanctuary predecessor (`rr-eog9.6` lobby,
`.7` CLI/replay, `.8` tests); roadside resources remain `rr-yx26`.
Optional measured follow-up (`rr-dkwg`): cut14HP of accepted escort sewer damage in
0x22,0x21,0x11. Forecast the complete key pickup/door/block handoff, not just
walking. C1 is already green; no stronger no-damage claim is made.

The historical room0x50 report is intact. No follower, keys, sword, capacity
or progression was poked. Alternate dungeon-entry/Survival evidence remains
separate. Broader offline tests retain the known fake sword-exit fixture
failure `rr-mott`; it predates this change and does not affect the ROM proof.

Reconstruct both ROM-test predecessor pins from the published tape (cwd `snes/`):

```bash
QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
uv run python alttp/scripts/audit_power_on.py replay \
  --tape alttp/recordings/clean_escape_20261002/controller.tape.npz \
  --capture 20698:DevClean_Ball --capture 21805:DevClean_Zelda \
  --out alttp/recordings/clean_escape_20261002/replay.json
```

Final verification: 14 existing/big-key ROM cases plus two whole-escort offset
replay cases pass; 332 offline cases pass with only the known `rr-mott` fixture
excluded; eight repository docs checks pass.
