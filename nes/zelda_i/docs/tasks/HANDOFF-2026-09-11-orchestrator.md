# Handoff — 2026-09-11 orchestrator (continue from reactive combat)

No STATUS claim. `route_eligible=false`. Four worker commits on `main`
plus this leftover-honesty edit. Suite: **1331 passed, 3 deselected**.

## Read this first

`uv run python nes/zelda_i/scripts/clean_tip.py` — `tip()` is still
`None`. Next open is `l1_tf` / `clear33_key`.

## What landed

| Commit | What |
|--------|------|
| `c05e7115` | Tracking/postmortem leftovers after `3f90a42c`: room-change `_prev` drop, age-1 unknown hp=128 is a shot, projectile rank nudge only if `closing_on`. |
| `c6d81cfe` | L1 `0x33` occupancy walk. `$6530` dump: no `0xB0` block; tile 244 at `(96,160)`. Stall `(88,165)` was RIGHT into that cell. |
| `688341da` | L6 `0x7a` measurement: `0x24_E` contacts at `(88,133)` vs parked `(96,125)`, d=8, ttc=0, `dodgeable=False`. Combat policy unchanged. |
| `78bbe906` | L1 `0x23` swing census + water-bar occupancy from `$6530`. |

Combined-tree M5 (`run_once(natural_entry=True, tag="rr_npv8_landed")`):
`clear33_key` red, leftover `0x33 (94,173)` mode 5 health `0x21` keys 1
TF `0x00`, `0x33_needs_heart` heart_wait=180, end 13091, stage 2443f.
Hits: Stalfos `0x2a` N + E, not fatal. Deaths 0.

## Open, in priority order

1. **L1 `0x33` heart at lo=1** — occupancy is done. The fail-closed
   contract waits 180f on the key tile then `0x33_needs_heart` because
   this seed never drops a heart. Next lever is a heart (combat/RNG) or
   a contract change to DONE at `0x21` (that sends `0x23` in at one
   heart; isolated 0x23 at `0x20` already dies on contact). Dump before
   believing a drop-rate story. Bead `rr-npv.8`.
2. **L1 `0x23` lo≥2** — census: failure to *arrive*, not `should_swing_at`.
   14 slashes killed the south-corridor Goriya; the other two sat on the
   NE channel while occupancy mashed UP into water. Water bar is now
   seeded from `$6530`. No Clean lo≥2 measurement exists until 0x33
   clears. Isolated `Level1Cleared33` is 1 heart.
3. **L6 `level6_east_key_0x7a`** — still the health leak. Sword-in-place,
   `off_line_step`, and a y=173 stand each turned the *green* east-key
   stage red (0x59 on the same axis). Need a stand cell that is not a
   live 0x24 firing row. Do not peel: `dodgeable` is False. Do not
   re-run the 0x78 east-waist chase.

## Review leftovers not landed

`Room33ScoopController.step` still bypasses `super().step` on scoop /
key-walk (silent DamageLog on those frames). `test_dungeon.py` is over
1k LOC. Root `scratch/` still cached (37 probes). `stash@{0}` still
stale. `z3-json-data` still has three deleted PNGs.

## Method

Read the per-stage damage census before tuning the room that went red.
Dump `$6530` before believing a wall. Link walks 1 px/frame;
`threat.dodgeable` False means no position table works at that pose.
