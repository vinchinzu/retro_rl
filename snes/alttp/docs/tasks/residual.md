# ALttP route residual — 2026-10-01

The Clean continuous tip remains room `0x50`. No STATUS promotion.

## rr-rkpv — required sewer key combat

The old escort chased the nearest rat. On the real rescued-Zelda sewer
predecessor, nearby alcove rat slot 4 is not the key carrier; slot 0 carries
`sprite_die_action == 1` at `$0CBA`. The ROM marks the required drop, so the
controller reads that marker instead of inferring inventory intent from
distance. It stops on the drop, then independently collects it and requires
`$F36F >= 1`. No marker or inventory writes.

The controller forecasts its complete approach, strike, hitstun, and death
wait. This avoids the old mismatch where a guard forecast walking straight
into a rat although the live controller would stop and swing. Guard detours
also preserve room, floor, and layer; forecasts stop at a completed stage.
No broad escort navigation rewrite was landed.

On `DevEscort_RatKey`, the old nearest-rat fight fails after 1438f and one
2HP rat hit. New combat passes **12/12 offsets (0–11), zero damage**,
197–282f (mean 205.1f). Fight plus key pickup passes committed forward
replay at offsets 0 and 7: 220f/228f, zero desync/loads/writes or damage.
Controller restores are disclosed: 178/167; no non-rollout loads or writes.
Evidence: `recordings/escort_guard_20261001/rat_offsets.log`,
`rat_baseline.json`, `rat_replay.json`, and `rat_replay_tests.log`.

The pin is a development scan captured 10639 committed frames into a
guarded prefix from `DevAudit_zelda_replay`. That source pin inherits
**11 Survival refills, 22 hearts**, from its 26078f power-on prefix. Its
provenance is not interchangeable with the older seven-refill Sanctuary
run. The rejected guarded prefix is not production route evidence.

Rat-key leave RAM (offset 0): room `0x21`, module/submodule `0x07/0x00`,
`(781,1207)`, floor `0xFF`, layer 0, control ready, HP `16`, sword/lamp 1,
follower 1, keys 1. The same-pin replay agrees on every frame.

## Whole-route handoff and replay

The real power-on rat fight passes in269f with HP22→22; pickup and later
transit still take damage. The changed timing exposed a middle-block stall:
a rat hit moved Link off the push column, but the old loop held UP for400f
at `(874,632)`. The targeted fix detects sideways displacement and returns
to the mapped approach. Actual recovery: first push14f to `(883,645)`,
re-align29f to `(888,648)`, second push62f to `(888,600)`. No extra refill.

**Power-on Survival controller and replay pass, 36755f**. The replay has
zero desync, loads, and deaths; seven disclosed HP writes restore14 hearts.
Capacity/progression writes remain0. The controller discloses3523 search
restores (`actual_search_restores`); these do not occur in the replay.
Evidence: `recordings/escort_guard_20261001/final_controller.json`,
`final_controller.tape.npz`, and `escort_poweron_replay.json`.
The preceding `controller.json`/`failed_replay.json` are the reproduced
block-stall red: synchronization passes but the route goal fails.

Sanctuary leave RAM: room `0x12`, module/submodule `0x07/0x00`,
`(1272,568)`, control ready, HP10/24, sword/lamp1, follower0, keys0,
progress2, flags`0x15`. `alttp.screen_glance` passes the leave band, and
the replay independently accepts completed Zelda delivery. This is
assisted development proof, not a Clean promotion.

## Remaining Clean work

`rr-vlor`: earn the castle big key without ball-and-chain refills. Read
its big-key marker (`$0CBA == 2`), measure the flail beyond the soldier's
body origin, and evaluate the complete combat policy at real predecessor
offsets. The earlier scratch Clean attempt dies in room `0x80`; treating
that weapon as only a nearby body is insufficient.

`rr-e8x7`: remove escort transit damage one measured room at a time.
Baseline no-refill suffix from `DevAudit_zelda_replay` dies in room `0x32`
at7789f, losing24HP: green soldier2, blue soldier1, rat1, rope3 hits.
The broad guarded walker was rejected: offsets0 and7 reach room `0x51`
with16/20HP but stall `mantle_dialogue` at `(754,2626)`. Do not weaken the
exact mantle handoff or guard to turn that stall green. Restore a real
first-hit predecessor, inspect collision and the hit trail, and change only
that transit/combat leg. Then require its actual successor, multiple
offsets, and committed replay before composing a whole no-refill escort.

The Zelda I lessons applied here are required-item combat, measured hit
causes, whole-policy ROM forecasts, offset sweeps, and separate CRC tape
replay. A successful inherited-Survival suffix still does not satisfy the
no-assist power-on gate. Preserve the existing Eastern approach proof and
its `rr-eog9.6` entrance follow-up; that work is separate from Clean escape.

Verification: 131 targeted offline tests pass; nine existing room-engine
and rollout ROM regressions pass; two new rat fight/pickup replay cases
pass. Complete Sanctuary power-on Survival controller and replay pass.
Docs gate: eight tests pass. `rr-rkpv` is complete; the Clean tasks stay open.

Did not STATUS-promote, treat a save-state pin as power-on, write follower,
sword, keys, capacity, or progression, copy approach geometry into Python,
or overwrite `recordings/verified_tip_run.json`.
