# ALTTP status

A save-state pin is development input; continuous proof is a power-on replay.

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M1 |
| Best verified result | C1: Clean power-on to completed Sanctuary |
| Last verification | 2026-10-02 |
| Runtime class | Bronze |
| Intervention class | Clean |
| Continuous tip | Sanctuary, room `0x12`, Zelda delivered |
| Machine report | `recordings/clean_escape_20261002/replay.json` |
| Next maturity gate | C2: Eastern Palace prize from the real Sanctuary predecessor |

## Verified facts

The committed button tape replays from an actual core reset (`NONE`) with
blank SRAM prepared before frame 1. **36078 frames; zero desync, state loads,
RAM writes, refill assists, or deaths.** Final room `0x12`, module/submodule
`0x07/0x00`, `(1272,568)`, HP `10/24`, sword/lamp 1, follower 0, progress 2,
flags `0x15`. The Sanctuary story predicate passes, not just room arrival.
Normal enemy heart drops are collected through play; HP is never written.

Development controller forecasts are disclosed: 4141 rollout restores plus
seven secret-hole candidate restores. They are absent from the published
forward replay. The controller is development evidence; the replay is Clean.
Inputs and every-frame full-RAM CRCs are in
`recordings/clean_escape_20261002/controller.tape.npz`.

Composition: boot, uncle sword and lamp, secret-passage exit, courtyard and
main hall, `0x50 → 0x01 → 0x72 → 0x82 → 0x81`, required small key and big
key, Zelda rescue, castle return, mantle, sewers, right lever, Sanctuary
story. Required items and progression are earned by controller inputs.
The first main-hall leg uses a mapped north approach then cardinal corridor
waypoints; the old tile path caught the corridor lip.

The big-key carrier is selected by `$0CBA == 2`. Complete approach, swing
and recovery forecasts include the flail; eight frames are committed before
replanning. The no-assist arrival at frame 20698 passes offsets 0, 7, 11
without damage, and every committed frame replays with no loads/writes.
The rescued-Zelda arrival at frame 21805 passes complete escort plus forward
replay at offsets 0 and 7. Pins remain development-only.

The historical four-stage `full_tip` helper and
`recordings/verified_tip_run.json` still describe the earlier room-`0x50`
checkpoint. The Sanctuary audit uses the full `spine.hops` composition.
Older Survival runs in `recordings/sanctuary_20261001/` remain assisted
history, separate from C1 proof. Eastern development evidence has not earned C2.

ROM sha1 `6d4f10a8b10e10dbe624cb23cf03b88bb8252973`.
The loader prefers the local `refs/z3-json-data/` checkout, then the committed
JSON dump. Imports never download reference data.
