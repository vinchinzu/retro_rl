# ALTTP rando status

Planner owns this file. A save-state pin is not a continuous clear.

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M1 |
| Best verified result | JP FirstPlay to uncle fighter sword (`house_to_uncle`, natural_entry) |
| Last verification | 2026-08-09 |
| Runtime class | Bronze |
| Intervention class | Clean |

## Verified facts

ROM is `roms/zelda3_jp.sfc` (JP 1.0, xxh32 `0x8AC8FD15`). Boot uses `alttp.startup` into `custom_integrations/ALTTPRando-Snes/FirstPlay.state`: module `0x07`, room `0x04` (Link's House), control ready.

`house_to_uncle` (`z3_links_house` to `z3_uncle_sword`) plays the vanilla `alttp` opening from that predecessor: wake, lamp, house exit, overworld to the castle, then `castle_to_sword`. Evidence is `recordings/house_to_uncle.json` and `recordings/house_to_uncle.evidence.json`. Clean: no progression writes. One predecessor state load. Not continuous from power-on name select, and not a shuffled seed.

Multi-seed opening dry-run (`alttp_rando.opening_tip_campaign`, 2026-08-09): fixture seeds `1337` / `1338` / `1339`, S/T 3/3 claimable, threshold 2. Substrate is vanilla JP FirstPlay. Seed source is the fixture packages. Spoiler oracle is false. This is not shuffled-ROM robustness. Report: `docs/opening_tip_seed_campaign_dry.json`.

Sanctuary and Eastern tips are not verified. A patched randomizer ROM is not wired.
