# Residual — rr-8t4.1 L7-A pond drain / enter 0x79

**Spine bead:** `rr-8t4.1` (`in_progress`). Do not close it. Acceptance is
power-on `--through level7-entry` from `MEASURED_POST_L6_EXIT`. Do not
STATUS. Do not push unless asked.

## This sitting

Interior chapters were already composed (`9dac146a`). Survival still dies
before pond `0x42`. Three agents + parent compose:

### Pond drain module (wired)

`level7/pond.py` `Level7PondDrainController` / `make_pond_drain_controller`:
pause-select recorder slot 5 (no `$0656` poke), 12×B, stairs `(96,132)`
tile 114, dest play `0x79` `(120,205)`. `writes=0`,
`route_eligible=False`. Unit 17/17. Live 2/2 drain skipped: no
`0x42`+whistle=1 pin (`OW_L7Pond` is whistle=0). Do not poke Whistle.

`make_pond_entry_controller` now returns that controller. Food poke +
rupee top-up stay.

### Post-L6 overworld (partial)

`0x22` is a boxed mountain graveyard (PNG `l7_p22w7_final.png`). West →
`0x21` Magical Sword grave is **dead** (leftover `(90,165)` tile 38, 41
occupancy misses). North is the L6 cave (UP = mode 16). South corridor
is the only walk-off.

Greened `POST_L6_TO_POND_HOPS` (not the dead `0x25` pocket):

```
0x22 ↓0x32 →0x33 ↑0x23 →0x24 ↑0x14 ←0x13
```

`l7_p14w` 1/1 from `Level6ExitOverworld`, 1698f, leftover **OW `0x13`
`(240,189)`** mode 5, whistle 1, food 0. `_after_hops` succeeds only on
pond `0x42`; the prefix fails closed there (`post_l6_path_exhausted_unmeasured`).

`POST_L6_TO_BAIT_HOPS` `0x22→0x25` is a DEAD SPUR (kept for bait-micro
tests, not the spine default).

### path.py Gut

`path.py` 2813 → 1123. New `west.py` 1073 (soft ~1000, slightly over).
Merged `Room0DClear*` → `stairs0d.py`, `Room4A`/`Room1B`/`Room39` →
`digdogger.py`, `Room38Up` → `hungry.py`, `Room1ACandle` → `cellar.py`.
Public names still import from `zelda_i.level7.path`.

## Measured leave (do not invent)

`MEASURED_POST_L6_EXIT`: OW `0x22` `(112,125)` mode 5, TF `0x3F`, keys 2,
bombs 8, rupees 42, Rod/Bow/Whistle 1, Food 0, Candle 0, 8 HC full.

## Next live boundary

`0x13` `(240,189)` west/south toward the western forest band, then reuse
`0x64` gap / `0x53` inland-left / `0x52` boulder onto pond `0x42`, then
the new drain controller. Do not retry `0x22` LEFT. Do not RIGHT
`0x24→0x25`.

Do not fill `MEASURED_POST_L7_EXIT` from the TF-0 fixture leftover.

## Integrity

deaths 0, progression_writes 0, capacity_writes 0, position_writes 0,
whistle poke 0. Food poke allowed (Survival, disclosed). Unit: 163
passed on the L7 hop/pond/interior modules.
