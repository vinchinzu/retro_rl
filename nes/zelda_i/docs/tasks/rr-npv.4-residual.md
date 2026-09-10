# rr-npv.4 — Clean L8 Entrance→TF no retopup

Fixture-live only. `route_eligible=false`. Do not STATUS. Do not close the bead.

## Landed this sitting

Leftover-relative interior hops call `dungeon.door_hop.door_band_goal` (not copied):

- north column `_north_door`, 0x3C `north_3c_step`: off-column leftover uses door x, never UP into a wall at x=208
- west/south/east `RoomHopSpec` steps bind the door band from leftover
- Gleeok heart dest is RAM slot 19 (`$83/$97`) + hc bit, not frozen `(32,192)`

`SPINE_L8_RETOPUP` is already empty when `allow_pokes=False` (Wave 0). Unit test pins that.

## Unit tests

165 passed (touched L8 files minus `test_level8_spine_wiring.py`).

`test_level8_spine_wiring.py` does not collect: L5 parallel lane
`level5/path.py` `RamWaitHop.pred` is a required dataclass field after HopController defaults. Not this lane. Retopup assertion is duplicated in `test_level8_suffix.py`.

## ROM glance — first red (stop)

One trial. `Level8InteriorReconFixture` play `0x7E` `(120,205)` TF `0x7F`, bombs 8 keys 9 MK 0, `--from-enter --clean` (no infinite life, no retopup). Tag `l8_npv4_enter`.

```text
[0] level8_north_manhandla_bomb  1700f  play 0x5E (120,189) bombs 8→7 keys 9  succ
[1] level8_darknut_key_up         824f  play 0x5E (96,163) mode 17  link_death
```

Final: L8 `0x5e` `(96,163)` mode 17 TF `0x7F` MK 0 keys 9 bombs 7 rupees 255 hc 3 health `0x20`. Writes 0. Bomb spend is the 0x6E wall, not `SPINE_L8_RETOPUP`.

Gate (TF `0x80`, MK earned, deaths 0) is red. First red of the sitting — not retried.

## Leftover

Clean 0x5E darknut clear without infinite life. Pin leftover after manhandla: play `0x5E` `(120,189)` mode 5 TF `0x7F` bombs 7 keys 9 MK 0. Next: a heart-safe 0x5E policy, or Survival assist for combat only (not this Clean bead's STATUS). Knockback leftover-relative door/heart hops are unit-green; not ROM-proven past 0x5E.

## Blockers

- Clean combat in 0x5E (type `0x0C` HP128) kills Link (`hc=3`, no refill)
- `test_level8_spine_wiring.py` collection: L5 `RamWaitHop` dataclass (other lane)
- Full Entrance→TF without retopup still unrun past the death
