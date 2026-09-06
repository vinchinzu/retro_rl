# Residual — rr-6o7.1 L8-A post-L7 bush 0x6D

**Spine bead:** `rr-6o7.1` (closing). Power-on `--through level8-entry` is
**2/2 byte-identical**. Do not STATUS.

## Proof

`recordings/l8_entry_green.json` and `l8_entry_green2.json`:

```
ok=true  stop=level8_entry_live  set_state=0  continuous=true
end_frame 263330 both
L8 play 0x7E (120,205) mode 5  TF 0x7F  candle 2  selected=4
deaths 0  progression_writes=0  capacity_writes=0
status_claim=false
level8_post_l7_to_bush 6080f
level8_select_red_candle 150f
level8_burn_bush_enter 321f  observed_entry_room=0x7E
```

Walk: leftover `0x42` `(96,93)` LEFT to x=24, DOWN the west sand, RIGHT
to the x=112 gap, then the reverse pond hops to `0x6D` `(48,61)`. Do not
use `Level7PondToLevel8BushController` (south-shore fixture, fail-closes
on the north strip). Burn: drop to the aim row, fire at 2px, keep RIGHT
through mode 16 and level=8 mode=2 load.

`MEASURED_LEVEL8_ENTRY_TOPOLOGY` is the spine default (`0x7E`,
`route_eligible=True`). Isolated `make_post_l7_to_bush_controller()` still
defaults empty hops.

## Next

`rr-6o7.2` Magical Key from this leftover. `--through level8-magic-key`.
Do not STATUS.
