# rr-npv.4 — Clean L8 Entrance→TF no retopup

Fixture-live only. `route_eligible=false`. Do not STATUS. Do not close the bead.

## Landed this sitting

0x2E select-before-leave stays. No LEFT of 112. On-column 0x55/0x56 peels
RIGHT to 128 while `x in [112,128)` **and cooldown==0**. On cooldown,
`(128,181)` + inbound 0x55 commits LEFT toward 112 and the next in-band
step stays LEFT (not RIGHT back to 128). One-frame UP+B, never hold UP.
Lip y>181 still UP-only. STAND_Y=181. `magic_key.py` 943 (merged
`_column_btn` / `_arrow_fire` peel). `north_column.py` untouched (996).
West-stream `(50,149)` `(88,181)` `(89,181)` stays blocked. No pokes.

Units: leftover `(128,181)` cooldown + inbound 0x55 → LEFT `column_peel`,
not RIGHT/idle/UP. Second step at x=120 still LEFT, not RIGHT. Mid-band
x=120 cooldown=0 still RIGHT. One-frame UP+B still not hold UP.
`test_east_edge_shoots_not_idle_on_stand_55` still UP/UP+B at cooldown=0.
`test_level8*.py` **277 passed**.

## ROM glance — first red (stop)

One trial. `Level8InteriorReconFixture` play `0x7E` `(120,205)` TF `0x7F`,
`--from-enter --clean --no-video`. Tag `l8clr_lab_h15`.

```text
[0] level8_north_manhandla_bomb: succ=True failed=False f=1700 -> 0x5e [120, 189] m5 hc=3 notes=['arrived_0x5e_120_189']
[1] level8_darknut_key_up: succ=True failed=False f=2757 -> 0x1e [120, 205] m5 hc=3 notes=['arrived_0x1e_120_205']
[2] level8_blue_gohma: succ=False failed=True f=142 -> 0x1e [128, 181] m17 hc=3 notes=['link_death']
```

Final: L8 `0x1e` `(128,181)` mode 17 TF `0x7F` MK 0 keys 8 bombs 2 rupees 252
hc 3 health `0x20`. Deaths 1. Writes 0. Shots 3 (rupees 255→252). **B=arrows**.
Did **not** walk off STAND_Y. Same death tile as h11/h13/h14; 16f longer
and one extra shot vs h14 (126f, shots 2).

Screenshot `nes/zelda_i/recordings/l8clr_lab_h15_final.png`: CONTINUE/SAVE/RETRY,
Blue Gohma north of center (eye open), Link at east stand x=128 y=181,
wooden arrow mid-column, 0x55 south/SW/SE of the stand, HUD B=arrows,
rupees 252, three empty hearts.

Commit LEFT toward 112 then at x=112 remaining cooldown RIGHT-peels back
to 128 (`lx > COLUMN_X_MIN` is false → RIGHT). Oscillation period stretched
128↔112, leftover tile unchanged.

## Leftover

Clean 0x1E Blue Gohma. Entry: play `0x1e` `(120,205)` mode 5 keys 8 bombs 2
hc 3 health `0x21` TF `0x7F` rupees 255 B=arrows. Death: `(128,181)` mode 17
health `0x20` deaths 1 rupees 252 shots 3. B=arrows.

## Blocked

**cooldown-oscillate 3/3 blocked** (h13 126f shots 2 / h14 126f shots 2 /
h15 142f shots 3). All `(128,181)` mode 17 B=arrows STAND_Y. Do not resume
LEFT/RIGHT peel on `[112,128]` at y=181. West-stream still blocked.
Do not STAND_Y=162. Do not dodge-only 6px left. Do not walk-into-body
south lip. `route_eligible=false`.

## Next hop

**Blocked.** New class required (not cooldown LEFT/RIGHT in the column
band). Do not spawn another 128↔112 peel try.
