# Residual — rr-6o7.2 L8-B Magical Key (0x0F two-ladder return)

**Spine bead:** `rr-6o7.2` (claimed this sitting). Discovered: `rr-gw0x`
(closed: 0x1E type `0x33` HP96, 3 connecting arrows), `rr-5eb2`
(four-head Gleeok model). **Not** `rr-tne2` (closed). True `--through level8`
power-on is blocked on `rr-8t4.3` (post-L7 leave) and `rr-6o7.1` (natural
bush entry). This sitting is fixture-live only: `natural_entry=false`,
`route_eligible=false`. Do not STATUS. Do not close `rr-6o7.2` (acceptance
is power-on `--through level8-magic-key`).

## Frontier

`Level8InteriorMKReconFixture` — L8 mode-9 cellar `$EB=0x0F` `(136,141)`,
keys 8, bombs 6, bow 1, arrows 1, rupees 247, Magical Sword 3, Candle 2,
Magic Key **1**, TF `0x7F`, selected B = arrows. Two-ladder passage
(west/east ladders, pit). `room_item_id=0x0B`. Four HP-0 keese residual
(not census). Tile 36 on the Magical Key pad.

Settled this sitting: from `Level8Interior1FReconFixture`, sword-clear the
mixed 0x1F census (E1 no-clear 0x68 south-face UP was hitstun-blocked),
west 0x68 `(96,144)` slides DOWN to `(96,160)`, vacated gap → centre stairs
`(128,141)` → mode-9 cellar `0x0F`, natural `ADDR_MAGIC_KEY` 0→1 at the
pedestal. 2/2 E2/E3, 9074 frames. Return **not** taken: y=141 LEFT/RIGHT
is pit tile 250.

## Next live boundary

Two-ladder return from cellar `0x0F` toward hypothesized Gleeok `0x3C`.
L1 bow policy: DOWN to the floor, under the pit, UP the west ladder
`(48,93)`. Do not LEFT/RIGHT at y=141. Do not start the Gleeok fight.
Do not poke Magic Key / TF / doors / Gleeok HP. Keys 8→8 and bombs 6→6
unless a new gate spends one.

## Parallel this sitting

1. **rr-5eb2** — model written: `docs/tasks/rr-5eb2-gleeok-model.md`. ROM
   claims body type `0x45` (AttrC/D of hypothesized room `0x3C`); **not
   live**. South-stand dy=22, Magical Sword 9 hits (160+96+96+96). Bead
   stays open until a live census. Do not set `GLEEOK_FOUR_HEAD_OBJECT_TYPE`.
2. **Durable north-column controllers** — `0x7E→0x1E` is one-frame policy
   in `level8/north_column.py`. Gohma / Magical Key stairs / Gleeok stay
   fail-closed. Do not green `L8_THROUGH`. `topology.magic_key_room` stays
   unset on `LIVE_RECON_LEVEL8_TOPOLOGY`.

## Integrity

No poke of arrows, rupees, Magic Key, TF, doors, or Gleeok HP. Survival
health refill only. Leave proof is RAM + `zelda_i.screen_glance`, not MP4.

## Durable controllers / power-on seam

`--through level8-entry` / `level8-magic-key` / `level8` stay valid named
spine stops and still refuse to green: default `UNMEASURED_POST_L7_HANDOFF`
fails closed, `LIVE_RECON_L8_OVERRIDES` keeps `route_eligible=False`, and
`L8_THROUGH` is not greened. `make_magic_key_stairs_controller` stays
fail-closed. True power-on remains blocked on `rr-8t4.3` and `rr-6o7.1`.
