# Residual — rr-6o7.2 L8-B Magical Key (0x1F stairs cellar)

**Spine bead:** `rr-6o7.2` (claimed this sitting). Discovered: `rr-gw0x`
(closed: 0x1E type `0x33` HP96, 3 connecting arrows), `rr-5eb2`
(four-head Gleeok model). **Not** `rr-tne2` (closed). True `--through level8`
power-on is blocked on `rr-8t4.3` (post-L7 leave) and `rr-6o7.1` (natural
bush entry). This sitting is fixture-live only: `natural_entry=false`,
`route_eligible=false`. Do not STATUS.

## Frontier

`Level8Interior1FReconFixture` — L8 play `0x1F` `(16,141)`, keys 8, bombs 6,
bow 1, arrows 1, rupees 246, Magical Sword 3, Candle 2, Magic Key **0**,
TF `0x7F`, selected B = arrows. Census: 2× `0x16` pols_voice HP160, 2×
`0x0C` HP128, 2× `0x0B` darknut HP64 (`room_obj_count=6`). Centre stairs
sprite `0x68` at ~(96,144) (PNG; not room population). `room_item_id=0x03`.
`cur_opened_doors`/`open_doorway_mask` = 0 on arrival (west mouth is an
OPEN doorway; byte 0 is not a closed-door claim).

Settled this sitting (rr-gw0x): 0x1E body type `0x33` HP96, **3 connecting
wooden arrows** (HP 96→64→32→0), 9 shots loosed (rupees 255→246), east
kill-clear shutter (`cur_opened_doors` 0x04→0x0D, PNG east black, mask
stayed 0x04). `BLUE_GOHMA_ARROWS_REQUIRED = 3` is the live connect count.
Colour not asserted (never `0x34`). 2/2 D1b/D2, 1731 frames.

## Next live boundary

Magic Key cellar in `0x1F`: take the centre stairs (`0x68` ~(96,144)),
pick up `ADDR_MAGIC_KEY`, return. Do not assume a clear is required
before the stairs. Do not poke Magic Key / TF / doors. Keys 8→8 and
bombs 6→6 unless a new gate spends one.

## Parallel this sitting

1. **rr-5eb2** — ROM + L4/L6 Gleeok model **before** any live L8 boss fight.
   Do not assume body type `0x45`. No Gleeok room fixture exists yet.
2. **Durable north-column controllers** — promote the already-settled
   `0x7E→0x1F` chain into `level8/path.py` so a later power-on compose can
   attach it. Keep Gohma/Gleeok factories fail-closed (Gohma notes may
   record measured type `0x33` / 3 connects). Do not green `L8_THROUGH`.

## Integrity

No poke of arrows, rupees, Magic Key, TF, doors, or Gleeok HP. Survival
health refill only. Leave proof is RAM + `zelda_i.screen_glance`, not MP4.
