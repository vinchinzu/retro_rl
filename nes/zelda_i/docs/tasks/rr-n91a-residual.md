# Residual — rr-n91a L7-C 0x0D ROM stair-list (2026-09-04)

Did not STATUS. Did not poke. Did not touch `level8/**`, `AGENTS.md`, or
Agent A's `rr-6o7.2` leftover. Live 2/2 deferred (Agent A holds the
emulator). ROM-only leave.

## Verified

- Local ROM `nes/zelda_i/roms/Legend of Zelda, The.nes` iNES header
  `4E45531A…`. Scratch `scratch/dump_l7_stairs.py` reads PRG only.
- L9 stairway list at DC `0x19C10` is still `60 70 72 75 67 77 00 4F`
  (calibrates the +52 offset from ZeldaHacks LevelInfo).
- L6 cellar `0x08` AttrA=`0x3A` AttrB=`0x1D` matches live cellar08.
- L9 cellar `0x77` AttrA=`0x52` AttrB=`0x03` matches live Patra dests.
- L7 LevelInfo (PRG `0x199E4`): entrance `0x79` (live), tf_room `0x2B`,
  boss `0x2A`, stairway +52 = PRG `0x19A18` / iNES `0x19A28`.

## ROM claims (not live)

First-quest L7 stairway list PRG **`0x19A18`** / iNES **`0x19A28`**:

```
7B 4A FF FF FF FF FF FF
```

`LevelInfo_CellarRoomIdArray` (first 6) = **`0x7B`, `0x4A`**. `0xFF` unused.

UW L7–9 block PRG `0x18A00` (AttrA @ +room, AttrB @ +128+room):

| Cellar | AttrA (left, x<$80) | AttrB (right, x>=$80) | Kind |
|--------|---------------------|-----------------------|------|
| **`0x7B`** | **play `0x29`** | **play `0x0D`** | tunnel |
| `0x4A` | play `0x1A` | play `0x1A` | treasure (live Red Candle) |

CheckWarps from play `0x0D` hits cellar `0x7B` (AttrB). InitMode9 from
`0x0D` spawns the **right** ladder `x=$C0`. CheckSubroom UP at `Y<$40`
and `X>=$80` returns to `0x0D`. Forward dest is the **left** ladder
`x=$30` → play **`0x29`**.

Play-room doors (N/S/W/E, secret=AttrE&7):

| Room | N | S | W | E | secret |
|------|---|---|---|---|--------|
| `0x0D` | wall | wall | bomb | wall | **block_stairs (5)** |
| `0x29` | wall | wall | wall | bomb | none |
| `0x2A` | wall | wall | bomb | shutter | foes_item |
| `0x2B` | wall | wall | open | wall | none |

`0x0D` has no cardinal exit except the live west bomb back to `0x0C`.
The only forward gate is the secret staircase into `0x7B`.

`0x29` E-bomb pairs with `0x2A` W-bomb (LevelInfo boss). `0x2A` E-shutter
then `0x2B` W-open (tf_room).

Table: `level7/stairs.py`. Tests: `tests/test_level7_stairs.py` (ROM-byte
lock when the local ROM is present).

**NOSE_CELLAR predecessor (ROM-only):** play **`0x0D`** (AttrB). Far side
**`0x29`**. `NOSE_CELLAR.ram_id` stays `None` (no walk-on).

## Assumed

- aldonunez CheckWarps / InitMode9 / CheckSubroom comments match this
  cart (calibrated on live L6 `0x08` and L9 `0x77`).
- Walkthrough "tip of the nose → stairs → bomb into Aquamentus" is the
  same 0x0D → 0x7B → 0x29 → 0x2A chain. Walkthroughs are not proof.

## Dead beliefs

- **Dead: 0x0D is not a cellar predecessor.** ROM AttrB of `0x7B` is
  `0x0D`. The 4+ failed south-face vectors are walk-on geometry, not a
  wrong-room proof.
- **Dead: the `(204,88)` poke's 0x7B passage is a return-only dead end
  with no far side.** AttrA=`0x29` ≠ AttrB=`0x0D`. The poke spawned on
  the right/B ladder (`source=0x0D`) and UP at x=192 is CheckSubroom
  AttrB → `0x0D`. Floor y≈189 UP cannot CheckSubroom (`Y<$40` required);
  the far side is the **left** ladder `x=$30`. Same miss as L6 cellar08
  climbing the source ladder.
- **Dead: some other L7 play room CheckWarps into 0x7B.** Only `0x0D`
  and `0x29` are endpoints. No other UW room has AttrA/B = `0x0D`.

Not dead: the 0x0D south-face squeeze. ROM says the stairs are real
(`secret=5`); it does not make `(192,160)` reachable.

## Plan

Do **not** brute-force the squeeze this sitting. Next:

1. From `Level7Interior0DNoseCellarReconFixture` (or a fresh disclosed
   poke, still `route_eligible=false`): cellar-cross 0x7B like L6
   cellar08 — DOWN to floor, LEFT to `x=$30`, UP left ladder — dest
   claim play **`0x29`**. One trial only, and only if the emulator is
   free.
2. TAS south-face squeeze is a **later** sitting.
3. Do not take a heavier `$EB` stand-in this sitting.

## Leftover

Unchanged pin `Level7Interior0DClearedReconFixture`: L7 play **`0x0D`**
mode 5 `(63,149)`, `room_all_dead=1`, `0x68` at the NE plug.
`route_eligible=false`. Glance not re-taken (no emulator).
