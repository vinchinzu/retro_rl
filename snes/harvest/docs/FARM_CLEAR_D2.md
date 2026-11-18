# Day 2 perfect-pass checklist

Spring Day 2 is complete only when every item below is observed on the actual
loaded farm. The Composer checks two settled observations before accepting it.

**Verified from power-on, 2026-10-31:** `recordings/d2_perfect/power_on.json`
(`success`, 374,794 frames, 702.9 s). Checkpoint: `LN_d2_perfect_poweron`.

- [x] Zero bushes/weeds (`0x03`), including every farm quadrant.
- [x] Zero fence posts, including the house paddock and the y=31 wall.
- [x] Zero loose stones, small rocks, large boulders, damaged boulders, and stumps.
- [x] Eight potatoes planted and all eight watered.
- [x] Nine pasture tiles planted using the free grass-seed bag from the shed.
- [x] A real shipping-bin deposit before 17:00, recorded with a bin-value increase.
- [x] Empty hands, stable input, and a loaded farm; interior zero counts never pass.
- [x] Continuous Clean power-on evidence: zero RAM writes, mid-run loads, or assists.

Grass grows without watering: the ROM night update sends grass stages `>=0x70`
to the growth branch without the crop rain/watering check (`bank_82.asm`,
`CODE_82A899`). Water all eight potatoes; spending can water on pasture does
not accelerate its growth.

## Route and cash flow

Grape → shipping/shop → shed hoe + potato seeds → pocket clear → plant and
water potatoes → sweep all weeds → haul all posts and stones to ponds → hammer
all boulders → axe all stumps → sow the free grass bag. The daily clock stops
advancing after 18:00; finish the mandatory work before sleeping. Spa trips
restore stamina through controller inputs and then resume the unfinished work.

Plant and water before the long field cleanup, so the first crop grows that
night. The free grass bag starts pasture growth without another seed purchase.
Eight shipped potatoes are worth 640G at the native 80G price; the Day 2 receipt
proves establishment, while a later natural harvest must prove that cash flow.
This checklist does not certify a highest-score economy or a completed Year 1.

The owning policy is `harvest/planner/d2_work.py:next_d2_spec`; it splices one
child at a time through `DayPlanTask`. Partial quotas are progress, not a
whole-farm pass. The report's `d2_farm.final_status` records the final clauses,
including `planted_grass` and remaining seed stock.

## Shed and refill handoffs

The shed exits at tile `(26,30)`. Walk south beyond its facade, then approach
open yard `(25,32)`. The old `(25,28)` target is on the shed roof. Westward
motion from the threshold hits the facade; upward corrections re-enter the
shed. Tool acquisition must propagate an exit failure and finish on a loaded
farm, not merely observe the equipped-tool byte.

After a spa return at the west farm gate, stale-map recovery must head east
into the farm. Both debris clearers use the live-position `yard_load_action`;
a fixed shed escape walks back onto the path. Bounded rock/stump clears fail
immediately if they leave the farm. Pocket clear retains its explicit approach
route from the path.

## Movement and stamina invariants

| Tile | IDs | `FARM_WALKABLE` | Travel BFS | Clear |
|------|-----|-----------------|------------|-------|
| Soil / path | `0x00`, `0x01`, `0xA0`–`0xA3`, … | yes | yes | - |
| WEED | `0x03` | yes (ROM) | **no** | adjacent lift |
| STONE / FENCE | `0x04` / `0x05` | no | no | adjacent |
| Small ROCK | `0x06` | **no** | **no** | adjacent hammer |
| Stump 2×2 | `0x09`–`0x0C` | **no** | **no** | adjacent axe |
| Large rock 2×2 | `0x0D`–`0x10` | **no** | **no** | adjacent hammer, ROM 6 hits / 8-swing stam budget |
| Rock damage | `0x11`–`0x14` | **no** | **no** | same 2×2, keep swinging |

Push-facing (`player_action==0` + no pixel motion) must not seal the
**approach cell** next to a rock/stump being cleared.

Tools strike from adjacent standable ground. Hammer/axe swings remain
stationary: changing direction resets the ROM hit counter. A large object needs
six registered hits at two stamina each; the eight-swing starting budget is 16.
A grass bag needs nine hoed tiles and the native sow stamina, so its D2 phase
requests a refill when fewer than 20 stamina remain.

Weeds can be lifted at zero stamina. Posts and stones must actually be tossed
away, with empty hands afterward. Count neither a carried object nor a
viewport unload as cleared debris. Only the explicit shipper scene may consume
A pulses after 17:00; a tool animation is not shipping dialogue.

## Verification

Nearby regression pins: `y1_complete/progress_f254000` and
`y1_complete/progress_f255000`. Run from the repo root:

```bash
QT_QPA_PLATFORM=offscreen HEADLESS=1 uv run pytest \
  snes/harvest/tests/test_d2_tools.py -q -m rom -k d2_hammer
QT_QPA_PLATFORM=offscreen uv run pytest snes/harvest/tests -q -m 'not rom and not ml'
```

The short ROM checks prove a stable shed exit and a real boulder decrease through
the Composer. Final acceptance uses the existing runner from `snes/harvest`:

```bash
QT_QPA_PLATFORM=offscreen HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 \
  --power-on --stop-after-d2-clear --save-end-state LN_d2_perfect_poweron \
  --out recordings/d2_perfect/power_on.json
```

Current replay receipts and any residual work are in
[tasks/rr-20w-run16-defects.md](tasks/rr-20w-run16-defects.md).
