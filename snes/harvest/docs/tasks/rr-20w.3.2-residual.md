## Residual — rr-20w.3.2 CROP_ESTABLISH nav_hoe_ring after harvest

**Status:** unit GREEN + second-plot ROM GREEN. Campaign replant running.
Do not STATUS.

### Why it timed out

Run7 D15–D18: `nav_hoe_ring_0_down` / `_1_down` after a pocket harvest.
Farmer was already on the notch `(209,457)` heading for a face-down stand
nudged 5px north to `(216,451)` (chebyshev 7). Ring nav used radius **3**
and `soft_radius=3`, so it never arrived.

D14 was a second hole: `hoe_until_tilled timeout tid=0x08` — watered
soil is already plantable, not in `HOED_OR_PLANTED`.

A radius-10 ball around the 5px-away nudge accepted the *away* neighbor
(live miss: stand `(18,29)` landed `(17,29)`, hoe target `(19,29)` dist=2).

### Fix

- Ring nav aims at stand **tile center**, radius/soft **7** (south edge
  arrives; neighbor tile chebyshev 8 stays out).
- Face is a 1f tap every 8f, not a hold (holding walks onto the target).
- Standing on the target steps off before Y.
- `HOED_OR_PLANTED` includes watered tilled `0x08`.
- With `ram`, skip ring tiles already tilled/crop (no nav to hoe 0x08).

### ROM

| Hop | Pin | Frames | Result |
|-----|-----|--------|--------|
| second ring establish | `Y1_D3_PostShop` | 3111 | 18:02→18:10, (19,28) 8×0x54, bag spent, 16 planted, farm |

Campaign: `Y1_D3_Morning --end-of-spring` in flight (`run8_replant.log`).

### Non-claims

No STATUS. Did not start from `Y1_D2_Morning_After_D1`. Did not record a
BFS-closable walk. Did not treat CrossMap origin-return as shop success.
Did not claim rr-3ae8 (refill).
