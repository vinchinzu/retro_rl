## Result — rr-dbu.9 Alpha PB return through Moat

GREEN pure one-hop chain from the natural first-Alpha-PB pin through West
Ocean. Not continuous; no STATUS promotion.

| Hop | Rooms | Dual result | Controller |
|---|---|---:|---|
| Alpha PB escape | `0xA3AE` → `0xA322` | **2102f** ×2 | RAM progress watchdog + reactive jump/multi-shot recovery |
| Caterpillar climb | `0xA322` → `0x962A` | **1869f** ×2 | promoted human RLE from exact natural landed pin |
| Elevator connector | `0x962A` → `0x948C` | **627f** ×2 | promoted human RLE |
| Kihunter → Moat | `0x948C` → `0x95FF` | **1844f** ×2 | promoted human RLE |
| Moat spark | `0x95FF` → `0x93FE` | **3010f** ×2 | existing reactive `play_moat_cross` |

Final pin: West Ocean `0x93FE` `(49,1163)` pose `1`; items `0x3105`, beams
`0x1007`, PB capacity/count `5`.

The Alpha room did not need a trained model. Its enemy-state mismatch is
handled by a deterministic forward-progress watchdog: after 42 frames without
meaningful x progress, jump and multi-shot diagonally, then resume the route.

### Non-claims

- Did not promote a continuous tip beyond Ice.
- Did not claim the human RLEs are robust from arbitrary room-entry states;
  they are dual-green from their named natural predecessor pins.
