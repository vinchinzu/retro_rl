# MK1 SNES RAM map

Read-only WRAM via `get_ram()`. Confirmed: `data.json` + GameHacking.org USA
(CRC32 `DEF42945`) + object stride from P1 X `0x00DA` / P2 X `0x0174`.

| Name | Addr | Hex | Notes |
|------|-----:|-----|-------|
| game_mode | 34 | `0x0022` | Title/fight codes use 07/08/09 |
| match_counter | 10 | `0x000A` | 0 = M1 … 11 = Shang |
| p2_character | 36 | `0x0024` | 0–6 roster, 7 Goro, 8 Shang Tsung |
| timer | 290 | `0x0122` | Round timer, max ~154 |
| continue_timer | 999 | `0x03E7` | Continue screen |
| p2_rounds | 1207 | `0x04B7` | |
| p1_health | 1209 | `0x04B9` | Max **161** |
| p2_health | 1211 | `0x04BB` | Max **161** |
| p1_rounds | 6510 | `0x196E` | |
| p1_character | 6514 | `0x1972` | Liu Kang = **3** |
| p1_x / p1_y | 218 / 219 | `0x00DA` / `0x00DB` | Fighter object P1 |
| p2_x / p2_y | 372 / 373 | `0x0174` / `0x0175` | Stride `0x9A` from P1 |
| p1_state | 274 | `0x0112` | Object +`0x38` (anim/attack; probe) |
| p2_state | 430 | `0x01AE` | Same field on P2 |

High WRAM sprite tables `0x7688` / `0x7788` (Hacc) are optional if `get_ram()`
is long enough — not required for v3 obs.

Hitboxes in `ram.py` are **derived** AABBs from X/Y + facing (hurt 28×80
stand / 28×48 crouch; attack 40×24 when state ≠ 0). Policies see overlap /
in-range bits plus raw state bytes — not pixels.

v3 observation is 20 floats (`snapshot_features`). Incompatible with v1
(9-dim) and v2 (13-dim) MLP zips.

Character select cursor (hold ~8f, then release): `Cage --DOWN--> Kano
--DOWN--> Raiden --RIGHT--> Liu Kang`. RIGHT from Cage drops to the bottom
row (Sub-Zero / Sonya). Confirm with Y or A. `p1_character` tracks the cursor.
