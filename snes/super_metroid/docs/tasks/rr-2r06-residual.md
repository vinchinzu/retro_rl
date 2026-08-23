## Residual — rr-2r06 Ice → Moat continuous compose

### Intent
Wire the dual-green K5 + K6 pure stack onto the continuous spine as
`--to alpha_pb` / `--to moat`. Compose from the Ice leave pin first.
Do **not** STATUS-promote until power-on dual continuous is green.

### One change
This hop: `bat_to_red` zero-settle water recovery + Red-bottom floor seat.
Default CLI tip stays `ice`.

### Names (not Wave)

`bat_to_red` is **Skree Boost Room** (`0xA3DD`, also called Bat Room) LEFT
into **Red Tower** bottom (`0xA253`). Not Norfair Bat Cave (`0xB07A`).
Wave is already held (`beams 0x1007` Charge+Spazer+Wave+Ice). Next item
on this stack is **Alpha Power Bombs**, then Moat / West Ocean.

High path: three dry pipe platforms (reverse of `play_bat_to_below_spazer`).
Water under the pipes is a different climb — HJ without Gravity does not
spin-jump out (`h_underwaterCrouchJumpDownGrab`).

### Source state
`scratch/post_ice_ceres_successor.state` (continuous Ice leave, Ceres-successor).

### Verify paste
```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py compose ice-to-moat \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_ceres_successor.state \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_to_moat_compose.state
```

### Acceptance
- [x] `--to alpha_pb` / `--to moat` registered (parent ice → alpha_pb → moat)
- [x] Spine hops Ice return + K5 reverse + K6 + Moat spark
- [x] Named-pin `bat_to_red` dual GREEN **698f** ×2 `(216,2422)` p82 (5f settle)
- [x] Zero-settle live Bat `bat_to_red` dual GREEN **768f** ×2 `(216,2443)` p165
- [x] Ice-pin compose GREEN through `bat_to_red` `(216,2443)` p165 @ 10642f
- [x] Ice-pin Red leave: checkpoint `bottom_floor → lower_ripper_1` GREEN **335f** (clearance-gated double WJ)
- [x] Ice-pin r1 pin: checkpoint `lower_ripper_1 → lower_ripper_2` dual GREEN **156f** ×2 `(125,2255)` p1
- [x] Ice-pin chain bottom → r2 GREEN **501f** `(115,2255)` p1
- [x] Ice-pin r2 pin: checkpoint `lower_ripper_2 → lower_ripper_3` dual GREEN **108f** ×2 `(140,2159)` p1
- [x] Ice-pin chain bottom → r3 GREEN **635f** `(130,2159)` p1
- [x] Ice-pin r3 pin: checkpoint `lower_ripper_3 → lower_ripper_4` dual GREEN **141f** ×2 `(155,2023)` p1
- [x] Ice-pin chain bottom → r4 GREEN **809f** `(146,2023)` p1
- [ ] Ice-pin compose GREEN (West Ocean `0x93FE`) — RED at `red_to_hellway` (product RLE)
- [ ] Ice-pin compose dual exact
- [ ] Power-on `--to moat` dual continuous (planner STATUS)

### Zero-settle live Bat (`post_ice_live_bat` / `post_ice_below_to_bat_pure`)

Same RAM at settle 0: pose **12** crouch `(472,139)` y_sub=65535 leftover
left momentum. kpdr.py's 5f idle drops that sill into morph-fall pose 42 —
the 698f dual was that accidental fall, not compose-continuous.

Public policy: stand out of the crouch, run up, spin-jump the dry pipes.
If she falls in the pool: crouch-jump + down-grab, not mash-A into the
pipe underside. https://wiki.supermetroid.run/Skree_Boost_Room

| | frames | seconds | clock |
|---|---:|---:|---|
| before (zero-settle mash-A) | 2200 | 36.606 | 00:36.67 |
| after (stand + water CJ/DG + floor seat) | 768 | 12.779 | 00:12.80 |
| Δ | −1432 | −23.827 | faster (RED→GREEN) |

Dual exact **768f** ×2, leave `(216, 2443)` p165. Floor y matches the
718f successor; pose is land-165 not run-10 (mom=0 x_sub=0 vs mom=1
x_sub=20480). Did not overwrite `post_ice_bat_to_red_pure` (718f /
`(216,2443)` p10). `kpdr.py pure bat-to-red` is now zero-settle.

### Compose probe (2026-08-23, after water recovery)

From `post_ice_ceres_successor` (`kpdr.py compose ice-to-moat`):

| Hop | Leave pin | Frames | Note |
|-----|-----------|-------:|------|
| ice_to_snake | `(472,395)` p10 | 538 | matches pure dual |
| ice_snake_to_tutorial | `(39,127)` p81 | 2959 | matches |
| ice_tutorial_to_gate | `(807,131)` p81 | 3938 | matches |
| ice_gate_to_business | `(41,907)` p25 | 4826 | matches |
| ice_business_to_warehouse | `(37,139)` p138 | 8342 | faster than 10255f thrash dual |
| warehouse_to_east | `(216,364)` p26 | 8627 | matches |
| east_to_glass | `(216,395)` p26 | 8894 | matches |
| glass_to_west | `(216,139)` p10 | 9117 | matches |
| west_to_below | `(472,393)` p82 | 9389 | matches |
| below_to_bat | `(472,139)` p12 | 9874 | sill pin; matches dual |
| **bat_to_red** | `(216,2443)` p165 | 10642 | GREEN 768f; was water timeout |
| **red_to_hellway** | RED | 16824 | Product RLE+IBJ still walks back to Bat. Ice checkpoints from this leave are green through lower_ripper_4 (see below). 718f successor still greens 6199f from p10 mom=1. |

### Ice checkpoints from Ice-pin Red leave `(216,2443)` p165

Public policy: Hi-Jump + Ice. Double WJ up the right wall onto frozen
Ripper 1, standing hop onto Rippers 2 and 3, crouch-jump onto Ripper 4.
Freeze only on the facing (right) side. Do not walk on the ice while
shooting (falls off). Do not RIGHT+A from aim-up (pose 81 falls through).
https://wiki.supermetroid.run/Red_Tower

| | frames | seconds | clock |
|---|---:|---:|---|
| product hop from this leave (prior) | 6267 | 104.278 | 01:44.45 |
| | RED to Bat `0xA3DD` (`red_to_hellway_from_seat`) | | |
| after edge 1 `bottom → r1` (clearance-gated double WJ) | 335 | 5.574 | 00:05.58 |
| after edge 2 `r1 → r2` dual exact | 156 | 2.596 | 00:02.60 |
| after edge 3 `r2 → r3` dual exact | 108 | 1.797 | 00:01.80 |
| after edge 4 `r3 → r4` dual exact | 141 | 2.346 | 00:02.35 |
| chain p165 → r3 | 635 | 10.566 | 00:10.58 |
| chain p165 → r4 | 809 | 13.461 | 00:13.48 |

Dual exact **141f** ×2 on r3→r4, leave `(155, 2023)` p1. Standing hop
apex is ~3px above r4 and falls through; crouch-jump apex **2008**
clears. Chain **809f** leaves `(146, 2023)` p1 from
`scratch/bat_zero_settle_eq216_leave.state`. Not wired into
`play_red_to_hellway` (still the 6199f tape body).

```bash
uv run python snes/super_metroid/scripts/probe/red_ice_climb.py \
  --source snes/super_metroid/scratch/bat_zero_settle_eq216_leave.state \
  --edge chain --phase-offsets 0
uv run python snes/super_metroid/scripts/probe/red_ice_climb.py \
  --source snes/super_metroid/scratch/red_ice_p165_ripper3.state \
  --edge 4
```

### Next action (required)
- **One change:** checkpoint `lower_ripper_4 → tunnel_floor` (solid left
  alcove ~x104 y1883). Crouch-jump from r4 clears height (apex ~1847)
  but x≈155 is open shaft; need leftward travel onto the seat. Do not
  walk off r4 ice.
- **Source:** `scratch/red_ice_p165_ripper4.state` / chain leave
  `(146,2023)` p1
- Do not mash more `bat_to_red` subpixel
- Keep `post_ice_bat_to_red_pure` (718f p10) until a Hellway leave greens
  the successor. Do not replace product `play_red_to_hellway` until the
  checkpoint climb actually reaches Hellway.

### Non-claims
- Did not STATUS-promote past Ice
- Did not change `DEFAULT_CONTINUOUS_TIP`
- Compose from a pin is not continuous evidence
- Did not clobber `post_ice_bat_to_red_pure` (718f successor)
- 698f named-pin dual was 5f-settle morph-fall, not zero-settle compose
