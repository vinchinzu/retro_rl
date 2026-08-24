## Residual — rr-2r06 Ice → Moat continuous compose

### Intent
Wire the dual-green K5 + K6 pure stack onto the continuous spine as
`--to alpha_pb` / `--to moat`. Compose from the Ice leave pin first.
Do **not** STATUS-promote until power-on dual continuous is green.

### One change
This hop: power-on `--to moat` dual continuous (scratch). Default CLI
tip stays `ice`. Do **not** STATUS-promote.

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
uv run python snes/super_metroid/scripts/record/continuous.py --to moat --no-video \
  --report snes/super_metroid/scratch/moat_poweron.json \
  --state-output snes/super_metroid/scratch/post_moat_poweron.state
# → GREEN 0x93FE (49,1163) p1 frames=175526 ×2 exact dual; max PB 5
# Ice-pin compose (not continuous evidence):
uv run python snes/super_metroid/scripts/probe/kpdr.py compose ice-to-moat \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_ceres_successor.state \
  --output snes/super_metroid/scratch/post_ice_to_moat_compose.state \
  --no-red-diag
# → GREEN 0x93FE (49,1163) p1 frames=28597 ×2 exact dual; max PB 5
```

### Acceptance
- [x] `--to alpha_pb` / `--to moat` registered (parent ice → alpha_pb → moat)
- [x] Spine hops Ice return + K5 reverse + K6 + Moat spark
- [x] Named-pin `bat_to_red` dual GREEN **698f** ×2 `(216,2422)` p82 (5f settle)
- [x] Zero-settle live Bat `bat_to_red` dual GREEN **768f** ×2 `(216,2443)` p165
- [x] Ice-pin compose GREEN through `bat_to_red` `(216,2443)` p165 @ 10642f
- [x] Ice-pin Red leave: checkpoint `bottom_floor → lower_ripper_1` dual GREEN **228f** ×2 `(94,2351)` p1 (freeze+step-off+standing hop; no WJ)
- [x] Ice-pin r1 pin: checkpoint `lower_ripper_1 → lower_ripper_2` dual GREEN **156f** ×2 `(125,2255)` p1
- [x] Ice-pin chain bottom → r2 GREEN **501f** `(115,2255)` p1
- [x] Ice-pin r2 pin: checkpoint `lower_ripper_2 → lower_ripper_3` dual GREEN **108f** ×2 `(140,2159)` p1
- [x] Ice-pin chain bottom → r3 GREEN **635f** `(130,2159)` p1
- [x] Ice-pin r3 pin: checkpoint `lower_ripper_3 → lower_ripper_4` dual GREEN **141f** ×2 `(155,2023)` p1
- [x] Ice-pin chain bottom → r4 GREEN **809f** `(146,2023)` p1
- [x] Ice-pin r4 pin: checkpoint `lower_ripper_4 → tunnel_floor` dual GREEN **69f** ×2 `(107,1883)` p2
- [x] Ice-pin chain bottom → tunnel GREEN **878f** `(107,1883)` p2
- [x] Ice-pin tunnel → mid floor dual GREEN **1210f** ×2 `(142,1625)` p65
- [x] Ice-pin mid floor → thin seat dual GREEN **2974f** ×2 `(86,587)` p2
- [x] Ice-pin chain bottom → thin seat GREEN **5062f** `(86,587)` p2
- [x] Ice-pin thin seat → upper_ripper_1 dual GREEN **94f** ×2 `(102,495)` p1
- [x] Ice-pin chain bottom → upper_ripper_1 GREEN **5156f** `(102,495)` p1
- [x] Ice-pin ur1 → ur2 dual GREEN **130f** ×2 `(119,391)` p1
- [x] Ice-pin ur2 → ur3 dual GREEN **158f** ×2 `(134,295)` p1
- [x] Ice-pin ur3 → ur4 dual GREEN **59f** ×2 `(144,207)` p1
- [x] Ice-pin chain bottom → upper_ripper_4 GREEN **5503f** `(144,207)` p1
- [x] Ice-pin ur3 → Hellway dual GREEN **283f** ×2 `0xA2F7` `(39,139)` p11 (ordinary left-door; 163f/`(237,139)` was door-slot fire)
- [x] Ice-pin Hellway → Caterpillar from that leave dual GREEN **2110f** ×2 `0xA322` `(39,1419)` p11
- [x] Ice-pin chain bottom → Hellway door-slot GREEN **5726f** `(237,139)` p11 (not ordinary settle; first hop no WJ; mid→thin still period WJ)
- [x] Ice-pin `play_red_to_hellway` wired: checkpoint climb to ordinary left-door dual GREEN **5846f** ×2 `0xA2F7` `(39,139)` p11
- [x] Ice-pin Caterpillar → Alpha PB dual GREEN **1372f** ×2 `0xA3AE` `(341,171)` p138 max PB 5
- [x] Ice-pin compose dual GREEN through Alpha PB **20016f** ×2 (Hellway 5846f + Caterpillar 2110f + PB hop 1418f)
- [x] Ice-pin compose dual GREEN through West Ocean **28597f** ×2 `0x93FE` `(49,1163)` p1 max PB 5
- [x] Power-on `--to moat` dual continuous **175526f** ×2 `0x93FE` `(49,1163)` p1 max PB 5 (scratch `moat_poweron.json` + `_dual.json`; Ice prefix **146937f**)
- [x] Graph: `moat_to_kihunter` reverse door (spark-setup leave-back)
- [x] Over-ocean spark from that leave dual GREEN **627f** ×2 `0xCA08` `(57,139)` p1
- [ ] Planner STATUS promote `--to moat` (default CLI stays `ice`)
- [ ] Wire `--to ws` (over-ocean spark is pin-green, not a continuous tip)

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

### Compose probe (2026-08-24, Ice climb wired)

From `post_ice_ceres_successor` (`kpdr.py compose ice-to-alpha-pb`):

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
| **red_to_hellway** | `(39,139)` p11 | 16488 | Ice climb **5846f** ×2 ordinary left-door. Was RED (6199f tape walked back to Bat). |
| **hellway_to_caterpillar** | `(39,1419)` p11 | 18598 | **2110f** ×2 from Ice leave |
| **caterpillar_to_alpha_pb** | `(341,171)` p138 | 20016 | Compose hop **1418f**; isolated dual **1372f** ×2; max PB 5 |
| **alpha_pb_to_caterpillar** | `(39,1931)` p11 | 21438 | Compose hop **1422f**; saved-pin dual **2102f** ×2 p164 |
| **caterpillar_to_elevator** | `(128,294)` p155 | 23307 | **1869f** (matches isolated dual) |
| **elevator_to_kihunter** | `(392,697)` p144 | 23934 | **627f** (saved-pin dual **626f**) |
| **kihunter_to_moat** | `(39,139)` p9 | 25778 | **1844f** ×2 |
| **moat_cross** | `(49,1163)` p1 | 28597 | Compose hop **2819f**; saved-pin dual **2941f**; named isolated **3010f** |

### Ice checkpoints from Ice-pin Red leave `(216,2443)` p165

Public policy: Hi-Jump + Ice. Freeze Ripper 1 from the floor, step ~28px
off the ice column, standing hop onto it (same-column jump bonks the
underside). Standing hop onto Rippers 2 and 3, crouch-jump onto Ripper 4,
then crouch-jump left onto the tunnel alcove, WJ to the thin seat, freeze
the y≈520 Ripper from that solid seat and standing-hop the upper ice
ladder to y=207. Freeze only on the facing (right) side. Do not walk on
ice while shooting. Do not RIGHT+A from aim-up (pose 81 falls through).
https://wiki.supermetroid.run/Red_Tower

| | frames | seconds | clock |
|---|---:|---:|---|
| product hop from this leave (prior) | 6267 | 104.278 | 01:44.45 |
| | RED to Bat `0xA3DD` (`red_to_hellway_from_seat`) | | |
| before edge 1 `bottom → r1` (double WJ) | 335 | 5.574 | 00:05.58 |
| after edge 1 `bottom → r1` (freeze+step-off+standing hop, no WJ) | 228 | 3.794 | 00:03.80 |
| Δ edge 1 | −107 | −1.780 | faster, no WJ |
| after edge 2 `r1 → r2` dual exact (old WJ pin) | 156 | 2.596 | 00:02.60 |
| after edge 3 `r2 → r3` dual exact | 108 | 1.797 | 00:01.80 |
| after edge 4 `r3 → r4` dual exact | 141 | 2.346 | 00:02.35 |
| after edge 5 `r4 → tunnel` dual exact | 69 | 1.148 | 00:01.15 |
| this tape r1 → r2 (from new leave) | 228 | 3.794 | 00:03.80 |
| this tape r2 → r3 | 241 | 4.010 | 00:04.01 |
| this tape r3 → r4 | 231 | 3.844 | 00:03.85 |
| chain p165 → r3 | 697 | 11.597 | 00:11.61 |
| chain p165 → r4 | 928 | 15.441 | 00:15.46 |
| chain p165 → tunnel | 997 | 16.589 | 00:16.61 |
| tunnel → mid floor dual exact | 1210 | 20.134 | 00:20.17 |
| chain p165 → mid floor | 2207 | 36.723 | 00:36.79 |
| mid floor → thin seat dual exact | 2974 | 49.485 | 00:49.57 |
| chain p165 → thin seat | 5181 | 86.208 | 01:26.37 |
| thin seat → upper_ripper_1 dual exact | 94 | 1.564 | 00:01.57 |
| chain p165 → upper_ripper_1 | 5275 | 87.772 | 01:27.93 |
| ur1 → ur2 dual exact | 130 | 2.163 | 00:02.17 |
| ur2 → ur3 dual exact | 158 | 2.629 | 00:02.63 |
| ur3 → ur4 dual exact | 59 | 0.982 | 00:00.98 |
| ur3 → Hellway dual exact (door-slot fire) | 163 | 2.712 | 00:02.72 |
| ur3 → Hellway dual exact (ordinary left-door) | 283 | 4.709 | 00:04.72 |
| Hellway → Caterpillar from `(39,139)` p11 | 2110 | 35.109 | 00:35.17 |
| Ice-pin `play_red_to_hellway` (wired climb) | 5846 | 97.273 | 01:37.43 |
| Ice-pin Caterpillar → Alpha PB isolated | 1372 | 22.829 | 00:22.87 |
| Ice-pin compose Ice → Alpha PB | 20016 | 333.052 | 05:33.60 |
| Ice-pin Alpha PB → Caterpillar isolated | 2102 | 34.976 | 00:35.03 |
| Ice-pin Caterpillar → elevator | 1869 | 31.099 | 00:31.15 |
| Ice-pin elevator → Kihunter | 626 | 10.416 | 00:10.43 |
| Ice-pin Kihunter → Moat | 1844 | 30.683 | 00:30.73 |
| Ice-pin Moat spark isolated | 2941 | 48.936 | 00:49.02 |
| Ice-pin Alpha PB pin → West Ocean | 9355 | 155.660 | 02:35.92 |
| Ice-pin compose Ice → West Ocean | 28597 | 475.833 | 07:56.62 |
| chain p165 → Hellway door-slot | 5726 | 95.276 | 01:35.43 |

Upper ice ladder is dual-exact through ordinary Hellway left-door.
Product ur3→ur4 remains the verified 10–28px aim-then-shot hop
(**59f** ×2, leave `(144, 207)` p1). The Hellway edge attaches from the
same ur3 pin, freezes ur4 in that band, then **skips the 8f ur4 settle**:
12f UP+X+A, A-only through the 1-tile hole at x≈134, RIGHT only once
y≤140, shoot-walk the blue door, **keep RIGHT until gs=8 x≤80**.
Stopping on the first Hellway `room_id` left the Red Tower door slot
`(237,139)` p11 — x underflows to 65522 and a Samus Eater eats the
traverse. Ordinary leave is **283f** ×2 `0xA2F7` `(39,139)` p11.
Successor `hellway_to_caterpillar` from that pin is dual **2110f** ×2
Caterpillar `(39,1419)` p11 (product from the 6199f tape leave was
**2218f** at the same Caterpillar coords). Zero-settle the successor —
5f idle drops the airborne door seat into a plant.
https://wiki.supermetroid.run/Hellway
Wired into `play_red_to_hellway` (Ice+HJ bottom floor). Tape body remains
the fallback when that seat is absent. Mid→thin is still the 2974f period WJ.

```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py compose ice-to-moat \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_ceres_successor.state \
  --output snes/super_metroid/scratch/post_ice_to_moat_compose.state \
  --no-red-diag
# → GREEN 0x93FE (49,1163) p1 frames=28597 ×2 exact dual; max PB 5
# Alpha PB pin onward:
uv run python snes/super_metroid/scripts/probe/kpdr.py compose alpha-pb-to-moat \
  --source snes/super_metroid/scratch/post_ice_to_alpha_pb_compose.state \
  --output snes/super_metroid/scratch/post_ice_alpha_pb_to_moat_compose.state \
  --no-red-diag
# → GREEN 0x93FE (49,1163) p1 frames=9355 ×2 exact dual
```

Public policy: jump the Alpha PB platforms back to the right door; do not
fall into the floor eaters. Caterpillar climb / elevator / Kihunter are
the dual-green human RLEs from that return seat. Moat is the existing
reactive spark. https://wiki.supermetroid.run/Alpha_Power_Bomb_Room
https://wiki.supermetroid.run/The_Moat

### Power-on `--to moat` (2026-08-24)

First run RED integrity: unknown Moat → Kihunter reverse at **172796f**
leave `(21,139)` p12 (play_moat_cross standing setup
`play_leave_moat_to_kihunter`). Route still reached West Ocean **175526f**.
Hand-authored `moat_to_kihunter` on SPEED_GRAPH (same pattern as Speed-return
reverse doors). Dual re-run exact **175526f** ×2, integrity green,
loads/prog/deaths 0. Spark-reentry `kihunter_to_moat` @175000 pose 201.

| | frames | seconds | clock |
|---|---:|---:|---|
| Ice-pin compose Ice → West Ocean | 28597 | 475.833 | 07:56.62 |
| Power-on `--to moat` (scratch dual) | 175526 | 2920.624 | 48:45.43 |
| Ice prefix (Ceres-successor) | 146937 | 2444.924 | 40:48.95 |
| Power-on post-Ice | 28589 | 475.700 | 07:56.48 |
| Over-ocean spark from power-on leave | 627 | 10.433 | 00:10.45 |

```bash
uv run python snes/super_metroid/scripts/probe/west_ocean_spark.py pure-ws \
  --source snes/super_metroid/scratch/post_moat_poweron.state \
  --out snes/super_metroid/scratch/post_moat_poweron_wo_to_ws.state
# → GREEN 0xCA08 (57,139) p1 frames=627 ×2 exact dual
```

https://wiki.supermetroid.run/West_Ocean

### Next action (required)
- **One change:** wire `--to ws` (`rr-p2bw`; `play_west_ocean_over_ocean_spark`
  from the power-on West Ocean leave). Planner STATUS for `moat` is
  `rr-g3nj`. Do not STATUS-promote from this residual.
- Mid→thin is still the 2974f period WJ (p2 alcove ceiling caps a
  standing hop at y≈1219; next is freeze-Geega or 1–2 bombs then one
  jump onto p3).
- Do not mash more `bat_to_red` subpixel
- Keep `post_ice_bat_to_red_pure` (718f p10) and the 6199f tape leave.

### Non-claims
- Did not STATUS-promote past Ice
- Did not change `DEFAULT_CONTINUOUS_TIP`
- Did not write `recordings/moat.json` (scratch only)
- Did not wire `--to ws` / ship interior
- Did not replace mid→thin period WJ
- Did not clobber `post_ice_bat_to_red_pure` (718f successor)
- 698f named-pin dual was 5f-settle morph-fall, not zero-settle compose
