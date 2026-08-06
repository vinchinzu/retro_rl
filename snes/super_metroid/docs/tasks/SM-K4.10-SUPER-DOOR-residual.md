# Residual — SM-K4.10-SUPER / rr-re9 (Double Chamber Super door + Wave)

## Result

PARTIAL — path geometry settled; Super sill + Wave PLM still RED

## Files changed

- `routes/kpdr/k4_wave.py` — post-gate Super approach rewritten to **missile-ledge runway**:
  - `_dc_missiles_and_runway` — pack on y≈139, backup under gate ~x425
  - `_dc_ledge_dash_and_launch` — dash to edge ~x575, spin launch (peaks y≈69)
  - `_dc_super_door_push` — Super red door when sill reached
  - **Banned:** open-loop WJ on spike floor (old pin ~(920,311) `wj_r`)
- `scripts/probe/watch_pure_hop.py` — headed pure hop viewer (pygame)
- Scratch research pins (gitignored): `dev_dc_post_gate_open`, `dev_dc_super_runway`,
  `dev_dc_runway_edge`, `dev_dc_post_missiles`

## Verify paste

```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py pure double-chamber-to-wave \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_single_to_double_chamber_pure.state \
  --no-red-diag
# RED: Wave door missed; room=0xADAD pose=25 xy=(920,323) missiles=20 frames=1791
# reason=…_mid_air  (high launch, drop at door column — not floor WJ)

# Headed:
uv run python snes/super_metroid/scripts/probe/watch_pure_hop.py double-chamber-to-wave \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_single_to_double_chamber_pure.state \
  --speed 0.35 --scale 3
```

## Acceptance

- [x] Gate open remains GREEN (rr-dbu.10) — do not thrash human RLE seat
- [x] Path uses **same ledge as missiles** (y≈139), not spike floor
- [ ] Super sill ~(920–940,139) pure, then Wave `0xADDE` beams `|= 0x0001` dual
- [ ] Residual PROCESS on final GREEN

## Geometry (verified this wave)

| Fact | Value |
|------|------:|
| Gate open | dual solid ~(488,139); human RLE f4650–5200 seat x∈[370,375] y≤139 |
| Missile ledge | solid y≈139, x≈**414–608** (past open gate) |
| Missile pack | ~x494–500; pure sees missiles 15→20 |
| Runway | backup left to **~x420–425** on **same ledge** (longest dash) |
| Edge launch | dash RIGHT+B to **~x575–582**, then RIGHT+B+A |
| Launch peak | **y≈69–71** around x≈630–640 (above door sill height) |
| High cross | air reaches ~x780 y≈170 before drop |
| Fail pin (new) | **(920, 323)** pose 25 mid-air — door column, low |
| Fail pin (old) | (920, 311) floor open-loop `wj_r` — **retired** |
| Door sill (place) | solid ~(920–960, 139); Super red → Wave `0xADDE` |
| Wave chozo | ~(171,120) bit `0x0001` (place-offline only) |
| Human tape | ends ADAD ~(494,139) — **no Super / Wave** on tape |

## Path recipe (intended product)

1. Gate open (done, human RLE) → past bars on upper ledge  
2. Missiles on ledge y≈139  
3. Backup under gate ~x425 **same ledge**  
4. Face right, dash to edge ~x575  
5. Spin launch high (peak y≲80)  
6. **One better launch + single WJ**, *or* **2–3 WJ** off right column while still high  
7. Land sill y≈139 x≳920 → Super → Wave → PLM  

## Residual risks / RED cause

- Launch is “almost perfect” visually; missing **wall contact timing** while high
  (x≳750–900, y≲200) before free-fall to y~320
- Low dash momentum at edge (mom≈2, speed_counter≈1) — longer B runway or
  better launch window may enable **one WJ**
- Double/triple WJ from mid-high contact is the safe ladder if single-WJ launch
  stays short
- Spike floor is a hard fail zone — never resume open-loop WJ below y≈280

## Next action (required)

- **Next card ID:** rr-re9 continue (same bead)
- **One change (pick one):**
  1. **Preferred:** tune ledge launch (earlier/later A, more speed) so one WJ at
     right column lands sill ~(920,139), **or**
  2. From high air x∈[780,920] y∈[100,200]: 2–3 `walljump_once` (into=RIGHT,
     flip=LEFT) chain — dual pure GREEN sill then Super+PLM
- **Source state:** `scratch/post_single_to_double_chamber_pure.state`  
  Research pins: `dev_dc_super_runway.state`, `dev_dc_runway_edge.state`  
  Iterate WJ with `watch_pure_hop.py` headed

## Non-claims

- No Wave beam bit / continuous `--to wave` / STATUS promote
- Gate open human RLE not reopened for thrash
- Place-based sill/Wave collect is not pure evidence

## Probe pin (PARTIAL)

room=0xADAD pose=25 x=920 y=323 door_transition=0 frames=1791  
last_pin=post_single_to_double_chamber_pure missiles=20 beams=0x0000  
path=missile-ledge runway + high launch; missing high WJ to sill
