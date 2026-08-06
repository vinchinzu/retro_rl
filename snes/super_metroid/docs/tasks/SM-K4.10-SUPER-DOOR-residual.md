# Residual — SM-K4.10-SUPER / rr-re9 (Double Chamber Super door + Wave)

## Result

GREEN — pure dual Super red door → Wave `0xADDE` beams `|= 0x0001`

## Files changed

- `routes/kpdr/k4_wave.py` — missile-ledge runway + **high door-column classic WJ**:
  - `_dc_missiles_and_runway` — pack on y≈139, backup under gate ~x425
  - `_dc_ledge_dash_and_launch` — dash edge **x600**, spin launch, wall contact
    ~(923,238), classic away WJ (`LEFT×3` + `LEFT+A×6` via `walljump_once` /
    `WallJumpTiming`), left follow 8f, RIGHT arc to sill ~(929,116)
  - `_dc_super_door_push` — Super red door when sill reached
  - **Banned:** open-loop WJ on spike floor (old pin ~(920,311) `wj_r`)
- Scratch research pins (gitignored): `dev_dc_door_contact_high`, etc.

## Verify paste

```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py pure double-chamber-to-wave \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_single_to_double_chamber_pure.state \
  --no-red-diag
# dual ×2: success=true room=0xADDE beams=0x0001 frames=2327 xy≈(171,123)
```

## Acceptance

- [x] Gate open remains GREEN (rr-dbu.10) — do not thrash human RLE seat
- [x] Path uses **same ledge as missiles** (y≈139), not spike floor
- [x] Super sill pure, then Wave `0xADDE` beams `|= 0x0001` dual
- [x] Residual PROCESS on final GREEN

## Geometry (verified this wave)

| Fact | Value |
|------|------:|
| Gate open | dual solid ~(488,139); human RLE f4650–5200 seat x∈[370,375] y≤139 |
| Missile ledge | solid y≈139, x≈**414–608** (past open gate) |
| Missile pack | ~x494–500; pure sees missiles 15→20 |
| Runway | backup left to **~x420–425** on **same ledge** |
| Edge launch | dash RIGHT+B to **~x600**, then RIGHT+B+A |
| Launch peak | **y≈60** (edge 600) |
| Wall contact | **(923, 238)** pose 25, vx=0 |
| Door WJ | classic away: LEFT×3 + LEFT+A×6, left follow 8f |
| Sill land | **~(929, 116)** vy=0 → Super push |
| Wave room | `0xADDE` beams `0x0001` frames **2327** dual |
| Fail pin (old) | (920, 323) mid_air no WJ — retired |
| Fail pin (older) | (920, 311) floor open-loop `wj_r` — **retired** |

## Path recipe (product)

1. Gate open (human RLE) → past bars on upper ledge  
2. Missiles on ledge y≈139  
3. Backup under gate ~x425 **same ledge**  
4. Face right, dash to edge ~x600  
5. Spin launch high  
6. Door-column contact → **one classic away WJ** (not floor)  
7. Left spin carry → RIGHT arc onto sill → Super → Wave PLM  

## Residual risks / RED cause

- (cleared) High launch alone dropped at door column without WJ
- Classic away timing is one-knob group (`delay_into=3`, `into_frames=6`,
  left follow 8); edge x600 couples with contact height
- Spike floor still refuse zone — never resume open-loop WJ below y≈280

## Next action (required)

- **Next card ID:** rr-l0u (continuous `--to wave`) or rr-dbu.11 Ice — planner
- **One change:** wire pure Wave hop into continuous tip after stabilize if needed
- **Source state:** `scratch/post_single_to_double_chamber_pure.state`  
  Post pure: capture `post_double_chamber_to_wave_pure.state` when composing

## Non-claims

- No continuous tip / STATUS promote (rr-l0u after)
- Gate open human RLE not reopened for thrash
- Place-based sill/Wave is not pure evidence (pure dual is)

## Probe pin (GREEN)

room=0xADDE pose=137 x=171 y=123 door_transition=0 frames=2327  
beams=0x0001 last_pin=post_single_to_double_chamber_pure  
path=missile-ledge runway + door-column classic WJ + Super + PLM
