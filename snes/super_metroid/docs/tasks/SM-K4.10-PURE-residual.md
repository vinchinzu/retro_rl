# Residual — SM-K4.10-PURE / rr-re9 (Pure Wave Beam PLM collect)

## Result

RED

## Files changed

- `routes/kpdr/k4_wave.py` — `_dc_open_blue_gate` one-knob: Kamer-top wait +
  R-angle (not UP+RIGHT) peak X+R + fall-volley X; `_dc_wait_kamer_top`;
  hop brake into seat band
- prior scaffold: `WAVE_BEAM_MASK`, registry, probe `double-chamber-to-wave`,
  `source_states` successor pin

## Verify paste

```bash
uv run pytest snes/super_metroid/tests/test_k4_norfair_scaffold.py snes/super_metroid/tests/test_k4_speed_branches.py -q
# 41 passed

uv run python snes/super_metroid/scripts/probe/kpdr.py pure double-chamber-to-wave \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_single_to_double_chamber_pure.state \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_double_chamber_to_wave_pure.state \
  --expect-room 0xADAD \
  --no-red-diag
# RED: Wave door missed; gate still closed (hard-stop bottom x≈475)
```

## Acceptance

- [x] Discover Wave room `0xADDE` + beam mask `0x0001` (place+collect verified)
- [x] Source: pure post Single→Double (`post_single_to_double_chamber_pure`) at `0xADAD` ~(61,139)
- [ ] Pure controller GREEN to Wave PLM collect
- [x] Units + registry wired
- [x] Residual PROCESS schema
- [ ] Dual pure / graph edge promote

## Geometry (verified this wave)

| Fact | Value |
|------|------:|
| Entry | Double Chamber `0xADAD` top-left ~(61,139) post Single→Double pure |
| Hop seat | pure hop lands Kamer ~(384,151) then cycles |
| **Kamer** | left-of-gate platform **y 139↔219** (~200f half-period); shoot only at **top y≤145** |
| Blue gate | solid hard-stop **x≈411** upper path; switch is **top** mechanism |
| Shot line | SM diagonal = **R** (angle-up), **not** UP+RIGHT |
| Human tape | `speed_to_ice_moat` f4650–5140: seat ~(378,139), R+X stand, peak A+R+X y≈104–111, fall pure X y122–160 (pose 19), walk through to x428+ |
| Human | **never entered Wave** (recording ends ADAD ~(494,139)) |
| Impacts | missiles/beam **do** explode on bars (ammo drains without assist) but PLM stays closed |
| Jump-over | spin from Kamer top peaks ~y104; max upper x≈411–414 against bars |
| Past gate | place ~(520,140) solid; Super red door ~(940,139) → Wave; chozo Wave bit `0x0001` |
| Bottom | hard-stop x≈475 (mid wall/pit) |

## Path recipe (intended)

1. Top: hop_run RIGHT → spin toward gate seat x∈[365,390]
2. **Wait Kamer top** y≤145
3. Gate: missiles then beam — R stand shots → peak X+R → fall X volley → walk probe
4. Right: hop/walljump to door sill y≲180 x≳920; Super red door → Wave
5. Wave: hop RIGHT to chozo; collect bit 0x01

## Residual risks / RED cause

- **Blue gate PLM does not open** from pure pin despite:
  - Correct weapon select (beam/missiles; ammo drains when assist off)
  - Kamer-top seating matching human y≈139–145
  - R-angle peak shots + fall volleys matching human button tape
  - Visible impact flashes/explosions on gate column
  - Natural re-entry SC→DC→gate same RED (not save PLM corruption)
- Bars hard-stop x≈411; projectiles hit **bars** more than proven **switch** hitbox
- sm-json: obstacle A clears from node 1→4 with only heat (normal shot-open);
  4→1 needs Wave or gate-glitch (switch faces left — we are on correct side)
- Human through-walk after their volley is **not reproduced** when replaying
  equivalent inputs from pure hop seat (max upper x≈411)
- Place-based Wave collect remains **not** pure evidence

## Next action (required)

- **Next card ID:** rr-re9 retry / SM-K4.10-PURE-GATE2 (one-knob: switch hitbox)
- **One change:** Prove a single projectile that toggles gate open (frame dump of
  open animation), or find alternate pure route past bars (bottom climb to right
  sill without upper gate). Prefer live RGB after exact human button replay from
  a Kamer-top save mid-room.
- **Source state:** `scratch/post_single_to_double_chamber_pure.state`

## Non-claims

- Did not STATUS-promote / continuous compose / forge progression RAM
- Did not mark graph edge `double_chamber_to_wave` continuous
- Place-based Wave collect is **not** pure-green evidence
- Human tape is seat/open research only (no Wave room in that recording)

## Probe pin (RED)

room=0xADAD pose=81 x=475 y=409 door_transition=0 frames=2995 last_pin=post_single_to_double_chamber_pure gate not cleared; beams=0x0000
