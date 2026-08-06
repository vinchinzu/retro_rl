> **Board split:** `rr-dbu.10` = blue gate open only → `rr-re9` = Super door + Wave PLM.
> Stop inventing shot knobs without PLM open proof (animation or x≳480 solid).

# Residual — SM-K4.10-GATE / rr-dbu.10 (Double Chamber blue gate open)

## Result

RED

## Files changed (this session)

- none for gate geometry (only `_DC_DOOR_*` rename while fixing Bubble `_DOOR_X`
  collision — no gate behavior change)

## Verify paste

```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py pure double-chamber-to-wave \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_single_to_double_chamber_pure.state \
  --no-red-diag
# RED: Wave door missed; room=0xADAD pose=81 xy=(475,409) frames=2995 beams=0x0000
```

## Acceptance

- [ ] Pure proves gate open (open animation **or** walk x≳480 y≲200 solid) dual
- [x] Residual PROCESS pin; one-knob policy documented
- [x] Does not claim Wave beam bit

## Geometry (verified)

| Fact | Value |
|------|------:|
| Entry | Double Chamber `0xADAD` ~(39–61,139) post Single→Double pure |
| Kamer | left-of-gate y **139↔219** (~200f half); shoot only at **top y≤145** |
| Blue gate bars | hard-stop **x≈411** upper path |
| Pure fail pin | bottom hard-stop ~(475,409) after failed upper path |
| Past gate | solid ~(480–520,139); Super red door ~(940,139) → Wave `0xADDE` |

## Human tape open timeline (`speed_to_ice_moat` DC segment)

Assist ammo (missiles never drain on tape). Gate open **is** on tape:

| frame | xy | notes |
|------:|----|-------|
| 4650 | (378,139) | seat; sel=1 missiles; pose 1 |
| 4652–4710 | seat | R hold + X+R pulses |
| 4712–4731 | peak | A+R / A+X+R; peak band y≈100–111 |
| 4834–4848 | fall | pure X pose 19 y≈122 |
| 4964 / 5005 | seat | **SELECT** sel 1→2→**0** (beam for final) |
| 5022–5054 | second volley | A then X+R on beam |
| 5083–5125 | approach | B+RIGHT (+A); still left of bars |
| **5126** | **(413,135)** | first x>411 airborne B+RIGHT |
| **5132** | **(421,139)** | **solid pose 9 walk** — gate clear |
| 5200 | (477,139) | past-gate platform |
| 5206 | (494,139) | missile pack (+5); tape end still ADAD |

Human never enters Wave. Prior pure/human-replay experiments under
`debug/wave_recon/` and `scratch/dev_gate_*` did **not** reproduce open from
pure pin (max upper x≈411).

## Residual risks / RED cause

- Impacts on **bars** ≠ proven **switch** hitbox; PLM WRAM open bit not mapped
  in-repo (`red_diag` marks PLM records blocked)
- Human SELECT + long dual-volley + Kamer phase may matter; naive R-angle
  scaffold still ends bottom ~(475,409)
- Do **not** add more shot cadence knobs until one of:
  1. Frame dump shows gate open animation after a single known projectile, or
  2. Exact human button replay from a **Kamer-top pure save** reaches x≳480, or
  3. Alternate pure path (bottom climb) documented with pins

## Next action (required)

- **Next card ID:** rr-dbu.10 continue (gate open only)
- **One change:** PLM truth — capture Kamer-top pure mid-state, replay human
  f4650–5200 buttons **exactly** (or prove delta); if still RED, map switch
  hitbox via sm-json / projectile pixel log — **not** another angle knob
- **Source state:** `scratch/post_single_to_double_chamber_pure.state`

## Non-claims

- Did not STATUS-promote / continuous Wave tip
- Did not claim Wave beam bit / Super door pure (rr-re9)
- Human tape is seat/open research only

## Probe pin (RED)

room=0xADAD pose=81 x=475 y=409 door_transition=0 frames=2995 last_pin=post_single_to_double_chamber_pure gate not cleared; beams=0x0000
