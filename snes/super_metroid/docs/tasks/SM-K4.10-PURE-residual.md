> **Board split:** `rr-dbu.10` = blue gate open only → `rr-re9` = Super door + Wave PLM.
> Gate open GREEN dual pure. Wave door still RED (rr-re9).

# Residual — SM-K4.10-GATE / rr-dbu.10 (Double Chamber blue gate open)

## Result

GREEN (gate open only)

## Files changed (this session)

- `routes/kpdr/k4_wave.py` — one-knob: human tape f4650–5200 RLE from Kamer seat x∈[370,375] y≤139; replace scaffold R/peak/fall attempts
- `custom_integrations/SuperMetroid-Snes/scratch/dev_gate_kamer_top_pure.state` — pure Kamer mid-state (research)
- `custom_integrations/SuperMetroid-Snes/scratch/dev_gate_open_pure.state` — post-open pure pin ~(488,139)
- `debug/wave_recon/gate_human_replay.py` — seat+replay research harness
- `debug/wave_recon/human_replay_pure/report.json` — early RED exact-replay delta log

## Verify paste

```bash
# Dual gate-open only (product controller hop + _dc_open_blue_gate):
# trial0/1: ok=True xy=(488,139) pose=9 frames=1171 beams=0x0000
# DUAL_GATE True

uv run python snes/super_metroid/scripts/probe/kpdr.py pure double-chamber-to-wave \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_single_to_double_chamber_pure.state \
  --no-red-diag
# exit 0 JSON success=false (Wave door = rr-re9)
# pin: room=0xADAD pose=25 xy=(920,311) missiles=20 frames=2371
# missiles=20 proves past-gate pack; x=920 is Super-door column (gate cleared)
```

## Acceptance

- [x] Pure proves gate open (walk x≳480 y≲200 solid) dual
- [x] Residual PROCESS pin; one-knob policy documented
- [x] Does not claim Wave beam bit

## Geometry (verified)

| Fact | Value |
|------|------:|
| Entry | Double Chamber `0xADAD` ~(61,139) post Single→Double pure |
| Open seat | **x∈[370,375] y≤139** standing; settle **8f** then human RLE |
| Blue gate bars | hard-stop **x≈411** until open |
| Dual open pin | solid **(488,139)** pose 9 after human f4650–5200 |
| Past gate | missile pack → Super column ~(920,·); Wave `0xADDE` still unopened |

## Human tape → pure (PLM truth)

Exact buttons from `speed_to_ice_moat_human.json` f4650–5200 open the gate
**from pure natural seat** when seat pin matches the human band:

| seat | result |
|------|--------|
| natural x∈[370,375] y≤139 + 8f settle + human RLE | **GREEN dual** (488,139) |
| natural x≈387 / y≈149 / wide band only | RED hard-stop ~411 or fall bottom |
| scaffold R/peak/fall shot knobs | RED bottom ~(475,409) |

Key frame match after green path: pose sequence tracks human (5→105→19→SELECT
1→2→0→105→ approach); first solid past bars ~f5132 class; end x≳480.

## Residual risks

- Full `double-chamber-to-wave` still RED at Super door (rr-re9) — not this bead
- Human RLE is long (~551f); fragile if hop seat drifts outside 370–375 / y>139
- Knockback mid-RLE aborts open (desync); rare on dual runs so far
- Do not re-introduce multi-attempt shot-knob thrash without PLM proof

## Next action (required)

- **Next card ID:** rr-re9 (Super door + Wave PLM collect)
- **One change:** open Super/missile red door at ~(940,139) + Wave chozo 0x0001 from post-gate pure pin
- **Source state:** `scratch/dev_gate_open_pure.state` or `post_single_to_double_chamber_pure` + gate open

## Non-claims

- Did not STATUS-promote / continuous Wave tip
- Did not claim Wave beam bit / Super door pure (rr-re9)
- Probe vehicle RED on Wave door is expected; gate open is the dbu.10 claim only

## Probe pin (GREEN gate)

room=0xADAD pose=9 x=488 y=139 door_transition=0 frames=1171 last_pin=post_single_to_double_chamber_pure gate open dual; beams=0x0000

## Probe pin (full vehicle — Wave door RED, gate cleared)

room=0xADAD pose=25 x=920 y=311 door_transition=0 frames=2371 missiles=20 beams=0x0000
