# TASK SM-K4.4-PURE-R19: Enemy-phase-aware fire + Super door (Phase D+E)

## Recipe step

1. Pure controller Phase D then Phase E. Geometry green before continuous.

## Model

Grok / Luna

## Wave type

implement

## Own files only

- `routes/kpdr/bubble_mountain_primitives.py` — phase wait / clear
- `routes/kpdr/bubble_mountain_mid.py` — fire with phase_wait
- `routes/kpdr/bubble_mountain_params.py` — phase + door constants
- `routes/kpdr/bubble_mountain.py` — Super door sticky right WJ
- `tests/test_k4_norfair_scaffold.py`
- residual + tip boards

## Source and contract

- Full pure GREEN from:
  `scratch/post_rising_tide_to_bubble_pure.state` → ordinary `0xB07A`
- Caps: Morph, Bombs, Missiles, Supers, Hi-Jump, Varia — **no Speed**

## Context

R18: pure max-left seat earns p132 + pose 84 (min_y≈159) but Phase D hard-caps
mx200≈251 unless Geruta AI phase matches a lucky live dump. Zeroing HP alone
does not unlock. Next class: wait/clear that preserves seat, or left-wall loop.

## Do

1. Idle on max-left fire seat until Geruta slots 4/6 hit proven clear geometry
2. Fire product recipe (no arm-pump) → `top_reached`
3. Sticky right Super door → Bat Cave
4. Full pure probe GREEN + residual

## Acceptance

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_to_bat_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_pure_pin_r19.json --no-red-diag
# success=true room=0xB07A
```

## Done when residual closes

See `SM-K4.4-PURE-R19-residual.md`.
