## Residual — SM-K4-CATH-04

### Result
GREEN

### Files changed
- `super_metroid/routes/kpdr/k4_norfair.py` — replaced scaffold
  `play_rising_tide_to_bubble` with platforming Hi-Jump cross + continuous
  RIGHT+B+X door pressure on the right blue door (x≥930, stay y≤170) →
  ordinary `0xACB3`.
- `super_metroid/scripts/probe/kpdr.py` — pure choice `rising-tide-to-bubble`
  (+ import / play map).
- `super_metroid/tests/test_k4_norfair_scaffold.py` — pure-segment registration
  for `rising_tide_to_bubble`.
- `super_metroid/docs/tasks/SM-K4-CATH-04.md` — card.
- `super_metroid/docs/tasks/SM-K4-CATH-04-residual.md` — PROCESS residual.
- `super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state`
  — pure GREEN successor.

### Verify paste

```text
uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
exit 0
11 passed in 0.16s

uv run python super_metroid/scripts/probe/kpdr.py pure rising-tide-to-bubble \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_cathedral_to_rising_tide_pure.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --pin-json super_metroid/debug/rising_tide_to_bubble_pure_pin.json
exit 0
success=true roomIdHex=0xACB3 samusX=39 samusY=634 pose=25 doorTransition=0 frames=2609 controllerOnly=true
```

Re-verify: same room / frames band (**2609f** ×2).

### Acceptance

- [x] Source loads at `0xAFA3` (CATH-03 successor pin band)
- [x] Ordinary `0xACB3` without warp / item grants
- [x] Successor state only if pure GREEN
- [x] Unit/registration green
- [x] Residual PROCESS fields; no continuous/STATUS claim

### Residual risks

- Pure-green only; no continuous re-record, graph promote, or STATUS claim.
- **Door altitude is load-bearing:** blue door ledge is y≤~165; falling under
  the platform (y>170) walks past the shell without transition. Controller
  backs left and charged-Hi-Jumps back onto the ledge.
- Continuous RIGHT+B+X door pressure works; deliberate grounded plant-and-shoot
  alone was unreliable (missed shell / enemy knockback).
- Mid-room low platforms need charged Hi-Jumps (y>150 grounded); pure run-jump
  cadence alone stalled ~x379 in lava pits.
- Morph-roll path max_x≈523 only — not sufficient without Gravity.
- Contact damage + unlimited-energy assist can stunlock pose 137/138 — spin-escape.

### Next action (required)

- **Next card ID:** `SM-K4.4-PURE` (or `SM-K4-BUBBLE-BAT` living card)
- **One change:** Pure controller Bubble Mountain `0xACB3` → Bat Cave `0xB07A`
  (`bubble_to_bat_cave`, requires Supers) from the captured pure successor.
- **Source state:**
  `custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state`

### Non-claims

- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.
- Cathedral pure stack CATH-01…04 is pure-only; continuous tip remains
  power-on → Frog Save until planner compose/stabilize.

### Probe pin (if pure/geometry) — mandatory metrics

```text
room=0xACB3 pose=25 x=39 y=634 door_transition=0
frames=2609 dwell=not reported last_pin=room=0xACB3 pose=25 x=39 y=634 door_transition=0
```
