## Residual — SM-K4-CATH-02

### Result
GREEN

### Files changed
- `super_metroid/routes/kpdr/k4_norfair.py` — replaced scaffold with bomb-drop
  through left morph-tunnel floor → floor cross to ~x620 → climb constrained to
  x∈[560, 680] (extreme-right wall x≈730 is a dead climb) → mid plant / high-band
  Super door open into ordinary `0xA788`.
- `super_metroid/scripts/probe/kpdr.py` — pure choice
  `cathedral-entrance-to-cathedral` (landed with CATH-02 implement).
- `super_metroid/tests/test_k4_norfair_scaffold.py` — pure-segment registration.
- `super_metroid/docs/tasks/SM-K4-CATH-02.md` — task card.
- `super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_cathedral_entrance_to_cathedral_pure.state`
  — pure GREEN successor.

### Verify paste

```text
uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
exit 0
9 passed in 0.16s

uv run python super_metroid/scripts/probe/kpdr.py pure cathedral-entrance-to-cathedral \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_business_to_cathedral_entrance_pure.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_cathedral_entrance_to_cathedral_pure.state \
  --pin-json super_metroid/debug/cathedral_entrance_to_cathedral_pure_pin.json
exit 0
success=true roomIdHex=0xA788 samusX=39 samusY=124 pose=81 doorTransition=0 frames=909 controllerOnly=true
```

Re-verify: same room / frames band (909f).

### Acceptance

- [x] Source loads at `0xA7B3`
- [x] Ordinary `0xA788` without warp / item grants
- [x] Successor state only if pure GREEN
- [x] Unit/registration green
- [x] Residual PROCESS fields; no continuous/STATUS claim

### Residual risks

- Pure-green only; no continuous re-record, graph promote, or STATUS claim.
- Climb x-band is load-bearing: drifting to x≈730 against the right wall stalls.
- Upper left path from Business spawn is solid at x≈91 — bomb floor is required.
- Mid shelf standing gate is fragile if enemies knock Samus off before plant.

### Next action (required)

- **Next card ID:** `SM-K4-CATH-03`
- **One change:** Pure controller Cathedral `0xA788` → Rising Tide `0xAFA3`
  (green Super / PB door per graph) from the captured pure successor.
- **Source state:**
  `custom_integrations/SuperMetroid-Snes/scratch/post_cathedral_entrance_to_cathedral_pure.state`

### Non-claims

- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.

### Probe pin (if pure/geometry) — mandatory metrics

```text
room=0xA788 pose=81 x=39 y=124 door_transition=0
frames=909 dwell=not reported last_pin=room=0xA788 pose=81 x=39 y=124 door_transition=0
```
