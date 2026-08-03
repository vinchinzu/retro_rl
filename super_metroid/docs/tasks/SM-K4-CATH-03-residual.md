## Residual — SM-K4-CATH-03

### Result
GREEN

### Files changed
- `super_metroid/routes/kpdr/k4_norfair.py` — replaced scaffold `play_cathedral_to_rising_tide` with ridge cross + drop to **lower-right** green Super door (~x700–730 / y340–400 near lava) → Super open → ordinary `0xAFA3`.
- `super_metroid/scripts/probe/kpdr.py` — pure choice `cathedral-to-rising-tide` (+ import / play map).
- `super_metroid/tests/test_k4_norfair_scaffold.py` — pure-segment registration for `cathedral_to_rising_tide`.
- `super_metroid/docs/tasks/SM-K4-CATH-03-residual.md` — PROCESS residual.
- `super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_cathedral_to_rising_tide_pure.state`
  — pure GREEN successor.

### Verify paste

```text
uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
exit 0
10 passed in 0.23s

uv run python super_metroid/scripts/probe/kpdr.py pure cathedral-to-rising-tide \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_cathedral_entrance_to_cathedral_pure.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_cathedral_to_rising_tide_pure.state \
  --pin-json super_metroid/debug/cathedral_to_rising_tide_pure_pin.json
exit 0
success=true roomIdHex=0xAFA3 samusX=39 samusY=139 pose=9 doorTransition=0 frames=1162 controllerOnly=true
```

Re-verify: same room / frames band (~1162f).

### Acceptance

- [x] Source loads at `0xA788` (CATH-02 successor pin band)
- [x] Ordinary `0xAFA3` without warp / item grants
- [x] Successor state only if pure GREEN
- [x] Unit/registration green
- [x] Residual PROCESS fields; no continuous/STATUS claim

### Residual risks

- Pure-green only; no continuous re-record, graph promote, or STATUS claim.
- **Door altitude is load-bearing:** green Super door is lower-right near lava
  (~y350–380), not upper y≈120. Upper-ledge supers (y≈300–330) miss the shell.
- Contact-damage + unlimited-energy assist can stunlock pose 137/138 — controller
  jump-escapes; idle-plant is a known failure mode.
- Ridge late gaps (~x450–560) need longer Hi-Jumps; short crest hops strand progress.
- Reverse doorway fixture spawn y≈120 is **not** the pure door lip for this hop.

### Next action (required)

- **Next card ID:** `SM-K4-CATH-04`
- **One change:** Pure controller Rising Tide `0xAFA3` → Bubble Mountain `0xACB3`
  from the captured pure successor.
- **Source state:**
  `custom_integrations/SuperMetroid-Snes/scratch/post_cathedral_to_rising_tide_pure.state`

### Non-claims

- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.

### Probe pin (if pure/geometry) — mandatory metrics

```text
room=0xAFA3 pose=9 x=39 y=139 door_transition=0
frames=1162 dwell=not reported last_pin=room=0xAFA3 pose=9 x=39 y=139 door_transition=0
```
