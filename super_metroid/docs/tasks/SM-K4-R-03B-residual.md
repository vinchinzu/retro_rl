## Residual — SM-K4-R-03B

### Result
RED

### Files changed
- `routes/kpdr/kraid_return.py` — rewrote `play_zeela_to_warehouse_return`
  to reverse the bottom roll and attempt staged reverse climbs before the
  upper-left Warehouse door; added floor-door and destination guards.
- `docs/tasks/SM-K4-R-03B-residual.md` — recorded the failed pure probe and
  next single geometry change.

### Verify paste
```bash
uv run python super_metroid/scripts/probe/kpdr.py pure zeela-to-warehouse-return \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_to_zeela_return.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_zeela_to_warehouse_return.state
```
Exit 1 — reverse climb does not leave the lower floor:

```text
success: false
error: zeela_to_warehouse_return: second-drop climb stalled
roomIdHex: 0xA471
samusX: 122
samusY: 409
pose: 65
frame: 5660
door_transition: 0
controllerOnly: true
```

### Acceptance
- [ ] Pure green post-Kihunter→Zeela source → ordinary `0xA6A1` — fail; the
  controller remains on the lower Zeela floor.
- [x] Floor-door guard present; transitions with `y > 250` fail loudly.
- [x] `play_kihunter_to_zeela_return` untouched.

### Residual risks
- The source reaches the expected lower Zeela band, but the attempted
  unmorphed jump/beam reverse of the second drop does not lift Samus.
- Pure green, continuous wiring, and STATUS promotion remain blocked.

### Next action (required)
- **Next card ID:** SM-K4-R-03C
- **One change:** Capture the lower-floor shaft geometry and replace only the
  second-drop vertical maneuver with the confirmed upward-entry sequence.
- **Source state:** `scratch/post_kihunter_to_zeela_return.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.
- Did not claim pure-green.

### Probe pin
```text
room=0xA471 pose=65 x=122 y=409 door_transition=0
```
