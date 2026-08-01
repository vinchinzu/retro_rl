## Residual — SM-K4-R-03

### Result
RED

### Files changed
- `routes/kpdr/kraid_return.py` — rewrote `play_zeela_to_warehouse_return`
  morph left-push + attempted "shaft" climb + left door (wrong class)

### Verify paste
```bash
uv run python super_metroid/scripts/probe/kpdr.py pure zeela-to-warehouse-return \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_to_zeela_return.state
```
Exit 1 — shaft climb stalled mid door-transition on floor band:

```text
success: false
error: zeela_to_warehouse_return: Warehouse shaft climb stalled
roomIdHex: 0xA471
samusX: 19
samusY: 395
pose: 16
frame: 350
door_transition: 1
controllerOnly: true
```

### Acceptance
- [ ] Pure green post-R-02 source → `0xA6A1` — **fail**
- [x] Residual schema

### Residual risks
- Source is bottom-right Zeela (`x≈403 y≈362`) after kihunter down-drop.
- Floor-left morph push reaches `x=19 y=395` door transition — **wrong band**
  (not Warehouse top-left). Residual notes "Energy Tank" class wrong door;
  pin is floor-left transition, not upper Warehouse.
- Forward geometry is multi-level drops (`play_zeela_to_kihunter`); reverse
  must **climb reverse of those drops** before left Warehouse door.

### Next action (required)
- **Next card ID:** SM-K4-R-03B
- **One change:** Maneuver-class rewrite — reverse of forward
  `play_zeela_to_kihunter` (bottom reverse-roll → climb mid → climb top →
  left Warehouse door). Fail-loud if `door_transition` while `y>250`.
- **Source state:** `scratch/post_kihunter_to_zeela_return.state`

### Non-claims
- Did not STATUS-promote
- Did not forge progression/capacity/door/event/boss RAM
- Not continuous evidence
- Did not claim pure-green

### Probe pin
```text
room=0xA471 pose=16 x=19 y=395 door_transition=1
```
