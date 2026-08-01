## Residual — SM-K4-R-03D

### Result
RED

### Files changed
- `routes/kpdr/kraid_return.py` — replaced the Zeela second-drop wall-run with the recon forward-drop reverse-shot cadence and bounded setup roll.
- `docs/tasks/SM-K4-R-03D-residual.md` — records the failed pure probe and pin.

### Verify paste
`uv run python super_metroid/scripts/probe/kpdr.py pure zeela-to-warehouse-return --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_to_zeela_return.state --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_zeela_to_warehouse_return.state`

Exit code: 1

Relevant output:
```text
zeela_to_warehouse_return: floor door transition during second-drop climb: SuperMetroidState(frame=308, game_state=11, phase=<GameplayPhase.ROOM_TRANSITION: 'room_transition'>, room_id=42097, area_index=1, door_transition=1, transition_direction=6, samus_x=20, samus_y=396, velocity_x=0, velocity_y=2, pose=82, health=199, max_health=199, reserve_health=0, max_reserve_health=0, missiles=15, max_missiles=15, super_missiles=5, max_super_missiles=5, power_bombs=0, max_power_bombs=0, selected_item=0, equipped_items=4101, collected_items=4101, event_flags=(1, 0, 0, 0, 0, 0, 1, 0))
roomIdHex=0xA471 samusX=20 samusY=396 pose=82 frame=308
```

### Acceptance
- [ ] Pure green to `0xA6A1`: fail; the probe stopped in `0xA471` during a floor-door transition.
- [x] Floor-door guard retained; no `0xA4B1` success claim.
- [x] `kihunter→zeela` untouched.

### Residual risks
- The forward-drop reverse-shot cadence reaches the floor-door guard before entering the middle band; no warehouse transition is evidenced.
- The generated output state was not produced because the controller raised before reaching the requested exit.
- This is not pure-green, continuous evidence, or a STATUS promotion.

### Next action (required)
- **Next card ID:** SM-K4-R-03E
- **One change:** constrain the leftward portion of the second-drop reverse-shot cadence so it cannot reach the floor-door lane before the climb clears the middle band.
- **Source state:** `scratch/post_kihunter_to_zeela_return.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.

### Probe pin (if pure/geometry)
room=0xA471 pose=82 x=20 y=396 door_transition=1
