## Residual — SM-K4-R-03E

### Result
RED

### Files changed
- `routes/kpdr/kraid_return.py` — added one hard RIGHT recenter during the second-drop reverse-shot leftward phase when below the middle band and at the floor-door x threshold.
- `docs/tasks/SM-K4-R-03E-residual.md` — records the pure-probe result and redesign handoff.

### Verify paste
`uv run python super_metroid/scripts/probe/kpdr.py pure zeela-to-warehouse-return --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_to_zeela_return.state --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_zeela_to_warehouse_return.state`

Exit code: 1

Relevant output:
```text
zeela_to_warehouse_return: first-drop climb stalled: SuperMetroidState(frame=5912, game_state=8, phase=<GameplayPhase.ORDINARY_GAMEPLAY: 'ordinary_gameplay'>, room_id=42097, area_index=1, door_transition=0, transition_direction=6, samus_x=41, samus_y=395, velocity_x=0, velocity_y=0, pose=2, health=199, max_health=199, reserve_health=0, max_reserve_health=0, missiles=15, max_missiles=15, super_missiles=5, max_super_missiles=5, power_bombs=0, max_power_bombs=0, selected_item=0, equipped_items=4101, collected_items=4101, equipped_beams=0, collected_beams=0, timer_type=0, escape_timer_frames=131, escape_timer_seconds=41, escape_timer_minutes=0, num_enemies=3, enemies_killed=1, enemy0_x=89, enemy0_y=152, enemy0_hp=30, enemy0_spritemap=58132, event_flags=(1, 0, 0, 0, 0, 0, 0, 0), boss_bits=(4, 3, 0, 0, 0, 0, 1, 0))
roomIdHex=0xA471 samusX=41 samusY=395 pose=2 frame=5912
```

### Acceptance
- [ ] Pure green to `0xA6A1`: fail; the controller stalled in `0xA471` during first-drop climb.
- [x] No silent `0xA4B1` / floor-door success: the floor-door guard remained active and no floor-door transition occurred.

### Residual risks
- The hard RIGHT recenter prevents the R-03D floor-door transition but leaves the first-drop setup at the lower floor (`x=41, y=395`), so no Warehouse transition is evidenced.
- The generated output state was not produced because the controller raised before reaching the requested exit.
- This is not pure-green, continuous evidence, or a STATUS promotion.

### Next action (required)
- **Next card ID:** PLANNER-GATE
- **One change:** Redesign the Zeela second-drop/first-drop transition rather than adding another R-03 cadence variant.
- **Source state:** `super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_to_zeela_return.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.

### Probe pin (if pure/geometry)
room=0xA471 pose=2 x=41 y=395 door_transition=0
