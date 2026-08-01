# TASK SM-K4-R-03B: Zeela→Warehouse — reverse of forward drops (not floor-left)

## Recipe step
1 pure controller

## Model
Luna

## Wave type
implement

## Own files only
- `routes/kpdr/kraid_return.py` (`play_zeela_to_warehouse_return` only)
- optional residual: `docs/tasks/SM-K4-R-03B-residual.md`

## Context
- SM-K4-R-03 **RED**: floor-left door transition pin
  `room=0xA471 pose=16 x=19 y=395 door_transition=1` — never Warehouse.
- Source: `scratch/post_kihunter_to_zeela_return.state` room `0xA471`
  `x≈403 y≈362` (bottom-right after kihunter down-drop, ~post bottom bomb-roll band).
- Forward continuous hop `play_zeela_to_kihunter` in
  `routes/kpdr/kraid_approach.py` is the geometry map to reverse:
  1. top-left (Warehouse door band) → first drop
  2. middle roll → second drop to floor `y≥395`
  3. bottom bomb-roll right to `x≥400` → up door to Kihunter
- Reverse from source therefore:
  1. bottom reverse-roll left off the kihunter tunnel band
  2. climb reverse second drop (to mid band)
  3. climb reverse first drop (to top / `y` upper band)
  4. face **left**, open blue door, enter Warehouse `0xA6A1`
- Warehouse door is **upper-left**, not floor-left. Floor-left LEFT spam is
  the wrong class (R-03 failure).
- Do **not** retouch `play_kihunter_to_zeela_return` (Wave 9 GREEN).
- Graph / continuous / STATUS: no.

## Read first
- `routes/kpdr/kraid_approach.py` — `play_zeela_to_kihunter` (forward)
- `routes/kpdr/kraid_return.py` — current `play_zeela_to_warehouse_return`
- `docs/tasks/SM-K4-R-03-residual.md`
- `docs/tasks/PROCESS.md` residual schema

## Do
1. **One maneuver-class change:** rewrite `play_zeela_to_warehouse_return` as
   reverse of forward `play_zeela_to_kihunter` (climb levels first, then left
   Warehouse door). Mirror the drop/roll bands from the forward controller;
   use existing helpers (`ensure_morph`, `unmorph`, `select_weapon`, `hold`,
   `wait_ordinary_room`, Hi-Jump vertical if needed).
2. **Guards (required):**
   - Fail-loud if `door_transition` while `samus_y > 250` (floor wrong door).
   - Fail-loud if exit room is not ordinary Warehouse `0xA6A1`.
   - Target upper band before left door: prefer `samus_y <= 200` (or tighter
     if forward settle lands higher) before door-shot/left push.
3. Optional capture:
   `--output .../scratch/post_zeela_to_warehouse_return.state`
4. Residual with PROCESS schema if still RED (pin + next card + one change).
5. No graph promote, continuous, STATUS, RAM forge, other hops.

## Acceptance
- [ ] Pure green from post-kihunter→zeela source → ordinary `0xA6A1`
      **or** residual with pin + next card + one change
- [ ] Floor-door guard present (no silent floor transition success)
- [ ] `play_kihunter_to_zeela_return` untouched

## Verify
```bash
uv run python super_metroid/scripts/probe/kpdr.py pure zeela-to-warehouse-return \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_to_zeela_return.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_zeela_to_warehouse_return.state

# Optional second run if first green (stability):
uv run python super_metroid/scripts/probe/kpdr.py pure zeela-to-warehouse-return \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_to_zeela_return.state
```

## Do not
- Touch `play_kihunter_to_zeela_return` / climb redesign
- Promote graph / continuous / STATUS
- Floor-left morph-only approach (known RED class)
- Claim continuous evidence
