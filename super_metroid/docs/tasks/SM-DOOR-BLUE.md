# TASK SM-DOOR-BLUE: Blue-door / shell diagnostic at Kraid left exit

## Recipe step
diagnostics (supports pure geometry — **not** pure green)

## Model
Luna

## Own files only
- `scripts/probe/kraid_door_blue_recon.py` (**create**)
- `docs/tasks/SM-DOOR-BLUE-report.md` (**create**)
- optional gitignored: `debug/kraid_door_blue_recon.json`

Do **not** edit `varia_return.py`, continuous, STATUS, progression.

## Context
DOOR-PHASE rejected “pose 138 is the whole story” and left **closed blue door
vs wrong trigger height** unresolved. Shots fired; `door_transition` never
left 0. This card must **instrument door-relevant RAM** if readable via
existing `ram.py` / state fields — or honestly report which fields are
**unavailable** in harness (that is a valuable failure mode).

## Read first (all)
- `docs/tasks/SM-DOOR-PHASE-report.md`
- `scripts/probe/kraid_door_phase_recon.py` (session pattern)
- `scripts/probe/kraid_left_door_recon.py`
- `ram.py` / SuperMetroidState fields (door_transition, game_state, etc.)
- `routes/kpdr/varia_return.py` shot sequence (read only)

## Do (harness-stress diagnostic)
1. CLI boots
   `scratch/post_varia_to_kraid_pure.state` with resource assist only.
2. Run a short scripted approach to left lip + 4 standing left shots (mirror
   controller shot pattern) while sampling **every frame**:
   room, pose, x, y, door_transition, transition_direction, game_state,
   enemy0_hp, and **any** additional door/PLM/BTS fields already exposed.
3. If door open-state is **not** in SuperMetroidState, do **not** invent RAM
   writes. Instead: document missing fields and propose the minimal ram.py
   probe field (read-only) for a **future** card — do not implement deep
   WRAM reverse-engineering beyond what is already patterned in the package.
4. Write report:
   - Table: frame windows where door_transition changed (likely empty)
   - Did shots change any sampled field?
   - Ranked: closed door vs geometry vs other
   - One recommended next geometry **or** ram-field card
5. Explicit non-claims.

## Residual required
- Last pin
- Harness gap list (what we cannot see)
- Files + command paste

## Do not
- Edit controller for “maybe green”
- Forge door open flags
- Continuous / STATUS

## Acceptance
- [ ] Probe script runs exit 0
- [ ] Report with honest harness limits
- [ ] No controller / continuous churn

## Verify commands
```bash
uv run python super_metroid/scripts/probe/kraid_door_blue_recon.py \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_to_kraid_pure.state
```
