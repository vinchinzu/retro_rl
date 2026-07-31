# TASK SM-DOOR-PLM: Minimal read-only door/PLM peek for Kraid left exit

## Recipe step
harness / diagnostics (read-only RAM — **not** pure green)

## Model
Luna

## Own files only
- `ram.py` (**add** only small read-only field(s) if an existing pattern fits)
- `scripts/probe/kraid_door_plm_recon.py` (**create**)
- `docs/tasks/SM-DOOR-PLM-report.md` (**create**)
- optional: `tests/test_ram.py` (**add** one decode/unit assert if field added)

Do **not** edit varia_return, continuous, STATUS, progression.

## Context
DOOR-BLUE: harness cannot see blue-door open state / PLM activation. Shots fire;
`door_transition` never leaves 0. Without a readable open-state, geometry cards
spin blindly. This card adds the **smallest** read-only peek patterned on
existing `ram.py` / SuperMetroidState fields — or honestly documents that no
safe pattern exists without deeper reverse engineering.

## Read first
- `docs/tasks/SM-DOOR-BLUE-report.md`
- `ram.py` SuperMetroidState + parse helpers
- `scripts/probe/kraid_door_blue_recon.py`
- any existing door_def / PLM / projectile peeks in package

## Do
1. Search for existing SM community offsets or local comments for door state /
   PLM (read-only). Prefer fields already partially wired.
2. If a **bounded, documented** read fits the package style:
   - Add field(s) to state parse
   - Unit test decode smoke (no emu required if pure int decode)
   - Probe script: boot post_varia_to_kraid_pure, approach + 4 shots, sample
     new field(s) every frame; write JSON + report
3. If **no** safe field found within ~1 hour effort / clear offset:
   - Do **not** invent offsets. Report blocked with links/notes of what was
     searched; recommend planner external RAM map work
4. Explicit non-claims; never write door/PLM RAM.

## Residual required
- Field added OR blocked reason
- Sample table during shots if field exists
- Planner next one sentence

## Do not
- Forge open door for green
- Edit controller geometry
- continuous / STATUS

## Acceptance
- [ ] Probe exit 0 **or** blocked residual without fake fields
- [ ] If field added: test_ram / parse smoke green
- [ ] Report written

## Verify commands
```bash
# if field added:
uv run pytest super_metroid/tests/test_ram.py -q
uv run python super_metroid/scripts/probe/kraid_door_plm_recon.py \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_to_kraid_pure.state
```
