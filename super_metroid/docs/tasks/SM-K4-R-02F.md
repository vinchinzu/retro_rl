# TASK SM-K4-R-02F: Kihunter→Zeela — vertical launch cadence only

## Recipe step
1 pure controller (one-knob geometry residual)

## Model
Luna

## Wave type
implement

## Own files only
- `routes/kpdr/kraid_return.py` (`play_kihunter_to_zeela_return` only)
- optional residual: `docs/tasks/SM-K4-R-02F-residual.md`

## Context (minimal)
- SM-K4-R-02E **RED**: launch x-band + right-cap still timeout in source room.
  Pin: `room=0xA4DA pose=1 x=470 y=395 door_transition=0 frame=336`.
- Baby guard ok (no `0xA521`). Zeela window `96..160` still not reached.
- Prior planner probe: brief LEFT then climb under shot blocks got
  **min_y≈291 @ x≈379 y≈299** (stuck under ceiling — need better UP-shot /
  Hi-Jump cadence, not more RIGHT into door).
- If `docs/tasks/SM-KIHUNTER-CLIMB-RECON-report.md` exists when you start,
  prefer its **best launch numbers** for cadence only; do not rewrite x-band
  or Zeela window constants in this card.
- Keep: climb success only if `room==0xA4DA` + `y<280` + `door_transition==0`;
  fail loud on Baby; post-climb Zeela x∈[96,160].

## Read first
- `routes/kpdr/kraid_return.py` (`play_kihunter_to_zeela_return`)
- `docs/tasks/SM-K4-R-02E-residual.md`
- optional: `docs/tasks/SM-KIHUNTER-CLIMB-RECON-report.md` if present
- `docs/SOURCE_STATES.md` (`post_baby_to_kihunter`)

## Do
1. **One knob — vertical launch / shot-block cadence only:**
   - Hold x-band setup + right-cap from R-02E (or recon-recommended setup
     frames if report is ready — still one conceptual knob: **cadence**).
   - Retune A/B/UP/X phase lengths so shot blocks clear and Samus reaches
     true upper land (`y<280` in-room) without walking into x≥480.
2. ≤3 cadence variants. Residual with in-Kihunter pin + best min_y if still red.
3. On pure green: `--output` → `scratch/post_kihunter_to_zeela_return.state`.
4. No R-03, continuous, graph, STATUS, RAM forge. No Zeela window retune.

## Acceptance
- [ ] Pure green → ordinary `0xA471` **or** residual with pin + best min_y
- [ ] Fail loud on `0xA521`
- [ ] `uv run pytest super_metroid/tests/test_controller_common.py -q` green

## Verify
```bash
uv run python super_metroid/scripts/probe/kpdr.py pure kihunter-to-zeela-return \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_baby_to_kihunter_return.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_to_zeela_return.state
uv run pytest super_metroid/tests/test_controller_common.py -q
```
