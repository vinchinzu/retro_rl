# TASK SM-K4-R-02B: Kihunter→Zeela — avoid Baby Kraid wrong door

## Recipe step
1 pure controller (one-knob geometry residual)

## Model
Luna

## Wave type
implement

## Own files only
- `routes/kpdr/kraid_return.py` (`play_kihunter_to_zeela_return` only)
- optional residual: `docs/tasks/SM-K4-R-02B-residual.md`

## Context (minimal)
- SM-K4-R-02 RED residual: shot-block/Hi-Jump **climb works**, but upper
  left traverse enters **Baby Kraid `0xA521`** instead of Zeela `0xA471`.
- Final pin (R-02): `room=0xA521 pose=105 x=65522 y=116 door_transition=1`
  error `upper traverse crossed wrong door`.
- Source (required):
  `custom_integrations/SuperMetroid-Snes/scratch/post_baby_to_kihunter_return.state`
  → room **`0xA4DA`** Warehouse Kihunter, lower-right alcove.
- Zeela exit is **blue down** at upper-left block band `[7, 15]` — not the
  right-side Baby Kraid hatch. MapRando room 81 node 2→5.
- Continuous tip still Varia-only; this is reverse pure K3.6 only.

## Read first
- `routes/kpdr/kraid_return.py` (`play_kihunter_to_zeela_return` + residual notes)
- `docs/tasks/SM-K4-R-02-residual.md`
- `docs/SOURCE_STATES.md` row `post_baby_to_kihunter`
- `routes/kpdr/kraid_approach.py` (forward reverse geometry **hint only**, no edit)

## Do
1. **One knob:** door-specific upper stop / positioning so left traverse never
   crosses into Baby Kraid hatch before Zeela down-door.
   - Detect wrong-room early (`0xA521`) and fail with a clear residual pin
     (do not silently continue).
   - Prefer x-band / y-band gates for Zeela vertical door before DOWN+shot.
   - Keep the working shot-block + Hi-Jump climb; do not rewrite climb from zero.
2. Do **not** implement zeela→warehouse (SM-K4-R-03).
3. On pure green: save `scratch/post_kihunter_to_zeela_return.state` via `--output`.
4. No graph verification promote; no continuous / STATUS.

## Do not
- Multi-room compose past Zeela
- Progression/door RAM forges
- Claim continuous
- Parallel-edit other modules in `kraid_return.py` hops beyond this function

## Acceptance
- [ ] Pure green from named source → ordinary `0xA471` Zeela
- [ ] Never claims green if end room is `0xA521`
- [ ] `uv run pytest super_metroid/tests/test_controller_common.py -q` green
- [ ] Residual PROCESS schema + pin if still red
- [ ] Optional source capture on green

## Verify commands
```bash
uv run python super_metroid/scripts/probe/kpdr.py pure kihunter-to-zeela-return \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_baby_to_kihunter_return.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_to_zeela_return.state
uv run pytest super_metroid/tests/test_controller_common.py -q
```

## Done when
Pure exits 0 into Zeela, or residual after ≤2–3 door-window strategies with
last pin and **one** next knob (not “rewrite whole climb”).
