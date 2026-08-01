# TASK SM-KIHUNTER-CLIMB-RECON: Lower-alcove climb height sweep (diagnostic)

## Recipe step
diagnostic recon (not pure green claim)

## Model
Luna

## Wave type
implement

## Own files only
- `scripts/probe/kihunter_climb_recon.py` (**create**)
- `docs/tasks/SM-KIHUNTER-CLIMB-RECON-report.md` (**create**)

Do **not** edit `routes/kpdr/kraid_return.py` (owned by SM-K4-R-02E).

## Context
- Pure climb stuck: best planner probe hit **min_y≈291** at **x≈379 y≈299**
  under shot-block ceiling; pure-LEFT pins **x≈357**; uncapped RIGHT → Baby
  at **x≈492**.
- Source: `scratch/post_baby_to_kihunter_return.state` (`0xA4DA`, start ~x465 y378).
- Need measured launch table: left-frames × right-cap × shot pattern →
  min_y / final x/y / whether y&lt;280 while still in room.

## Read first
- `docs/tasks/SM-K4-R-02D-residual.md`, `SM-K4-R-02E.md`
- style: `scripts/probe/kihunter_zeela_recon.py`

## Do
1. Probe script: fresh boot per trial, natural inputs only (no place_samus
   required for climb table; optional upper-warp only if labeled development).
2. Sweep a small grid (e.g. left 0..32 step 4; right-cap 450/460/470/475;
   ≤3 shot/jump patterns). Record min_y, min/max x, final room/pose, whether
   true upper land (`room==0xA4DA`, `y<280`, `door_transition==0`).
3. Report: best launch recipe for R-02E residual consumers (numeric).
4. Never claim pure green. No route edits.

## Acceptance
- [ ] Report with table + recommended launch numbers
- [ ] Non-claims

## Verify
```bash
uv run python super_metroid/scripts/probe/kihunter_climb_recon.py \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_baby_to_kihunter_return.state
test -f super_metroid/docs/tasks/SM-KIHUNTER-CLIMB-RECON-report.md
```
