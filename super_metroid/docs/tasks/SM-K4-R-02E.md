# TASK SM-K4-R-02E: Kihunter→Zeela — lower-alcove launch geometry only

## Recipe step
1 pure controller (one-knob geometry residual)

## Model
Luna

## Wave type
implement

## Own files only
- `routes/kpdr/kraid_return.py` (`play_kihunter_to_zeela_return` only)
- optional residual: `docs/tasks/SM-K4-R-02E-residual.md`

## Context (minimal)
- SM-K4-R-02D **RED**: climb timed out in source room (honest — no Baby
  false-positive). Pin:
  `room=0xA4DA pose=2 x=357 y=395 door_transition=0 frame=340`.
- x≈357 is the **lower hard wall** warned in older comments; pure-LEFT
  strategies re-plant there and never clear shot blocks.
- Source start: **x≈465 y≈378**. Uncapped RIGHT hits Baby at **x≈492**.
- Planner probe (not pure evidence): brief LEFT then **RIGHT-capped**
  Hi-Jump/shoot got best height **min_y≈291** near setup **x≈360–380**
  (cap ~470). Need one more band of height for `y < 280` while still
  `0xA4DA` / `door_transition==0`.
- Keep from 02D (do **not** retune in this card):
  - climb success only if `room==0xA4DA` and `y<280` and `door_transition==0`
  - fail loud on Baby `0xA521`
  - post-climb Zeela door window **x∈[96,160]** (recon)
- Continuous / graph / STATUS off-limits.

## Read first
- `routes/kpdr/kraid_return.py` (`play_kihunter_to_zeela_return`)
- `docs/tasks/SM-K4-R-02D-residual.md`
- `docs/tasks/SM-KIHUNTER-RECON-report.md` (door band only — already applied)
- `docs/SOURCE_STATES.md` (`post_baby_to_kihunter`)

## Do
1. **One knob — lower-alcove launch only** (setup + climb input pattern):
   - Target launch band roughly **x∈[360,420]** (left of Baby door, right of
     the x≈357 pin wall). Do not pure-LEFT into the wall for the whole climb.
   - While climbing: **hard cap** so Samus does not walk/spin into **x≥480**
     (east hatch). Prefer LEFT brake when near cap; keep shooting UP into
     shot blocks + Hi-Jump/spin until true upper land.
   - Leave post-climb Zeela window `96..160` and Baby guards unchanged.
2. ≤3 bounded launch variants (e.g. left-frames × right-cap). Residual with
   **in-Kihunter** pin if still red (best min_y / final x/y if useful).
3. On pure green: `--output` → `scratch/post_kihunter_to_zeela_return.state`.
4. No R-03, continuous, graph, STATUS, RAM forge.

## Do not
- Change the recon Zeela door x-window constants in the same card
- Edit `kraid_approach.py` or multi-room compose
- Claim green if end room is not ordinary `0xA471`

## Acceptance
- [ ] Pure green → ordinary `0xA471` **or** residual with in-Kihunter pin +
      best observed min_y / x during climb
- [ ] Fail loud on `0xA521`
- [ ] `uv run pytest super_metroid/tests/test_controller_common.py -q` green

## Verify
```bash
uv run python super_metroid/scripts/probe/kpdr.py pure kihunter-to-zeela-return \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_baby_to_kihunter_return.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_to_zeela_return.state
uv run pytest super_metroid/tests/test_controller_common.py -q
```

## Done when
Pure exit 0 into Zeela, or residual after ≤3 launch knobs with PROCESS schema
and one next change (still launch-only unless climb is green).
