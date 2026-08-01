# TASK SM-K4-R-03D: Zeela→Warehouse — forward-drop reverse-shot climb class

## Recipe step
1 pure controller

## Model
Luna

## Wave type
implement

## Own files only
- `routes/kpdr/kraid_return.py` (`play_zeela_to_warehouse_return` second-drop /
  climb setup only)
- optional residual: `docs/tasks/SM-K4-R-03D-residual.md`

## Context
- SM-ZEELA-CLIMB-RECON (**done**): 24 trials, `best_min_y=331` from
  **`forward_drop_reverse_shot`** (breaks y=395 floor pin). Report:
  `docs/tasks/SM-ZEELA-CLIMB-RECON-report.md`.
- Class sketch: open reverse drop with `UP+X`, then jump/leftward shot cadence
  (not standing A spam / wall-run hold — those stayed min_y=398).
- Caveats from recon:
  - Natural setup stuck at `strategy_start_x=281` (band targets not reached)
  - Best still only y=331 (need ≤325 mid, then ≤200 for warehouse door)
  - `morph_left_bomb_cycle` hits min_y=334 but wrong room `0xA4B1` — **do not use**
- R-03C Hi-Jump wall-run failed same floor pin.
- Source: `scratch/post_kihunter_to_zeela_return.state`
- Do not touch `play_kihunter_to_zeela_return`. Keep floor-door guard.

## Do
1. **One change class:** replace second-drop climb with recon
   `forward_drop_reverse_shot` class + **one** named setup geometry tweak so
   launch is not stuck at natural x=281 (move toward mid-drop lane ~x=105–130
   if possible, without morph-left into `0xA4B1`).
2. Pure probe same verify command; optional `--output` if green.
3. Residual: pin + best_min_y if still red; next card one change only.

## Acceptance
- [ ] Pure green → `0xA6A1` **or** residual with pin + next card
- [ ] Floor-door guard retained; no `0xA4B1` success claim
- [ ] kihunter→zeela untouched

## Verify
```bash
uv run python super_metroid/scripts/probe/kpdr.py pure zeela-to-warehouse-return \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_to_zeela_return.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_zeela_to_warehouse_return.state
```

## Do not
- continuous / STATUS / graph promote
- morph-left into Energy Tank `0xA4B1` as success
- Full hop rewrite unrelated to climb class
