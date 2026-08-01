# TASK SM-K4-R-04: Pure `warehouse-to-business` after reverse Zeela (blocked until R-03B)

## Recipe step
1 pure controller

## Model
Luna

## Wave type
implement

## Own files only
- `scripts/probe/kpdr.py` (register pure choice `warehouse-to-business` only)
- `routes/kpdr/warehouse.py` (`play_warehouse_to_business` only if pure needs retune)
- optional residual: `docs/tasks/SM-K4-R-04-residual.md`
- optional capture: `scratch/post_warehouse_to_business_return.state` via pure `--output`

## Context
- **Blocked until** SM-K4-R-03B pure green + source
  `scratch/post_zeela_to_warehouse_return.state` room `0xA6A1`.
- Forward elevator hop already exists: `play_warehouse_to_business`
  (Warehouse elevator DOWN → Business Center `0xA7DE`). Reverse pure chain
  reuses this continuous hop; may already pure-green without retune.
- Pure CLI currently **missing** `warehouse-to-business` in
  `scripts/probe/kpdr.py` choices/map — wire it first (import already used
  via registry patterns; mirror `business-to-warehouse`).
- Graph comment: reverse path reuses continuous edge `warehouse_to_business`.
- After this green, reverse pure reaches Business floor — next planner gate
  is continuous compose design (still no continuous until full reverse
  integrity judgment).
- Do not edit `business_climb.py` / continuous / STATUS.

## Do
1. Wire pure CLI choice `warehouse-to-business` → `play_warehouse_to_business`.
2. Pure probe from post-R-03B warehouse source.
3. If RED, one-knob retune elevator position only in `play_warehouse_to_business`.
4. Capture exit state for Business floor if green.
5. Residual schema; no STATUS.

## Acceptance
- [ ] Pure green → ordinary Business `0xA7DE` **or** residual
- [ ] Source for Business floor captured if green

## Verify
```bash
# Only after R-03B source exists:
uv run python super_metroid/scripts/probe/kpdr.py pure warehouse-to-business \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_zeela_to_warehouse_return.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_warehouse_to_business_return.state
```

## Do not
- Dispatch while R-03B is still RED / source missing
- continuous.py / STATUS / business_climb geometry
