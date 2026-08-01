# TASK SM-K4-R-04B: Planner redesign — warehouse reverse (Zeela door → elevator)

## Recipe step
1 pure controller (planner redesign — **not** one-knob elevator retune)

## Model
Planner (Grok)

## Wave type
implement

## Own files only
- `routes/kpdr/warehouse.py` (`play_warehouse_to_business` reverse approach when
  source is right-ledge; keep elevator DOWN path for left-side continuous)
- optional residual: `docs/tasks/SM-K4-R-04B-residual.md`

## Context
The natural Varia-lineage probe from `post_varia_continuous_to_warehouse`
(right Zeela-door ledge) is **GREEN**: the reverse branch reaches the lower
lip, clears both stack tiers, and reuses the left-side elevator closeout.

Geometry findings (R-04 residual):

| Fact | Detail |
|------|--------|
| Super-block wall | floor left hard-stops **x≈325** y=315 |
| Forward open | left side x≈75–100 y≈139, 3 supers face RIGHT |
| Forward cross | spin RIGHT+B+A on **upper** y≈139–155 through x=300–360 |
| Floor after open | still blocked left at 325 — passage is upper |
| Open from right upper | tried; stack did not open a left path |

## Source
`scratch/post_varia_continuous_to_warehouse.state` room `0xA6A1` x≈728

## Do
1. When `samus_x > 400`, run reverse approach class (do not break left-side
   continuous elevator hop used on power-on chain).
2. Goal: ordinary Business `0xA7DE`; capture
   `scratch/post_warehouse_to_business_return.state` if green.
3. Residual if RED: multi-strategy pin table + next planner action (not Luna
   cadence spam on elevator settle frames).
4. Continuous promotion is planner-only after the pure gate; it is now covered
   by two integrity-green `--to business` runs.

## Acceptance
- [x] Pure green → ordinary `0xA7DE` (1,421f)
- [x] Continuous left-side `warehouse_to_business` still valid (no break)
- [x] `uv run pytest super_metroid/tests/test_controller_common.py -q` green

## Verify
```bash
uv run python super_metroid/scripts/probe/kpdr.py pure warehouse-to-business \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_continuous_to_warehouse.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_warehouse_to_business_return.state
# Optional left-side sanity (continuous-like elevator source if available):
# uv run python super_metroid/scripts/probe/kpdr.py pure warehouse-to-business \
#   --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/red_to_warehouse_controller.state
```
