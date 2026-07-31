# SM-TIGHTEN-01C — Business settle trim residual

## Change

The eight platform settles were changed from 20f to 12f:

- `business_1339_settle`: 20 -> 12
- `business_1227_settle`: 20 -> 12
- `business_1147_settle`: 20 -> 12
- `business_987_settle`: 20 -> 12
- `business_907_settle`: 20 -> 12
- `business_843_settle`: 20 -> 12
- `business_779_settle`: 20 -> 12
- `business_elevator_settle`: 20 -> 12

`business_1067_settle` remains 30f. P2/P3, the elevator-center settle, and the
floor-recover settle remain unchanged.

## Pure isolation

The documented source files are present, but the required pure-isolation
choice is not wired in the current probe CLI. The attempted command exited
with code 2 and reported `business-to-warehouse` as an invalid segment choice:

```text
kpdr.py pure: error: argument segment: invalid choice: 'business-to-warehouse'
```

Available source fixtures include:

- `custom_integrations/SuperMetroid-Snes/scratch/continuous_like_business_climb_entry.state`
- `custom_integrations/SuperMetroid-Snes/scratch/business_to_warehouse_function.state`
- `custom_integrations/SuperMetroid-Snes/scratch/post_varia_to_kraid_pure.state`

The required probe command, once the pure choice is wired by its owning task,
is:

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure business-to-warehouse \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/continuous_like_business_climb_entry.state
```

No pure result was obtained because the documented source exists but the
required pure choice is missing. Pure isolation is not continuous integrity
evidence.

## Planner gate and rollback

The planner must re-record `--to kraid --no-video` before claiming any savings.
There is no continuous or STATUS savings claim in this note. If continuous
fails, immediately revert all eight labels above from 12f back to 20f.
