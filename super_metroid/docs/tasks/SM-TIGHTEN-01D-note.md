# SM-TIGHTEN-01D — Business 12f settle baseline

## Baseline

`routes/kpdr/business_climb.py` already has 12f on all eight platform
settles:

- `business_1339_settle`
- `business_1227_settle`
- `business_1147_settle`
- `business_987_settle`
- `business_907_settle`
- `business_843_settle`
- `business_779_settle`
- `business_elevator_settle`

The pure-green setup is **four setup jumps** (`RIGHT, LEFT, LEFT, RIGHT`)
with these 12f settles. The planner pure run cleared
`business-to-warehouse` in approximately **3,721f**, ending in Warehouse
(`0xA6A1`). The 30f `business_1067_settle`, 8f elevator-center settle, and 15f
floor-recovery settle are not part of the eight platform-settle baseline.

## Pure verification

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure business-to-warehouse \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/continuous_like_business_climb_entry.state
```

Expected result: pure GREEN, approximately 3,721f, with the final room
Warehouse (`0xA6A1`). This is controller-isolation evidence only; it is not
continuous natural-entry or integrity evidence.

## Planner continuous recipe

The planner must re-record the continuous prefix before claiming this baseline
is safe on the natural chain:

```bash
uv run python super_metroid/scripts/record/continuous.py --to kraid --no-video
uv run python super_metroid/scripts/export/split_dwell.py \
  super_metroid/recordings/start_to_kraid.json --reasons --top 20
```

Compare the `business_to_warehouse` split in the new report against the prior
baseline of approximately **2,257f**. The relevant dwell fields are the split
id `business_to_warehouse`, its frame count, reason totals, and the start/end
frame range. The planner should also check the report's integrity fields:
zero state loads, zero progression writes, and natural Kraid entry.

If the continuous run fails in the Business climb, roll back all eight
platform settles from **12f to 20f** before investigating other knobs. Do not
change the four-jump setup as part of this rollback.

## Explicit non-claims

- This note does not claim a continuous-green result or any dwell savings.
- This note does not promote `STATUS.md`, the progression graph, or a tracker.
- Pure isolation does not prove natural predecessor entry, zero state loads,
  zero progression writes, or completion of the continuous tip.
- No progression, capacity, equipment, boss/event, door-state, or map-state RAM
  was forged or written for this baseline.
