# SM-PURE-ISO Source Map

These are controller-only probes. A pure probe uses the named source state and
resource assists, but it is not a continuous integrity run: it does not prove
natural predecessor entry, zero state loads, zero progression writes, or a
continuous tip. Continuous re-records and STATUS promotion remain planner
gates.

## Pure Choices

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure hj-shaft-to-business \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/hj_shaft_to_business_source.state

uv run python super_metroid/scripts/probe/kpdr.py pure business-to-warehouse \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/continuous_like_business_climb_entry.state
```

## Required Sources

| Segment | Required source | Expected start room | State status |
|---|---|---:|---|
| `hj-shaft-to-business` | `custom_integrations/SuperMetroid-Snes/scratch/hj_shaft_to_business_source.state` | `0xAA41` | **MISSING** |
| `business-to-warehouse` | `custom_integrations/SuperMetroid-Snes/scratch/continuous_like_business_climb_entry.state` | `0xA7DE` | present and room-validated |

The available `hijump_to_business_composed.state` is room `0xA7DE`, so it is
not a valid `hj-shaft-to-business` source. The available
`business_to_warehouse_function.state` is room `0xA6A1`, an output state, so it
is not the required Business entry either.

To validate a newly captured natural shaft source before running the controller:

```bash
uv run python -c 'from pathlib import Path; from super_metroid.dev.common import make_dev_env, boot_from_state; from super_metroid.ram import parse_env_state; p=Path("super_metroid/custom_integrations/SuperMetroid-Snes/scratch/hj_shaft_to_business_source.state"); e=make_dev_env(); boot_from_state(e,p); s=parse_env_state(e); print(f"room=0x{s.room_id:04X} pose={s.pose} x={s.samus_x} y={s.samus_y}"); e.close()'
```

The pure probe itself is the required segment validation once the source is
available:

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure hj-shaft-to-business \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/hj_shaft_to_business_source.state
```

Do not create a green source by door-warping into the shaft. The source must be
captured from the natural predecessor chain by the planner/reviewer.

## Regression Target

The harness isolates the climb that exposed the SM-TIGHTEN-01B regression:
the continuous re-record failed at `business_1227_land` with Samus on the
`y=1419` floor, then the floor-recover retry failed at
`business_1339_ground` (`y=1291`). A pure Business source can catch that
failure in minutes instead of waiting for the approximately 27-minute full
continuous run, but a pure pass still does not make the continuous run green.
