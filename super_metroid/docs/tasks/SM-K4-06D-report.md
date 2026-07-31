# SM-K4-06D Weapon Door Recon Report

## Scope

Source:
`custom_integrations/SuperMetroid-Snes/scratch/post_varia_to_kraid_pure.state`
(`0xA59F`, Kraid's Room, start `x=463 y=395 pose=10`). Each mode booted a
fresh copy of that source. The probe used normal weapon selection and inputs,
plus `UnlimitedResourcesAssist` for energy/ammo attrition only. It sampled
room, door transition, pose, position, selected item, and ammo after every
emulator frame.

This is a bounded development diagnostic. It is **not pure-green evidence**,

## Run

Command:

```bash
uv run python super_metroid/scripts/probe/kraid_door_weapon_recon.py \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_to_kraid_pure.state \
  --mode all
```

Exit code: `0`.

Stdout:

```text
mode=beam available=True selected=True frames=1200 rooms=['0xA59F'] door_transition!=0=False last_pin={'room': '0xA59F', 'pose': 138, 'x': 37, 'y': 395}
mode=missile available=True selected=True frames=1200 rooms=['0xA59F'] door_transition!=0=False last_pin={'room': '0xA59F', 'pose': 138, 'x': 37, 'y': 395}
mode=super available=True selected=True frames=1200 rooms=['0xA59F'] door_transition!=0=False last_pin={'room': '0xA59F', 'pose': 138, 'x': 37, 'y': 395}
output=super_metroid/debug/kraid_door_weapon_recon.json
```

## Results

| Mode | Available | Selected | Door transition? | Room change? | Rooms observed | Final pin |
|---|---|---|---|---|---|---|
| beam | yes | yes | no | no | `0xA59F` | `0xA59F`, pose `138`, `x=37`, `y=395` |
| missile | yes | yes | no | no | `0xA59F` | `0xA59F`, pose `138`, `x=37`, `y=395` |
| super | yes | yes | no | no | `0xA59F` | `0xA59F`, pose `138`, `x=37`, `y=395` |

All sampled frames had `door_transition == 0`. No mode changed rooms. The
three attempts also converged to the same left-side pose and coordinate pin,
so changing weapon type did not produce a distinguishable door response in
this choreography.

## Source Ammo

| Resource | Current | Capacity |
|---|---:|---:|
| Missiles | 15 | 15 |
| Supers | 5 | 5 |
| Power bombs | 0 | 0 |

The source had both missile and Super capacity, so both non-beam attempts were
actually selected and fired. The assist telemetry recorded zero energy writes,
zero ammo writes, zero progression writes, and zero capacity writes in every
mode.

## Recommendation

No weapon clearly opens the door. Do not change the production controller to a
missile or Super experiment based on this run. The next card should investigate
the trigger geometry or read-only door/PLM/BTS state, using the same source and
bounded choreography, before changing production route behavior.

## Non-Claims

- No pure-green claim.
- No continuous claim.
- No STATUS promotion.
- No graph or route promotion.
- No progression, capacity, room, event, boss, or door RAM was forged or
  written. The only assist writes permitted were resource-attrition writes;
  this run recorded none.
