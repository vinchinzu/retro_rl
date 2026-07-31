# SM-DOOR-BLUE Diagnostic Report

## Scope

Source:
`custom_integrations/SuperMetroid-Snes/scratch/post_varia_to_kraid_pure.state`
(`0xA59F`, Kraid's Room, start `x=463 y=395 pose=10`). The probe used only
`UnlimitedResourcesAssist`, sampled after boot and after every emulator frame,
and did not write progression, capacity, room, event, boss, or door state.

This is a bounded harness diagnostic. It is **not pure-green evidence**, is
**not continuous evidence**, and does **not** promote STATUS.

## Run

Command:

```bash
uv run python super_metroid/scripts/probe/kraid_door_blue_recon.py \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_to_kraid_pure.state
```

Stdout:

```text
frames=1200 rooms=['0xA59F'] door_transition!=0=False output=super_metroid/debug/kraid_door_blue_recon.json
last_pin={'room': '0xA59F', 'pose': 138, 'x': 37, 'y': 395}
```

The sequence reached the left side with the prior approach, backed off for 10
frames, ran the unmorph and face-left recovery, fired four 4-frame standing
left-shot / 18-frame fuse cycles, then continued with a bounded spin-push.
Every frame was sampled. The probe completed with exit code 0.

## Door Transition Windows

There were no frames where `door_transition` was nonzero.

| Window | Frame range | Phases | Values | Result |
|---|---:|---|---|---|
| none | none | none | none | `door_transition` was `0` for all 1,201 samples, including boot |

Additional stable observations:

| Field | Observation |
|---|---|
| Rooms | `0xA59F` only; no room transition |
| `transition_direction` | Stable at `5` |
| `game_state` | Stable at `8` (ordinary gameplay) |
| `enemy0_hp` | Stable at `1000`, including all shot windows |
| `door_definition_ptr` | Stable at decimal `37458` (`0x9252`) |
| `knockback_timer` | Stable at `0` |
| Final pin | `0xA59F`, pose `138`, `x=37`, `y=395` |

## Shot Effects

The four shot/fuse cycles covered frames `192..279`, with Samus at
`y=427` and `x=155..127`. The sampled fields that changed during the shot
windows were pose, X, and the exposed `invincibility_timer`. The following
door-relevant observations did **not** change:

- `door_transition` stayed `0`.
- `transition_direction` stayed `5`.
- `door_definition_ptr` stayed `0x9252`.
- `enemy0_hp` stayed `1000`.
- `knockback_timer` stayed `0`.
- The room stayed `0xA59F`.

The probe therefore confirms that inputs were reaching the scripted shot
windows and that Samus moved through the shell area, but it does not show an
open-door state or a transition trigger.

## Harness Limits

`SuperMetroidState` and existing `ram.py` helpers expose the navigation fields
used here, plus read-only `door_def_ptr`, invincibility, and knockback peeks.
The current harness does not expose:

- Door open-state / door state-machine internals.
- PLM records or PLM activation state.
- Door BTS / tile-collision metadata.

No new RAM constants or reverse-engineered fields were added in this card. The
stable `door_definition_ptr` is only an identifier/pointer observation; it is
not evidence that the blue door is open or closed.

## Ranked Hypotheses

1. **Wrong trigger height / lip geometry remains the leading hypothesis.** The
   run reached `x=127` while standing and firing at `y=427`, then the bounded
   spin-push ended pinned at `x=37`, `pose=138`. No trigger occurred despite
   the shot timing being exercised, so the exact vertical trigger band or
   shell collision geometry remains untested.
2. **Closed blue door remains plausible but unproven.** Four standing shots
   did not produce a transition, but the harness cannot observe the door's
   open-state or PLM activation state and therefore cannot separate a closed
   door from a valid door whose trigger geometry was missed.
3. **Other collision or route-state mismatch is possible.** The stable enemy
   HP and knockback timer provide no evidence that enemy damage or knockback
   caused the failure. A door-definition pointer mismatch, BTS collision, or
   hidden door-state predicate cannot be ruled out with the current fields.

## Recommended Next Card

Add a **read-only door/PLM/BTS RAM reconnaissance card** that first identifies
the minimal exposed field needed to distinguish closed-door state from trigger
geometry; do not write the candidate field or change the controller until that
field is available and validated.

## Residuals And Non-Claims

- Last pin: room `0xA59F`, pose `138`, `x=37`, `y=395`.
- Harness gap list: door open/state-machine internals, PLM records/activation,
  and door BTS/tile-collision metadata are unavailable.
- No pure-green claim.
- No continuous claim.
- No STATUS promotion.
- No progression, capacity, event, boss, room, or door RAM was forged or
  written. Resource assist was enabled only for energy/ammo attrition and
  reported zero writes and zero progression/capacity writes.
