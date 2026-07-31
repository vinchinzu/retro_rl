# SM-DOOR-PHASE Diagnostic Report

## Scope

Source for both modes:
`custom_integrations/SuperMetroid-Snes/scratch/post_varia_to_kraid_pure.state`
(`0xA59F`, Kraid's Room, start `x=463 y=395 pose=10`). The probe used only
`UnlimitedResourcesAssist` and read full-bank state for event/boss telemetry.
It did not write progression, capacity, room, event, or boss state.

This is a bounded diagnostic. It is **not pure-green evidence**, is **not
continuous evidence**, and does **not** promote STATUS.

## Phase Run

Command:

```bash
uv run python super_metroid/scripts/probe/kraid_door_phase_recon.py \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_to_kraid_pure.state \
  --mode phases --frames 2000
```

The six phases were executed separately and sampled every emulated frame. The
run used 1,209 frames because the bounded choreography completed before the
2,000-frame cap.

| Phase | Frames spent | X range | Y range | Pose range | Door transition max | Outcome |
|---|---:|---:|---:|---|---:|---|
| approach | 141 | 178..463 | 284..433 | 10, 26, 167 | 0 | Reached `x=178`, still `0xA59F` |
| lip_backoff | 10 | 155..173 | 427..427 | 1, 9, 38 | 0 | Backed off without transition |
| unmorph | 26 | 157..157 | 400..427 | 3, 75, 77 | 0 | Unmorph primitive completed |
| face_left_release | 14 | 156..158 | 406..428 | 47, 82, 165 | 0 | Left-facing release completed |
| door_shots | 88 | 127..155 | 427..427 | 2, 10, 12 | 0 | Four shot/fuse cycles, no transition |
| spin_push | 930 | 37..127 | 284..427 | 1, 2, 37, 38, 76, 78, 82, 138, 165 | 0 | Remained in `0xA59F`; final `x=37 y=307 pose=82` |

Observed across all phase samples:

- Rooms: only `0xA59F`.
- `door_transition != 0`: never observed.
- `transition_direction`: stable at `5`.
- Enemy 0 HP: stable at `1000`.
- Boss bits: stable at `[4, 3, 0, 0, 0, 0, 1, 0]`.
- Selected item: stable at `0`.

## Y Sweep

Each attempt booted a fresh copy of the required source state. Inputs were
normal movement only: floor walk, an 18-frame short hop, or a 36-frame medium
hop during the left approach, followed by the same four shot/fuse cycles and
an 80-frame brief spin. The three attempts used 666 frames each, for 1,998
frames total.

| Attempt | Peak left X | Y range | Final pose | Final X/Y | Room change? | Door transition? |
|---|---:|---:|---:|---|---|---|
| floor_walk | 36 | 316..429 | 82 | 36/374 | N | N |
| short_hop | 36 | 316..436 | 82 | 36/374 | N | N |
| medium_hop | 36 | 290..433 | 82 | 36/374 | N | N |

All attempts ended in room `0xA59F`. No sample saw a room other than
`0xA59F`, and no sample saw a nonzero `door_transition`.

## Ranked Hypotheses

1. **Wrong height / lip geometry remains the leading hypothesis.** The
   production-like phase choreography reached the left side and the Y sweep
   changed the jump arc, including a medium-hop range down to `y=290`, but no
   door trigger occurred. This supports testing the exact trigger Y band or a
   more deliberate jump landing rather than assuming the floor lip is valid.
2. **Pose-138 pin is not the whole failure.** The original walk probe stayed
   at pose 138 and `x=85`; this instrumented run exercised backoff, unmorph,
   shots, and spin recovery, and ended at `x=37 pose=82`. The lip remains
   blocked, but the exact old pose pin was recoverable.
3. **Closed blue door is more plausible than before, but unproven.** Four
   standing left-facing beam-shot cycles produced no door transition. The
   probe cannot distinguish an unopened door from a trigger-position failure.
4. **Enemy interaction is weakly supported only.** Enemy 0 remained at 1000
   HP throughout both modes, but unchanged HP does not prove collision is the
   cause or that the enemy blocks the trigger.
5. **Door-definition / pin mismatch remains unresolved.** The stable direction
   value `5` was observed, but this diagnostic intentionally did not read or
   modify door-definition state.

## Future SM-K4-06b Recipes

Try exactly one geometry change per bounded run and retain the same source,
telemetry, and no-write rules:

1. **Jump approach Y:** replace only the floor-level approach with one fixed
   short-hop timing and hold the left input through the target Y band before
   firing.
2. **Backoff duration:** keep all other timings fixed and test only 6, 10,
   and 14 frames of RIGHT before unmorph.
3. **Re-face timing:** keep the 10-frame backoff and shots fixed; vary only the
   LEFT face hold and release durations.
4. **Shot timing:** keep position and facing fixed; vary only the shot hold or
   fuse interval, one parameter per run.
5. **Spin-push timing:** keep the successful recovery sequence fixed and vary
   only the first spin-push delay or brief-spin duration.

The single recommended next controller change is **one fixed short-hop Y
approach primitive** before the existing door-shot sequence. Do not combine it
with backoff, re-face, shot, or spin changes in that card.

## Claims And Residuals

- Not pure green.
- Not continuous.
- No STATUS promotion.
- No progression or capacity RAM was forged or written.
- Last phase-mode pin: room `0xA59F`, pose `82`, `x=37`, `y=307`.
- Last Y-sweep pin for every attempt: room `0xA59F`, pose `82`, `x=36`,
  `y=374`.
- No sample in either mode saw `door_transition != 0` or a room other than
  `0xA59F`.
