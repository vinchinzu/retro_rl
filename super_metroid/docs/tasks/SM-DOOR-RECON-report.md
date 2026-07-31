# SM-DOOR-RECON Report

## Run

Command:

```bash
uv run python super_metroid/scripts/probe/kraid_left_door_recon.py \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_to_kraid_pure.state \
  --frames 600
```

The bounded probe exited 0 after exactly 600 emulated frames. It used the
resource-only `UnlimitedResourcesAssist`; it did not write room, item,
capacity, event, or boss state. The JSON artifact is
`debug/kraid_left_door_recon.json`.

## Observed Trajectory

| Point | Frame | Room | Pose | X | Y | Game state | Door transition | Direction | Enemy 0 HP | Selected item |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Start | 0 | `0xA59F` | 10 | 463 | 395 | 8 | 0 | 5 | 1000 | 0 |
| Left approach, first pinned sample | 170 | `0xA59F` | 138 | 85 | 427 | 8 | 0 | 5 | 1000 | 0 |
| End | 600 | `0xA59F` | 138 | 85 | 427 | 8 | 0 | 5 | 1000 | 0 |

The approach moved left from `x=463` to `x=85`. Y rose from `395` to `427`
while approaching the floor. Pose `138` first appeared at the sampled frame
170 and remained through the end of the run. The four requested standing beam
shot attempts did not produce a visible state change in the sampled fields.

Full observed ranges:

- X: `85..463`
- Y: `395..427`
- Pose: `10`, `42`, and `138`
- Game state: only `8`
- Door transition: only `0`
- Transition direction: only `5`
- Enemy 0 HP: only `1000`
- Rooms: only `0xA59F`

Boss bytes were stable for every sample: `[4, 3, 0, 0, 0, 0, 1, 0]`.
The area-selected byte was `3` (the Brinstar/Kraid slot in this state). No
boss-bit mutation was observed.

## Interpretation

The room never changed, and no door-transition state was observed. This is
not pure-green evidence. The probe confirms the existing failure shape: left
movement reaches `x=85`, then Samus remains in pose `138` at the lower lip
(`y=427`) without opening or traversing the blue door.

Possible causes, ordered by what this run supports:

1. **Wrong height / lip geometry.** The attempt reaches the floor-level lip,
   but the blue-door trigger may require a different Y band or a jump/spin
   approach rather than holding LEFT at `y=427`.
2. **Pose pin / collision lock.** Pose `138` appears at the exact leftward
   stopping point and never recovers under the bounded walk-and-shot script.
   The controller's backoff and unmorph sequence was not exercised by this
   recon script.
3. **Door remains closed.** `door_transition=0` for all samples means no
   transition was initiated. It does not distinguish a closed blue door from
   a failed trigger due to Samus' position or pose.
4. **Enemy interaction.** Enemy 0 was present at `1000` HP and unchanged. It
   may contribute collision or pinning, but this run does not establish that
   it blocks the door.
5. **Pin / door-definition mismatch.** The state exposes a stable transition
   direction value of `5`, but this probe did not read or alter the door
   definition pointer and cannot identify the intended left-door pin.

## Recommended Next Geometry Card

1. Start from the same source state and instrument the existing
   `play_kraid_to_eye_return` phases separately: approach, right backoff,
   `unmorph`, four door shots, and spin-push. Record the same fields at every
   frame around each phase.
2. Test a small, explicit Y sweep near the left lip using normal movement only
   (for example, floor, one jump arc, and a higher landing), with each attempt
   capped and sourced from a fresh copy of the named state. Do not teleport
   Samus for a green claim; if a placement is used for diagnosis, label it
   development-only.
3. Compare standing beam shots after confirmed unmorph and left-facing pose
   against the current run. Record whether any shot changes a door-open byte,
   door transition, or room state; do not write those bytes.
4. If the lip remains pinned, vary only one controller primitive at a time:
   backoff duration, re-face duration, shot timing, and spin-push timing.
   Keep the total budget bounded at roughly 2,000 frames.
5. A successful geometry probe should still be marked `controller_dev` only.
   A planner must later establish natural predecessor entry and continuous
   integrity before any route or status promotion.
