# TASK SM-ROOM-BOOT-01: Bootstrap next unstarted easy item rooms

## Recipe step
room practice bootstrap (dual-track)

## Model
Luna

## Wave type
implement

## Own files only
- entry fixtures under `custom_integrations/SuperMetroid-Snes/room_*.state`
  for the listed problems only (+ provenance if written)
- optional note: `docs/tasks/SM-ROOM-BOOT-01-note.md`

## Context
- Easy teleport-open set ranks 52–56 are residual-stuck or parked.
- Next easy ranks 57+ are **unstarted** (no teleport fixture).
- Bootstrap only — do not invent full clear policies this card.
- Practice track only.

## Do
1. Bootstrap doorway fixtures for **up to 4** of these (stop early if tooling
   fails; residual lists which landed):
   - `room_a890_from_a8b9_to_a8b9` Ice Beam Room
   - `room_a447_from_a408_to_a408` Spazer Room
   - `room_a15b_from_a130_to_a130` Hopper Energy Tank Room
   - `room_a1d8_from_a1ad_to_a1ad` Billy Mays' Room
2. Use:
   ```bash
   uv run python super_metroid/scripts/room/run_problem.py bootstrap PROBLEM_ID
   uv run python super_metroid/scripts/room/run_problem.py teleport PROBLEM_ID
   ```
3. Residual: table of problem → teleport ok / fail + next scaffold card ids.
4. No continuous / STATUS / kpdr spine.

## Acceptance
- [ ] ≥1 new teleport-ready fixture **or** residual with bootstrap pin/error
- [ ] Dual-track non-claim

## Verify
```bash
uv run python super_metroid/scripts/room/run_problem.py teleport room_a890_from_a8b9_to_a8b9
# (repeat for any others that bootstrapped)
```
