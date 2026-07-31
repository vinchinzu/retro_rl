# TASK SM-K4-R-01: Geometry attempt `eye_to_baby_return` pure (chained source)

## Recipe step
1 pure controller (geometry — **may red; residual quality is the stress test**)

## Model
Luna

## Own files only
- `routes/kpdr/kraid_return.py` (**edit** `play_eye_to_baby_return` only)
- optional: `docs/tasks/SM-K4-R-01-residual.md`

Do **not** edit varia_return (except if you need a read), continuous, STATUS,
progression verification, or other reverse hops.

## Context
- Scaffold exists from SM-K4-R-SCAFFOLD (naive spin-push).
- **Blocked natural source** until `kraid_to_eye_return` is pure green and
  saves `scratch/post_kraid_to_eye_return.state` (or equivalent).
- This card is intentionally aggressive: either
  (A) pure-green from a **named** eye-room source if it exists, or
  (B) **exit with blocked residual** if no valid source — do **not** door-warp
  invent a green claim.

## Read first
- `routes/kpdr/kraid_return.py`
- `routes/kpdr/kraid_approach.py` forward eye/baby geometry
- `routes/kpdr/varia_return.py` door open patterns
- `scripts/probe/kpdr.py` pure choices for `eye-to-baby-return`
- list `scratch/*eye*` / post_kraid states

## Do
1. Search scratch for a state that loads into `ROOM_KRAID_EYE` (0xA56B) with
   ordinary gameplay post-Varia return context. Document path or MISSING.
2. If MISSING: improve scaffold **lightly** only if there is a clear door-open
   gap vs forward controller (document); pure probe not runnable — EXIT with
   residual “blocked on source,” no force-pass.
3. If source exists: tune **one** hop (door open + directional exit) for pure
   green; stop after 2–3 strategies; residual pin if red.
4. Never promote graph edge verification.

## Residual required
- Source path or MISSING
- Pure result or blocked
- Next planner step (produce source via 06B chain)

## Do not
- Warp to fake pure green
- continuous / STATUS / multi-hop compose
- Edit all four reverse hops in one card

## Acceptance
- [ ] Honest pure green **or** blocked/red residual
- [ ] pytest controller_common green
- [ ] No verification promotion

## Verify commands
```bash
uv run pytest super_metroid/tests/test_controller_common.py -q
# only if source exists:
# uv run python super_metroid/scripts/probe/kpdr.py pure eye-to-baby-return \
#   --source <named eye-room state>
```
