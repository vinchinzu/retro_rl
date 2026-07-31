# TASK SM-HJ-SRC: Capture pure source for `hj-shaft-to-business`

## Recipe step
harness / scratch state (enables pure HJ return isolation)

## Model
Luna

## Own files only
- `scripts/probe/` or `scripts/export/` helper if needed (**create** small)
- `docs/tasks/SM-HJ-SRC-note.md` (**create**)
- optional gitignored state under
  `custom_integrations/SuperMetroid-Snes/scratch/hj_shaft_to_business_source.state`

Do **not** edit continuous.py, STATUS, hijump_return knobs, business_climb.

## Context
SM-PURE-ISO marked `hj-shaft-to-business` source **MISSING** (need room
`0xAA41` HJ shaft). Available `hijump_to_business_composed.state` is wrong room.
Without this source, 02B/02C gray-door + bomb-tunnel trims cannot pure-gate.

## Read first
- `docs/tasks/SM-PURE-ISO-note.md`
- `scripts/probe/kpdr.py` pure + save-state options
- `routes/kpdr/hijump_return.py` entry contract for `play_hj_shaft_to_business`
- `routes/kpdr/rooms.py` ROOM_HJ_SHAFT
- existing pure segments that end in HJ shaft if any

## Do
1. Prefer a **controller-only natural** chain from an existing valid scratch:
   e.g. pure hop(s) that land ordinary in `0xAA41`, then save state via probe
   CLI if supported (`--output` / save helpers). No door-warp for “green.”
2. If only dev warps exist, you may create a state labeled
   `developmentOnly` in the note — **must not** claim pure continuous evidence;
   still useful for mechanical bomb-tunnel tuning if room matches.
3. Validate: boot state → room `0xAA41`, ordinary gameplay; print x/y/pose.
4. Run pure `hj-shaft-to-business` once; paste exit code + residual.
5. Note documents path + room hex + pure result + non-claims.

## Residual required
- State path + room validation paste
- Pure HJ result
- Whether state is natural vs developmentOnly

## Do not
- continuous STATUS
- Claim natural-entry continuous from warp state
- Edit 02B knobs in this card

## Acceptance
- [ ] State file exists with room 0xAA41 **or** blocked residual with why
- [ ] Pure attempted if state exists
- [ ] Note complete

## Verify commands
```bash
# after state exists:
uv run python super_metroid/scripts/probe/kpdr.py pure hj-shaft-to-business \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/hj_shaft_to_business_source.state
```
