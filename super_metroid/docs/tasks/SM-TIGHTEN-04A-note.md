# SM-TIGHTEN-04A: Patch A Note

## Scope

Only the `main_shaft_entry_settle` site in
`routes/spore_spawn_controller.py` was changed. Dachora settle and descent
cadence were not changed. Patch B and Patch C remain unimplemented.

## Old-to-new control flow

- Old: unconditionally idle for 1,000 frames with reason
  `main_shaft_entry_settle`, then begin the four 60-frame main-shaft descent
  legs.
- New: poll for up to 360 frames. Each unsuccessful poll idles for one frame
  with the same `main_shaft_entry_settle` reason. The settle exits early when
  Samus is in the recorded x band `118..126`, has an established standing-ish
  pose, and has zero vertical velocity. On timeout it raises `TimeoutError`
  rather than starting the descent from an unconfirmed pose. The same four
  descent legs then run unchanged.

## Timeout

The selected cap is **360 frames** (6 seconds at 60 Hz), within the requested
300-400-frame range.

## Verification ownership and residual

The planner should re-record the continuous prefix and compare the split:

```bash
uv run python super_metroid/scripts/record/continuous.py --to spore --no-video
```

No 600-800 frame saving is claimed. The current 2,806-frame
`green_brinstar_main_shaft` dwell remains the comparison baseline until a
green re-record and split-dwell report exist. This patch does not establish
continuous integrity, natural-entry evidence, or a STATUS promotion.

## Non-claims

- No continuous run was recorded by this card.
- No dwell reduction or performance saving is claimed.
- No Patch B or Patch C behavior was changed.
- No progression, capacity, equipment, boss-bit, room, door, or map state was
  written or forged.
- No continuous verification or STATUS promotion is claimed.
