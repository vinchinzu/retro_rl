# SM-TIGHTEN-P2B Result

Tried the three-jump setup candidate `("LEFT", "LEFT", "RIGHT")` in both
the normal Business setup and the floor-recover re-climb. This differs from
the Wave-4 candidate `("RIGHT", "LEFT", "LEFT")`, which was pure-red on the
same source.

## Pure gate

The required pure probe exited `0` and reached Warehouse (`roomIdHex=0xA6A1`)
at frame `3467`, with `samusX=37`, `samusY=139`, and `pose=138`.

## Final tree state

The three-jump tuple remains in both setup loops. Settles and `runup_907`
were not changed.

## Residual risk

- This is pure controller evidence only, not continuous natural-entry or
  integrity evidence.
- The planner must run
  `uv run python super_metroid/scripts/record/continuous.py --to kraid --no-video`
  before any continuous claim, followed by the required multi-run reliability
  gate if that run succeeds.
- No STATUS promotion was made, and no progression or capacity RAM was forged.
