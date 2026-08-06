## Residual — rr-siz / SM-CLEAN-BOMBS

### Result
SUPERSEDED → GREEN via **SM-CLEAN-BT-ECONOMY** (rr-5of)

Original compose RED (death in BT on hash-pinned policy under Clean). Economy
one-knob landed clean hybrid + kite defaults. See
[`SM-CLEAN-BT-ECONOMY-residual.md`](SM-CLEAN-BT-ECONOMY-residual.md).

### Verify paste
```bash
uv run python snes/super_metroid/scripts/record/continuous.py \
  --to bombs --clean --no-video \
  --report snes/super_metroid/recordings/bombs_clean.json
# [GREEN] frames=49321 room=0x92FD (dual reverify same)
```

### Next action (required)
- **Next card ID:** SM-CLEAN-STAB / SM-CLEAN-STATUS (planner)
- **One change:** STATUS secondary promote for Clean bombs (dual already green)
- **Source state:** n/a
