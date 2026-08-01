## Residual — SM-K4-SPEEDWAY-PURE

### Result

PARTIAL

### Files changed

- `routes/kpdr/k4_norfair.py` — Business→Frog controller complete; Speedway remains scaffolded.
- `routes/continuous.py` — accepted Frog Save tip registered.

### Verify paste

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure business-to-frog-save \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_business_continuous.state
# GREEN: 0xB167 Frog Savestation, 1,190f

uv run python super_metroid/scripts/record/continuous.py --to frog --no-video
# GREEN: 114,923f, twice; all integrity flags true
```

### Acceptance

- [x] Business→Frog pure green from the accepted predecessor.
- [x] Frog Save continuous twice from power-on.
- [ ] Frog Save→Speedway pure green.

### Residual risks

- `play_frog_save_to_speedway` is still the bounded scaffold; no natural
  Frog Save exit geometry has been accepted.

### Next action (required)

- **Next card ID:** SM-K4-SPEEDWAY-PURE
- **One change:** replace the Frog Save→Speedway scaffold hold loop with one
  natural right-door geometry sequence.
- **Source state:** `scratch/post_frog_continuous.state` (room `0xB167`).

### Non-claims

- No progression, capacity, door, event, boss, or room-state writes.
- Frog Save→Speedway is not continuous evidence.

### Probe pin

room=0xB167 pose=11 x=39 y=139 door_transition=0
frames=114923 (accepted continuous endpoint)
last_pin=room=0xB167 pose=11 x=39 y=139
