## Residual — rr-qpkd Pure Double → Single Chamber return

### Result
**GREEN dual pure** — Double Chamber `0xADAD` → Single Chamber `0xAD5E`.
Exact dual **1101f ×2**. Not continuous. Part of Wave→Business stack
(`rr-vqv3`); stack remains open.

### Files changed
- `routes/kpdr/wave/double_to_single.py` — pure controller (ledge hop → Super
  column drop → floor LEFT + morph tunnel → left blue door y395)
- `routes/kpdr/wave/geometry.py` — `DTS_*` bands
- `routes/kpdr/wave/__init__.py`, `k4_wave.py`, `k4_norfair.py`, `registry.py`,
  `routes/kpdr/__init__.py`
- `scripts/probe/kpdr.py` — `double-to-single-chamber` pure CLI
- `source_states.py` — `post_double_to_single_chamber_pure` fingerprint
- `tests/test_k4_wave_return_scaffold.py` — registry / geometry unit

### Verify paste
```bash
uv run pytest snes/super_metroid/tests/test_k4_wave_return_scaffold.py -q
# → 4 passed

uv run python snes/super_metroid/scripts/probe/kpdr.py pure double-to-single-chamber \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_wave_to_double_chamber_pure.state \
  --expect-room 0xADAD \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_double_to_single_chamber_pure.state
# → GREEN room=0xAD5E xy=(216,630) frames=1101 (×2 exact dual)
```

### Acceptance
- [x] Pure hop `0xADAD` → `0xAD5E` dual green
- [x] Export post state for Single→Bubble predecessor
- [x] Source is `post_wave_to_double_chamber_pure` (not stale post_double_to_wave)
- [ ] Full Wave→Business stack (parent `rr-vqv3` remains open)
- [x] No STATUS promote continuous Ice

### Probe pin (dual exact)
room=0xAD5E pose=82 x=216 y=630 door_transition=0 frames=1101 last_pin=dual exact
door_def_ptr=0x9612 (Double bottom-left blue)

### Residual risks
1. Settle pin is deep Single shaft ~(216,630) not human tape first-frame
   ~(20,395). Next hop Single→Bubble must climb from this natural pure pin
   (or re-settle) — do not invent a fake top pin.
2. Morph tunnel band `DTS_MORPH_TUNNEL_X` is tape-derived; heated Funes can
   still knock mid-floor (escape_kb LEFT).
3. Parent stack `rr-vqv3` still needs Single→Bubble→Farm→Speedway→Frog→Business.

### Next action (required)
- **Next card ID:** `rr-u0y8` — Pure Single → Bubble return (`0xAD5E`→`0xACB3`)
- **One change:** one-hop pure from `post_double_to_single_chamber_pure`
- **Source state:** `scratch/post_double_to_single_chamber_pure.state` ~(216,630)

### Non-claims
- Did not STATUS-promote continuous `--to ice`
- Did not claim dual continuous green Ice
- Did not finish Wave→Business full stack
- Did not implement Single→Bubble pure yet
