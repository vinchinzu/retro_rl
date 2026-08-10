## Residual — rr-u0y8 Pure Single → Bubble return

### Result
**GREEN dual pure** — Single Chamber `0xAD5E` → Bubble Mountain `0xACB3`.
Exact dual **817f ×2**. Not continuous. Part of Wave→Business stack
(`rr-vqv3`); stack remains open.

### Files changed
- `routes/kpdr/wave/single_to_bubble.py` — pure controller (deep LEFT+A hop
  over morph slope → wall A/RIGHT+A → mid-low y523 → floor y395 → wall climb
  → top y139 left blue door)
- `routes/kpdr/wave/geometry.py` — `STB_*` bands
- `routes/kpdr/wave/__init__.py`, `k4_wave.py`, `k4_norfair.py`, `registry.py`,
  `routes/kpdr/__init__.py`
- `scripts/probe/kpdr.py` — `single-to-bubble` pure CLI
- `source_states.py` — `post_single_to_bubble_pure` fingerprint
- `tests/test_k4_wave_return_scaffold.py` — registry / geometry unit

### Verify paste
```bash
uv run pytest snes/super_metroid/tests/test_k4_wave_return_scaffold.py -q
# → 5 passed

uv run python snes/super_metroid/scripts/probe/kpdr.py pure single-to-bubble \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_double_to_single_chamber_pure.state \
  --expect-room 0xAD5E \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_single_to_bubble_pure.state
# → GREEN room=0xACB3 xy=(472,395) frames=817 (×2 exact dual)
```

### Acceptance
- [x] Pure hop `0xAD5E` → `0xACB3` dual green
- [x] Export post state for Bubble→Farm predecessor
- [x] Source is `post_double_to_single_chamber_pure` deep pin ~(216,630) — climbed, no fake top pin
- [ ] Full Wave→Business stack (parent `rr-vqv3` remains open)
- [x] No STATUS promote continuous Ice

### Probe pin (dual exact)
room=0xACB3 pose=12 x=472 y=395 door_transition=0 frames=817 last_pin=dual exact
door_def_ptr=0x95CA (Bubble right sill from Single top-left leave)

### Residual risks
1. Deep floor has morph-slope trap ~x167 — pure LEFT falls to y779+; LEFT+A hop
   required (human f6926–6960). Do not simplify deep approach to walk-only.
2. Floor y395 wall climb launch must start x≤88; x≥95 wall-misses and falls
   back to y395.
3. Mid-air y≈507 pose 81 is not a solid mid-low land — stop_when needs pose in
   ledge set and x≥70.
4. Parent stack `rr-vqv3` still needs Bubble→Farm→Speedway→Frog→Business.

### Next action (required)
- **Next card ID:** `rr-czg9` — Pure Bubble → Upper Norfair Farm (`0xACB3`→`0xAF72`)
- **One change:** one-hop pure from `post_single_to_bubble_pure`
- **Source state:** `scratch/post_single_to_bubble_pure.state` ~(472,395)

### Non-claims
- Did not STATUS-promote continuous `--to ice`
- Did not claim dual continuous green Ice
- Did not finish Wave→Business full stack
- Did not implement Bubble→Farm pure yet
