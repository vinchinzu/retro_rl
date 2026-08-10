## Residual — rr-81ek Pure Ice Tutorial → Gate return (K5 hop 2)

### Result
GREEN — pure dual exact **969f** ×2. Tutorial left → Ice Gate. Not continuous.
No STATUS change. Parent **rr-dbu.8** stays open (Alpha PB stack incomplete).

### Files changed
- `routes/kpdr/ice/tutorial_to_gate.py` — hybrid RLE-to-mid + morph tunnel + gap/door
- `routes/kpdr/data/ice_tutorial_to_gate_rle.json` — cleaned human left→mid (thrash stripped)
- `routes/kpdr/ice/geometry.py` — Tutorial door / mid RLE / gate settle constants
- `routes/kpdr/ice/__init__.py` — export `play_ice_tutorial_to_gate`
- `routes/kpdr/registry.py` / `scripts/probe/kpdr.py` — `ice_tutorial_to_gate` segment
- `routes/kpdr/k5/__init__.py` — hop map mark GREEN
- `source_states.py` / `docs/SOURCE_STATES.md` — `post_ice_tutorial_to_gate_pure`
- `tests/test_k4_ice_scaffold.py` — registry + RLE load

### Verify paste
```bash
uv run pytest snes/super_metroid/tests/test_k4_ice_scaffold.py -q
# → passed

uv run python snes/super_metroid/scripts/probe/kpdr.py pure ice-tutorial-to-gate \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_snake_to_tutorial_pure.state \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_tutorial_to_gate_pure.state
# → GREEN room=0xA815 xy=(807,131) pose=81 frames=969 (×2 exact dual)
```

### Acceptance
- [x] Pure Tutorial left → Ice Gate dual green from `post_ice_snake_to_tutorial_pure`
- [x] Morph tunnel open-loop (double-DOWN midair); no boyon angled thrash RLE
- [x] Export `scratch/post_ice_tutorial_to_gate_pure.state` for next hop
- [ ] Full K5 stack / continuous / STATUS (parent rr-dbu.8)

### Residual risks
1. Gate settle pin ~(807,131) is mid-room after 280f settle (not top-left door lip ~(494,139)); Gate→Business must handle mid-top entry band.
2. Partial mid RLE is pin-sensitive on first gap hop; multi-attempt retry exists if mid miss.
3. Morph tunnel height is exact (pipe y≈120) — crouch/land pose 164 pin must stand first.

### Next action (required)
- **Next card:** Pure Ice Gate → Business return (K5 hop 3) — create if missing
- **One change:** pure controller Gate → Business `0xA7DE` from `post_ice_tutorial_to_gate_pure`
- **Source state:** `scratch/post_ice_tutorial_to_gate_pure.state`

### Non-claims
- Did not STATUS-promote continuous past Ice
- Did not claim Alpha PB pure/continuous green
- Did not invent Moat hops

### Probe pin
- Dual GREEN: room=0xA815 pose=81 x=807 y=131 frames=969 ×2
  source=`post_ice_snake_to_tutorial_pure` items `0x3105`
