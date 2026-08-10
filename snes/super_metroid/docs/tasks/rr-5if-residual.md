## Residual — rr-5if Pure Snake → Ice PLM

### Result
GREEN — dual pure exact match **1756f** ×2 Ice PLM collect

### Files changed
- `routes/kpdr/ice/geometry.py` — tunnel y band (365–395), false ledge y409, mid shelf
- `routes/kpdr/ice/snake_to_ice.py` — 2WJ climb + morph-drop + mid-shelf human RLE recovery + PLM
- `tests/test_k4_ice_scaffold.py` — unit bands + registry

### Verify paste
```bash
uv run pytest snes/super_metroid/tests/test_k4_ice_scaffold.py -q
# → 6 passed

uv run python snes/super_metroid/scripts/probe/kpdr.py pure ice-snake-to-ice \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_acid_to_snake_pure.state
# → GREEN room=0xA890 xy=(187,120) frames=1756 beams=0x1007 (×2 exact)
```

### Acceptance
- [x] 2WJ / platform-hop bands from pure handoff (not freeze ladder)
- [x] Package under `routes/kpdr/ice/` (no Wave megafile)
- [x] Dual pure GREEN Snake → Ice PLM (**1756f** ×2, beams `0x1007`)
- [x] Parent stack gate `rr-dbu.11` CLOSED (outbound pure documented)
- [ ] Continuous tip (compose only: `rr-dbu.7`; no STATUS without dual continuous green)

### Residual risks / findings
1. Climb GREEN: floor ~(216,651) → top y139 platform hops.
2. Tunnel floor is **y=377 only**; false ledge **y~409** is a morph trap (must not treat as tunnel).
3. Primary: morph-drop from right platform y267; recovery: mid-shelf ~(197,507) human RLE f15354–15470.
4. End states: `scratch/post_ice_snake_to_ice_pure.state` (+ `_dual`).

### Next action (required)
- **Next card ID:** `rr-dbu.7` (compose continuous `--to ice`)
- **One change:** tip wiring / graph / catalog only — no new Snake→Ice knobs
- **Source state:** `scratch/post_ice_snake_to_ice_pure.state` (Ice PLM collected)

### Non-claims
- Did not STATUS-promote continuous `--to ice`
- Did not forge progression RAM / freeze ladder

### Probe pin
- Dual GREEN: room=0xA890 pose=81 x=187 y=120 beams=0x1007 frames=1756 ×2
