## Residual — rr-5if Pure Snake → Ice PLM

### Result
PARTIAL — 2WJ climb bands landed + controller scaffold; morph-tunnel entry RED

### Files changed
- `routes/kpdr/ice/geometry.py` — Snake climb/tunnel bands + `has_ice`
- `routes/kpdr/ice/snake_to_ice.py` — pure hop scaffold (climb + tunnel attempt + PLM)
- `routes/kpdr/ice/__init__.py`, `registry.py`, `scripts/probe/kpdr.py` — wire `ice_snake_to_ice`
- `tests/test_k4_ice_scaffold.py` — unit bands + registry

### Verify paste
```bash
uv run pytest snes/super_metroid/tests/test_k4_ice_scaffold.py -q
# pure (tunnel RED — right column, too low):
uv run python snes/super_metroid/scripts/probe/kpdr.py pure ice-snake-to-ice \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_acid_to_snake_pure.state
# → RED room=0xA8B9 pose=30 x=219 y=553 frames=1623 (past wall, missed tunnel y~377)
```

### Acceptance
- [x] 2WJ / platform-hop bands from pure handoff (not freeze ladder)
- [x] Package under `routes/kpdr/ice/` (no Wave megafile)
- [ ] Dual pure GREEN Snake → Ice PLM
- [ ] Continuous tip (blocked on pure stack)

### Residual risks / findings
1. **Climb GREEN path (live probe):** floor ~(216,651) → L1 y587 → L2 y523 → L3 y459 → L4 y395 → L5 y331 → L6 y267 → L7 y203 → top y139 via alternating platform hops (tape first-climb shape).
2. **Center wall x=171** is solid at mid height from the left shaft. Morph at x=171 y~409 does not enter the tunnel.
3. **Top cross** (y139) is the only clean way past the wall; shoot-down opens the right-column shelf (~y155 → y~270).
4. **Morph tunnel** is on the right column at y~377 x≥200 (human successful roll f15425). Entry from right column after top cross is the remaining knob — not freeze thrash, not left-wall morph.
5. Standing morph needs `ensure_morph` (double-tap DOWN); held DOWN only crouches.

### Next action (required)
- **Next card ID:** rr-5if (continue) | knob: right-column morph tunnel entry
- **One change:** From right platform ~y267 (post top cross + shoot-down), land the y~377 tunnel ledge and morph-roll to Ice door; then dual pure PLM collect
- **Source state:** `scratch/post_ice_acid_to_snake_pure.state`

### Non-claims
- Did not STATUS-promote / continuous `--to ice` / freeze ladder
- Climb bands proven in live probe; full hop not dual-GREEN yet

### Probe pin
- Climb isolation (live): room=0xA8B9 pose=9 x=120 y=139 (top after platform hops)
- Pure hop RED: room=0xA8B9 pose=30 x=219 y=553 door_transition=0 frames=1623
  (reached right column morph; tunnel band y~377 missed — fell low)
