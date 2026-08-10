## Residual — rr-czg9 Pure Bubble → Upper Norfair Farm

### Result
**GREEN dual pure** — Bubble Mountain `0xACB3` → Upper Norfair Farm `0xAF72`.
Exact dual **1566f ×2**. Not continuous. Part of Wave→Business stack
(`rr-vqv3`); stack remains open.

### Files changed
- `routes/kpdr/wave/bubble_to_farm.py` — pure controller (mid climb → upper
  drop morph → bottom tunnels → bottom-most left blue door)
- `routes/kpdr/wave/geometry.py` — `BTF_*` bands
- `routes/kpdr/wave/__init__.py`, `k4_wave.py`, `k4_norfair.py`, `registry.py`,
  `routes/kpdr/__init__.py`
- `scripts/probe/kpdr.py` — `bubble-to-farm` pure CLI
- `source_states.py` — `post_bubble_to_farm_pure` fingerprint
- `tests/test_k4_wave_return_scaffold.py` — registry / geometry unit

### Verify paste
```bash
uv run pytest snes/super_metroid/tests/test_k4_wave_return_scaffold.py -q
# → 6 passed

uv run python snes/super_metroid/scripts/probe/kpdr.py pure bubble-to-farm \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_single_to_bubble_pure.state \
  --expect-room 0xACB3 \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_to_farm_pure.state
# → GREEN room=0xAF72 xy=(472,139) frames=1566 (×2 exact dual)
```

### Acceptance
- [x] Pure hop `0xACB3` → `0xAF72` dual green
- [x] Export post state for Farm→Speedway predecessor
- [x] Source is `post_single_to_bubble_pure` right mid pin ~(472,395)
- [ ] Full Wave→Business stack (parent `rr-vqv3` remains open)
- [x] No STATUS promote continuous Ice

### Probe pin (dual exact)
room=0xAF72 pose=10 x=472 y=139 door_transition=0 frames=1566 last_pin=dual exact
door_def_ptr=0x956A (Farm right door from Bubble bottom-most left leave)

### Residual risks
1. Bottom morph shelf ~x251 y745 needs a full RIGHT run to ~x360 before LEFT
   tunnels; early LEFT stalls against tube geometry.
2. Human tape thrash (idle/X at shelf) was shortened but RIGHT length kept —
   do not collapse the 130f RIGHT into a short reactive without re-proving.
3. Parent stack `rr-vqv3` still needs Farm→Speedway→Frog→Business; Farm→
   Speedway **needs Speed** (Boost Blocks).

### Next action (required)
- **Next card ID:** `rr-z13h` — Pure Farm → Frog Speedway (`0xAF72`→`0xB106`)
- **One change:** one-hop pure from `post_bubble_to_farm_pure` (Speed loadout)
- **Source state:** `scratch/post_bubble_to_farm_pure.state` ~(472,139)

### Non-claims
- Did not STATUS-promote continuous `--to ice`
- Did not claim dual continuous green Ice
- Did not finish Wave→Business full stack
- Did not implement Farm→Speedway pure yet
