## Residual — rr-z13h Pure Farm → Frog Speedway

### Result
**GREEN dual pure** — Upper Norfair Farm `0xAF72` → Frog Speedway `0xB106`.
Exact dual **329f ×2**. Not continuous. Part of Wave→Business stack
(`rr-vqv3`); stack remains open.

### Files changed
- `routes/kpdr/wave/farm_to_speedway.py` — pure controller (LEFT run/hop → left blue door)
- `routes/kpdr/wave/geometry.py` — `FTS_*` bands + `SPEED_BOOSTER_MASK` / `has_speed`
- `routes/kpdr/wave/__init__.py`, `k4_wave.py`, `k4_norfair.py`, `registry.py`,
  `routes/kpdr/__init__.py`
- `scripts/probe/kpdr.py` — `farm-to-speedway` pure CLI
- `source_states.py` — `post_farm_to_speedway_pure` fingerprint (right entry)
- `tests/test_k4_wave_return_scaffold.py` — registry / geometry / has_speed unit

### Verify paste
```bash
uv run pytest snes/super_metroid/tests/test_k4_wave_return_scaffold.py -q
# → 8 passed

uv run python snes/super_metroid/scripts/probe/kpdr.py pure farm-to-speedway \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_to_farm_pure.state \
  --expect-room 0xAF72 \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_farm_to_speedway_pure.state
# → GREEN room=0xB106 xy=(2008,139) frames=329 (×2 exact dual)
```

### Acceptance
- [x] Pure hop `0xAF72` → `0xB106` dual green
- [x] Export post state for Speedway→Frog predecessor
- [x] Source is `post_bubble_to_farm_pure` right-top pin ~(472,139)
- [x] Speed loadout present (`collected_items` bit `0x2000`)
- [ ] Full Wave→Business stack (parent `rr-vqv3` remains open)
- [x] No STATUS promote continuous Ice

### Probe pin (dual exact)
room=0xB106 pose=10 x=2008 y=139 door_transition=0 frames=329 last_pin=dual exact
door_def_ptr=0x970E (Speedway right entry from Farm left leave)
collected_items=0x3105 (includes Speed)

### Residual risks
1. Settle is Speedway **right** ~(2008,139) — next hop must run LEFT through
   mid-room Boost Blocks (needs Speed) to Frog Save on the left. Human tape
   transition frames can show x≈18 briefly; do not fingerprint left entry.
2. Open-loop RLE is short (~200f human leave); reactive door budget covers
   stalls. Mid-room hop window ~x320 is geometry from tape, not thrash.
3. Parent stack `rr-vqv3` still needs Speedway→Frog→Business.

### Next action (required)
- **Next card ID:** `rr-05dp` — Pure Speedway → Frog Save (`0xB106`→`0xB167`)
- **One change:** one-hop pure from `post_farm_to_speedway_pure` (Speed loadout)
- **Source state:** `scratch/post_farm_to_speedway_pure.state` ~(2008,139)

### Non-claims
- Did not STATUS-promote continuous `--to ice`
- Did not claim dual continuous green Ice
- Did not finish Wave→Business full stack
- Did not implement Speedway→Frog pure yet
