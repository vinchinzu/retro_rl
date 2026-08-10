## Residual — rr-05dp Pure Speedway → Frog Save

### Result
**GREEN dual pure** — Frog Speedway `0xB106` → Frog Save `0xB167`.
Exact dual **621f ×2**. Not continuous. Part of Wave→Business stack
(`rr-vqv3`); stack remains open.

### Files changed
- `routes/kpdr/wave/speedway_to_frog.py` — pure controller (B+LEFT dash → left blue door)
- `routes/kpdr/wave/geometry.py` — `STF_*` bands (+ export FTS/STF/has_speed in `__all__`)
- `routes/kpdr/wave/__init__.py`, `k4_wave.py`, `k4_norfair.py`, `registry.py`,
  `routes/kpdr/__init__.py`
- `scripts/probe/kpdr.py` — `speedway-to-frog-save` pure CLI
- `source_states.py` — `post_speedway_to_frog_save_pure` fingerprint (right entry)
- `tests/test_k4_wave_return_scaffold.py` — registry / geometry unit

### Verify paste
```bash
uv run pytest snes/super_metroid/tests/test_k4_wave_return_scaffold.py -q
# → 9 passed

uv run python snes/super_metroid/scripts/probe/kpdr.py pure speedway-to-frog-save \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_farm_to_speedway_pure.state \
  --expect-room 0xB106 \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_speedway_to_frog_save_pure.state
# → GREEN room=0xB167 xy=(216,122) frames=621 (×2 exact dual)
```

### Acceptance
- [x] Pure hop `0xB106` → `0xB167` dual green
- [x] Export post state for Frog→Business predecessor
- [x] Source is `post_farm_to_speedway_pure` right pin ~(2008,139)
- [x] Speed loadout present (`collected_items` bit `0x2000`)
- [ ] Full Wave→Business stack (parent `rr-vqv3` remains open)
- [x] No STATUS promote continuous Ice

### Probe pin (dual exact)
room=0xB167 pose=82 x=216 y=122 door_transition=0 frames=621 last_pin=dual exact
door_def_ptr=0x97DA (Frog Save right entry from Speedway left leave)
collected_items=0x3105 (includes Speed)

### Residual risks
1. Settle is Frog Save **right** ~(216,122) pose 82 (spin/land) — next hop
   must navigate LEFT past the save tube to Business left door. Human ordinary
   was ~(216,139) p10; product pin is slightly higher with residual spin.
2. Mid-room Boost Blocks require Speed; controller hard-requires the bit and
   reports min_x stall if charge fails.
3. Parent stack `rr-vqv3` still needs Frog→Business (`rr-vsjy`).

### Next action (required)
- **Next card ID:** `rr-vsjy` — Pure Frog Save → Business (`0xB167`→`0xA7DE`)
- **One change:** replace scaffold with one-hop pure from
  `post_speedway_to_frog_save_pure`
- **Source state:** `scratch/post_speedway_to_frog_save_pure.state` ~(216,122)

### Non-claims
- Did not STATUS-promote continuous `--to ice`
- Did not claim dual continuous green Ice
- Did not finish Wave→Business full stack
- Did not implement Frog→Business pure yet
