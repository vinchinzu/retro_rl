## Residual — rr-vsjy Pure Frog Save → Business

### Result
**GREEN dual pure** — Frog Save `0xB167` → Business `0xA7DE`.
Exact dual **347f ×2**. Not continuous. Final hop of Wave→Business stack
(`rr-vqv3`); stack complete pure dual green.

### Files changed
- `routes/kpdr/wave/frog_to_business.py` — pure controller (LEFT tube clear → left blue door)
- `routes/kpdr/wave/geometry.py` — `FTB_*` bands
- `routes/kpdr/wave/__init__.py`, `k4_wave.py`, `k4_norfair.py`, `k4_business_frog.py` (scaffold → re-export)
- `scripts/probe/kpdr.py` — `frog-save-to-business` pure CLI
- `source_states.py` — `post_frog_save_to_business_pure` fingerprint (Business floor)
- `tests/test_k4_wave_return_scaffold.py` — registry / geometry unit

### Verify paste
```bash
uv run pytest snes/super_metroid/tests/test_k4_wave_return_scaffold.py -q
# → 10 passed

uv run python snes/super_metroid/scripts/probe/kpdr.py pure frog-save-to-business \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_speedway_to_frog_save_pure.state \
  --expect-room 0xB167 \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_frog_save_to_business_pure.state
# → GREEN room=0xA7DE xy=(216,1419) frames=347 (×2 exact dual)
```

### Acceptance
- [x] Pure hop `0xB167` → `0xA7DE` dual green
- [x] Export post state for Ice/compose predecessor
- [x] Source is `post_speedway_to_frog_save_pure` right pin ~(216,122)
- [x] Scaffold `play_frog_save_to_business` replaced (wave package pure)
- [x] Full Wave→Business stack dual pure green (parent `rr-vqv3`)
- [x] No STATUS promote continuous Ice

### Probe pin (dual exact)
room=0xA7DE pose=12 x=216 y=1419 door_transition=0 frames=347 last_pin=dual exact
door_def_ptr=0x9816 (Business floor entry from Frog Save left leave)
collected_items=0x3105 (includes Speed)

### Residual risks
1. Settle is Business **floor** ~(216,1419) — Frog door is the floor-right blue
   door used by continuous Business→Frog. Ice Super green is **mid-shaft**
   (~y900–940 left). Compose/Ice continuous must climb or use an elev-style
   pin; do not treat floor pin as Ice Super source without a climb hop.
2. Save tube still needs Hi-Jump pulses from right pin; flat B+LEFT can stall.
3. Graph edge `frog_save_to_business` remains `unverified` in SPEED_GRAPH
   (controller_dev promote optional; pure dual is product evidence here).

### Next action (required)
- **Next card:** compose Wave→Business return into continuous Ice prefix
  (after stack) / intermediate tip — **still no continuous Ice STATUS**
  without dual continuous green.
- Parent stack `rr-vqv3` closable on pure dual complete.

### Non-claims
- Did not STATUS-promote continuous `--to ice`
- Did not claim dual continuous green Ice
- Did not compose ice-prefix continuous hops yet
- Did not climb Business floor → Super lip
