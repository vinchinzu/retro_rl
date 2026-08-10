## Residual — rr-vqv3 Wave→Business pure return stack

### Result
**COMPLETE pure dual** — all **seven** hops dual pure GREEN
(Wave→Double→Single→Bubble→Farm→Speedway→Frog→Business). Not continuous.
Do **not** STATUS-promote continuous Ice until dual continuous green after
compose.

### One-hop beads (room IDs) — discovered under this stack

| Order | Bead | Hop | Rooms | Pure dual | Controller |
|------:|------|-----|-------|----------:|------------|
| 1 | `rr-pd0i` ✓ | Wave → Double | `0xADDE` → `0xADAD` | **560f** ×2 | `play_wave_to_double_chamber` |
| 2 | `rr-qpkd` ✓ | Double → Single return | `0xADAD` → `0xAD5E` | **1101f** ×2 | `play_double_to_single_chamber` |
| 3 | `rr-u0y8` ✓ | Single → Bubble return | `0xAD5E` → `0xACB3` | **817f** ×2 | `play_single_to_bubble` |
| 4 | `rr-czg9` ✓ | Bubble → Farm | `0xACB3` → `0xAF72` | **1566f** ×2 | `play_bubble_to_farm` |
| 5 | `rr-z13h` ✓ | Farm → Speedway | `0xAF72` → `0xB106` | **329f** ×2 | `play_farm_to_speedway` |
| 6 | `rr-05dp` ✓ | Speedway → Frog Save | `0xB106` → `0xB167` | **621f** ×2 | `play_speedway_to_frog_save` |
| 7 | `rr-vsjy` ✓ | Frog Save → Business | `0xB167` → `0xA7DE` | **347f** ×2 | `play_frog_save_to_business` |

Tape order (rr-dbu.12 Phase B return): Wave → Double → Single → Bubble →
Farm → Speedway → Frog Save → Business. Then existing Ice pure from Business.

Package: return hops under `routes/kpdr/wave/` (`wave_to_double`,
`double_to_single`, `single_to_bubble`, `bubble_to_farm`, `farm_to_speedway`,
`speedway_to_frog`, `frog_to_business`, + geometry `WAVE_*` / `DTS_*` /
`STB_*` / `BTF_*` / `FTS_*` / `STF_*` / `FTB_*`).

### Files changed (stack)
- `routes/kpdr/wave/wave_to_double.py` — pure Wave leave → Double
- `routes/kpdr/wave/double_to_single.py` — pure Double → Single return
- `routes/kpdr/wave/single_to_bubble.py` — pure Single → Bubble return
- `routes/kpdr/wave/bubble_to_farm.py` — pure Bubble → Farm return
- `routes/kpdr/wave/farm_to_speedway.py` — pure Farm → Speedway return
- `routes/kpdr/wave/speedway_to_frog.py` — pure Speedway → Frog Save return
- `routes/kpdr/wave/frog_to_business.py` — pure Frog Save → Business return
- `routes/kpdr/wave/geometry.py` — `WAVE_*` + `DTS_*` + `STB_*` + `BTF_*` + `FTS_*` + `STF_*` + `FTB_*`
- `routes/kpdr/wave/__init__.py`, `k4_wave.py`, `k4_norfair.py`, `registry.py`
- `routes/kpdr/k4_business_frog.py` — scaffold re-exports pure
- `scripts/probe/kpdr.py` — pure CLI segments
- `source_states.py` — handoffs through `post_frog_save_to_business_pure`
- `tests/test_k4_wave_return_scaffold.py` — unit registry / predicates

### Verify paste
```bash
uv run pytest snes/super_metroid/tests/test_k4_wave_return_scaffold.py -q
# → 10 passed

uv run python snes/super_metroid/scripts/probe/kpdr.py pure wave-to-double-chamber \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/dev_wave_collected.state \
  --expect-room 0xADDE \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_wave_to_double_chamber_pure.state
# → GREEN room=0xADAD xy=(984,139) frames=560 (×2 exact dual)

uv run python snes/super_metroid/scripts/probe/kpdr.py pure double-to-single-chamber \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_wave_to_double_chamber_pure.state \
  --expect-room 0xADAD \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_double_to_single_chamber_pure.state
# → GREEN room=0xAD5E xy=(216,630) frames=1101 (×2 exact dual)

uv run python snes/super_metroid/scripts/probe/kpdr.py pure single-to-bubble \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_double_to_single_chamber_pure.state \
  --expect-room 0xAD5E \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_single_to_bubble_pure.state
# → GREEN room=0xACB3 xy=(472,395) frames=817 (×2 exact dual)

uv run python snes/super_metroid/scripts/probe/kpdr.py pure bubble-to-farm \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_single_to_bubble_pure.state \
  --expect-room 0xACB3 \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_to_farm_pure.state
# → GREEN room=0xAF72 xy=(472,139) frames=1566 (×2 exact dual)

uv run python snes/super_metroid/scripts/probe/kpdr.py pure farm-to-speedway \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_to_farm_pure.state \
  --expect-room 0xAF72 \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_farm_to_speedway_pure.state
# → GREEN room=0xB106 xy=(2008,139) frames=329 (×2 exact dual)

uv run python snes/super_metroid/scripts/probe/kpdr.py pure speedway-to-frog-save \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_farm_to_speedway_pure.state \
  --expect-room 0xB106 \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_speedway_to_frog_save_pure.state
# → GREEN room=0xB167 xy=(216,122) frames=621 (×2 exact dual)

uv run python snes/super_metroid/scripts/probe/kpdr.py pure frog-save-to-business \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_speedway_to_frog_save_pure.state \
  --expect-room 0xB167 \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_frog_save_to_business_pure.state
# → GREEN room=0xA7DE xy=(216,1419) frames=347 (×2 exact dual)
```

### Acceptance
- [x] One-hop beads with room IDs from human tape Phase B only
- [x] First hop dual pure green Wave → Double
- [x] Second hop dual pure green Double → Single
- [x] Third hop dual pure green Single → Bubble
- [x] Fourth hop dual pure green Bubble → Farm
- [x] Fifth hop dual pure green Farm → Speedway
- [x] Sixth hop dual pure green Speedway → Frog Save
- [x] Seventh hop dual pure green Frog Save → Business
- [x] Full stack dual pure green to Business
- [ ] Compose ice-prefix hops / intermediate tip (after stack)
- [x] No STATUS promote continuous Ice

### Residual risks
1. **`post_double_chamber_to_wave_pure.state` is STALE** — loads Double
   `0xADAD` ~(923,311) not Wave. Use `dev_wave_collected` (Wave
   `0xADDE` ~(171,120) beams `0x0001`) until re-captured from continuous
   Wave tip (product beams `0x1005`).
2. Single→Bubble deep morph-slope trap ~x167 requires LEFT+A hop; floor climb
   launch x≤88. See `rr-u0y8-residual.md`.
3. Bubble→Farm bottom shelf needs full RIGHT to ~x360 before LEFT tunnels.
   See `rr-czg9-residual.md`.
4. Farm→Speedway settles Speedway **right** ~(2008,139); Speedway→Frog dual
   GREEN (rr-05dp 621f×2). See `rr-z13h-residual.md` / `rr-05dp-residual.md`.
5. Frog→Business dual GREEN (rr-vsjy 347f×2) settles Business **floor**
   ~(216,1419). Ice Super is mid-shaft — compose must climb / re-pin. See
   `rr-vsjy-residual.md`.

### Next action (required)
- **Compose** Wave tip → Business return chain into continuous Ice prefix
  (or intermediate tip). Existing Ice pure stack starts Business → Gate;
  continuous dual still blocked until compose + dual continuous green.
- **Do not** STATUS-promote continuous Ice without dual continuous green.

### Non-claims
- Did not STATUS-promote continuous `--to ice`
- Did not claim dual continuous green Ice
- Did not compose ice-prefix continuous hops yet
- Did not climb Business floor → Super lip as part of this stack

### Probe pins
- rr-pd0i dual: room=0xADAD pose=12 x=984 y=139 frames=560 exact
- rr-qpkd dual: room=0xAD5E pose=82 x=216 y=630 frames=1101 exact
- rr-u0y8 dual: room=0xACB3 pose=12 x=472 y=395 frames=817 exact
- rr-czg9 dual: room=0xAF72 pose=10 x=472 y=139 frames=1566 exact
- rr-z13h dual: room=0xB106 pose=10 x=2008 y=139 frames=329 exact
- rr-05dp dual: room=0xB167 pose=82 x=216 y=122 frames=621 exact
- rr-vsjy dual: room=0xA7DE pose=12 x=216 y=1419 frames=347 exact
