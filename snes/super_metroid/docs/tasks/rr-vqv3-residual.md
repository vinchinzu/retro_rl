## Residual — rr-vqv3 Wave→Business pure return stack

### Result
PARTIAL — stack beads split from human tape Phase B; first **four** hops
dual pure GREEN (Wave→Double, Double→Single, Single→Bubble, Bubble→Farm).
Remaining 3 hops open. Not continuous. Do **not** STATUS-promote continuous Ice.

### One-hop beads (room IDs) — discovered under this stack

| Order | Bead | Hop | Rooms | Pure dual | Controller |
|------:|------|-----|-------|----------:|------------|
| 1 | `rr-pd0i` ✓ | Wave → Double | `0xADDE` → `0xADAD` | **560f** ×2 | `play_wave_to_double_chamber` |
| 2 | `rr-qpkd` ✓ | Double → Single return | `0xADAD` → `0xAD5E` | **1101f** ×2 | `play_double_to_single_chamber` |
| 3 | `rr-u0y8` ✓ | Single → Bubble return | `0xAD5E` → `0xACB3` | **817f** ×2 | `play_single_to_bubble` |
| 4 | `rr-czg9` ✓ | Bubble → Farm | `0xACB3` → `0xAF72` | **1566f** ×2 | `play_bubble_to_farm` |
| 5 | `rr-z13h` | Farm → Speedway | `0xAF72` → `0xB106` | open | TBD (needs Speed) |
| 6 | `rr-05dp` | Speedway → Frog Save | `0xB106` → `0xB167` | open | TBD (needs Speed) |
| 7 | `rr-vsjy` | Frog Save → Business | `0xB167` → `0xA7DE` | open | replace scaffold |

Tape order (rr-dbu.12 Phase B return): Wave → Double → Single → Bubble →
Farm → Speedway → Frog Save → Business. Then existing Ice pure from Business.

Package: return hops under `routes/kpdr/wave/` (`wave_to_double`,
`double_to_single`, `single_to_bubble`, `bubble_to_farm`, + geometry
`WAVE_*` / `DTS_*` / `STB_*` / `BTF_*`). One module per hop until multi-room
reverse solidifies.

### Files changed (stack so far)
- `routes/kpdr/wave/wave_to_double.py` — pure Wave leave → Double
- `routes/kpdr/wave/double_to_single.py` — pure Double → Single return
- `routes/kpdr/wave/single_to_bubble.py` — pure Single → Bubble return
- `routes/kpdr/wave/bubble_to_farm.py` — pure Bubble → Farm return
- `routes/kpdr/wave/geometry.py` — `WAVE_*` + `DTS_*` + `STB_*` + `BTF_*`
- `routes/kpdr/wave/__init__.py`, `k4_wave.py`, `k4_norfair.py`, `registry.py`
- `scripts/probe/kpdr.py` — pure CLI segments
- `source_states.py` — `dev_wave_collected`, post Wave/Double/Single/Bubble handoffs
- `tests/test_k4_wave_return_scaffold.py` — unit registry / predicates

### Verify paste
```bash
uv run pytest snes/super_metroid/tests/test_k4_wave_return_scaffold.py -q
# → 6 passed

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
```

### Acceptance
- [x] One-hop beads with room IDs from human tape Phase B only
- [x] First hop dual pure green Wave → Double
- [x] Second hop dual pure green Double → Single
- [x] Third hop dual pure green Single → Bubble
- [x] Fourth hop dual pure green Bubble → Farm
- [ ] Full stack dual pure green to Business
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
4. Farm/Speedway reverse needs **Speed** (Boost Blocks) — product loadout OK
   after Wave continuous parent.
5. `play_frog_save_to_business` remains scaffold until `rr-vsjy`.

### Next action (required)
- **Next card ID:** `rr-z13h` — Pure Farm → Frog Speedway
- **One change:** one-hop pure from `post_bubble_to_farm_pure` (Speed loadout)
- **Source state:** `scratch/post_bubble_to_farm_pure.state` ~(472,139)

### Non-claims
- Did not STATUS-promote continuous `--to ice`
- Did not claim dual continuous green Ice
- Did not finish Wave→Business full stack
- Did not compose ice-prefix continuous hops yet

### Probe pins
- rr-pd0i dual: room=0xADAD pose=12 x=984 y=139 frames=560 exact
- rr-qpkd dual: room=0xAD5E pose=82 x=216 y=630 frames=1101 exact
- rr-u0y8 dual: room=0xACB3 pose=12 x=472 y=395 frames=817 exact
- rr-czg9 dual: room=0xAF72 pose=10 x=472 y=139 frames=1566 exact
