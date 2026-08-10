## Residual — rr-vqv3 Wave→Business pure return stack

### Result
PARTIAL — stack beads split from human tape Phase B; first hop
**Wave → Double dual pure GREEN**. Remaining 6 hops open. Not continuous.
Do **not** STATUS-promote continuous Ice.

### One-hop beads (room IDs) — discovered under this stack

| Order | Bead | Hop | Rooms | Pure dual | Controller |
|------:|------|-----|-------|----------:|------------|
| 1 | `rr-pd0i` ✓ | Wave → Double | `0xADDE` → `0xADAD` | **560f** ×2 | `play_wave_to_double_chamber` |
| 2 | `rr-wu5r` | Double → Single return | `0xADAD` → `0xAD5E` | open | TBD |
| 3 | `rr-vcu5` | Single → Bubble return | `0xAD5E` → `0xACB3` | open | TBD |
| 4 | `rr-w6z9` | Bubble → Farm | `0xACB3` → `0xAF72` | open | TBD (rev farm_to_bubble) |
| 5 | `rr-gz7y` | Farm → Speedway | `0xAF72` → `0xB106` | open | TBD (needs Speed) |
| 6 | `rr-bnc8` | Speedway → Frog Save | `0xB106` → `0xB167` | open | TBD (needs Speed) |
| 7 | `rr-i6p1` | Frog Save → Business | `0xB167` → `0xA7DE` | open | replace scaffold |

Tape order (rr-dbu.12 Phase B return): Wave → Double → Single → Bubble →
Farm → Speedway → Frog Save → Business. Then existing Ice pure from Business.

Package: first hop under `routes/kpdr/wave/wave_to_double.py` (+ geometry WR_*).
Further hops stay one-module until multi-room reverse solidifies.

### Files changed
- `routes/kpdr/wave/wave_to_double.py` — pure Wave leave → Double
- `routes/kpdr/wave/geometry.py` — `WAVE_DOOR_X` / leave frames / settle
- `routes/kpdr/wave/__init__.py`, `k4_wave.py`, `k4_norfair.py`, `registry.py`
- `scripts/probe/kpdr.py` — `wave-to-double-chamber` pure CLI
- `source_states.py` — `dev_wave_collected` + post handoff; stale note on
  `post_double_chamber_to_wave_pure`
- `tests/test_k4_wave_return_scaffold.py` — unit registry / predicates

### Verify paste
```bash
uv run pytest snes/super_metroid/tests/test_k4_wave_return_scaffold.py -q
# → 3 passed

uv run python snes/super_metroid/scripts/probe/kpdr.py pure wave-to-double-chamber \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/dev_wave_collected.state \
  --expect-room 0xADDE \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_wave_to_double_chamber_pure.state
# → GREEN room=0xADAD xy=(984,139) frames=560 (×2 exact dual)
```

### Acceptance
- [x] One-hop beads with room IDs from human tape Phase B only
- [x] First hop dual pure green Wave → Double
- [ ] Full stack dual pure green to Business
- [ ] Compose ice-prefix hops / intermediate tip (after stack)
- [x] No STATUS promote continuous Ice

### Residual risks
1. **`post_double_chamber_to_wave_pure.state` is STALE** — loads Double
   `0xADAD` ~(923,311) not Wave. Use `dev_wave_collected` (Wave
   `0xADDE` ~(171,120) beams `0x0001`) until re-captured from continuous
   Wave tip (product beams `0x1005`).
2. Double→Single return is the hard geometry hop (Super column drop + floor).
3. Farm/Speedway reverse needs **Speed** (Boost Blocks) — product loadout OK
   after Wave continuous parent.
4. `play_frog_save_to_business` remains scaffold until `rr-i6p1`.

### Next action (required)
- **Next card ID:** `rr-wu5r` — Pure Double → Single Chamber return
- **One change:** one-hop pure from `post_wave_to_double_chamber_pure` tape geometry
- **Source state:** `scratch/post_wave_to_double_chamber_pure.state` ~(984,139)

### Non-claims
- Did not STATUS-promote continuous `--to ice`
- Did not claim dual continuous green Ice
- Did not finish Wave→Business full stack
- Did not compose ice-prefix continuous hops yet

### Probe pin (rr-pd0i dual)
room=0xADAD pose=12 x=984 y=139 door_transition=0 frames=560 last_pin=dual exact
