## Residual — rr-dbu.8 K5 Alpha PB pure stack

### Result
PARTIAL — first pure one-hop dual GREEN (Ice → Snake return). Full K5 stack
to Alpha PB PLM still open. Not continuous. No STATUS change.

### One-hop map (tape Phase B return + Phase C)

| Order | Hop | Rooms | Pure dual | Controller / note |
|------:|-----|-------|----------:|-------------------|
| 0 | Ice → Snake | `0xA890` → `0xA8B9` | **538f** ×2 ✅ | `play_ice_to_snake` |
| 1 | Snake → Tutorial | `0xA8B9` → `0xA865` | ⬜ | next; mid-right ~(472,395) → top door |
| 2 | Tutorial → Gate | `0xA865` → `0xA815` | ⬜ | |
| 3 | Gate → Business | `0xA815` → `0xA7DE` | ⬜ | |
| 4 | Business → Warehouse | `0xA7DE` → `0xA6A1` | reuse? | `play_business_to_warehouse` exists |
| 5–10 | tunnels reverse | Wh→East→Glass→West→Below→Bat | ⬜ | reverse of red_stack |
| 11 | Bat → Red Tower | `0xA3DD` → `0xA253` | ⬜ | climb |
| 12 | Red → Hellway | `0xA253` → `0xA2F7` | ⬜ | long Red Tower (~7k human) |
| 13 | Hellway → Caterpillar | `0xA2F7` → `0xA322` | ⬜ | |
| 14 | Caterpillar → Alpha PB PLM | `0xA322` → `0xA3AE` | ⬜ | first PB capacity |

Tape: `tasks/speed_to_wave_ice_moat_human.json` (rr-dbu.12). Packages:
`routes/kpdr/ice/` (return) + `routes/kpdr/k5/` (outbound map).

### Files changed
- `routes/kpdr/ice/ice_to_snake.py` — Ice→Snake return pure hop
- `routes/kpdr/ice/geometry.py` — `ICE_LEAVE_*` leave constants
- `routes/kpdr/ice/__init__.py` — export return hop + doc
- `routes/kpdr/registry.py` / `scripts/probe/kpdr.py` — `ice_to_snake` segment
- `routes/kpdr/k5/__init__.py` — K5 hop map (no controllers yet)
- `tests/test_k4_ice_scaffold.py` — registry + leave constants
- `source_states.py` / `docs/SOURCE_STATES.md` — `post_ice_to_snake_pure`

### Verify paste
```bash
uv run pytest snes/super_metroid/tests/test_k4_ice_scaffold.py -q
# → 8 passed

uv run python snes/super_metroid/scripts/probe/kpdr.py pure ice-to-snake \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_snake_to_ice_pure.state \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_to_snake_pure.state
# → GREEN room=0xA8B9 xy=(472,395) frames=538 (×2 exact dual)
```

### Acceptance
- [x] First tape-backed pure one-hop dual green (Ice→Snake return)
- [x] Package layout: return in `ice/`; K5 map under `k5/`
- [x] No invented hops beyond tape Phase B/C table
- [ ] Full pure stack through Alpha PB PLM
- [ ] Continuous tip / STATUS (planner only after dual continuous)

### Residual risks
1. Snake→Tutorial is a long mid-column climb (~1.2k human frames) — hard-room candidate.
2. Red Tower human stretch ~7k frames (freeze thrash) — prefer clean climb, not thrash RLE.
3. Tunnel reverses may reuse geometry of outbound red_stack but need natural-entry pure pins.
4. `business_to_warehouse` exists but needs re-verify from post-Ice Business handoff (not pre-Ice).

### Next action (required)
- **Next card ID:** `rr-bf29` Pure Ice Snake → Tutorial return (K5 hop 1)
- **One change:** pure controller Snake mid-right ~(472,395) → Tutorial `0xA865`
- **Source state:** `scratch/post_ice_to_snake_pure.state`

### Non-claims
- Did not STATUS-promote continuous past Ice
- Did not claim Alpha PB pure/continuous green
- Did not invent Moat approach hops (`rr-dbu.9`)

### Probe pin
- Dual GREEN: room=0xA8B9 pose=10 x=472 y=395 frames=538 ×2
  source=`post_ice_snake_to_ice_pure` beams/items via handoff `0x3105`
