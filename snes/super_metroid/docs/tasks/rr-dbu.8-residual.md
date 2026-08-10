## Residual — rr-dbu.8 K5 Alpha PB pure stack

### Result
PARTIAL — two pure one-hop dual GREENs on Ice return (Ice→Snake, Snake→Tutorial).
Full K5 stack to Alpha PB PLM still open. Not continuous. No STATUS change.

### One-hop map (tape Phase B return + Phase C)

| Order | Hop | Rooms | Pure dual | Controller / note |
|------:|-----|-------|----------:|-------------------|
| 0 | Ice → Snake | `0xA890` → `0xA8B9` | **538f** ×2 ✅ | `play_ice_to_snake` |
| 1 | Snake → Tutorial | `0xA8B9` → `0xA865` | **2386f** ×2 ✅ | `play_ice_snake_to_tutorial` |
| 2 | Tutorial → Gate | `0xA865` → `0xA815` | ⬜ | next |
| 3 | Gate → Business | `0xA815` → `0xA7DE` | ⬜ | |
| 4 | Business → Warehouse | `0xA7DE` → `0xA6A1` | reuse? | `play_business_to_warehouse` exists |
| 5–10 | tunnels reverse | Wh→East→Glass→West→Below→Bat | ⬜ | reverse of red_stack |
| 11 | Bat → Red Tower | `0xA3DD` → `0xA253` | ⬜ | climb |
| 12 | Red → Hellway | `0xA253` → `0xA2F7` | ⬜ | long Red Tower (~7k human) |
| 13 | Hellway → Caterpillar | `0xA2F7` → `0xA322` | ⬜ | |
| 14 | Caterpillar → Alpha PB PLM | `0xA322` → `0xA3AE` | ⬜ | first PB capacity |

Tape: `tasks/speed_to_wave_ice_moat_human.json` (rr-dbu.12). Packages:
`routes/kpdr/ice/` (return) + `routes/kpdr/k5/` (outbound map).

### Files changed (this hop: rr-bf29)
- `routes/kpdr/ice/snake_to_tutorial.py` — Snake→Tutorial return pure hop
- `routes/kpdr/ice/geometry.py` — Tutorial door / drop constants
- registry / probe / source_states / k5 hop map / ice scaffold tests

### Verify paste
```bash
uv run pytest snes/super_metroid/tests/test_k4_ice_scaffold.py -q

uv run python snes/super_metroid/scripts/probe/kpdr.py pure ice-snake-to-tutorial \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_to_snake_pure.state \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_snake_to_tutorial_pure.state
# → GREEN room=0xA865 xy=(39,127) frames=2386 (×2 exact dual)
```

### Acceptance
- [x] First tape-backed pure one-hop dual green (Ice→Snake return)
- [x] Second hop dual green (Snake→Tutorial return)
- [x] Package layout: return in `ice/`; K5 map under `k5/`
- [ ] Full pure stack through Alpha PB PLM
- [ ] Continuous tip / STATUS (planner only after dual continuous)

### Residual risks
1. Snake climb multi-attempt (L3 pin-sensitive) — OK dual but not single-pass.
2. Red Tower human stretch ~7k frames — prefer clean climb, not thrash RLE.
3. Tunnel reverses may reuse geometry of outbound red_stack but need natural-entry pure pins.
4. `business_to_warehouse` exists but needs re-verify from post-Ice Business handoff.

### Next action (required)
- **Next card ID:** `rr-81ek` Pure Ice Tutorial → Gate return (K5 hop 2)
- **One change:** pure controller Tutorial ~(39,127) → Ice Gate `0xA815`
- **Source state:** `scratch/post_ice_snake_to_tutorial_pure.state`

### Non-claims
- Did not STATUS-promote continuous past Ice
- Did not claim Alpha PB pure/continuous green
- Did not invent Moat approach hops (`rr-dbu.9`)

### Probe pin
- Dual GREEN hop0: room=0xA8B9 pose=10 x=472 y=395 frames=538 ×2
- Dual GREEN hop1: room=0xA865 pose=81 x=39 y=127 frames=2386 ×2
