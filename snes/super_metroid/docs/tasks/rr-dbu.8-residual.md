## Residual — rr-dbu.8 K5 Alpha PB pure stack

### Result
PARTIAL — eight pure one-hop dual GREENs on Ice return + Business→Warehouse +
Warehouse→East + East→Glass + Glass→West (Ice→Snake, Snake→Tutorial,
Tutorial→Gate, Gate→Business, Business→Warehouse, Warehouse→East, East→Glass,
Glass→West). Full K5 stack to Alpha PB PLM still open. Not continuous. No
STATUS change.

### One-hop map (tape Phase B return + Phase C)
| Order | Hop | Rooms | Pure dual | Controller / note |
|------:|-----|-------|----------:|-------------------|
| 0 | Ice → Snake | `0xA890` → `0xA8B9` | **538f** ×2 ✅ | `play_ice_to_snake` |
| 1 | Snake → Tutorial | `0xA8B9` → `0xA865` | **2386f** ×2 ✅ | `play_ice_snake_to_tutorial` |
| 2 | Tutorial → Gate | `0xA865` → `0xA815` | **969f** ×2 ✅ | `play_ice_tutorial_to_gate` |
| 3 | Gate → Business | `0xA815` → `0xA7DE` | **879f** ×2 ✅ | `play_ice_gate_to_business` |
| 4 | Business → Warehouse | `0xA7DE` → `0xA6A1` | **10255f** ×2 ✅ | `play_business_to_warehouse` (Super fall+ladder) |
| 5 | Warehouse → East | `0xA6A1` → `0xCF80` | **285f** ×2 ✅ | `play_warehouse_to_east` (reverse east_to_warehouse) |
| 6 | East → Glass | `0xCF80` → `0xCEFB` | **253f** ×2 ✅ | `play_east_to_glass` (reverse glass_to_east) |
| 7 | Glass → West | `0xCEFB` → `0xCF54` | **211f** ×2 ✅ | `play_glass_to_west` (reverse west_to_glass) |
| 8–10 | tunnels reverse | West→Below→Bat | ⬜ | reverse of red_stack |
| 11 | Bat → Red Tower | `0xA3DD` → `0xA253` | ⬜ | climb |
| 12 | Red → Hellway | `0xA253` → `0xA2F7` | ⬜ | long Red Tower (~7k human) |
| 13 | Hellway → Caterpillar | `0xA2F7` → `0xA322` | ⬜ | |
| 14 | Caterpillar → Alpha PB PLM | `0xA322` → `0xA3AE` | ⬜ | first PB capacity |

Tape: `tasks/speed_to_wave_ice_moat_human.json` (rr-dbu.12). Packages:
`routes/kpdr/ice/` (return) + `routes/kpdr/k5/` (outbound reverse).

### Files changed (this hop: rr-85c4)
- `routes/kpdr/k5/glass_to_west.py` — LEFT run_shoot reverse of west_to_glass
- source_states / SOURCE_STATES / k5 hop map / residual

### Verify paste
```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py pure glass-to-west \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_east_to_glass_pure.state \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_glass_to_west_pure.state
# → GREEN room=0xCF54 xy=(216,139) frames=211 (×2 exact dual)
```

### Acceptance
- [x] First tape-backed pure one-hop dual green (Ice→Snake return)
- [x] Second hop dual green (Snake→Tutorial return)
- [x] Third hop dual green (Tutorial→Gate return)
- [x] Fourth hop dual green (Gate→Business return)
- [x] Fifth hop dual green (Business→Warehouse return)
- [x] Sixth hop dual green (Warehouse→East return)
- [x] Seventh hop dual green (East→Glass return)
- [x] Eighth hop dual green (Glass→West return)
- [x] Package layout: return in `ice/`; K5 map under `k5/`
- [ ] Full pure stack through Alpha PB PLM
- [ ] Continuous tip / STATUS (planner only after dual continuous)

### Residual risks
1. Snake climb multi-attempt (L3 pin-sensitive) — OK dual but not single-pass.
2. Business→Warehouse **10255f** — multi-attempt after Super fall; optimizable.
3. Red Tower human stretch ~7k frames — prefer clean climb, not thrash RLE.
4. West pin mid-floor p10 — next reverse hop LEFT to Below Spazer.
5. Tunnel reverses may reuse geometry of outbound red_stack but need natural-entry pure pins.

### Next action (required)
- **Next card:** Pure West Tunnel → Below Spazer return (K5 hop 8)
- **One change:** pure West `0xCF54` → Below `0xA408` reverse of below_spazer_to_west
- **Source state:** `scratch/post_ice_glass_to_west_pure.state`

### Non-claims
- Did not STATUS-promote continuous past Ice
- Did not claim Alpha PB pure/continuous green
- Did not invent Moat approach hops (`rr-dbu.9`)

### Probe pin
- Dual GREEN hop0: room=0xA8B9 pose=10 x=472 y=395 frames=538 ×2
- Dual GREEN hop1: room=0xA865 pose=81 x=39 y=127 frames=2386 ×2
- Dual GREEN hop2: room=0xA815 pose=81 x=807 y=131 frames=969 ×2
- Dual GREEN hop3: room=0xA7DE pose=25 x=41 y=907 frames=879 ×2
- Dual GREEN hop4: room=0xA6A1 pose=138 x=37 y=139 frames=10255 ×2
- Dual GREEN hop5: room=0xCF80 pose=26 x=216 y=364 frames=285 ×2
- Dual GREEN hop6: room=0xCEFB pose=12 x=216 y=395 frames=253 ×2
- Dual GREEN hop7: room=0xCF54 pose=10 x=216 y=139 frames=211 ×2
