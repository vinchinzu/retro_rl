## Residual — rr-dbu.11 Post-Wave Ice pure stack

### Result
GREEN — outbound pure serial stack dual-GREEN through Ice PLM collect.
Not continuous. Parent of hop beads; closes product pure gate for Ice PLM.

### One-hop beads (room IDs) — discovered under this stack

| Order | Bead | Hop | Rooms | Pure dual GREEN | Controller |
|------:|------|-----|-------|----------------:|------------|
| 1 | `rr-fg3` ✓ | Business → Ice Gate | `0xA7DE` → `0xA815` | **894f** ×2 | `play_business_to_ice_gate` |
| 2 | `rr-9t4` ✓ | Ice Gate → Acid | `0xA815` → `0xA75D` | **370f** ×2 | `play_ice_gate_to_acid` |
| 3 | `rr-5cf` ✓ | Acid → Ice Snake | `0xA75D` → `0xA8B9` | **652f** ×2 | `play_ice_acid_to_snake` |
| 4 | `rr-5if` ✓ | Ice Snake → Ice PLM | `0xA8B9` → `0xA890` | **1756f** ×2 | `play_ice_snake_to_ice` |

Tape order (rr-dbu.12): Business → Gate → **Acid** (not Tutorial-first) → Snake (prefer 2WJ) → Ice PLM.
Package: `routes/kpdr/ice/` (geometry + four hop modules). Premature invent cards
`rr-dbu.2`–`.6` folded earlier; real hops are the four above.

### Files (stack package; no new code this close)
- `routes/kpdr/ice/` — package from day 1
- `docs/SOURCE_STATES.md` — pure handoff pins
- hop residuals: `docs/tasks/rr-5if-residual.md` (PLM pin)

### Verify paste (2026-08-09 re-verify, this session)
```bash
uv run pytest snes/super_metroid/tests/test_k4_ice_scaffold.py -q
# → 7 passed

uv run python snes/super_metroid/scripts/probe/kpdr.py pure business-to-ice-gate \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_business_continuous.state
# → GREEN room=0xA815 xy=(1752,651) frames=894

uv run python snes/super_metroid/scripts/probe/kpdr.py pure ice-gate-to-acid \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_business_to_ice_gate_wave_speed_pure.state
# → GREEN room=0xA75D xy=(470,139) frames=370

uv run python snes/super_metroid/scripts/probe/kpdr.py pure ice-acid-to-snake \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_gate_to_acid_pure.state
# → GREEN room=0xA8B9 xy=(216,651) frames=652

uv run python snes/super_metroid/scripts/probe/kpdr.py pure ice-snake-to-ice \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_acid_to_snake_pure.state
# → GREEN room=0xA890 xy=(187,120) frames=1756 (×2 exact this session)
```

### Acceptance
- [x] One-hop beads with room IDs (rr-fg3 / rr-9t4 / rr-5cf / rr-5if)
- [x] Pure dual green Ice PLM (`rr-5if` 1756f ×2, beams `0x1007`)
- [x] No intentional continuous tip / STATUS promote in hop beads
- [x] Stack re-verify serial pure GREENS (this residual)

### Residual risks
1. Hops are **serial pure**, not continuous power-on → Ice. Compose is `rr-dbu.7`.
2. Gate→Acid needs **Speed** loadout (Boost Blocks); Wave+Speed pin
   `post_business_to_ice_gate_wave_speed_pure`.
3. Return path Ice→Snake→Tutorial→Gate→Business still optional (not product gate).
4. Do **not** STATUS-promote continuous `--to ice` without dual continuous green.

### Next action (required)
- **Next card ID:** `rr-dbu.7` (compose continuous `--to ice`) — planner-owned tip wiring
- **One change:** graph/catalog/`TipSpec` compose after pure greens; no new hop knobs
- **Source state:** pure chain handoffs under `scratch/post_ice_*`; continuous predecessor is Wave/`business` tip

### Non-claims
- Did not STATUS-promote continuous `--to ice`
- Did not claim dual continuous green Ice
- Did not implement return path pure hops
- Did not invent K5/Moat pure without tape gates
