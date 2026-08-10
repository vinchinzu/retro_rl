## Residual — rr-3gh9 Pure Business Super → Warehouse return (K5 hop 4)

### Result
GREEN — pure dual exact **10255f** ×2. Super lip → Warehouse elev exit.
Not continuous. No STATUS change. Parent **rr-dbu.8** stays open (Alpha PB stack incomplete).

### Files changed
- `routes/kpdr/business_climb.py` — Super/midshaft floor-fall first; Charge multi-attempt ladder (classic 14→8 lead, cont-tuned retries); mid-right floor re-anchor on recover
- `source_states.py` / `docs/SOURCE_STATES.md` — `post_ice_business_to_warehouse_pure`
- `routes/kpdr/k5/__init__.py` — hop 4 mark GREEN
- `docs/tasks/rr-dbu.8-residual.md` — hop map + next card

### Verify paste
```bash
uv run pytest snes/super_metroid/tests/test_kpdr_dev.py \
  snes/super_metroid/tests/test_source_states_and_ram_cache.py \
  snes/super_metroid/tests/test_k4_ice_scaffold.py -q

uv run python snes/super_metroid/scripts/probe/kpdr.py pure business-to-warehouse \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_gate_to_business_pure.state \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_business_to_warehouse_pure.state
# → GREEN room=0xA6A1 xy=(37,139) pose=138 frames=10255 (×2 exact dual)
```

### Acceptance
- [x] Pure Business Super lip → Warehouse dual green from `post_ice_gate_to_business_pure`
- [x] Accept Super band ~(41,907) p25 via floor fall + elev climb
- [x] Export `scratch/post_ice_business_to_warehouse_pure.state` for next hop
- [x] Classic floor climb entry still GREEN (no warehouse spine first-try rewrite)
- [ ] Full K5 stack / continuous / STATUS (parent rr-dbu.8)

### Residual risks
1. Frame cost high (**10255f**) — multi-attempt Charge ladder after Super fall; not optimized.
2. Elev exit pose 138 (turn residual) — next reverse hop must accept upper-left Warehouse band.
3. Cont-tuned retries after classic 14/8; enemy noise on Business floor can force later rows.
4. Did not restructure ice `_climb_business_floor_to_elevator` to share helper (parity only).

### Next action (required)
- **Next card:** Pure Warehouse → East Tunnel return (K5 hop 5) — reverse of `east_to_warehouse`
- **One change:** pure Warehouse elev exit → East `0xCF80` from `post_ice_business_to_warehouse_pure`
- **Source state:** `scratch/post_ice_business_to_warehouse_pure.state`

### Non-claims
- Did not STATUS-promote continuous past Ice
- Did not claim Alpha PB pure/continuous green
- Did not invent Moat hops

### Probe pin
- Dual GREEN: room=0xA6A1 pose=138 x=37 y=139 frames=10255 ×2
  source=`post_ice_gate_to_business_pure` items `0x3105` beams `0x1007`
