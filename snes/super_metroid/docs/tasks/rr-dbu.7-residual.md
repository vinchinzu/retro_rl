## Residual — rr-dbu.7 Continuous tip `--to ice` compose

### Result
COMPOSE WIRED — TipSpec / spine / catalog / graph edges for continuous
`--to ice` after Ice pure stack greens (`rr-dbu.11`). **Not** dual continuous
green. **Not** STATUS-promoted. Default CLI tip remains `wave`.

### Wire (this bead)

| Layer | Change |
|-------|--------|
| TipSegment | `ice` parent=`wave`, final Ice PLM, aliases `ice_beam` / `k4_11` |
| SpineHop ×4 | `business_to_ice_gate` → `ice_gate_to_acid` → `ice_acid_to_snake` → `ice_snake_to_ice` |
| TipSpec | Generated via `hops.py` `ICE_ONLY_HOPS` + `register_tips` |
| Graph | Outbound Ice edges spine-emitted `continuous` via `continuous_edges_for_tips(..., "ice")`; Tutorial return path stays hand-authored `unverified` |
| Catalog | ContinuousTip + NamedRoute derived; `DEFAULT_CONTINUOUS_TIP` still `wave` |

### Verify paste (unit / no emulator)
```bash
uv run pytest snes/super_metroid/tests/test_continuous_tips.py \
  snes/super_metroid/tests/test_k4_speed_branches.py \
  snes/super_metroid/tests/test_k4_ice_scaffold.py -q
# → 33+ passed (compose + ice scaffold)

# Catalog smoke
uv run python -c "from super_metroid.routes.catalog import get_continuous_tip, DEFAULT_CONTINUOUS_TIP; \
  t=get_continuous_tip('ice'); print(t.tip_id, t.aliases, DEFAULT_CONTINUOUS_TIP)"
# → ice ('ice_beam', 'k4_11', 'k4.11') wave
```

### Acceptance
- [x] TipSpec / catalog / graph compose from pure stack greens
- [x] CLI `--to ice` resolves (compose identity)
- [x] No STATUS promote of continuous Ice
- [x] Default tip remains `wave`
- [ ] Dual continuous green power-on → Ice (blocked — see risks)

### Residual risks
1. **Room gap Wave → Business.** Wave tip ends `0xADDE`; ice hops start
   `0xA7DE` (`play_business_to_ice_gate` requires Business). Human tape
   Phase B return (Wave→Double→Single→Bubble→Farm→Speedway→Frog→Business)
   has **no pure stack** yet. Continuous dual will RED until return pure lands
   and is composed as ice-prefix hops (or an intermediate tip).
2. Gate→Acid still needs **Speed** (Boost Blocks); product loadout from Wave
   parent is correct once return exists.
3. Tutorial return path edges remain `unverified` (outbound skips Tutorial).

### Next action (required)
- **Next card ID:** `rr-vqv3` — Wave→Business pure return stack (one-hop pure
  beads from post-Wave continuous; human tape Phase B return)
- **Then:** continuous dual `--to ice` stabilize (no STATUS until dual green)
- **Source state:** `recordings/wave.json` endpoint / Wave checkpoint; pure
  Ice chain under `scratch/post_ice_*`

### Non-claims
- Did not STATUS-promote continuous `--to ice`
- Did not claim dual continuous green Ice
- Did not implement Wave→Business return pure hops
- Did not change default CLI tip away from `wave`
