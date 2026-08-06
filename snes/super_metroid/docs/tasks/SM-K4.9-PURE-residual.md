# Residual — SM-K4.9-PURE / rr-g1b (Pure Single → Double Chamber)

## Result

GREEN

## Files changed

- `routes/kpdr/k4_wave.py` — `play_single_to_double_chamber` (shaft drop → missile red → gap hop)
- `routes/kpdr/k4_norfair.py` / `registry.py` / `__init__.py` — re-export + segment id
- `scripts/probe/kpdr.py` — `single-to-double-chamber` pure CLI
- `tests/test_k4_norfair_scaffold.py` — registry unit
- `source_states.py` — successor `post_single_to_double_chamber_pure` + pin bounds

## Verify paste

```bash
uv run pytest snes/super_metroid/tests/test_k4_norfair_scaffold.py snes/super_metroid/tests/test_k4_speed_branches.py -q
# 40 passed

uv run python snes/super_metroid/scripts/probe/kpdr.py pure single-to-double-chamber \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_to_single_chamber_pure.state \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_single_to_double_chamber_pure.state \
  --expect-room 0xAD5E \
  --no-red-diag
# success=true room=0xADAD frames=700 (×2 match)
```

## Acceptance

- [x] Discover exit room `0xADAD` Double Chamber (code + sm-json; not BACKLOG hex swap)
- [x] Source: pure post Bubble→Single (`post_bubble_to_single_chamber_pure`) at `0xAD5E` ~(39,133–139)
- [x] Pure controller GREEN to ordinary Double Chamber
- [x] Units + dual pure re-run match
- [x] Residual PROCESS schema
- [ ] Graph edge promote / continuous tip (planner after pure green)

## Geometry (resolved)

| Fact | Value |
|------|------:|
| Entry | Single Chamber `0xAD5E` left shaft top ~(39,139) post Bubble→Single pure |
| Mid ledge | y≈267; walk LEFT to drop column x≈60 |
| Floor seat | upper red door platform ~(124,395) |
| Door | Second Top Right red (missiles) block ≈[15,23] → Double Top Left |
| Gap | short walk to ~x145 + 12f RIGHT+B+A spin-hop then RIGHT |
| Exit | ordinary Double Chamber `0xADAD` ~(39,139) after settle |

## Path recipe

1. Top: walk RIGHT with beam shots to x≈130–150; fall to mid y≈267
2. Mid: walk LEFT to x≈60; drop with RIGHT drift → floor ~(75–100,395)
3. Floor: walk to missile seat ~x124; SELECT missiles; stationary X volley ~110f
4. Fuse, walk-up to ~x145, spin-hop gap, hold RIGHT through red door → Double

## Residual risks

- Source is pure Bubble→Single pin only (not continuous-like)
- BACKLOG/KPDR_TRACKER still list swapped Single/Double hex (0xADAD/0xAD5E); code + sm-json use Double=`0xADAD`
- Heated rooms; gap hop is timing-sensitive if red door not fully open
- Settle pin ~(39,139) is top-left Double after transition (not human mid-door ~(238,395) flash)

## Next action (required)

- **Next card ID:** rr-re9 / SM-K4.10-PURE (Pure Wave Beam PLM collect) — or planner name
- **One change:** Pure Double Chamber → Wave room / PLM collect from `post_single_to_double_chamber_pure`
- **Source state:** `scratch/post_single_to_double_chamber_pure.state`

## Non-claims

- Did not STATUS-promote / continuous compose / forge progression RAM
- Did not mark graph edge `single_to_double_chamber` continuous
- Did not pure Wave collect / Ice next hops

## Probe pin (GREEN)

room=0xADAD pose=9 x=39 y=139 door_transition=0 frames=700 last_pin=post_bubble_to_single_chamber_pure dual match
