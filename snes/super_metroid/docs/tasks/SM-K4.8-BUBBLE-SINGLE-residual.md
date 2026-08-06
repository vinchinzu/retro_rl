# Residual — SM-K4.8-PURE / rr-yzv (Bubble → Single Chamber Wave pure)

## Result

GREEN

## Files changed

- `routes/kpdr/k4_wave.py` — `play_bubble_to_single_chamber` (top drop → mid-right blue)
- `routes/kpdr/k4_norfair.py` / `registry.py` / `__init__.py` — re-export + segment id
- `scripts/probe/kpdr.py` — `bubble-to-single-chamber` pure CLI
- `tests/test_k4_norfair_scaffold.py` — registry unit
- `source_states.py` — successor `post_bubble_to_single_chamber_pure` + use_for

## Verify paste

```bash
uv run pytest snes/super_metroid/tests/test_k4_norfair_scaffold.py snes/super_metroid/tests/test_k4_speed_branches.py -q
# 39 passed

uv run python snes/super_metroid/scripts/probe/kpdr.py pure bubble-to-single-chamber \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_speed_return_to_bubble_pure.state \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_to_single_chamber_pure.state \
  --expect-room 0xACB3 \
  --no-red-diag
# success=true room=0xAD5E frames=421 (×2 match)
```

## Acceptance

- [x] Discover exit room `0xAD5E` Single Chamber (SM-K4.8 / room_ids; not BACKLOG 0xADAD swap)
- [x] Source: pure post-Speed-return Bubble (`post_speed_return_to_bubble_pure`) at `0xACB3` ~(472,115)
- [x] Pure controller GREEN to ordinary Single Chamber
- [x] Units + dual pure re-run match
- [x] Residual PROCESS schema
- [ ] Graph edge promote / continuous tip (planner after pure green)

## Geometry (resolved)

| Fact | Value |
|------|------:|
| Entry | Bubble Mountain `0xACB3` ~(472,115) post Speed return settle |
| Drop band | top walk LEFT → shaft x∈[370,400] (human ~381) |
| Floor sill | middle-right blue door ~(492,395) block [31,23] |
| Exit | ordinary Single Chamber `0xAD5E` left shaft ~(39,133) after settle |

## Path recipe

1. Bubble top-right: walk LEFT to drop band x≈385
2. Step off lip / fall shaft to floor/mid y≥360
3. Mid platforms: short left hop if x≪360, else RIGHT down to sill
4. Floor: RIGHT hop/run to door x≥470 y≈395
5. RIGHT+X + dash through middle-right blue → Single Chamber

## Residual risks

- Source is pure Speed-return pin only (not continuous-like Bubble seat)
- BACKLOG/KPDR_TRACKER still list swapped Single/Double hex (0xADAD/0xAD5E); code + sm-json use Single=`0xAD5E`
- Heated Single Chamber entry; next hop needs missiles for red door mid-path

## Next action (required)

- **Next card ID:** rr-g1b / SM-K4.9-PURE (Pure Single → Double Chamber)
- **One change:** Pure Wave-path hop from `post_bubble_to_single_chamber_pure`
- **Source state:** `scratch/post_bubble_to_single_chamber_pure.state`

## Non-claims

- Did not STATUS-promote / continuous compose / forge progression RAM
- Did not mark graph edge `bubble_to_single_chamber` continuous
- Did not pure Double Chamber / Wave collect / Ice next hops

## Probe pin (GREEN)

room=0xAD5E pose=81 x=39 y=133 door_transition=0 frames=421 last_pin=post_speed_return_to_bubble_pure dual match
