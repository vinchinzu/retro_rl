# Residual — SM-K4.7-SPEED-RETURN / rr-g4i (Speed return → Bubble pure)

## Result

GREEN

## Files changed

- `routes/kpdr/speed_return.py` — `play_speed_return_to_bubble` (Speed→Hall→Bat→Bubble)
- `routes/kpdr/k4_norfair.py` / `registry.py` / `__init__.py` — re-export + segment id
- `scripts/probe/kpdr.py` — `speed-return-to-bubble` pure CLI
- `tests/test_k4_norfair_scaffold.py` — registry unit
- `source_states.py` — successor `post_speed_return_to_bubble_pure` + use_for

## Verify paste

```bash
uv run pytest snes/super_metroid/tests/test_k4_norfair_scaffold.py snes/super_metroid/tests/test_k4_speed_branches.py -q
# 38 passed

uv run python snes/super_metroid/scripts/probe/kpdr.py pure speed-return-to-bubble \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_speed_collected.state \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_speed_return_to_bubble_pure.state \
  --expect-room 0xAD1B \
  --no-red-diag
# success=true room=0xACB3 frames=2158 (×2 match; also post_speed_hall_to_speed_pure)
```

## Acceptance

- [x] Discover exit room `0xACB3` Bubble Mountain (SM-K4.7 / BACKLOG / KPDR_TRACKER)
- [x] Source: pure post-Speed (`post_speed_collected` / `post_speed_hall_to_speed_pure`) at `0xAD1B`
- [x] Pure controller GREEN to ordinary Bubble
- [x] Units + dual pure re-run match
- [x] Residual PROCESS schema
- [ ] Graph edge promote / continuous tip (planner after pure green)

## Geometry (resolved)

| Fact | Value |
|------|------:|
| Entry | Speed Room `0xAD1B` ~(169,123) items `0x3105` |
| Speed → Hall | LEFT blue door → Hall right lip ~x=3032 |
| Hall reverse | **LEFT+B** incline → left blue → Bat top-right |
| Bat shelf→cavity | morph bombs at x≈165–175 |
| Bat cavity→floor | power **DOWN+X** at tight hole **x∈[148,154]** |
| Bat floor→door | LEFT hop rhythm (flat walk falls lava) |
| Exit | ordinary Bubble `0xACB3` ~(472,115) after settle |

## Path recipe

1. Speed Room: walk/dash LEFT, shoot blue, enter Hall
2. Speed Hall: hold LEFT+B entire reverse incline; left blue to Bat
3. Bat top-right: morph bomb shelf floor into cavity ~(168,251)
4. Align hole band x∈[148,154], DOWN+X + hop → floor y≈395
5. Hop-left over lava gaps, shoot bottom-left blue → Bubble

## Next action (required)

- **Next card ID:** rr-yzv / SM-K4.8-PURE (Pure Bubble → Single Chamber Wave)
- **One change:** Pure Wave-path hop from `post_speed_return_to_bubble_pure`
- **Source state:** `scratch/post_speed_return_to_bubble_pure.state`

## Non-claims

- Did not STATUS-promote / continuous compose / forge progression RAM
- Did not mark graph edge `speed_return_to_bubble` continuous
- Did not pure Wave / Ice / Moat next hops

## Probe pin (GREEN)

room=0xACB3 pose=82 x=472 y=115 door_transition=0 frames=2158 last_pin=post_speed_collected dual match
