# SM-K4-06C Residual

## Scope

Changed only the `kraid_return_short_hop` and
`kraid_return_approach_settle` hold lengths in
`routes/kpdr/varia_return.py`. The source state for each attempt was:

```text
custom_integrations/SuperMetroid-Snes/scratch/post_varia_to_kraid_pure.state
```

All post-approach door choreography remained unchanged. No progression,
capacity, door, event, or boss RAM was forged or written.

## Pure Attempts

| Attempt | Short hop | Settle | Result | Final pin | Door transition max |
|---|---:|---:|---|---|---:|
| 1 | 12f | 6f | RED / timeout | `0xA59F`, pose `82`, `x=37`, `y=289` | `0` |
| 2 | 24f | 20f | RED / timeout | `0xA59F`, pose `82`, `x=37`, `y=307` | `0` |

Attempt 2 is retained as the controller values because its final pin matches
the prior 06B residual while using the longer bounded settle. No attempt
observed a room change or `door_transition != 0`.

## Commands

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure kraid-to-eye-return \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_to_kraid_pure.state
```

Both bounded runs exited with code `1` at the left eye-door exit timeout.

## Next Primitive

The next planner-authorized primitive should be a read-only door/PLM/BTS
reconnaissance step, or a single shot-Y-band test once such a field is
available. Do not use free spin as the next geometry change.

## Non-Claims

- Not pure green.
- Not continuous evidence.
- No STATUS or graph promotion.
- No progression, capacity, door, event, or boss RAM was forged or written.
- Last pin: room `0xA59F`, pose `82`, `x=37`, `y=307`.
