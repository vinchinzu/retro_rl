# SM-K4-06B Residual

## Attempt

Changed only the `kraid_return_approach` choreography in
`routes/kpdr/varia_return.py`: stage with a bounded left walk, perform one
fixed 18-frame `A+LEFT` short hop, and settle neutrally for 12 frames. The
existing lip backoff, unmorph, face, release, four shot/fuse cycles, spin-push,
and lip recovery timings were unchanged.

## Pure Probe

Command:

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure kraid-to-eye-return \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_to_kraid_pure.state
```

Exit code: `1`.

The controller timed out in the left eye-door exit. Last pin:

- Room: `0xA59F`
- Pose: `82`
- Position: `x=37`, `y=307`
- `door_transition`: `0`
- No frame observed `door_transition != 0`.

## Next Authorization

Pure green was not reached. The planner should authorize exactly one next
primitive: vary the fixed short-hop landing/settle timing while keeping the
approach staging, lip backoff, re-face, shot/fuse, and spin-push timings fixed.
This attempt does not claim continuous evidence, STATUS promotion, or graph
promotion. No progression, capacity, door, event, or boss RAM was forged or
written.
