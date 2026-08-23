## Residual — rr-av5s Pure Red Tower → Hellway return (K5 hop 12)

### Result
GREEN — dual exact **6199f** ×2 from `post_ice_bat_to_red_pure` ~(216,2443)
via `warehouse_to_red_human` hop 6 body (`play_red_to_hellway`). Ordinary
Hellway `0xA2F7` ~(42,153) p29. Not continuous. No STATUS change.

Ice-ladder RAM rewrite in `red_to_hellway.py` remains residual research;
product hop is the dual-green tape body (skip enter-pin frame).

### Verify paste
```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py pure red-to-hellway \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_bat_to_red_pure.state \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_red_to_hellway_pure.state \
  --no-red-diag
# → GREEN room=0xA2F7 xy=(42,153) p29 frames=6199 (×2 exact dual)
```

### Acceptance
- [x] Segment wired + ROOM_HELLWAY
- [x] Dual green Red bottom → Hellway `0xA2F7` from product pin
- [x] Export `scratch/post_ice_red_to_hellway_pure.state` (+ dual)
- [ ] Hellway → Caterpillar from this leave pin (open-loop hop 7 RED)
- [ ] Full K5 stack / continuous / STATUS (planner)

### Next action (required)
- **Next card:** Hellway `0xA2F7` → Caterpillar `0xA322` (K5 hop 13)
- **One change:** dual-green from `post_ice_red_to_hellway_pure` ~(42,153) p29
- **Source:** `post_ice_red_to_hellway_pure` (warehouse hop 6 body; not hop-7 live pin)

### Non-claims
- Did not STATUS-promote continuous past Ice
- Did not close parent rr-dbu.8 (Alpha PB PLM still open)
- Did not dual-green Hellway→Caterpillar from the new leave pin
- Ceres-successor `--to ice` dual is now GREEN **146,937f** ×2 in scratch
  (`rr-ucl9` STATUS promote; not this card)

### Probe pin
- **HELLWAY dual end: room=0xA2F7 pose=29 x=42 y=153 frames=6199 exact ×2**
