## Residual — rr-av5s Pure Red Tower → Hellway return (K5 hop 12)

### Result
PARTIAL — scaffolding + lower right-wall WJ primitive greens to the
**right-pocket ledge** `0xA253` ~(225,2091). Full hop to Hellway `0xA2F7`
not dual green yet. Human tape final ascent RLE desyncs (enemy state differs
from pure Bat→Red pin). Not continuous. No STATUS change.

### Files changed
- `routes/kpdr/k5/red_to_hellway.py` — multi-phase climb (lower WJ / mid / upper)
- `routes/kpdr/k5/geometry.py` — RED_* climb + Hellway exit constants
- `routes/kpdr/k5/__init__.py` — export `play_red_to_hellway`
- `routes/kpdr/room_ids.py` + `rooms.py` — `ROOM_HELLWAY = 0xA2F7`
- `routes/kpdr/registry.py` / `__init__.py` / `scripts/probe/kpdr.py` — segment wiring
- `routes/kpdr/data/red_to_hellway_human_ascent.json` — tape final-ascent RLE seed
  (not dual-stable from pure pin; kept as recon data)

### Verify paste
```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py pure red-to-hellway \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_bat_to_red_pure.state \
  --no-red-diag
# → RED lower/mid: reaches right-pocket ~(219–225,2091) then stalls / falls
# Lower WJ cycle alone: GREEN pocket land y≈2091 from bottom ~(216,2443)
```

### Acceptance
- [x] Segment wired (`red-to-hellway` / `play_red_to_hellway`) + ROOM_HELLWAY
- [x] Source pin `post_ice_bat_to_red_pure` fingerprint validated
- [x] Lower right-wall WJ reaches y≈2091 pocket (reverse of lower descent band)
- [ ] Pure dual green Red bottom → Hellway
- [ ] Export `scratch/post_ice_red_to_hellway_pure.state` (+ dual)
- [ ] Full K5 stack through Alpha PB (parent rr-dbu.8)
- [ ] Continuous / STATUS (planner only)

### Residual risks
1. **Right-pocket ceiling** at ~(225,2091): vertical A peaks ~y1964 then lands
   same ledge; apex LEFT-WJ currently drops into free-fall (not mid platforms).
2. Human final-ascent RLE (f27448–Hellway) needs enemy/RNG state after ~4k
   thrash in Red — pure Bat entry pin desyncs early into Bat door / floor.
3. Mid bomb-floor (~y1600–1760) and upper zigzag not dual-proven from pure pin.
4. Prefer clean reverse of `play_red_tower_to_bat` bands over freeze thrash.

### Next action (required)
- **Next card:** **rr-av5s** (same) — mid shaft past right-pocket + bomb floor
- **One change:** from natural lower land ~(225,2091), open-loop apex-WJ + mid
  platform seat to y≤1880 (tunnel), then bomb-floor reverse to y≤1600
- **Source state:** `scratch/post_ice_bat_to_red_pure.state` (or dump
  `scratch/dev_red_lower_done.state` after lower if useful for isolation)

### Non-claims
- Did not dual-green Red→Hellway
- Did not STATUS-promote continuous past Ice
- Did not close parent rr-dbu.8
- Did not claim Alpha PB pure

### Probe pin
- PARTIAL lower: room=0xA253 pose=9 x=225 y=2091 (right-pocket after WJ)
- RED full hop: room=0xA253 pose=25 x=219 y=2132 frames≈3164 (stall)
