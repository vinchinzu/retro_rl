## Residual — rr-av5s Pure Red Tower → Hellway return (K5 hop 12)

### Result
PARTIAL — **lower past pocket ceiling GREEN**. From
`post_ice_bat_to_red_pure` ~(206,2443), right-wall WJ + re-catch reaches open
shaft **~(219,1942)** (past dead-end pocket ~(225,2091) / pure-A ceiling
~y1964). Mid double-bomb IBJ + bomb-floor / upper still RED — pure stalls at
y=1942 (hard right-wall height; shaft too wide for single-WJ left latch).
Not dual green. No STATUS change. Parent rr-dbu.8 stays open.

### Files changed
- `routes/kpdr/k5/red_to_hellway.py` — lower WJ + right-wall re-catch past
  pocket; Bat-door abort; mid IBJ double-bomb (18/30) + floor/upper shells
- `routes/kpdr/k5/geometry.py` — RED_* climb + Hellway exit constants (prior)
- wiring: `ROOM_HELLWAY`, probe `red-to-hellway`, registry (prior)
- scratch (gitignored): `dev_red_lower_past_pocket.state` ~(219,1942) after lower

### Verify paste
```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py pure red-to-hellway \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_bat_to_red_pure.state \
  --no-red-diag
# → RED mid: residualPinLine room=0xA253 pose=25 x=219 y=1942 frames≈7716
# Lower alone: GREEN ~(219,1942) in ~475f (export dev_red_lower_past_pocket.state)
```

### Acceptance
- [x] Segment wired (`red-to-hellway` / `play_red_to_hellway`) + ROOM_HELLWAY
- [x] Source pin `post_ice_bat_to_red_pure` fingerprint validated
- [x] Lower right-wall WJ reaches y≈2091 pocket (reverse of lower descent band)
- [x] Past pocket ceiling into open shaft y≈1942 (not y≤2090 alone)
- [ ] Pure dual green Red bottom → Hellway
- [ ] Export `scratch/post_ice_red_to_hellway_pure.state` (+ dual)
- [ ] Full K5 stack through Alpha PB (parent rr-dbu.8)
- [ ] Continuous / STATUS (planner only)

### Residual risks
1. **Right-wall hard ceiling ~y1942**: recatch oscillates 1942↔2091; cannot
   gain past without left wall / IBJ / freeze platforms.
2. **Shaft width**: single right-WJ + LEFT spin min x≈135 — never latches left
   wall (x≲50).
3. **IBJ mid nondeterministic**: one bottom-path trial peaked y=1799 then fell;
   not dual-stable from pure pin / post-lower 1942 seat.
4. **Bat door 0xA3DD**: floor RIGHT spins must abort x≥220 y≳2340; mid fall
   bias LEFT. Prefer clear_bat reentry before re-lower.
5. Human final-ascent RLE still desyncs (enemy thrash state).

### Next action (required)
- **Next card:** **rr-av5s** (same) — mid past y1942 → tunnel y≤1880 → bomb floor
- **One change:** from natural `dev_red_lower_past_pocket` ~(219,1942), open-loop
  midair HBJ / staggered IBJ / freeze-ripper platforms (Ice equipped) to y≤1880
- **Source state:** `scratch/dev_red_lower_past_pocket.state` or
  `scratch/post_ice_bat_to_red_pure.state`

### Non-claims
- Did not dual-green Red→Hellway
- Did not STATUS-promote continuous past Ice
- Did not close parent rr-dbu.8
- Did not claim Alpha PB pure
- Did not claim mid/upper dual green (only lower past pocket)

### Probe pin
- LOWER past pocket: room=0xA253 pose=25 x=219 y=1942 (~475f from pure bottom)
- FULL hop RED: room=0xA253 pose=25 x=219 y=1942 frames≈7716 (mid stall)
