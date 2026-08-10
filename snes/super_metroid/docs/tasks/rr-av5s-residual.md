## Residual — rr-av5s Pure Red Tower → Hellway return (K5 hop 12)

### Result
PARTIAL — **mid climb dual-stable to temporary floor ~y1606**. From
`post_ice_bat_to_red_pure` ~(206,2443):

1. Morph + double-bomb **IBJ 18/30 c150** from pure bottom (do **not**
   `climb_lower` first — desyncs IBJ) → dual tunnel peak **~y1820**
2. **Tunnel→midplat**: seat y1883 x≈104, UP+X, A→A+X→RIGHT+A+X hop →
   midplat **~y1720**, stand walk **x171**
3. Midplat IBJ 18/30 c171 dual peaks **temporary floor y1606** pose 49
4. Upper zigzag / Hellway exit still **RED** — temp floor lip; spin/bomb
   reverse not yet opening upper shaft to door `0xA2F7`

Not dual green Hellway. No STATUS. Parent **rr-dbu.8** stays open.

### Files changed
- `routes/kpdr/k5/red_to_hellway.py` — phased mid: bottom IBJ →
  `_tunnel_to_midplat` → midplat IBJ; skip lower-before-IBJ; upper floor
  handoff bomb/spin (still RED past y1600)
- geometry constants unchanged

### Geometry discoveries
- Human mid seats y2255/2159/2023 are **frozen rippers** (Ice held), not
  solid tiles — place-grid empty across open shaft except tunnel y1883 +
  right wall x≈228
- Bottom IBJ dual-stable only from pure pin; live `climb_lower` then IBJ
  stalls ~y1977 (enemy/block desync)
- Temp floor hard lip: IBJ dual peaks y1606 then settles; need open path up

### Verify paste
```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py pure red-to-hellway \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_bat_to_red_pure.state \
  --no-red-diag
# → RED: mid dual y1606; upper/Hellway not reached
```

Mid dual isolation:
```bash
# climb_mid after entry → (171,1606)p49 ×2
```

### Acceptance
- [x] Segment wired + ROOM_HELLWAY
- [x] Lower / pocket / spin (prior)
- [x] Bottom IBJ dual tunnel peak ~y1820
- [x] Tunnel→midplat dual
- [x] Midplat IBJ dual temporary floor ~y1606
- [ ] Upper shaft dual to top door band
- [ ] Pure dual green Red bottom → Hellway `0xA2F7`
- [ ] Export `scratch/post_ice_red_to_hellway_pure.state` (+ dual)
- [ ] Full K5 stack / continuous / STATUS (planner)

### Residual risks
1. **Temp floor y1600 lip**: dual peak not a solid stand into upper; bomb
   reverse + spin handoff still stalls ~y1600–1612
2. **climb_lower before IBJ**: kills dual bottom IBJ — keep lower recovery only
3. **Bat door 0xA3DD**: floor RIGHT morph must abort x≥220 y≳2340
4. Human frozen-ripper path nondeterministic for pure

### Next action (required)
- **Next card:** **rr-av5s** (same) — upper from dual floor pin y1606
- **One change:** open temp floor / Hi-Jump spin zigzag to top-right door,
  then RIGHT Hellway dual green + export
- **Source:** `post_ice_bat_to_red_pure` or scratch mid floor pin ~(171,1606)

### Non-claims
- Did not dual-green Red→Hellway
- Did not STATUS-promote continuous past Ice
- Did not close parent rr-dbu.8
- Did not export `post_ice_red_to_hellway_pure`

### Probe pin
- MID floor dual: room=0xA253 pose=49 x=171 y=1606 (climb_mid ×2)
- FULL hop RED: still not Hellway `0xA2F7` (upper residual)
