## Residual — rr-av5s Pure Red Tower → Hellway return (K5 hop 12)

### Result
PARTIAL — **upper period WJ dual-stable end ~(171,687)**. From
`post_ice_bat_to_red_pure` ~(206,2443):

1. Morph + double-bomb **IBJ 18/30 c150** from pure bottom (do **not**
   `climb_lower` first — desyncs IBJ) → dual tunnel peak **~y1820**
2. **Tunnel→midplat**: seat y1883 x≈104, UP+X, A→A+X→RIGHT+A+X hop →
   midplat **~y1720**, stand walk **x171**
3. Midplat IBJ 18/30 c171 dual peaks **temporary floor y1606** pose 49
4. **Human ascent RLE first 850f** from live floor pin → dual peak past
   temp floor **~(122,1459) p81** (both trials; mid-air, not solid)
5. **Spin-left seat** ~(37,1499) then **alternating period WJ** (16/6/8,
   short phases) dual end **~(171,687) p25** frames=11802 exact ×2
6. Hellway `0xA2F7` still **RED** — y420 peak during WJ chain; hard
   residual ~y390–420 cannot yet reach top door band y180

Not dual green Hellway. No STATUS. Parent **rr-dbu.8** stays open.

### Files changed
- `routes/kpdr/k5/red_to_hellway.py` — left seat after human850; alternating
  period WJ D-chain dual ~y687; no force-unmorph p81; no mid-shaft exit thrash
- `routes/kpdr/data/red_to_hellway_human_ascent.json` — (existing) first 850f
  only; remainder desyncs from pure pin
- `scripts/probe/red_upper_probe.py` — diagnostic upper recipes (dev)

### Geometry discoveries
- Human mid seats y2255/2159/2023 are **frozen rippers** (Ice held), not
  solid tiles
- Temp floor hard lip ~y1600: IBJ dual peaks y1606; **bombing from below
  opens floor and falls down** (outbound bombs from above)
- Human RLE@850 from **live** climb_mid end dual-stable to y1459; RLE
  after 850 / full human from floor peaks same y1459 then falls (tape
  enemy state differs)
- Pose 81 at handoff is **mid-air peak** (vy=0 temporary); force UP-unmorph
  drops ~100px; spin-left seats solid left ledge ~(37,1499) p2
- Solids (place map): y1400 band, y1284 right, y1028 right, y740 wide,
  y180 top door platforms; thin natural seat ~y587 x≈85–89
- Alternating period WJ (into 6 / flip 8 / period 16) dual climbs past
  y1459 → peak ~y420 → end pin y687; further residual thrash **loses**
  height (do not adaptive-thrash after dual pin)

### Verify paste
```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py pure red-to-hellway \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_bat_to_red_pure.state \
  --no-red-diag
# → RED dual exact: room=0xA253 pose=25 x=171 y=687 frames=11802 (×2)
# Hellway 0xA2F7 not reached; upper residual ~y420→door
```

### Acceptance
- [x] Segment wired + ROOM_HELLWAY
- [x] Lower / pocket / spin (prior)
- [x] Bottom IBJ dual tunnel peak ~y1820
- [x] Tunnel→midplat dual
- [x] Midplat IBJ dual temporary floor ~y1606
- [x] Past temp floor dual ~(122,1459) via human RLE@850
- [x] Upper period WJ dual past y1459 → end ~(171,687)
- [ ] Upper shaft dual to top door band y≤~220
- [ ] Pure dual green Red bottom → Hellway `0xA2F7`
- [ ] Export `scratch/post_ice_red_to_hellway_pure.state` (+ dual)
- [ ] Full K5 stack / continuous / STATUS (planner)

### Residual risks
1. **y420→y180 gap**: period WJ stalls / peaks ~y391–420; open shaft to top
   door platforms not dual-cleared yet
2. **climb_lower before IBJ**: kills dual bottom IBJ — keep lower recovery only
3. **Bat door 0xA3DD**: floor RIGHT morph must abort x≥220 y≳2340
4. Open-loop after intermediate save desyncs — record only from live mid
5. Adaptive thrash after dual upper pin falls to ~y1275 — stop at pin

### Next action (required)
- **Next card:** **rr-av5s** (same) — from dual upper ~(171,687) / peak ~y420
- **One change:** clear y420→top door y180 (solid seats / tighter WJ / shoot
  bomb blocks if any) without losing height; dual green Hellway + export
- **Source:** `post_ice_bat_to_red_pure` (live mid+human850+period WJ)

### Non-claims
- Did not dual-green Red→Hellway
- Did not STATUS-promote continuous past Ice
- Did not close parent rr-dbu.8
- Did not export `post_ice_red_to_hellway_pure`

### Probe pin
- MID floor dual: room=0xA253 pose=49 x=171 y=1606 (climb_mid ×2)
- PAST floor dual: room=0xA253 pose=81 x=122 y=1459 (mid+human@850 ×2)
- UPPER dual end: room=0xA253 pose=25 x=171 y=687 frames=11802 exact ×2
- FULL hop RED: still not Hellway `0xA2F7` (y420→door residual)
