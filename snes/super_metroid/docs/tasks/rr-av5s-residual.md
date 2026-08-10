## Residual — rr-av5s Pure Red Tower → Hellway return (K5 hop 12)

### Result
PARTIAL — **past temp floor dual-stable to ~(122,1459)**. From
`post_ice_bat_to_red_pure` ~(206,2443):

1. Morph + double-bomb **IBJ 18/30 c150** from pure bottom (do **not**
   `climb_lower` first — desyncs IBJ) → dual tunnel peak **~y1820**
2. **Tunnel→midplat**: seat y1883 x≈104, UP+X, A→A+X→RIGHT+A+X hop →
   midplat **~y1720**, stand walk **x171**
3. Midplat IBJ 18/30 c171 dual peaks **temporary floor y1606** pose 49
4. **Human ascent RLE first 850f** from live floor pin → dual peak past
   temp floor **~(122,1459) p81** (both trials)
5. Upper adaptive WJ / Hellway exit still **RED** — handoff pose 81 must
   not force-unmorph (taller hitbox falls); spin/WJ from p81 not yet
   dual-stable to top door `0xA2F7`

Not dual green Hellway. No STATUS. Parent **rr-dbu.8** stays open.

### Files changed
- `routes/kpdr/k5/red_to_hellway.py` — phased mid IBJ stack + human RLE@850
  floor breach; adaptive upper residual; bat abort; no bomb-through-floor
- `routes/kpdr/data/red_to_hellway_human_ascent.json` — (existing) human
  open-loop source; controller slices first 850f

### Geometry discoveries
- Human mid seats y2255/2159/2023 are **frozen rippers** (Ice held), not
  solid tiles
- Temp floor hard lip ~y1600: IBJ dual peaks y1606; **bombing from below
  opens floor and falls down** (outbound bombs from above)
- Human RLE@850 from **live** climb_mid end dual-stable to y1459; RLE
  recorded against a reloaded floor save desyncs (do not intermediate-save
  before open-loop)
- Pose 81 at dual handoff: force UP-unmorph drops ~100px; keep air poses
- Adaptive WJ after y1459 still stalls / falls (upper residual)

### Verify paste
```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py pure red-to-hellway \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_bat_to_red_pure.state \
  --no-red-diag
# → RED: mid dual y1606; human@850 dual y1459; Hellway not reached
```

Mid dual isolation:
```bash
# climb_mid after entry → (171,1606)p49 ×2
# + human RLE 850f → (122,1459)p81 ×2
```

### Acceptance
- [x] Segment wired + ROOM_HELLWAY
- [x] Lower / pocket / spin (prior)
- [x] Bottom IBJ dual tunnel peak ~y1820
- [x] Tunnel→midplat dual
- [x] Midplat IBJ dual temporary floor ~y1606
- [x] Past temp floor dual ~(122,1459) via human RLE@850
- [ ] Upper shaft dual to top door band
- [ ] Pure dual green Red bottom → Hellway `0xA2F7`
- [ ] Export `scratch/post_ice_red_to_hellway_pure.state` (+ dual)
- [ ] Full K5 stack / continuous / STATUS (planner)

### Residual risks
1. **Upper from y1459 p81**: spin/WJ adaptive not dual to door; unmorph
   drops progress
2. **climb_lower before IBJ**: kills dual bottom IBJ — keep lower recovery only
3. **Bat door 0xA3DD**: floor RIGHT morph must abort x≥220 y≳2340
4. Open-loop after intermediate save desyncs — record only from live mid

### Next action (required)
- **Next card:** **rr-av5s** (same) — upper from dual handoff ~(122,1459) p81
- **One change:** open-loop or latch WJ chain p81→top door band without
  force-unmorph; dual green Hellway + export
- **Source:** `post_ice_bat_to_red_pure` (live mid+human850 handoff)

### Non-claims
- Did not dual-green Red→Hellway
- Did not STATUS-promote continuous past Ice
- Did not close parent rr-dbu.8
- Did not export `post_ice_red_to_hellway_pure`

### Probe pin
- MID floor dual: room=0xA253 pose=49 x=171 y=1606 (climb_mid ×2)
- PAST floor dual: room=0xA253 pose=81 x=122 y=1459 (mid+human@850 ×2)
- FULL hop RED: still not Hellway `0xA2F7` (upper residual)
