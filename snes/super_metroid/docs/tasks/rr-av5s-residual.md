## Residual — rr-av5s Pure Red Tower → Hellway return (K5 hop 12)

### Result
PARTIAL — **first enemy-aware Ice/WJ checkpoint GREEN**, plus the earlier
thin-seat dual after period WJ + ice-ladder attempt. From
`post_ice_bat_to_red_pure` ~(206,2443):

0. New checkpoint policy: bottom floor → freeze lowest Ripper → consecutive
   WJ → grounded `lower_ripper_1` **y2351**, dual exact and GREEN over 31 patrol
   phases total (`0..240f`, step 8), **230–414f at 408–636 FPS**. Probe-only
   (not wired into product `RoomAutopilot` — red climb is human/probe until a
   verified reactive policy exists); `lower_ripper_1→2` remains planned.

1. Morph + double-bomb **IBJ 18/30 c150** from pure bottom → dual tunnel peak
   **~y1820**
2. **Tunnel→midplat** → midplat IBJ dual temporary floor **y1606** pose 49
3. **Human ascent RLE first 850f** → dual past temp floor **~(122,1459) p81**
4. **Spin-left seat** ~(37,1499) → **period WJ** phases 0–7 (no phase-8 fall)
   dual peak **~y420** end phase7 ~(89,436)
5. **Land thin natural seat** dual exact **(91,587) p3** frames=35259 ×2
6. Ice-freeze ripper ladder **wired** (tiers y520/416/320/232 morph-less high
   hop); first hop to y495 probe-green from live seat, tier-2+ offset freeze
   still residual — Hellway `0xA2F7` still **RED**

Not dual green Hellway. No STATUS. Parent **rr-dbu.8** stays open.

### Files changed
- `routes/kpdr/k5/red_ice_climb.py` — live enemy sensor + first checkpoint
  runner for probe scripts (not product AP)
- `routes/kpdr/data/red_tower_ice_checkpoint_plan.json` — full checkpoint /
  recovery tree; only edge 01 is marked verified
- `scripts/probe/red_ice_climb.py` — natural dual + patrol-phase sweep
- `scripts/export/red_ice_route_plan.py` — deterministic full-room and edge PNGs
- `routes/kpdr/k5/red_to_hellway.py` — drop WJ phase8 (falls y687); thin-seat
  land; Ice ripper ladder helpers (freeze + high hop, no ground-morph); residual
  thrash only when already near door
- `routes/kpdr/data/red_to_hellway_human_top.json` — human top hop RLE extract
  (dev reference; open-loop desyncs from pure enemy phase)
- `scripts/probe/red_top_gap_probe.py` — upper/peak capture + recipe sweep (dev)

### Geometry discoveries
- Period WJ dual peaks **~y420** (89,420); phase8 LEFT stop=150 **loses** height
  → old dual end (171,687) — **do not run phase8**
- Hard WJ ceiling ~y390–420 without platforms; pure solids: y740 wide, y180 top
  door, thin seat **~y587 x≈85–95**
- Upper rippers **0xD47F** at y≈520/416/320/232; Ice freezes (fr timer @+0x26);
  human stands on freeze tops y495/391/295/207 then RIGHT Hellway
- **Ground morph (pose 23) falls through frozen rippers**; air spin / crouch
  land (164/1) sticks — hop must not force DOWN morph
- Same-column freeze bonks hop underside; prefer min_dx≥12–14 for tiers above
  first; freeze projectile may still re-center enemy x
- Probe: live thin seat after full WJ → freeze520 → high hop dual-capable
  **(100,495)p1**; full ladder tier-2+ still flaky

### Verify paste
```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py pure red-to-hellway \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_bat_to_red_pure.state \
  --no-red-diag
# → RED dual exact: room=0xA253 pose=3 x=91 y=587 frames=35259 (×2)
# Hellway 0xA2F7 not reached; ice ladder residual y495→door
```

### Acceptance
- [x] Segment wired + ROOM_HELLWAY
- [x] Bottom → first frozen lower Ripper: live phase track + consecutive WJ,
  dual exact and 31-phase GREEN
- [ ] First → second frozen lower Ripper (next checkpoint edge)
- [x] Lower / pocket / spin (prior)
- [x] Bottom IBJ dual tunnel peak ~y1820
- [x] Tunnel→midplat dual
- [x] Midplat IBJ dual temporary floor ~y1606
- [x] Past temp floor dual ~(122,1459) via human RLE@850
- [x] Upper period WJ dual peak ~y420 (stop before phase8 fall)
- [x] Thin seat dual ~(91,587) p3 after WJ+ice attempt
- [ ] Ice ladder dual y495→391→295→207 → door y≤~220
- [ ] Pure dual green Red bottom → Hellway `0xA2F7`
- [ ] Export `scratch/post_ice_red_to_hellway_pure.state` (+ dual)
- [ ] Full K5 stack / continuous / STATUS (planner)

### Residual risks
1. **Ice ladder tier-2+**: min_dx freeze + high hop not dual-stable past y495
2. Freeze timer ~400f — long waits unfreeze seat ice and drop
3. climb_lower before IBJ / bomb floor / force-unmorph p81 / RLE past 850
4. Adaptive thrash after dual mid pins loses height

### Next action (required)
- **Next card:** **rr-av5s** (same) — from dual thin seat ~(91,587)
- **One change:** dual ice-ladder hop chain y495→391→295→207 → RIGHT Hellway
  0xA2F7 + export; tighten freeze offset + hop from live full-climb enemy phase
- **Source:** `post_ice_bat_to_red_pure` (live mid+human850+WJ+seat)

### Non-claims
- Did not dual-green Red→Hellway
- Did not STATUS-promote continuous past Ice
- Did not close parent rr-dbu.8
- Did not export `post_ice_red_to_hellway_pure`

### Probe pin
- MID floor dual: room=0xA253 pose=49 x=171 y=1606 (climb_mid ×2)
- PAST floor dual: room=0xA253 pose=81 x=122 y=1459 (mid+human@850 ×2)
- WJ peak dual: room=0xA253 best ~y420 (phase7)
- **THIN SEAT dual end: room=0xA253 pose=3 x=91 y=587 frames=35259 exact ×2**
- FULL hop RED: still not Hellway `0xA2F7` (ice ladder residual)
