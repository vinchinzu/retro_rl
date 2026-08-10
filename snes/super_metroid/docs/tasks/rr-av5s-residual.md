## Residual — rr-av5s Pure Red Tower → Hellway return (K5 hop 12)

### Result
PARTIAL — **lower past pocket GREEN** + **mid pocket-spin dual-stable to
~y1932**. From `post_ice_bat_to_red_pure` ~(216,2443):

1. Right-wall WJ → pocket ~(225,2091) → optional re-catch → open shaft
   **~(219,1942)** (right-wall **hard ceiling**, pure-A gain 0).
2. **Pocket spin** `B+LEFT+A` from ~(235,2091) → mid crouch seat
   **~(174–183, 1932–1942)** pose 26 (dual-stable; human-matched launch).
3. Tunnel y≤1880 / bomb floor y≤1600 / upper / Hellway still **RED** —
   ~50px short of `RED_TUNNEL_Y`; morph from crouch seat falls through thin
   lip; bottom-path IBJ 18/30 peaked **y1799 once** (not dual-stable).

Not dual green. No STATUS change. Parent **rr-dbu.8** stays open.

### Files changed
- `routes/kpdr/k5/red_to_hellway.py` — pocket spin mid (`_pocket_spin_mid`),
  pocket/wall-ceiling detectors, mid climb prefers spin→IBJ over recatch thrash;
  lower WJ + past-pocket re-catch retained
- `routes/kpdr/k5/geometry.py` — RED_* climb + Hellway exit constants (prior)
- wiring: `ROOM_HELLWAY`, probe `red-to-hellway`, registry (prior)
- scratch (gitignored): `dev_red_lower_past_pocket.state` ~(219,1942);
  `dev_red_pocket_seat.state` ~(235,2091)

### Verify paste
```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py pure red-to-hellway \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_bat_to_red_pure.state \
  --no-red-diag
# → RED mid: best_y≈1932–1942 (pocket-spin seat) or re-lower stall; not Hellway
```

### Acceptance
- [x] Segment wired (`red-to-hellway` / `play_red_to_hellway`) + ROOM_HELLWAY
- [x] Source pin `post_ice_bat_to_red_pure` fingerprint validated
- [x] Lower right-wall WJ reaches y≈2091 pocket
- [x] Past pocket ceiling into open shaft y≈1942 (right-wall hard ceiling)
- [x] Pocket spin reaches mid crouch ~y1932 x≈175 (dual-stable technique)
- [ ] Pure dual green Red bottom → Hellway
- [ ] Tunnel y≤1880 dual-stable
- [ ] Export `scratch/post_ice_red_to_hellway_pure.state` (+ dual)
- [ ] Full K5 stack through Alpha PB (parent rr-dbu.8)
- [ ] Continuous / STATUS (planner only)

### Residual risks
1. **Right-wall hard ceiling ~y1942**: pure-A gain 0; recatch oscillates
   1942↔2091. Leave via pocket spin, not more WJ.
2. **Tunnel lip ~y1932 crouch seat**: under ceiling; standing bonks; morph
   drops through thin geometry (~y1977) then free-fall.
3. **~50px gap to RED_TUNNEL_Y=1880**: human tape spends ~3k frames thrashing
   y1950→1900 (f23895–26986) then platform hops x≈185–203 y2255→1878.
4. **IBJ mid nondeterministic**: bottom 18/30 double-bomb once peaked y=1799
   then fell; not dual-stable from pure pin / post-spin seat.
5. **Shaft width**: single right-WJ + LEFT spin min x≈106–135 — never latches
   left wall (x≲50).
6. **Bat door 0xA3DD**: floor RIGHT spins must abort x≥220 y≳2340.
7. Human final-ascent RLE still desyncs (enemy thrash state).

### Geometry notes (human tape f23078–29947)
| Band | y | Technique |
|------|---|-----------|
| Lower | 2440→2090 | Right-wall WJ |
| Pocket launch | 2091 | Stand + B+LEFT+A spin |
| Mid thrash | 1950→1900 | Long human thrash / ledge hops |
| Tunnel | 1880→1600 | Morph + UP+X / bomb blocks |
| Floor | ~1600 | Temp bomb floor reverse |
| Upper | 1600→180 | Zigzag WJ to Hellway door |

### Next action (required)
- **Next card:** **rr-av5s** (same) — close y1932 → tunnel y≤1880 → floor
- **One change:** from natural mid crouch after pocket spin, human-style **ledge
  hop chain** on solid mid platforms (x≈185–203, y≈2255→2012→1909) *or*
  stabilize bottom IBJ 18/30 through y1799→1600 dual; then upper + Hellway exit
- **Source state:** `scratch/dev_red_pocket_seat.state` / `dev_red_lower_past_pocket`
  / `post_ice_bat_to_red_pure`

### Non-claims
- Did not dual-green Red→Hellway
- Did not STATUS-promote continuous past Ice
- Did not close parent rr-dbu.8
- Did not claim Alpha PB pure
- Did not claim tunnel/floor/upper dual green

### Probe pin
- LOWER past pocket: room=0xA253 pose=25 x=219 y=1942 (~475f from pure bottom)
- POCKET SEAT: room=0xA253 x=235 y=2091 p9
- MID spin seat: room=0xA253 pose=26 x≈174–183 y≈1932–1942 (from pocket spin)
- FULL hop RED: still not Hellway `0xA2F7` (mid residual)
