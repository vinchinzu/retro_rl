## Residual — rr-tlaq Phantoon fight pure (0xCD13)

**Status:** window 1 charge 300 GREEN. Stay-left morph rain: **died** in
`$D788` vs (48, 96). Full fight RED.
**Pin:** `scratch/post_ws_basement_to_phantoon.state`
**Probe:** `window --windows 2 --weapon beam --wait 6000`
**Report:** `scratch/phantoon_window_beam_w2_stayleft.json`

Do **not** STATUS-promote. Default CLI stays `ice`. Super-spray is not a hit.
Do **not** start another 16k until a measured W2 chip. Do not fire the
right fig-8 at x=219 (the body). Do not retry x≥230.

### Public policy (wiki)

https://wiki.supermetroid.run/Phantoon
https://wiki.supermetroid.run/Phantoon#Phantoon_First

KPDR beginner 4-round: charge when the eye opens, two more, repeat.
Charge+Missiles: 2 + 2 + charge × 4. Super **enrages**. Pin boots with
supers selected — `select_weapon` to missiles/beam before the first open.

Eye is **slot 1** (not enemy0 spritemap). Live flags:
- body function `$0FB2` `0xD60D` / `0xD767` / `0xD788` (open windows)
- eye IL `$0FD2` in `0xCC53–0xCC7A` (open anim) or `0xCC9D–0xCCD6` (look)
- intro is `$0FB2=0xD4A9…D5E7`; figure-8 starts ~f793; first open ~f1513–1594

### What works (verified this pass)

- **Seat** from the enter pin: idle fall to `(39, 187)` p1.
- **Window 1** charge **300** at `(104, 149)` p43 vs `(120, 108)` dy=41.
- Skip-right: `charge_window_ok` is left fig-8 only (park x at func change).
- Stay-left morph: x stayed **52–57** the whole skip/rain (no room-cross).
- Assist off: `energy_writes 0`, `missile_writes 0`.

### What fails

1. **Left morph cannot tank rain — halt.** Skip right open `(203, 83)`.
   Hold morph `x≤56` `(53–56, 201)` p29/65. Health 239→0 ~f4238.
   Death: pose **65** `(56, 201)`, `$D788`, body **(48, 96)**. Also −20
   on `$D788` vs (168, 64) 39→19. No left fig-8. Flames still hit the
   left corner (~−20 every few hundred frames through `D5E7`/`D82A`/`D788`).
2. **Right fig-8 at x=219 is the body.** Do not retry x≥230 (wall, p137).
3. **Swap-across-room** previously died faster (rolled through the wave).
   Super unused (correct).

### Rain dump (every ~30f, x=52–57)

239 until ~f2114 → 219 → 199 → 179 (right open skipped) → `D82A` 159…99 →
`D788` (200,114) 79 → (128,96) 59 → (88,64) 39 → (168,64) 19 → (48,96) **0**.

### Next actions (do not start another 16k first)

1. Left morph cannot tank this seed's rain at 239 HP. Next is **snipe /
   farm** so W2 starts closer to 299, or a **flame-snipe** from the left
   corner that is not a jump under (128, 96). Halt at first miss. Do not
   fire x=219.
2. Dual-green `scratch/post_phantoon_poweron.state` only after a kill
   (do **not** clobber `post_phantoon_defeated.state`).

```bash
# From repo root. Window 1 should still chip 300:
uv run python snes/super_metroid/scripts/probe/phantoon_combat.py window --windows 1 \
  --weapon beam --report snes/super_metroid/scratch/phantoon_window.json
```

### Non-claims

- Did not STATUS-promote past Ice
- Did not change `DEFAULT_CONTINUOUS_TIP`
- Did not write `recordings/ws.json`
- Did not append to `POST_ICE_SPINE` / `WS_ONLY_HOPS` / `--to ws`
- Did not close `rr-g3nj`
- Did not rewrite `play_ws_entrance_to_main` / `play_ws_main_to_basement` /
  `play_ws_basement_to_phantoon`
- Did not close `rr-tlaq` (full kill still RED)
