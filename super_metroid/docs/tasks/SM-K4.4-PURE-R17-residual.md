## Residual — SM-K4.4-PURE-R17

### Result
PARTIAL (RECON complete; double-WJ extracted as named primitive; pure Phase D
still red — integer-matched human seat is **not** velocity-matched)

### Files changed
- `routes/kpdr/bubble_mountain_primitives.py` — **new** micro-primitive library:
  stationary missile clear, walk-brake seat, `bubble_double_walljump_r15`,
  full R15 open-loop helper
- `routes/kpdr/bubble_mountain_params.py` — R17 stationary-X / human-seat params
- `routes/kpdr/bubble_mountain_mid.py` — call named double-WJ after fire spin
  (R16 fire/lip control flow preserved — no Phase C regress)
- `tests/test_k4_norfair_scaffold.py` — R17 param + primitive import tests
- `docs/tasks/BUBBLE_TECHNIQUES.md` — living technique → code map
- This residual; tip boards

### RECON load-bearing facts (velocity-matched)

| Fact | Detail |
|------|--------|
| Human pin | `scratch/bubble_human_runway.state` (27,395) **p2** |
| Human R15 open-loop | run21 + spin83 + L20 a4 R8 L24 a2 R14 + right-spin → **top=True** mx200=301 min_y=138 |
| Human first WJ | pose **132** at ~(264,297) |
| Human post_run | (54,395) p9 after 21f RIGHT+B (≈1.3 px/f) |
| Pure land (R16) | ~(53,395) p9 true_ground after lower |
| Pure LEFT+X walk | KB pose **138**, sticks ~x37, runway=False |
| Pure stationary X then walk | reaches x~26–30 grounded (blocker clear without KB) |
| Pure dump (27,395) **p2** | integer-matched human seat; post_run often (42–55,395) p9 |
| Pure from that dump + R15 | **no pose 132**; best free-air min_y≈228–255; mx200=0 |
| Pure run windup | 21f from pure seat gains ~15–28 px inconsistently vs human 27 px |
| Ceiling pocket | when height is good (min_y≲108 door-air class) mx200 hard-caps ~268–276 |
| Closed-loop WJ early | ruins even human path if triggered at wall_band before open-loop timing |
| Product pure R17 close | max_x=370 min_y=228 **phase_c_hit=True** launched=True top=False (R16 envelope) |

Diff critical window (human vs pure dump, aligned to R15 script):

| Phase | Human | Pure dump (27,395)p2 |
|-------|-------|----------------------|
| post_run | (54,395) p9 | (42–55,395) p9 (variable) |
| spin peak | min_y≈228 @ x~177 | min_y≈228–255, often short of wall |
| first WJ | p132 ~(264,297) | **never** |
| Phase D | mx200=301 top=True | mx200=0 top=False |

Debug artifacts (dev only, not hop GREEN):

- `debug/bubble_r17_recon.json` — pure vs human phase marks
- `debug/bubble_longjump_vxvy.json` — dx/dy by phase (RAM vx≈0 while running)
- `debug/bubble_to_bat_pure_pin_r17.json` — full pure pin

### Acceptance

- [x] Full-state RECON: pure fire path vs human double-WJ vs place-isolated
- [x] Named double-WJ primitive with documented entry guards
- [x] Unit green (27 scaffold tests)
- [x] No R16 Phase-C regress (`phase_c_hit=True` launched=True min_y=228)
- [ ] Full pure `top_reached` / Phase D — **red**
- [ ] Ordinary `0xB07A` — **red**

### Probe

```text
uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
# 27 passed

uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_pure_pin_r17.json --no-red-diag
# success=false
# max_x=370 min_y=228 phase_c_hit=True top_reached=False launched=True
# frames≈28714  (R16 envelope held; Phase D still red)
```

Human pin still GREEN for Phase D (dev isolation only):

```text
# bubble_save_runway_open_loop_r15 from bubble_human_runway.state
# → top=True end ~(301,143)
```

### Why Phase D is still red (not “more WJ period”)

1. **Velocity/subpixel mismatch** at integer-matched (27,395)p2: pure dumps do
   not reproduce human run windup or wall contact class (no p132).
2. **LEFT+X walk** is a known pure fail (KB); stationary clear helps seat but
   does not by itself deliver human post_run physics.
3. **mx200 pocket ~271** remains when height is earned without right-wall WJ.
4. Forbidden without new pin: another open-loop period/y-window tweak on the
   same arc. Next class must change approach velocity or closed-loop contact
   detection that actually earns p132.

### Rejected this session (do not re-ship without new pin)

| Attempt | Why |
|---------|-----|
| Aggressive LEFT+X + A-recover seat | launches to air p25; or enters Save 0xB0DD |
| Brake-walk mid control rewrite | pure regress launched=False / Save door |
| Closed-loop WJ on wall_band early | desyncs human open-loop (no Phase D) |
| Fixed run/spin grid on pure dump | 500+ trials, zero tops, zero p132 |
| Adaptive run to post_run x=54 | matches x sometimes; still no WJ |

### Shipped

```text
# bubble_mountain_primitives.bubble_double_walljump_r15
# after SAVE_RUN + SAVE_SPIN on fire seat:
B+A×4, B, idle×2, LEFT×2
LEFT+A ×20, A×4, RIGHT+A ×8   # WJ1
LEFT+A ×24, A×2, RIGHT+A ×14  # WJ2
RIGHT+B+A ×56                 # Phase D push
```

### Next action (required) — **continue spine, do not park**

- **Next card ID:** `SM-K4.4-PURE-R18`
- **Card:** [`SM-K4.4-PURE-R18.md`](SM-K4.4-PURE-R18.md)
- **One change class:** pure **velocity dump** at natural post-spin / wall-approach
  from the continuous-like fire path, then **short-horizon search** (or one
  closed-loop WJ trigger on **pose 132**) until `top_reached` on that dump;
  recompose full pure from CATH-04 source.
- **Source:** `scratch/post_rising_tide_to_bubble_pure.state`
- **Human refs:** `scratch/bubble_human_runway.state` / peak / ceiling
- **Techniques:** [`BUBBLE_TECHNIQUES.md`](BUBBLE_TECHNIQUES.md)
- **Ladder:** [`SM-K4.4-PHASE-LADDER.md`](SM-K4.4-PHASE-LADDER.md) § R18

### Non-claims

- Did not STATUS-promote / continuous tip advance.
- Did not close SM-K4.4-PURE pure GREEN to Bat.
- Integer seat match ≠ pure Phase D.
- Primitive extract alone is not hop GREEN.
- Did **not** park Bubble→Bat — spine next is R18.
