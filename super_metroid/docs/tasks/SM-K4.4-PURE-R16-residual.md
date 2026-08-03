## Residual — SM-K4.4-PURE-R16

### Result
PARTIAL (pure seats max-left fire solid and fires R15 open-loop; height
class improves to free-air ceiling min_y=228; Phase D / top still red)

### Files changed
- `routes/kpdr/bubble_mountain_params.py` — R16 lower shelves end on fire solid
  `(50,395)`; `SAVE_CLEAR_X_FRAMES` / `SAVE_EDGE_LEFT_FRAMES`
- `routes/kpdr/bubble_mountain.py` — lower stops on fire solid (not mid-iso
  float); mid_repin accepts save runway; avoid-door allows fire x≥25
- `routes/kpdr/bubble_mountain_mid.py` — max-left edge + X-clear left blocker;
  face-right settle before R15 open-loop
- `tests/test_k4_norfair_scaffold.py` — R16 shelf + fire-pin unit tests
- This residual; tip boards

### Load-bearing facts

| Fact | Detail |
|------|--------|
| Pure lower (R16 shelves) | lands fire solid ~(53,395) p9 true_ground |
| Prior lower | mid-iso float ~(107,365) → fell onto lip ~(79,427) |
| Mid-iso stop | **rejected** as lower success — steals fire path |
| Left blocker ~x37 | pure cannot walk left without X (missile); human free |
| Human pin R15 | Phase D GREEN min_y=134 mx200=300 from (27,395) p2 |
| Pure fire seat R15 | launches; free-air ceiling **min_y=228** mx200 short of 300 |
| Ceiling pocket | pure high class can hit min_y≈76–92 from door-edge pin, but mx200 hard-caps ~271 |
| Lip path R13 | min_y=260 phase_c still available as fallback when fire misses |

### Acceptance

- [x] Pure seats fire solid (not lip-first) from CATH-04 source
- [x] Unit green (26 scaffold tests)
- [x] No R13 Phase-C regress (`phase_c_hit=True` launched=True)
- [x] Height class improves: min_y=228 (was 260 lip)
- [ ] Full pure `top_reached` / Phase D — **red**
- [ ] Ordinary `0xB07A` — **red**

### Probe

```text
uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
# 26 passed

uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_pure_pin_r16.json --no-red-diag
# success=false
# max_x=370 min_y=228 phase_c_hit=True top_reached=False launched=True
# frames≈28714  (fire path; free-air ceiling, not R15 pin Phase D)
```

### Why top is still red

1. Pure fire seat (~x32–40 after max-left edge) is not velocity-matched to
   human pin (27,395) p2 — open-loop misses right-wall WJ contact class that
   clears the ceiling pocket into x≥300 @ y≤200.
2. From pure door-edge pins, height can reach min_y≈76–92 with shot-assisted
   WJ, but **mx200 hard-caps ~271** (ceiling pocket / KB at ~x219–271).
3. Human R15 double-WJ timings remain correct on `bubble_human_runway.state`.

### Rejected this session (do not re-ship without new pin)

| Attempt | Why |
|---------|-----|
| Prefer save over lip always (R15) | pure regress launched=False |
| Lip walk-left before HJ | pure loses Phase C |
| Mid-iso walk-left to fire | float falls to lip; never seats y395 solid |
| Arm-pump from human pin | never leaves ground class |
| Finish-only retune from pure x=20 p12 | mx200 hard-cap 271 all variants |
| Shoot-during-climb alone | KB drops; mx200 still 271 |

### Next action (required)

- **Next card ID:** `SM-K4.4-PURE-R17`
- **One change:** From pure fire-seat launch, earn **right-wall WJ contact** that
  pushes mx200 past 271 into Phase D (x≥300 y≤200) — or velocity-match pure
  seat to human pin (27,395) p2 so wired R15 double-WJ flips `top_reached`.
  Keep R16 fire lower + R13 floor-reclimb + R6 lip as regression.
- **Source:** `scratch/post_rising_tide_to_bubble_pure.state`
- **Human refs:** `scratch/bubble_human_runway.state` / peak / ceiling;
  `tasks/bubble_jump_try.json`

### Non-claims

- Did not STATUS-promote / continuous tip advance.
- Did not close SM-K4.4-PURE pure GREEN to Bat.
- min_y=228 free-air ≠ Phase D.
- Place finish from (300,y≤200) still holds separately.
