## Residual — SM-K4.4-PURE-R15

### Result
PARTIAL (double-WJ open-loop clears Phase D on human runway pin; full pure
still R13 envelope — does not seat max-left fire window)

### Files changed
- `routes/kpdr/bubble_mountain_params.py` — max-left fire window, R15 double-WJ
  timings (`SAVE_WJ_*` / `SAVE_WJ2_*` / `SAVE_RUNWAY_FIRE_X`)
- `routes/kpdr/bubble_mountain_mid.py` — stationary bug-clear spray; double WJ +
  right-spin finish; fire only in max-left window (do not steal lip seats)
- `routes/kpdr/bubble_mountain.py` — save-runway docstring
- `tests/test_k4_norfair_scaffold.py` — x=27 seat + R15 param unit tests
- `scripts/probe/bubble_r15_*.py` — dev recon (runway place / human pin / WJ micro)
- This residual; tip boards as needed

### Human + pin evidence (load-bearing)

| Fact | Detail |
|------|--------|
| Source pin | `scratch/bubble_human_runway.state` ~(27,395) p2 |
| Peak pin | `scratch/bubble_human_peak.state` ~(245,163) |
| Ceiling pin | `scratch/bubble_human_ceiling.state` ~(242,146) |
| Human baseline | run21 + spin83 + L23 a6 R16 → min_y=158 max_x@200=251 **no Phase D** |
| R15 double WJ | L20 a4 R8 **L24 a2 R14** + right-spin → **top=True** mx200=301 min_y=138 |
| Best height+D | same family + longer alt-WJ / push → min_y=76–84, mx200≥300 |
| Ceiling lip | cleared vertically (y76–84 ≪ pocket y142); right push at apex hits x≥300 |
| Arm pump | desyncs human pin launch (min_y stays 395) — rejected for this seat |
| Longer runway frames | r≠21 or spin≠83 from pin fails wall contact — timing razor-tight |
| Place-at-rest | cannot reproduce (velocity) — only human pin / pure seat |

Maprando 2→7: Running Jump into Right Side Walljump Climb (no Speed, consecutive WJ).

### Acceptance

- [x] Named double-WJ trajectory that clears Phase D on human runway pin
- [x] Unit green (24 scaffold tests)
- [x] Full pure holds R13 envelope: min_y=260 phase_c_hit launched (no regress)
- [ ] Full pure top_reached — **red** (pure seats lip, not max-left fire x)
- [ ] Ordinary `0xB07A` — **red**

### Why pure top is still red

1. Pin Phase D needs fire seat **x∈[25,60] y~395** then exact run21/spin83/double WJ.
2. Full pure still lands the **solid lip** (~x79 y427) and takes R6 lip HJ → height
   class min_y=260 / Phase C via floor-reclimb — never the R15 open-loop.
3. Forcing save-before-lip or lip→walk-left stole lip seats and regressed pure
   (`launched=False` / `phase_c_hit=False` / min_y≈295–365). Reverted.

### Rejected this session (do not re-ship without new pin)

| Attempt | Why |
|---------|-----|
| Prefer save runway over lip always | pure regress launched=False min_y~365 |
| Lip walk-left bias 28f before HJ | pure phase_c_hit=False min_y=295 |
| Arm-pump run from human pin | never leaves ground class (min_y=395) |
| Run frames ≠21 / spin ≠83 from pin | miss wall contact; height collapses to ~228 |
| Clear with LEFT walk reset | walks into Save 0xB0DD |
| Morph/ceil crawl from peak pin | max_x high only after deep fall |

### Shipped open-loop (controller, fire window only)

```text
# seat x∈[25,60] y~395 stand-pin
Y×8                    # stationary bug clear
RIGHT+B ×21            # max run (human)
RIGHT+B+A ×83          # spin-glide
B+A×4, B, idle×2, LEFT×2
LEFT+A ×20             # WJ1
A ×4
RIGHT+A ×8             # flip1
LEFT+A ×24             # WJ2
A ×2
RIGHT+A ×14            # flip2
RIGHT+B+A ×56          # Phase D right-spin  → x≥300 y≤200 on pin
```

### Probe pins

```text
# Human runway pin (dev) — Phase D GREEN
# win_L20R8L24R14L0_spin: min_y=138 mx200=301 top=True

uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
# 24 passed

uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_pure_pin_r15.json --no-red-diag
# success=false
# max_x=408 min_y=260 phase_c_hit=True top_reached=False launched=True
# frames≈30806  (R13 envelope; lip path)
```

### Next action (required)

- **Next card ID:** `SM-K4.4-PURE-R16`
- **One change:** Seat pure on **max-left save fire window** (x∈[25,60] y~395)
  without regressing lip height-class / Phase C — then R15 open-loop should
  flip `top_reached`. Prefer natural walk-to-save from mid-iso / post-lower,
  **not** lip-first steal. Alternate: human re-record that finishes ordinary
  `0xB07A` for a full open-loop lock.
- Keep R15 double-WJ params + R13 floor-reclimb + R6 lip as regression.
- **Source:** `scratch/post_rising_tide_to_bubble_pure.state`
- **Human refs:** `tasks/bubble_jump_try.json`,
  `scratch/bubble_human_runway.state` / `peak` / `ceiling`

### Non-claims

- Did not STATUS-promote / continuous tip advance.
- Did not close SM-K4.4-PURE pure GREEN to Bat.
- Pin Phase D ≠ pure GREEN.
- Place finish from (300,y≤200) still holds separately (R14 finish recon).
