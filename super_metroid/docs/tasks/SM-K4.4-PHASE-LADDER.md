# Bubble → Bat phase ladder (`0xACB3` → `0xB07A`)

Concrete hard-room split for **SM-K4.4-PURE** / R-series. General rules:
[`HARD_ROOM_SPLITS.md`](HARD_ROOM_SPLITS.md).

## Product contract

- Caps: Morph, Bombs, Missiles, Supers (≥1), Hi-Jump, Varia — **no Speed**.
- Full pure source (hop GREEN only from here):
  `scratch/post_rising_tide_to_bubble_pure.state` (CATH-04).
- Continuous tip remains Frog Save until planner compose/stabilize after pure
  GREEN to Bat.

## Phase checklist

| Phase | Name | Acceptance (full pure unless noted) | Status (post-R10) |
|-------|------|--------------------------------------|-------------------|
| **A** | Mid pin | `standing_mid_pinned=True` | **green** (R5) |
| **B** | Height class | `min_y≤280` after lip launch | **green** (R6; R9/R10 hold 260) |
| **C** | Usable right contact | first `x∈[300,395]`, `y∈[200,430]` in Bubble (not thrash-only `max_x`) | **red** — bottleneck |
| **D** | Top band | `y≤200` and `x≥300` | place-proven; natural red |
| **E** | Bat door | ordinary `0xB07A` | blocked on D |

Place-proven finish (isolation only — not natural proof):

- Air `(360, y≤370)` period-8 WJ → top.
- Grounded shelves `(360–388, 331–363)` LEFT charged HJ → top.
- Right hits Single Chamber outer-wall trap — shelf hop is **LEFT**.

Natural gap (load-bearing):

- Lip peak still `~(150, 260)`; at shelf height y≈360 pure is only ~x211.
- First `x≥340` still ~y467 (below usable Phase-C altitude).
- One-shot lip→right air band is too far at fall rate.
- R10 mid-high window `y≤450` engages earlier but does **not** create Phase C.

## Frozen green work

Do **not** edit without height/mid regression:

- R5 lower-left multi-hop (`_BUBBLE_LOWER_SHELVES`)
- R6 solid lip launch (`_BUBBLE_LIP_*`)
- Wrong-door avoid + cavity x cap (`_BUBBLE_CAVITY_X_MAX`)

Banned without new pin evidence (known fails):

- Lip run-up (pure min_y regress ~365)
- Floor WJ climb (unstable)
- Left-column top hunt (no solids y~174–394 left)
- Further `_BUBBLE_MIDHIGH_Y` / period-only tweaks without Phase-C pin

## Work shape for R11+

Stagnation: R7–R10 PARTIAL on top while Phase C still red → **triage**.

### R11a — RECON / capture (preferred first)

1. Full pure with Phase-C dump + stop:

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --dump-phase-c super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_right_contact_pure.state \
  --stop-at-phase-c --no-red-diag \
  --pin-json super_metroid/debug/bubble_phase_c_pin.json
```

2. If Phase C **never** hits: residual **BLOCKED on trajectory** — next card
   is mid-nub / launch redesign, **not** “WJ harder.”
3. If Phase C hits: record pin `(x,y,pose,vx,vy)` + state path. That dump is
   a **dev handoff**, not hop GREEN.

### R11b — IMPL climb-only (one named change)

From right-contact handoff **or** velocity-matched place:

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_right_contact_pure.state \
  --start-phase climb --no-red-diag
```

One named change: stabilize WJ / reseat into right air band
`(x≥340, y∈[280,370])` or grounded shelf, then reuse R9 LEFT shelf HJ /
period-8 WJ.

Climb-only top is **not** full hop GREEN.

### R11c / full compose — Phase D+E on CATH-04 source

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_to_bat_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_pure_pin.json
```

Only this path (full pure, no place, no climb-only) can close SM-K4.4-PURE.

## Code hooks

| Symbol / flag | Role |
|---------------|------|
| `bubble_phase_c_usable_right_contact` | Phase-C predicate |
| `BubblePhaseStop` | early exit for capture (`--stop-at-phase-c`) |
| `play_bubble_to_bat_cave(..., start_phase=)` | `auto` full path; `climb` skip lower/repin/launch |
| `--dump-phase-c` | save first Phase-C state (needs probe env) |
| `--start-phase climb` | climb-only pure probe |
| `--stop-at-phase-c` | stop at first Phase C (diagnostic success) |

## Living cards

| Card | Phase focus |
|------|-------------|
| R5–R6 residuals | A / B green history |
| R7–R10 residuals | approach thrash; top still red |
| **`SM-K4.4-PURE-R11`** | Phase C → D (capture + climb + full recheck) |

## Non-claims

- No continuous / STATUS from phase tooling.
- Place air success ≠ natural Phase C.
- Continuous tip stays Frog Save until Bat pure GREEN + planner compose.
