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
| **C** | Usable right contact | first `x∈[300,395]`, `y∈[200,430]` in Bubble (not thrash-only `max_x`) | **green** (R13 floor-reclimb; pin ~`(301,429)` marginal) |
| **D** | Top band | `y≤200` and `x≥300` | place-proven; natural red — bottleneck |
| **E** | Bat door | ordinary `0xB07A` | blocked on D |

Place-proven finish (isolation only — not natural proof):

- Air `(360, y≤370)` period-8 WJ → top.
- Grounded shelves `(360–388, 331–363)` LEFT charged HJ → top.
- Right hits Single Chamber outer-wall trap — shelf hop is **LEFT**.

Natural gap (load-bearing):

- Lip peak still `~(150, 260)`; one-shot lip→right air band too far at fall rate.
- R13 floor-reclimb after height class hits Phase C ~`(301,429)` (predicate green).
- That contact is **marginal**: place climb from dump best min_y≈427; no shelf.
- Recoverable finish still needs grounded shelf ~`(380,390)` or air `(360,y≤370)`.
- **R14 bottleneck:** raise right contact into shelf/air band, then Phase D.

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

Modules: `bubble_mountain_params.py` (constants), `bubble_mountain.py`
(predicates / lower / repin / door / product), `bubble_mountain_mid.py`
(single mid loop, start=launch|climb). Re-exported from `k4_norfair` for
compat. Product `play_bubble_to_bat_cave(session)` has **no** recon kwargs —
probe maps CLI flags to dev helpers.

| Symbol / flag | Role |
|---------------|------|
| `bubble_phase_c_usable_right_contact` | Phase-C predicate |
| `bubble_phase_d_top_band` / `bubble_phase_d_near_top` | Phase-D top checks |
| `bubble_on_launch_lip` / `bubble_on_right_shelf` | lip / shelf seats |
| `BubblePhaseStop` | early exit for capture (`--stop-at-phase-c`) |
| `play_bubble_to_bat_cave(session)` | product full pure (session only) |
| `play_bubble_climb_from_handoff` | dev: skip lower/repin/launch |
| `play_bubble_from_top_door` | dev: Super door only |
| `play_bubble_to_bat_cave_with_phase_capture` | probe full path + dump/stop |
| `bubble_lower_to_mid_pin` / `bubble_run_mid` / `bubble_top_super_door` | phase helpers (mid is one call) |
| `--dump-phase-c` | save first Phase-C state (needs probe env) |
| `--start-phase climb` | climb-only pure probe |
| `--stop-at-phase-c` | stop at first Phase C (diagnostic success) |

## Living cards

| Card | Phase focus |
|------|-------------|
| R5–R6 residuals | A / B green history |
| R7–R10 residuals | approach thrash; top still red |
| **`SM-K4.4-PURE-R11`** | spin-apex false-land fix; Phase C still red |
| **`SM-K4.4-PURE-R12`** | extract lip stand_pin restore (pure R11 envelope); trajectory IMPL still open — fall-gated WJ Phase C only with height regress |
| **`SM-K4.4-PURE-R13`** | floor-reclimb after height class → **Phase C green** on full pure; top still red (marginal y≈429 contact) |

## Non-claims

- No continuous / STATUS from phase tooling.
- Place air success ≠ natural Phase C.
- Continuous tip stays Frog Save until Bat pure GREEN + planner compose.
