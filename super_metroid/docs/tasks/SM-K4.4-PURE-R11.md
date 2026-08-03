# TASK SM-K4.4-PURE-R11: Phase C right contact → top (ladder)

## Recipe step

1. Pure controller. Geometry green before graph promote / continuous.

## Model

Luna

## Wave type

implement (after RECON capture if Phase C never hit on full pure)

## Own files only

- `routes/kpdr/k4_norfair.py` — **one named change** on climb/right-contact
  only (freeze R5 lower + R6 lip). Prefer right-wall WJ from natural first
  contact `(x≈300–380, y≈400–430)` **or** frame-tight mid-nub chain into
  right air band `(x≥340, y∈[280,370])`. Then reuse R9/R10 LEFT shelf HJ /
  period-8 WJ. **No lip run-up. No mid-high window-only tweaks.**
- `docs/tasks/SM-K4.4-PURE-R11-residual.md` — required PROCESS residual.
- Optional diagnostic: `scripts/probe/` recon helper (not pure proof).

Do not edit `continuous.py`, `STATUS.md`, CATH controllers, or progression ranks.

## Read first

- [`SM-K4.4-PHASE-LADDER.md`](SM-K4.4-PHASE-LADDER.md) — phase A–E checklist
- [`HARD_ROOM_SPLITS.md`](HARD_ROOM_SPLITS.md) — stagnation / RECON→IMPL rules
- `SM-K4.4-PURE-R10-residual.md` — max_x=349 min_y=260 pin; top red

## Source and contract

- Preferred source (full hop GREEN only):
  `custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state`
- Expected room: `0xACB3` Bubble Mountain
- Caps: Morph, Bombs, Missiles, Supers (≥1), Hi-Jump, Varia — **no Speed**

## Work shape (do not thrash full pure first)

R7–R10 already spent the 3-PARTIAL stagnation budget on the same top
checkbox while **Phase C** stayed red. Follow the ladder:

### 1) RECON / capture (R11a — do this first if no handoff state)

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --dump-phase-c super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_right_contact_pure.state \
  --stop-at-phase-c --no-red-diag \
  --pin-json super_metroid/debug/bubble_phase_c_pin.json
```

- Exit 0 + `phaseCHit` → pin `(x,y,pose,vx,vy)` + handoff state (dev only).
- Phase C never hits → residual **BLOCKED on trajectory** (mid-nub / launch
  redesign), **not** another WJ period.

### 2) IMPL climb-only (R11b — one named change)

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_right_contact_pure.state \
  --start-phase climb --no-red-diag
```

Climb-only top is **not** hop GREEN.

### 3) Full pure recheck (only claim path)

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_to_bat_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_pure_pin.json

uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
```

## Context

- R10: mid-high open-loop `y≤450`; full pure pin True, max_x=349 min_y=260,
  top red. Place air `(360,y≤370)` period-8 WJ → top still holds.
- Lip peak ~(150,260); first x≥340 ~y467. Approach gap, not finish gap.
- Place once: `(380,400)` WJ → shelf — must re-check with **velocity** match.

## Do

1. Capture Phase C or prove it never hits (trajectory residual).
2. One named climb/right-contact change; freeze R5/R6.
3. Climb-only iterate, then full pure for GREEN claim.
4. Residual PROCESS fields; no continuous/STATUS claim.

## Acceptance

- [ ] Phase C: natural usable right contact **or** honest BLOCKED trajectory
- [ ] Source loads at `0xACB3` (full pure)
- [ ] Full pure pin + min_y≤280 (no regression)
- [ ] Full pure top band y≤200 x≥300 in `0xACB3` (Phase D)
- [ ] Ordinary `0xB07A` if door lands; successor only if pure GREEN (Phase E)
- [ ] Unit/registration green
- [ ] Residual PROCESS fields; no continuous/STATUS claim
- [ ] Climb-only / phase-capture not presented as hop GREEN

## Residual routing

- GREEN full pure → compose/stabilize (planner)
- PARTIAL Phase C only → next climb knob with handoff state
- BLOCKED trajectory → mid-nub / launch redesign card (not period tweak)
