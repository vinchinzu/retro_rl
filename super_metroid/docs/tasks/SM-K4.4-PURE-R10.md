# TASK SM-K4.4-PURE-R10: Mid-high approach into right shelf band then top

## Recipe step

1. Pure controller. Geometry green before graph promote / continuous.

## Model

Luna

## Wave type

implement

## Own files only

- `routes/kpdr/k4_norfair.py` — after R9 open-loop, **one named intermediate
  re-seat / multi-hop** that puts Samus into place-proven right air band
  `(x≥340, y∈[280,340])` before falling past shelf height, then reuse R9
  period-8 WJ / no-A shelf land + LEFT shelf→top HJ. Leave R5 lower + R6 lip
  alone unless height regresses. Prefer offline recon first.
- `docs/tasks/SM-K4.4-PURE-R10-residual.md` — required PROCESS residual.
- Optional diagnostic only: `scripts/probe/` recon helper (not pure proof).

Do not edit `continuous.py`, `STATUS.md`, CATH controllers, or progression ranks.

## Source and contract

- Preferred source:
  `custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state`
- Expected room: `0xACB3` Bubble Mountain
- R10 target: full pure `top_reached=True`; ordinary `0xB07A` if door lands
- Caps: Morph, Bombs, Missiles, Supers (≥1), Hi-Jump, Varia — **no Speed**

## Context

- R9 residual: max_x=389 min_y=260 pin True top red; frames≈7.3k.
  Place shelves + air `(360,320)` period-8 WJ one-hop to top in isolation.
  Lip peak ~(130–165,260) reaches x≥250 only at y≈400+ (below shelf band).
  Cavity floor hops cannot climb to y≤390 (ceiling ~435).

## Do

1. One named change: intermediate mid-high approach into right air/shelf band.
2. Keep R5/R6/R9 shelf LEFT HJ + period-8 WJ; wrong-door avoid; cavity x cap.
3. Pure probe; successor only if full GREEN to Bat.
4. Residual PROCESS fields; no continuous/STATUS claim.

## Acceptance

- [x] Source loads at `0xACB3`
- [x] Full pure pin + min_y≤280 (no regression) — min_y=260
- [ ] Full pure top band y≤200 x≥300 in `0xACB3` — **not achieved**
- [ ] Ordinary `0xB07A` if door lands; successor only if pure GREEN
- [x] Unit/registration green
- [x] Residual PROCESS fields; no continuous/STATUS claim

**Result: PARTIAL** — see `SM-K4.4-PURE-R10-residual.md`. Next: `SM-K4.4-PURE-R11`.

## Verify

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_to_bat_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_pure_pin.json

uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
```

## Residual routing

- GREEN → compose/stabilize
- RED → next one named door/cross knob
