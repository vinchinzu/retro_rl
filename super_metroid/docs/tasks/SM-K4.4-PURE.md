# TASK SM-K4.4-PURE: Bubble Mountain → Bat Cave (pure)

## Recipe step

1. Pure controller. Geometry green before graph promote / continuous.

## Model

Luna

## Wave type

implement

## Own files only

- `routes/kpdr/k4_norfair.py` — add `play_bubble_to_bat_cave` (leave CATH
  geometry and other scaffolds).
- `scripts/probe/kpdr.py` — pure choice `bubble-to-bat-cave`.
- `tests/test_k4_norfair_scaffold.py` — registration.
- `docs/tasks/SM-K4.4-PURE-residual.md` — required PROCESS residual.

Do not edit `continuous.py`, `STATUS.md`, CATH controllers, or progression ranks.

## Source and contract

- Preferred source:
  `custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state`
- Expected room: `0xACB3` Bubble Mountain (CATH-04 pure GREEN successor:
  room=`0xACB3` pose=25 x=39 y=634 door_transition=0, **2609 frames**)
- Target: ordinary Bat Cave `0xB07A` through Bubble **top-right green Super
  door** (node 7, block `[31, 7]`, orientation right; graph
  `connection_233_acb3_7_to_b07a_1`; progression edge `bubble_to_bat_cave`,
  requires `super_missiles`).
- Caps: Morph, Bombs, Missiles, Supers (≥1), Hi-Jump, Varia — **no Speed**.
- One named controller only: `play_bubble_to_bat_cave`.

## Context

- Cathedral pure stack CATH-01…04 GREEN; first Bubble pure closeout done.
- Entry is mid-left node 3 (Rising Tide), not farm-bottom or save.
- Climb is 2×4 screens: mid-left y≈634 → top-right y≈112. Maprando: junction
  (node 9) → top-right uses **Walljump with HiJump**. Avoid left doors
  (Rising Tide y≈624, Save y≈368, Missiles Super y≈112) and mid-right Single
  Chamber (y≈368).
- Proven offline: place `(420,130)` Super-opens door → Bat in ~153f. Climb gap
  y≈350→150 is the hard phase.

## Do

1. Real geometry controller from pure Bubble successor to ordinary Bat Cave.
2. Register pure `bubble-to-bat-cave`.
3. Pure-probe GREEN → `scratch/post_bubble_to_bat_pure.state`.
4. Residual → Speed Hall pure or R1.

## Living residuals (spine)

| Card | Status |
|------|--------|
| R13–R18 | Phases A–C + fire seat; R18 Phase D enemy AI root cause |
| **R19** | **GREEN closeout** — [`SM-K4.4-PURE-R19-residual.md`](SM-K4.4-PURE-R19-residual.md) |

Techniques: [`BUBBLE_TECHNIQUES.md`](BUBBLE_TECHNIQUES.md) · ladder
[`SM-K4.4-PHASE-LADDER.md`](SM-K4.4-PHASE-LADDER.md).

## Acceptance

- [x] Source loads at `0xACB3` (CATH-04 pin band)
- [x] Ordinary `0xB07A` without warp / item grants
- [x] Successor state only if pure GREEN
- [x] Unit/registration green (30 scaffold)
- [x] Residual PROCESS fields; no continuous tip promote

## Verify

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_to_bat_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_pure_pin_r19.json --no-red-diag
# success=true roomIdHex=0xB07A frames=2012

uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
# 30 passed
```

## Residual routing

- **GREEN (R19)** → next: `SM-K4.4-GRAPH` compose / Bat → Speed Hall pure
  from `scratch/post_bubble_to_bat_pure.state`
- Continuous/STATUS tip remains Frog Save until planner compose

### PROCESS residual (required on exit)

Authoritative closeout: [`SM-K4.4-PURE-R19-residual.md`](SM-K4.4-PURE-R19-residual.md).
