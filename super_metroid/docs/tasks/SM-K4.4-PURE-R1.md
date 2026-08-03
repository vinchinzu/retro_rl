# TASK SM-K4.4-PURE-R1: Bubble mid walljump climb (one phase)

## Recipe step

1. Pure controller. Geometry green before graph promote / continuous.

## Model

Luna

## Wave type

implement

## Own files only

- `routes/kpdr/k4_norfair.py` — `play_bubble_to_bat_cave` mid phase (leave
  CATH geometry and other scaffolds).
- `scripts/probe/kpdr.py` — pure choice `bubble-to-bat-cave` (if not already).
- `tests/test_k4_norfair_scaffold.py` — registration.
- `docs/tasks/SM-K4.4-PURE-R1-residual.md` — required PROCESS residual.

Do not edit `continuous.py`, `STATUS.md`, CATH controllers, or progression ranks.

## Source and contract

- Preferred source:
  `custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state`
- Optional mid isolation:
  `scratch/post_bubble_mid_climb_pure.state` (lower phase pin ≈x112 y369)
- Expected room: `0xACB3` Bubble Mountain
- R1 target band: **stay in** `0xACB3`, reach top band **y≤200 / x≥300**
  without wrong-door exits
- Compose target (same controller, not R1 close): ordinary Bat Cave `0xB07A`
  via top-right green Super door (node 7)
- Caps: Morph, Bombs, Missiles, Supers (≥1), Hi-Jump, Varia — **no Speed**

## Context

- SM-K4.4-PURE residual: door open proven via place `(420,130)` (~153f);
  lower climb ~OK; hard gap mid walljump y≈350→150
- Maprando strat 154 (junction 9 → door 7): **Walljump with HiJump** —
  from platform near left (save-door height), run-jump to **cavity** right
  wall, walljump twice. Far outer right wall stalls at Single Chamber
  height (~y360).

## Do

1. One named phase: mid Bubble walljump climb (y≈350–400 → y≤200 / x≥300).
2. Keep wrong-door hard-avoid (Rising Tide / Save / Missiles Super / SC).
3. Pure probe; successor state only if full GREEN to Bat.
4. Residual PROCESS fields; no continuous/STATUS claim.

## Acceptance

- [ ] Source loads at `0xACB3` (CATH-04 pin band)
- [ ] Top band y≤200 / x≥300 still in `0xACB3` (R1 phase)
- [ ] Ordinary `0xB07A` without warp / item grants (compose)
- [ ] Successor state only if pure GREEN
- [ ] Unit/registration green
- [ ] Residual PROCESS fields; no continuous/STATUS claim

## Verify

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_to_bat_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_pure_pin.json

uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
```

## Residual routing

- GREEN → `SM-K4.5-PURE` (Bat Cave → Speed Hall) or compose/stabilize
- RED → next one named phase (mid WJ timing / platform pin / door compose)

### PROCESS residual (required on exit)

Result · Files changed · Verify paste · Acceptance · Residual risks ·
Next action · Non-claims · Probe pin.
