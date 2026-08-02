# TASK SM-K4-CATH-02: Cathedral Entrance → Cathedral (pure)

## Recipe step

1. Pure controller. Geometry green before graph promote / continuous.

## Model

Luna

## Wave type

implement

## Own files only

- `routes/kpdr/k4_norfair.py` — replace only
  `play_cathedral_entrance_to_cathedral` (leave CATH-01 + other scaffolds).
- `scripts/probe/kpdr.py` — add pure choice `cathedral-entrance-to-cathedral`.
- `tests/test_k4_norfair_scaffold.py` — registration if needed.
- `docs/tasks/SM-K4-CATH-02-residual.md` — required PROCESS residual.

Do not edit `continuous.py`, `STATUS.md`, CATH-01 geometry, or progression
verification ranks.

## Source and contract

- Preferred source:
  `custom_integrations/SuperMetroid-Snes/scratch/post_business_to_cathedral_entrance_pure.state`
- Expected room: `0xA7B3` Cathedral Entrance (CATH-01 pure GREEN successor:
  reload ≈ x=39 / y=139 / pose=11)
- Target: ordinary Cathedral `0xA788` through **right red Super door**
  (node 2, block `[47, 7]`, orientation right).
- Caps: Morph, Bombs, Missiles, Supers (≥5), Hi-Jump, Varia — **no Speed**.
- One named controller only.

## Context

- **Repath:** first Bubble = Cathedral climb (no Speed).
- Prior: `SM-K4-CATH-01` GREEN (~959f Business → Cathedral Entrance).
- Chain: CATH-01 → **CATH-02** → CATH-03 (rising tide) → CATH-04 (bubble).
- Room is ~3 screens wide; spawn is left door lip — run right, select Supers,
  open red door, settle ordinary in `0xA788`.

## Read first

- `routes/kpdr/k4_norfair.py` (`play_business_to_cathedral_entrance` door pattern)
- `routes/controller_common.py` (`select_weapon` — supers = 2)
- `docs/tasks/SM-K4-CATH-01-residual.md`
- Graph edge `connection_219_a7b3_2_to_a788_1` in `maps/full_room_graph.json`

## Do

1. Replace scaffold in `play_cathedral_entrance_to_cathedral` with real geometry.
2. Register pure `cathedral-entrance-to-cathedral` in `kpdr.py`.
3. Pure-probe GREEN → write
   `scratch/post_cathedral_entrance_to_cathedral_pure.state`.
4. Residual → `SM-K4-CATH-03` or R1.

## Acceptance

- [ ] Source loads at `0xA7B3`
- [ ] Ordinary `0xA788` without warp / item grants
- [ ] Successor state only if pure GREEN
- [ ] Unit/registration green
- [ ] Residual PROCESS fields; no continuous/STATUS claim

## Verify

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure cathedral-entrance-to-cathedral \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_business_to_cathedral_entrance_pure.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_cathedral_entrance_to_cathedral_pure.state \
  --pin-json super_metroid/debug/cathedral_entrance_to_cathedral_pure_pin.json

uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
```

## Residual routing

- GREEN → `SM-K4-CATH-03` (Cathedral → Rising Tide) or SRC catalog
- RED → `SM-K4-CATH-02-R1` one named phase; same entrance source
