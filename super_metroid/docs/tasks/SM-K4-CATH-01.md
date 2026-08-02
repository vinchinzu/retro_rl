# TASK SM-K4-CATH-01: Business Center → Cathedral Entrance (pure)

## Recipe step

1. Pure controller. Geometry green before graph promote / continuous.

## Model

Luna

## Wave type

implement

## Own files only

- `routes/kpdr/k4_norfair.py` — replace only
  `play_business_to_cathedral_entrance` (leave other cathedral scaffolds).
- `routes/kpdr/registry.py` / `__init__.py` — already registered; no rename.
- `scripts/probe/kpdr.py` — add pure choice `business-to-cathedral-entrance`.
- `tests/test_k4_norfair_scaffold.py` — registration if needed.
- `docs/tasks/SM-K4-CATH-01-residual.md` — required PROCESS residual.

Do not edit `continuous.py`, `STATUS.md`, Frog Save→Speedway geometry, or
progression verification ranks (graph edge already exists `unverified`).

## Source and contract

- Preferred source: `custom_integrations/SuperMetroid-Snes/scratch/post_business_continuous.state`
- Expected room: `0xA7DE` Business Center (integrity-green `--to business` endpoint)
- Target: ordinary Cathedral Entrance `0xA7B3` through Business **top-right**
  blue door (exit right → entry left).
- Caps: Morph, Bombs, Missiles, Supers, Hi-Jump, Varia — **no Speed grant**.
- One named controller only.

## Context

- **Repath (planner 2026-08-01):** first Bubble visit is Cathedral climb, not
  Frog Speedway. `SM-K4.2-PURE` RED: Speedway Boost Blocks need Speed.
- Graph: `business_to_cathedral_entrance` → … → `rising_tide_to_bubble`.
- Continuous tip may still be Frog Save; from Frog, pure
  `frog_save_to_business` is a separate card. This card starts from **Business**.
- Process: pure-first; residual → next card + one change.

## Read first

- `routes/kpdr/k4_norfair.py` (`play_business_to_frog_save` settle pattern)
- `docs/SOURCE_STATES.md` `post_business_continuous`
- `docs/tasks/SM-K4.2-PURE-residual.md` (why repath)
- `docs/routes/ROUTE_KPDR.md` K4 section

## Do

1. Replace scaffold in `play_business_to_cathedral_entrance` with real climb/
   door geometry from Business mid/top to top-right door into `0xA7B3`.
2. Register pure `business-to-cathedral-entrance` in `kpdr.py`.
3. Pure-probe GREEN → write
   `scratch/post_business_to_cathedral_entrance_pure.state`.
4. Residual next: `SM-K4-CATH-02` (Cathedral Entrance → Cathedral) or SRC.

## Acceptance

- [ ] Source loads at `0xA7DE`
- [ ] Ordinary `0xA7B3` without warp / item grants
- [ ] Successor state only if pure GREEN
- [ ] Unit/registration green
- [ ] Residual PROCESS fields; no continuous/STATUS claim

## Verify

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure business-to-cathedral-entrance \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_business_continuous.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_business_to_cathedral_entrance_pure.state \
  --pin-json super_metroid/debug/business_to_cathedral_entrance_pure_pin.json

uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
```

## Residual routing

- GREEN → `SM-K4-CATH-02` (cathedral entrance → cathedral) or SRC catalog
- RED → `SM-K4-CATH-01-R1` one named phase; same Business source

Chain: CATH-01 → CATH-02 → CATH-03 (rising tide) → CATH-04 (bubble) →
Bat Cave → Speed Hall → Speed.
