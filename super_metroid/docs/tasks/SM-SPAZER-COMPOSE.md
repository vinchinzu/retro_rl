# TASK SM-SPAZER-COMPOSE: Continuous tip --to spazer

## Recipe step
3 catalog tip + 4 continuous hops (scaffold; stabilize separate)

## Model
Luna

## Wave type
implement

## Own files only
- `routes/catalog.py` — tip_id `spazer` / artifact `start_to_spazer`
- `routes/continuous.py` — hops + `play_start_to_spazer` / `run_start_to_spazer`
- tests that lock tip registry aliases
- residual optional

Depends on: `SM-SPAZER-PURE` green + `SM-SPAZER-GRAPH`.

## Context
- Epic: [`SPAZER_EARLY.md`](SPAZER_EARLY.md)
- Secondary tip: power-on → Spazer collect (does **not** replace Frog / K4 tip).
- Prefix reuses Below Spazer continuous chain, then detour collect.

## Read first
- `routes/catalog.py` `below_spazer` tip block
- `routes/continuous.py` `play_start_to_below_spazer`
- `docs/ARCHITECTURE.md` tip-extension recipe

## Do
1. Tip `spazer` with aliases (`start_to_spazer`, `k2_2`).
2. Splits: Below Spazer splits + spazer detour hop names.
3. Wire CLI `continuous.py --to spazer` (no baseline claim until STAB).
4. Tests: tip resolves; no STATUS edit.

## Do not
- Claim integrity green without STAB record
- Fold into default post-Below hops (FOLD card)
- Overwrite unrelated tips

## Acceptance
- [ ] `--to spazer` selectable
- [ ] Unit/registry tests green
- [ ] Residual: next `SM-SPAZER-STAB`

## Verify commands
```bash
uv run pytest super_metroid/tests/test_start_to_supers.py -q -k spazer || true
# tip list / resolve smoke:
uv run python -c "from super_metroid.routes.catalog import get_continuous_tip; print(get_continuous_tip('spazer'))"
```
