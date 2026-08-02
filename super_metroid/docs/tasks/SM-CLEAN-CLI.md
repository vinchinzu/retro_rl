# TASK SM-CLEAN-CLI: `--clean` alias + uniform assist flags

## Recipe step
infra

## Model
Luna

## Wave type
implement

## Own files only
- `scripts/record/continuous.py`
- `routes/continuous.py` — early runners (`run_start_to_morph` / `bombs` /
  `spore` / `supers`) accept `unlimited_energy` / `unlimited_ammo` consistently
  without changing defaults
- `tests/` for CLI/flag wiring if present
- residual: `docs/tasks/SM-CLEAN-CLI-residual.md`

Do **not** invert defaults (assists remain **on**). Do not STATUS-promote.

## Context
- Today: `--no-unlimited-energy` / `--no-unlimited-ammo` exist.
- Morph/bombs runners historically take ammo only; energy flag may error via
  `run_to` when `supports_unlimited_energy` is false.
- Clean track needs a single ergonomic flag and no foot-guns.

## Read first
- `docs/CLEAN_TRACK.md`
- `scripts/record/continuous.py`
- `routes/continuous.py` (`run_to`, early runners)
- `routes/catalog.py` (`supports_unlimited_energy`)

## Do
1. Add `--clean` → sets both energy and ammo assists off (mutually document
   with the two long flags; if both specified, clean wins or error — pick one
   and document).
2. When clean: use clean artifact defaults (`SM-CLEAN-ARTIFACTS`).
3. Early runners: accept `unlimited_energy` kwarg as no-op when they only use
   ammo assist (or switch morph/bombs to `UnlimitedResourcesAssist` with both
   flags default True) — **behavior with defaults must match today**.
4. Print/report `intervention_class` or assist enable flags clearly in JSON.

## Acceptance
- [ ] Default run still resource-assisted
- [ ] `--clean` disables both assists
- [ ] `--to bombs --clean --no-video` does not error on energy flag
- [ ] Defaults for assisted morph/bombs telemetry unchanged when flags omitted

## Verify commands
```bash
uv run python super_metroid/scripts/record/continuous.py --help | rg clean
# smoke only if ROM available; else unit tests for flag parsing / run_to kwargs
```
