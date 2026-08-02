# TASK T4-CLEAN-CLI: `--clean` / assist-disable flags on continuous recorder

## Recipe step
infra

## Model
Luna

## Wave type
implement

## Own files only
- `scripts/record_full_hard_run.py` — CLI flags + assist gating only
- `tests/` — flag / path tests
- `docs/tasks/T4-CLEAN-CLI-residual.md`

Depends: prefer `T4-CLEAN-ARTIFACTS` path helper first (or land together).

## Context
- Today assists are hard-coded on (e-HP threshold + form-2 iframe hold).
- Clean needs: disable both; keep counting natural damage; fail integrity if
  any assist would have fired when clean (see INTEGRITY card).
- Defaults must stay assisted.
- Super Metroid analog: `SM-CLEAN-CLI` (`--clean`).

## Read first
- `docs/CLEAN_TRACK.md`
- `docs/ASSIST_CONTRACT.md`
- `scripts/record_full_hard_run.py` (assist write sites ~e-HP + iframe)

## Do
1. Add `--clean` that disables emergency HP **and** form-2 iframe hold.
2. Optionally expose long form: `--no-emergency-hp` and `--no-iframe-hold`
   (either alone is not full Clean; both required for Clean claim).
3. When `--clean` (or both long forms), default artifacts to clean stems.
4. Document flags in module docstring.
5. Residual → `T4-CLEAN-INTEGRITY` if asserts not done here.

## Acceptance
- [ ] Default run still applies both assists
- [ ] `--clean` applies zero resource/protection writes
- [ ] Clean path defaults isolated
- [ ] Residual PROCESS fields

## Verify commands
```bash
uv run python -m tmnt_iv.scripts.record_full_hard_run --help
# unit tests for flag matrix
uv run pytest tmnt_iv/tests/ -q -k "clean or assist or record"
```
