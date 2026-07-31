# TASK SM-TIGHTEN-04A: Guarded main-shaft entry settle (Patch A only)

## Recipe step
efficiency implement (early KPDR — **not** STATUS)

## Model
Luna

## Own files only
- `routes/spore_spawn_controller.py` (**edit** Patch A site only)
- `docs/tasks/SM-TIGHTEN-04A-note.md` (**create**)

Do **not** edit continuous.py, STATUS, kpdr reverse, progression verification.
Do **not** implement Patch B or C in this card.

## Context
- Report: `docs/tasks/SM-TIGHTEN-04-report.md`
- Split `green_brinstar_main_shaft` ~2,806f; Patch A targets
  `_hold(session, 1_000, reason="main_shaft_entry_settle")` → guarded settle
  with timeout ~300–400f and standing/x band from report
- Continuous prefix for verify is shorter than kraid: `--to spore` (still
  multi-minute — **planner residual**, not required for EXIT)
- Aggressive vs Wave-3: this **edits a live continuous controller** on the
  verified spine (pre-Warehouse), so residual must be ruthless about non-claims

## Read first (all)
- `docs/tasks/SM-TIGHTEN-04-report.md` Patch A
- `routes/spore_spawn_controller.py` around `main_shaft_entry_settle` /
  `play_main_shaft_to_spore_spawn` / elevator exit into shaft
- `routes/controller_common.py` wait helpers you may reuse
- `docs/STATUS.md` (read only — do not edit)

## Do
1. Replace fixed 1000f `main_shaft_entry_settle` with a **guarded** settle:
   - Poll until standing-ish pose + x band ≈118–126 (or report’s recorded band)
   - Cap timeout **300–400f** (pick one, document)
   - Keep reason label containing `main_shaft_entry_settle` for dwell tooling
2. Do **not** change Dachora settle (B) or descent cadence (C).
3. Unit smoke: import path / any existing spore controller tests if present.
4. Residual: speculative ~600–800f band **not claimed**; planner
   `continuous.py --to spore --no-video` then split_dwell compare vs 2,806f.
5. Rollback: restore 1000f hold if continuous fails.

## Residual required
- Exact old→new control flow
- Timeout chosen
- Continuous verify command for planner
- Non-claims list

## Do not
- Patch B/C
- continuous.py / STATUS
- Claim savings without re-record

## Acceptance
- [ ] Only Patch A site changed
- [ ] pytest relevant green (controller_common + any spore tests)
- [ ] Residual complete

## Verify commands
```bash
uv run pytest super_metroid/tests/test_controller_common.py super_metroid/tests/test_post_spore_controller.py -q
rg -n "main_shaft_entry_settle" super_metroid/routes/spore_spawn_controller.py
# planner:
# uv run python super_metroid/scripts/record/continuous.py --to spore --no-video
```
