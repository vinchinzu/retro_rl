# TASK SM-TIGHTEN-P2: Business climb setup jumps 4→3 (medium risk)

## Recipe step
efficiency implement (higher risk than settle trim — **not** continuous claim)

## Model
Luna

## Own files only
- `routes/kpdr/business_climb.py` (**edit** setup jump loop only)
- `docs/tasks/SM-TIGHTEN-P2-note.md` (**create**)

Do **not** edit continuous.py, STATUS, hijump_return, progression, settles.

## Context
- Report: `docs/tasks/SM-TIGHTEN-01-report.md` **P2**
- Setup jumps currently cost ~460f (4× release+setup+land); always run on floor return
- Wave-3 5f settle trim **broke continuous** — treat this as higher-risk geometry
- Speculative band ~115f (1 jump) — **not claimed**

## Read first (all)
- `docs/tasks/SM-TIGHTEN-01-report.md` P2 section
- `routes/kpdr/business_climb.py` setup jump loop (~lines 98–102 region) + fallback re-climb
- STATUS notes on business continuous-hardening (read only)

## Do (aggressive)
1. Reduce setup jump sequence from **4 to 3** directions only.
   Report suggestion: `("RIGHT", "LEFT", "LEFT")` or `("LEFT", "LEFT", "RIGHT")`.
   Pick **one** sequence; document which and why briefly in residual.
2. Do **not** change settle durations, runup_907, or platform hop gates.
3. Keep floor-recover / re-climb fallback intact (if 3-jump misses y=1339, retry path must still exist).
4. Smoke pytest; residual must require planner continuous `--to kraid` × **≥1**
   success before any claim (report asked for 3 — note that as planner residual).
5. If you can pure-probe from a Business floor source, paste result; else MISSING.

## Residual required
- Exact tuple before→after
- Risk: miss 1339 → fallback costs more than savings
- Continuous verify + multi-run residual for planner
- Explicit non-claims

## Do not
- Combine with settle/runup trims
- continuous.py / STATUS
- Force-pass if pure fails (pure optional)

## Acceptance
- [ ] Setup jumps are 3 only (one sequence)
- [ ] pytest green
- [ ] Residual with continuous multi-run planner gate

## Verify commands
```bash
uv run pytest super_metroid/tests/test_controller_common.py -q
rg -n "setup|1339|business_climb" super_metroid/routes/kpdr/business_climb.py | head -40
# planner:
# uv run python super_metroid/scripts/record/continuous.py --to kraid --no-video
```
