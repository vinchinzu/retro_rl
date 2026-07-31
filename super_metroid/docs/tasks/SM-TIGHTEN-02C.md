# TASK SM-TIGHTEN-02C: HJ return gray-door exit Recipe C

## Recipe step
efficiency implement (depends on Wave-3 02B still in tree)

## Model
Luna

## Own files only
- `routes/kpdr/hijump_return.py` (**edit** gray-door exit loop only)
- `docs/tasks/SM-TIGHTEN-02C-note.md` (**create**)

Do **not** edit business_climb, continuous, STATUS, progression.
Do **not** re-touch bomb-tunnel duty or settle_frames from 02B unless a one-line
conflict forces it (prefer leave 02B knobs).

## Context
- Report: `docs/tasks/SM-TIGHTEN-02-report.md` **Recipe C**
- Wave-3 02B already applied A+B (bomb %30<3, settle 180, floor 60, anchor 60)
- Planner continuous after 01B-revert still needs to prove 02B alone; this card
  stacks more aggression on gray exit
- Speculative ~150f — **not claimed**

## Read first
- `docs/tasks/SM-TIGHTEN-02-report.md` Recipe C
- `routes/kpdr/hijump_return.py` gray exit + sova cleanup
- full `play_hj_shaft_to_business`

## Do
1. Apply Recipe C only: replace periodic `Right+B+X` / `Right+B+A` gray exit
   with report’s suggested pattern (X only first frames to open, continuous B
   run, remove unnecessary A if flat approach).
2. Keep timeout / TimeoutError label family; keep sova cleanup.
3. Residual: interaction risk with 02B knobs; planner continuous `--to kraid`
   before claim; rollback C independently if fail.
4. pytest smoke.

## Residual required
- Diff of gray-exit loop only
- Non-claims + continuous gate

## Do not
- Rewrite bomb tunnel again
- continuous / STATUS
- Claim savings

## Acceptance
- [ ] Recipe C only
- [ ] pytest green
- [ ] Residual complete

## Verify commands
```bash
uv run pytest super_metroid/tests/test_controller_common.py -q
rg -n "hj_return_gray|gray" super_metroid/routes/kpdr/hijump_return.py
# planner:
# uv run python super_metroid/scripts/record/continuous.py --to kraid --no-video
```
