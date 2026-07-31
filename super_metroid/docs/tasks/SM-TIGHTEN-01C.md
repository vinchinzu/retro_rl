# TASK SM-TIGHTEN-01C: Safer re-try of business settle trim (12f) + pure-first gate

## Recipe step
efficiency implement (bounded — **pure isolation first**; continuous still planner)

## Model
Luna

## Own files only
- `routes/kpdr/business_climb.py` (**edit** platform settles only)
- `docs/tasks/SM-TIGHTEN-01C-note.md` (**create**)

Do **not** edit continuous.py, STATUS, hijump_return, progression.

## Context (honest Wave-3 failure)
- SM-TIGHTEN-01B applied 20→**5**f on 8 platform settles.
- Planner continuous `--to kraid` **RED**:
  - `business_1227_land` timeout — Samus on floor y=1419 (fell off)
  - floor-recover retry `business_1339_ground` fail y=1291
- Settles were **reverted to 20f** by planner. Hypothesis: 5f debounce too
  short before next hop commits while still unstable on lip.
- This card retries a **mid** value (**12f**) and **requires pure isolation
  attempt** before EXIT — not a blind continuous gamble.

## Read first (all)
- `docs/tasks/SM-TIGHTEN-01B-note.md`
- `docs/tasks/SM-TIGHTEN-01-report.md` (P1 only)
- `routes/kpdr/business_climb.py` `_business_high_jump_platforms`
- `docs/tasks/SM-PURE-ISO.md` (if pure choices land first; else document raw command)

## Do
1. Change **only** the same 8 labels that were 20→5→20:
   `business_1339_settle`, `1227`, `1147`, `987`, `907`, `843`, `779`,
   `business_elevator_settle` — set hold duration **20 → 12**.
2. Leave `business_1067_settle` at 30f; leave P2/P3/runup alone.
3. Attempt pure isolation if source exists:
   ```bash
   uv run python super_metroid/scripts/probe/kpdr.py pure business-to-warehouse \
     --source <documented matching state>
   ```
   If source MISSING: note MISSING and still EXIT with code change + residual
   (planner continuous remains the real gate).
4. Residual note must state:
   - Exact before/after (20→12)
   - Pure result or MISSING source
   - Planner must re-record `--to kraid --no-video` before any savings claim
   - If continuous fails: revert all 8 back to 20f immediately

## Residual required
- Label list + 12f
- Explicit non-claim of continuous / STATUS savings
- Rollback recipe one-liner

## Do not
- Use 5f again
- P2/P3
- Claim ~80f savings without re-record
- continuous.py / STATUS

## Acceptance
- [ ] Only those 8 settles at 12f
- [ ] pytest controller_common green
- [ ] Residual + pure attempt/MISSING documented

## Verify commands
```bash
uv run pytest super_metroid/tests/test_controller_common.py -q
rg -n "business_.*_settle" super_metroid/routes/kpdr/business_climb.py
# planner only:
# uv run python super_metroid/scripts/record/continuous.py --to kraid --no-video
```
