# TASK SM-TIGHTEN-02B: Implement HJ return bomb-tunnel + business settle trims

## Recipe step
efficiency implement (bounded — **not** STATUS / continuous promotion)

## Model
Luna

## Own files only
- `routes/kpdr/hijump_return.py` (**edit**)
- optional: `docs/tasks/SM-TIGHTEN-02B-note.md` (**create**)

Do **not** edit business_climb.py, continuous.py, STATUS.md, progression.

## Context
- Analysis: `docs/tasks/SM-TIGHTEN-02-report.md`
- Split: `hj_shaft_to_business` ~1,885f; bomb tunnel ~719f (38%); business settle ~524f
- Implement **Recipe A + Recipe B only** (not Recipe C gray-door rewrite)
- Continuous re-record is planner residual (~27 min)

## Read first (all)
- `docs/tasks/SM-TIGHTEN-02-report.md` (Recipes A/B, risks)
- `routes/kpdr/hijump_return.py` full `play_hj_shaft_to_business`
- `routes/controller_common.py` (`wait_ordinary_room`, morph helpers if used)
- `docs/tasks/SM-K4-05-dwell-report.md` (candidate rank context)

## Do (thorough)
1. **Recipe A — bomb tunnel frequency** (`hj_return_bomb_tunnel` loop ~lines 117–122):
   - Change bomb duty from `frame % 45 < 2` to **`frame % 30 < 3`**
   - Keep RIGHT crawl, early break `samus_x >= 350`, timeout 1100, TimeoutError label
   - Leave sova cleanup loop as-is unless a one-line comment is needed
2. **Recipe B — business settle trims** (same function, post gray door):
   - `wait_ordinary_room(..., settle_frames=280, ...)` → **`settle_frames=180`**
   - Floor wait loop `range(120)` → **`range(60)`** (`hj_return_business_floor`)
   - Climb anchor loop `range(100)` → **`range(60)`** (`hj_return_business_climb_anchor`)
   - Keep final brake 4f + anchor settle 20f
3. Do **not** rewrite gray door exit pattern (Recipe C out of scope)
4. Smoke:
   ```bash
   uv run pytest super_metroid/tests/test_controller_common.py -q
   ```
5. Residual note (final message / optional md):
   - Speculative ~300f + ~220f from report — **not claimed**
   - Planner continuous verify:
     `uv run python super_metroid/scripts/record/continuous.py --to kraid --no-video`
     then dwell compare `hj_shaft_to_business`
   - Risk: more bombs knock back in tunnel; 180f settle may be tight on slow loads
   - Revert recipe if continuous fails

## Residual required (super-clean)
- Diff of the three numeric knobs (before→after)
- What was not touched (Recipe C, business_climb, STATUS)
- Continuous verify command + failure rollback plan

## Do not
- Recipe C gray-door pattern rewrite
- continuous.py / STATUS / business_climb
- Claim savings without re-record
- Progression writes

## Acceptance
- [ ] A+B applied exactly as specified
- [ ] pytest green
- [ ] Residual lists continuous verify for planner

## Verify commands
```bash
uv run pytest super_metroid/tests/test_controller_common.py -q
rg -n "hj_return_bomb_tunnel|settle_frames|hj_return_business_floor|hj_return_business_climb_anchor" \
  super_metroid/routes/kpdr/hijump_return.py
```
