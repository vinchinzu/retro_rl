# TASK SM-PURE-ISO: Cheap continuous-regression pure probes for post-HJ climb

## Recipe step
harness (probe CLI + tests — **not** continuous re-record ownership)

## Model
Luna

## Own files only
- `scripts/probe/kpdr.py` (**edit** pure choices / wiring only)
- `routes/kpdr/registry.py` only if a segment id is missing for pure
- `tests/test_kpdr_dev.py` or `tests/test_controller_common.py` (**add** register/import asserts)
- `docs/tasks/SM-PURE-ISO-note.md` (**create** — usage + source map)

Do **not** edit continuous.py, STATUS, business_climb geometry knobs,
hijump_return knobs, or progression verification.

## Context
Wave-3 planner continuous re-record (`--to kraid`) **failed** after
SM-TIGHTEN-01B (settle 20→5): first fail `business_1227_land` (y=1419 floor),
retry fail `business_1339_ground` (y=1291). TIGHTEN-02B (HJ return) appeared
to reach Business. Full continuous is ~27 min — harness gap: no cheap pure
isolation for the climb that breaks after efficiency trims.

Known scratch sources (gitignored under integration):
- `scratch/continuous_like_business_climb_entry.state` (Business entry-ish)
- `scratch/business_to_warehouse_function.state` (if present / usable)
- `scratch/post_varia_to_kraid_pure.state` (already wired for reverse)

## Read first (all)
- `scripts/probe/kpdr.py` (pure segment map + boot_from_state)
- `routes/kpdr/registry.py` / `rooms.py`
- `routes/kpdr/business_climb.py` (`play_business_to_warehouse` entry contract)
- `routes/kpdr/hijump_return.py` (`play_hj_shaft_to_business`)
- `docs/tasks/QUEUE.md` residual on continuous fail
- `docs/ARCHITECTURE.md` pure-vs-continuous note if present

## Do (aggressive harness stress)
1. Wire pure probe choices (kebab-case) if missing:
   - `hj-shaft-to-business`
   - `business-to-warehouse`
2. Document **required** `--source` paths in `SM-PURE-ISO-note.md` with expected
   start room hex for each. If a scratch state is missing or wrong room, the
   note must say **MISSING** and print the probe command that creates/validates
   it (do not invent warps as green evidence).
3. On pure failure, ensure TimeoutError / CLI exit prints room, pose, x, y,
   frame (improve only if currently incomplete — keep change small).
4. Add unit tests: segment ids resolve via registry (no emu).
5. **Bonus (not required for EXIT):** run pure `business-to-warehouse` once from
   a named source if the state exists and room matches; paste exit code.
   Do not claim continuous green from pure alone.

## Residual required
- Segment ids wired + source map table
- Explicit: pure ≠ continuous integrity; continuous still planner gate
- What this would have caught on TIGHTEN-01B (settle regression)

## Do not
- Re-run full continuous and claim STATUS
- Change settle/runup geometry
- Door-warp into “green” pure

## Acceptance
- [ ] Pure CLI choices exist and are documented
- [ ] Registry/import tests green
- [ ] Residual names continuous still planner-only

## Verify commands
```bash
uv run pytest super_metroid/tests/test_controller_common.py super_metroid/tests/test_kpdr_dev.py -q
uv run python super_metroid/scripts/probe/kpdr.py pure --help 2>&1 | head -80
# optional if state present:
# uv run python super_metroid/scripts/probe/kpdr.py pure business-to-warehouse \
#   --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/continuous_like_business_climb_entry.state
```
