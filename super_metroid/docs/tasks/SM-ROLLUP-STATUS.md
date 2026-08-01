# TASK SM-ROLLUP-STATUS: Propose STATUS / tracker / board sync from QUEUE + reports

## Recipe step
docs | efficiency (rollup) — **proposal only, no silent promotion**

## Model
Flash

## Own files only
- `docs/tasks/SM-ROLLUP-STATUS-proposal.md` (**create** — full proposed text)
- Optionally draft-only snippets under `docs/tasks/` (never overwrite STATUS
  claim tables without planner approval)

**Do not edit** `docs/STATUS.md`, `docs/routes/KPDR_TRACKER.csv`, or
`docs/research/PATH_ROOM_BOARD.md` in this card. Planner applies approved diffs.

## Context
- Continuous tip verified claim stays planner-owned (see STATUS.md).
- QUEUE Waves 3–5 have continuous greens and dwell deltas **not** fully
  mirrored in STATUS / tracker / board.
- Process: `docs/tasks/PROCESS.md` §4.

## Read first (only these)
- `docs/STATUS.md`
- `docs/tasks/QUEUE.md` (Wave 4–5 rollups + residual planner work)
- `docs/routes/KPDR_TRACKER.csv` (header + K3 / reverse rows)
- `docs/research/PATH_ROOM_BOARD.md` (verification legend only if present)
- Latest reports if present:
  - `recordings/start_to_kraid.json`
  - `recordings/start_to_varia.json`
- `docs/SOURCE_STATES.md` (for “sources known” section)
- `scripts/export/split_dwell.py` usage in AGENTS.md (offline only)

## Do
1. Summarize **facts only** from QUEUE + JSON reports:
   - last continuous GREEN frame totals (`--to kraid` / `--to varia` if re-run)
   - integrity (loads / progression writes)
   - pure residuals still RED (door return, etc.)
   - dwell deltas measured but not promoted
2. Propose three small diffs (unified-diff or clearly marked before/after):
   - STATUS: verification date / frame tables **only if** multi-run stable is
     claimed by planner notes; otherwise propose “measured, not promoted” note
   - KPDR_TRACKER: reverse hop rows status (`unverified` / `controller_dev`)
     matching QUEUE honesty
   - PATH_ROOM_BOARD: only hop verification flags that changed in progression
     tests (no invent)
3. List **open pure blockers** with next card IDs from QUEUE.
4. Write `docs/tasks/SM-ROLLUP-STATUS-proposal.md` with residual schema footer.

## Do not
- Touch `routes/continuous.py` or any controller
- Claim continuous savings from single-run dwell noise
- Promote graph edges to `continuous` verification
- Paste absolute home paths

## Acceptance
- [ ] Proposal file exists with STATUS / tracker / board sections
- [ ] Each proposed change cites QUEUE or JSON field
- [ ] Explicit “planner must approve” banner at top
- [ ] Residual uses PROCESS residual schema (next card + one change)

## Verify commands
```bash
# Offline only — no emu required
test -f super_metroid/docs/STATUS.md
test -f super_metroid/docs/tasks/QUEUE.md
# If reports exist:
# uv run python super_metroid/scripts/export/split_dwell.py \
#   super_metroid/recordings/start_to_kraid.json --top 10
```

## Done when
Flash returns proposal path + residual. Planner merges or discards.
