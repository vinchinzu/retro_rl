# TASK SM-K4.1-PURE: Pure Frog Save → Speedway

## Alias

Living implement card: **[`SM-K4-SPEEDWAY-PURE.md`](SM-K4-SPEEDWAY-PURE.md)**  
Backlog row: `SM-K4.1-PURE` in [`BACKLOG.csv`](../routes/BACKLOG.csv).

Use either ID with dispatch (`SM-K4.1-PURE` or `SM-K4-SPEEDWAY-PURE`). Same
own-files, source, and acceptance.

## Recipe step
1 pure controller

## Model
Luna

## Wave type
implement

## Own files only
- `routes/kpdr/k4_norfair.py` — `play_frog_save_to_speedway` only
- `routes/kpdr/__init__.py` / `routes/kpdr/registry.py` — this segment only
- `scripts/probe/kpdr.py` — this pure choice only
- `tests/test_k4_norfair_scaffold.py` — import/registry only
- residual: `docs/tasks/SM-K4-FROG-SPEEDWAY-PURE-residual.md`

## Context (minimal)
- Continuous tip verified: power-on → Frog Savestation (`--to frog`, 114923f ×2)
- Source: `scratch/post_frog_continuous.state` expect room `0xB167`
- Target exit: Frog Speedway `0xB106` (door `0x980A`)
- Architecture: `docs/ARCHITECTURE.md` continuous tip-extension recipe
- Process: `docs/tasks/PROCESS.md`
- Wave board: `docs/tasks/WAVE-11.md`

## Read first (only these)
- `routes/kpdr/k4_norfair.py` — existing style
- `docs/SOURCE_STATES.md` — confirm source row
- `docs/tasks/SM-K4-SPEEDWAY-PURE.md` — full acceptance + verify commands
- one recent pure hop in `routes/kpdr/` — one-knob pattern

## Do
1. Implement pure controller hop Frog Savestation → Frog Speedway from the listed source.
2. Keep one named primitive / constant change only.
3. Reach ordinary gameplay in `0xB106` with green pure probe.

## Do not
- Touch `continuous.py` / `STATUS.md` / `progression.py` tip registration
- Claim continuous integrity
- Second interacting knob
- Progression / capacity / door / event writes

## Acceptance
- [ ] Pure probe green from `scratch/post_frog_continuous.state`
- [ ] Residual uses PROCESS schema (next card ID + one change)
- [ ] No unrelated file churn

## Verify commands
```bash
uv run python super_metroid/scripts/probe/kpdr.py pure frog-save-to-speedway \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_frog_continuous.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_frog_save_to_speedway_pure.state \
  --pin-json super_metroid/debug/frog_save_to_speedway_pure_pin.json

uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
```

## Residual routing
- GREEN → [`SM-K4-SPEEDWAY-SRC`](SM-K4-SPEEDWAY-SRC.md)
- RED → `SM-K4-FROG-SPEEDWAY-R1` (one named controller phase only)

## Done when
Executor returns residual. Graph + continuous tip stay with planner.

## Residual note (planner)
A residual claiming **GREEN** may already exist at
`SM-K4-FROG-SPEEDWAY-PURE-residual.md`. If re-verify is red/flaky, treat pure
as still open; do not STATUS-promote from residual alone.
