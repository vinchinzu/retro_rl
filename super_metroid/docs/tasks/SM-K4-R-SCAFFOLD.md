# TASK SM-K4-R-SCAFFOLD: Scaffold remaining K4 reverse hops (new module)

## Recipe step
1 pure controller (scaffold only — pure probe may fail without natural source)

## Model
Luna

## Own files only (do not edit others)
- `routes/kpdr/kraid_return.py` (**create**)
- `routes/kpdr/registry.py` (register only)
- `routes/kpdr/__init__.py` (export only)
- `scripts/probe/kpdr.py` (pure choices only)
- `tests/test_controller_common.py` (import/register asserts only)

Do **not** edit `varia_return.py`, `continuous.py`, `STATUS.md`, `progression.py`.

## Context
- Graph reverse edge ids (locked): `eye_to_baby_return` (0xA56B→0xA521),
  `baby_to_kihunter_return` (0xA521→0xA4DA), `kihunter_to_zeela_return`
  (0xA4DA→0xA471), `zeela_to_warehouse_return` (0xA471→0xA6A1)
- Style: `routes/kpdr/varia_return.py` scaffolds + forward reverse in
  `kraid_approach.py` / `warehouse.py`
- Pure green for these hops is **not** required; registration + import tests are

## Read first
- `routes/kpdr/varia_return.py`
- `routes/kpdr/kraid_approach.py` (baby/eye/kihunter/zeela forwards)
- `routes/kpdr/rooms.py`
- `routes/kpdr/registry.py`
- `scripts/probe/kpdr.py`
- `tests/test_controller_common.py`

## Do (thorough)
1. Create `routes/kpdr/kraid_return.py` with four `play_*` scaffolds:
   - `play_eye_to_baby_return` — ROOM_KRAID_EYE left toward baby
   - `play_baby_to_kihunter_return` — baby left toward kihunter
   - `play_kihunter_to_zeela_return` — kihunter up/return toward zeela
   - `play_zeela_to_warehouse_return` — zeela left toward warehouse
2. Each: `require_state`/`require_room`, weapon select, bounded LEFT/spin or
   directional push matching graph exit_direction, `wait_ordinary_room` target,
   docstring: controller_dev scaffold; not continuous evidence
3. Register segment ids matching graph edge ids in `KPDR_SEGMENTS`
4. Export from `routes/kpdr/__init__.py`
5. Wire pure probe CLI choices (kebab-case)
6. Unit tests: each importable via `get_segment(...)` (no emu)

## Do not
- Promote graph verification or STATUS
- Continuous compose
- Progression RAM writes / door warps inside pure controllers

## Acceptance
- [ ] `uv run pytest tests/test_controller_common.py -q` green (cwd=super_metroid)
  or from monorepo root:
  `uv run pytest super_metroid/tests/test_controller_common.py -q`
- [ ] All four segments registered
- [ ] Diff summary + residual (need natural sources for pure green)

## Verify commands
```bash
uv run pytest super_metroid/tests/test_controller_common.py -q
uv run python super_metroid/scripts/probe/kpdr.py list 2>/dev/null || true
# optional pure (expect fail without source — report only):
# uv run python super_metroid/scripts/probe/kpdr.py pure eye-to-baby-return --source <state>
```
