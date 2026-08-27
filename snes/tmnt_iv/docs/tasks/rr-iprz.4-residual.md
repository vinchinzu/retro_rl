# Residual — rr-iprz.4 Raph air/dash primitive

**Status:** Starbase KEEP jump is production (`raph_starbase_jump_action`).
Wave/dash primitive stayed **off** — live 8k `RaphFastStage7` did not KEEP
either branch vs poke. Do not STATUS.

## RaphFastStage7 8000f / heal=emergency

| Policy | Outcome | Damage | Heals | Min HP | Boss HP | Notes |
|--------|---------|--------|-------|--------|---------|-------|
| poke (no hook) | timeout | **214** | **3** | 9 | 0→0 | production |
| jump-kick + dash+Y | timeout | 277 | 4 | 8 | **0→172** | reached Leatherhead; +63 dmg |
| jump-kick only | timeout | 456 | 7 | 8 | 0→0 | max_hit 32; worse than poke |

Dash is faster (boss spawn inside 8k) but not less damage. Jump-only
duplicated WK `0xB0` jump_slash and chipped harder.

## What shipped

- `snes/tmnt_iv/tactics/raph_air.py` — `raph_starbase_jump_action` (period-4
  B+Y, reasons `raph_starbase_jump` / `raph_starbase_close_gap`)
- `fight_action` calls that helper; dash/wave parking lot deleted
- `snes/tmnt_iv/tests/test_raph_air.py`

## Exact next action

Stateful dash run-up (not `frame % 8`) on `RaphFastStage7` until damage
≤ 214 **and** boss_hp still reaches 172, then consider a dash helper.
Do not re-land stateless dash. Do not jump-kick `0xB0` on WK (already
in the `CombatProfile.jump_kinds` overlay).
