# Residual — SM-CLEAN-BOMBS

## Attempt
2026-08-02 clean continuous probes (`--to bombs --clean --no-video`)

## Scope this session
**Get to missiles** on Clean (no energy/ammo writes). **Do not re-solve Bomb
Torizo** — assisted BT path / `combat/bomb_torizo.py` model already owns the
fight. Clean tip only needs compose + integrity once the early prefix holds.

## Result
| Gate | Status | Notes |
|------|--------|-------|
| Morph clean | **GREEN** | see `SM-CLEAN-MORPH-residual.md` |
| First missiles (cap 5) | **GREEN** | frame **27,928** (same as assisted) |
| Blue Brinstar missiles (cap 10) | **GREEN** | frame **29,690** (same as assisted) |
| Construction + elevator return | **GREEN** | health 99 → Pit |
| Pit weapon settle | **fixed** | clean detour exits **beam** (sel=0); one blind SELECT was arming missiles — now only SELECT when sel≠0 |
| Climb → Flyway → BT room | **reached** | bombs collected; BT active |
| Bomb Torizo exit (Clean tip) | **not claimed** | out of scope this residual; use existing BT model |

## Missile detour log (clean, no ammo refill)

| Checkpoint | Frame | HP | Missiles | selected_item |
|------------|------:|---:|---------:|--------------:|
| post morph | 27074 | 99 | 0/0 | 0 |
| post two-missile detour | 30795 | 99 | **5/10** | **0** (beam) |
| post construction return | 32151 | 99 | 5/10 | 0 |
| post elevator return | 32503 | 99 | 5/10 | 0 |
| pit aligned | ~32800 | ~84 | 5/10 | 0 |

Assisted baseline at same boundary: missiles **10/10**, selected_item **1**
(missiles) — unlimited ammo refill during the detour.

Both natural expansions land on the same frames as assisted:

- `first_missiles` @ **27928** (`max_missiles` 0→5)
- `blue_brinstar_missiles` @ **29690** (`max_missiles` 5→10)

Clean resource counters through this prefix: **all zero** writes/restored;
`intervention_class=Clean`.

## Code change kept (not BT)
`routes/continuous.py` `play_start_to_bombs` pit settle: press SELECT only when
`selected_item != 0`, else one idle frame, then the historical 9-frame grounded
settle. Preserves assisted 1+9 budget when sel=1; does not flip clean beam→missiles.

**No** changes claimed to `combat/bomb_torizo.py` / hash-pinned BT spray for this
card closeout.

## Probe note (BT handoff only — not a tip claim)
Later exploratory handoff into the RAM fighter saw combat-active BT
(`spritemap 0xAA12`, HP 800) with ~3/10 missiles and ~90 HP after Flyway. That
path is **not** the Clean tip solution; leave BT on the existing model.

## Next
1. `SM-CLEAN-BOMBS` compose: full clean `--to bombs` using **existing** BT path
   (hash pin and/or landed `play_bomb_torizo_fight`) — tank damage OK, **deaths
   zero**, zero resource writes.
2. If RED only on BT ammo/DPS → `SM-CLEAN-BT-ECONOMY` one knob.
3. GREEN → `SM-CLEAN-STAB` ×2 then `SM-CLEAN-STATUS` secondary.

## Artifacts
- Green morph: `recordings/start_to_morph_clean.json`
- Partial bombs probe (missiles splits only; tip **not** green):
  `recordings/start_to_bomb_torizo_clean.json` (failed later; do not publish as tip)
- Assisted baselines `start_to_morph.json` / `start_to_bomb_torizo.json` untouched
