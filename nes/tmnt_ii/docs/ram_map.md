# RAM map — TMNT II (NES)

Partial map for M1–M3. Expand enemy slots during later stages.

| Address | Name | Notes | Confidence |
|---------|------|-------|------------|
| `0x004D` | lives | BEST counter on HUD | high |
| `0x0568` | health | LIFE bar; starts ~60 | high |
| `0x03F0` | score | PTS low byte; +1 per foot kill | high |
| `0x0200+` | OAM | Sprite y/tile/attr/x; Leo ~Y 100–175 | medium |

## Readiness

`is_level1_ready`: `0 < health < 200` and `lives >= 0`, plus optional
frame-mean gate so title screens do not false-trigger.

## Progress

First-wave segment uses **score ≥ 5** as the clear predicate (isolated
kills from `Level1`). Player screen X/Y estimated from OAM band sprites
(`player_screen_x` / `player_screen_y` in `ram.py`).

## Not yet mapped

- Enemy object slots / HP
- Camera / scroll / screen-lock flag
- Stage / area index
- Weapon / special meter
