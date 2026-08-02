# Residual — SM-CLEAN-MORPH

## Attempt
2026-08-02 `uv run python super_metroid/scripts/record/continuous.py --to morph --clean --no-video`

## Result
**GREEN**

## Evidence
| Field | Value |
|-------|------:|
| Report | `recordings/start_to_morph_clean.json` |
| Outcome | `morph_ball_acquired` |
| Frames | **27,074** (matches assisted morph) |
| Final room | `0x9E9F` Morph Ball Room |
| Morph | collected (`0x0004`) |
| State loads | 0 |
| Progression / capacity writes | 0 / 0 |
| Energy writes / restored | 0 / 0 |
| Ammo writes / restored | 0 / 0 |
| Intervention | **Clean** |
| Assisted `start_to_morph.json` | untouched |

## Splits (clean)
| Split | Frame |
|-------|------:|
| first_ceres_control | 10860 |
| ridley_countdown | 16414 |
| zebes_landing | 21799 |
| morph_ball | 27074 |

## Next
`SM-CLEAN-BOMBS` ★ — missiles detour + Bombs/BT (BT fight uses existing model; not re-solved here).
