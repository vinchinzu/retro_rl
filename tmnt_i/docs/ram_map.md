# RAM map — TMNT (NES)

Partial map used for M1 readiness. Expand during M2.

```text

ADDR_HEALTH_1 = 0x0077  # selected turtle health (probe: 128 on Area 1 map)
ADDR_GAMEOVER = 0x009E
ADDR_SCORE = 0x00C2

```

Readiness: `is_level1_ready` in `ram.py` (may also require a minimum frame mean so
title/info screens do not false-trigger).
