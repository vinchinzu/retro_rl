# RAM map — Zelda I (NES)

Partial map used for M1 readiness. Expand during M2.

```text

ADDR_MODE = 0x0012  # 5 = overworld play
ADDR_HEALTH = 0x066F  # heart fragments encoding (probe: 34 at start)
ADDR_SCREEN = 0x00EB  # overworld screen id (partial)

```

Readiness: `is_level1_ready` in `ram.py` (may also require a minimum frame mean so
title/info screens do not false-trigger).
