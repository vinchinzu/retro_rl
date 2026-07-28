# RAM map — Super Mario Bros. (NES)

Partial map used for M1 readiness. Expand during M2.

```text
ADDR_LIVES = 0x075a
ADDR_LEVEL_LO = 0x075c
ADDR_WORLD = 0x075f
```

Readiness: `is_level1_ready` in `ram.py` (may also require a minimum frame mean so
title/info screens do not false-trigger).
