# RAM map — Castlevania (NES)

Partial map used for M1 readiness. Expand during M2.

```text
ADDR_HEALTH = 0x0044
ADDR_LIVES = 0x002a
```

Readiness: `is_level1_ready` in `ram.py` (may also require a minimum frame mean so
title/info screens do not false-trigger).
