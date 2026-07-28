# RAM map — Contra (NES)

Partial map used for M1 readiness. Expand during M2.

```text
ADDR_LIVES = 0x0032
ADDR_FLAG = 0x0008
```

Readiness: `is_level1_ready` in `ram.py` (may also require a minimum frame mean so
title/info screens do not false-trigger).
