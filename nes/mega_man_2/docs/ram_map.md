# RAM map — Mega Man 2 (NES)

Partial map used for M1 readiness. Expand during M2.

```text
ADDR_LIVES = 0x00A8
ADDR_HEALTH = 0x06C0  # full bar often 28
```

Readiness: `is_level1_ready` in `ram.py` (may also require a minimum frame mean so
title/info screens do not false-trigger).
