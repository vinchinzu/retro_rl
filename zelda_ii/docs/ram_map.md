# RAM map — Zelda II (NES)

Partial map used for M1 readiness. Expand during M2.

```text

ADDR_LIFE = 0x0773  # probe: 127 in North Palace
ADDR_FLAG = 0x0769  # non-zero once file is active

```

Readiness: `is_level1_ready` in `ram.py` (may also require a minimum frame mean so
title/info screens do not false-trigger).
