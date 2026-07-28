# RAM map — Mike Tyson's Punch-Out!! (NES)

Partial map used for M1 readiness. Expand during M2.

```text
ADDR_HEALTH = 0x0391
ADDR_OPP_HEALTH = 0x0398
```

Readiness: `is_level1_ready` in `ram.py` (may also require a minimum frame mean so
title/info screens do not false-trigger).
