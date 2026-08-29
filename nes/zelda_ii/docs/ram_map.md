# RAM map — Zelda II (NES)

Partial map. M1 readiness uses magic; leave-palace stop uses engine mode.

Sources: Data Crystal / z2disassembly; live fceumm probe from `Level1` LEFT
walk (2026-08-29).

```text
ADDR_PLAYER_Y     = 0x0029  # side-scroll Y
ADDR_PAGE         = 0x003B  # side-scroll map page (North Palace starts at 1)
ADDR_PLAYER_X     = 0x004D  # side-scroll X low
ADDR_OW_Y         = 0x0073  # overworld tile Y (North Palace 52)
ADDR_OW_X         = 0x0074  # overworld tile X (North Palace 23)
ADDR_LIVES        = 0x0700
ADDR_ENGINE_MODE  = 0x0736  # 11 side-scroll play, 5 overworld play
ADDR_FLAG         = 0x0769  # non-zero once file is active
ADDR_LIFE         = 0x0773  # magic meter; probe 127 in North Palace
ADDR_HEALTH       = 0x0774  # life meter (death when 0 in play)
```

Engine mode `$0736` (leave-palace sequence): 11 (North Palace) → 16 → 1–4
(black load) → 5 (overworld play). Transition set `{1,2,3,4,16}`.

Readiness: `is_level1_ready` in `ram.py` (may also require a minimum frame mean so
title/info screens do not false-trigger).

Stop: `palace_exit_success` is engine mode 5.
