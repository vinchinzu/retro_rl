# RAM map — Super Mario Bros. 3 (NES)

Partial map used for M1 readiness and World 1-1 clear. Expand during broader M2.

```text
ADDR_HPOS         = 0x0090  # on-screen X
ADDR_VPOS         = 0x00A2  # Y in levels
ADDR_X_PAGE       = 0x0075  # coarse X in levels (8-block units; dual-use on map)
ADDR_IN_AIR       = 0x00D8
ADDR_HVEL         = 0x00BD
ADDR_MAP_Y        = 0x0078
ADDR_MAP_X        = 0x0079
ADDR_MAP_MOVE     = 0x007B  # remaining map-walk pixels
ADDR_MAP_TILE     = 0x00E5  # tile under Mario ($04 = 1-2 panel)
ADDR_MAP_OPERATION= 0x0729  # $0D = normal move/enter
ADDR_LIVES        = 0x0736
ADDR_WORLD        = 0x0727  # world number - 1
ADDR_FORM         = 0x0746
ADDR_AUTO_CONTROL = 0x0559  # non-zero at goal card / cutscene
ADDR_RETURN_MAP   = 0x0014
```

Progress proxy: `player_progress_x = x_page * 256 + hpos` while `x_page < 0x18`
(death/map writes large page values).

Readiness: `is_level1_ready` in `ram.py` (may also require a minimum frame mean so
title/info screens do not false-trigger).

Goal: `is_goal_auto` when `ADDR_AUTO_CONTROL != 0` after enough progress.
