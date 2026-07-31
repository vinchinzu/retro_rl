# SM-K4-05 Dwell Report — start_to_varia.json

**Date:** 2026-07-31
**Source:** `recordings/start_to_varia.json` (outcome=`varia_collected`, total_frames=101,954)

## Commands

```bash
uv run python super_metroid/scripts/export/split_dwell.py \
  super_metroid/recordings/start_to_varia.json --top 15
uv run python super_metroid/scripts/export/split_dwell.py \
  super_metroid/recordings/start_to_varia.json --reasons --top 20
```

## Top 5 dwell splits (controllable room time)

| Rank | Split ID | Room | Dwell (f) | Notes |
|------|----------|------|----------:|-------|
| 1 | `spore_spawn_activated` | 0x9DC7 Spore Spawn | 12,182 | Boss fight + exit (was 23k in old policy) |
| 2 | `bombs` | 0x9804 Bomb Torizo | 11,812 | Boss fight + item collection |
| 3 | `first_ceres_control` | 0xDF45 Ceres | 10,860 | Scripted intro sequence |
| 4 | `ridley_countdown` | 0xE0B5 Ceres Ridley | 5,554 | Scripted countdown + escape |
| 5 | `zebes_landing` | 0x91F8 Landing Site | 5,385 | Landing + transition |

## Top 5 action_reasons (policy/fight labels)

| Rank | Frames | Reason | Notes |
|------|------:|--------|-------|
| 1 | 13,143 | `policy_pit_to_post_torizo` | Climb + Bomb Torizo + Parlor — already spliced |
| 2 | 7,590 | `boot_intro_wait` | Title/menu — not tightenable |
| 3 | 5,170 | `fight_spore_spawn` | Already 4.5x improved |
| 4 | 3,913 | `fight_kraid` | Boss fight — not primed for tighten |
| 5 | 3,708 | `policy_two_missile_detour` | Side-trip policy |

## Tightenable candidates (controllable movement, not boss/scripted)

| Split ID | Dwell (f) | Room | Tighten Potential |
|----------|----------:|------|-------------------|
| `business_to_warehouse` | 2,257 | 0xA6A1 Warehouse | Business climb return — grounded gates + charge hops |
| `hj_shaft_to_business` | 1,885 | 0xA7DE Business | Hi-Jump shaft return descent |
| `terminator_energy_tank` | 4,693 | 0x990D Terminator | E-Tank detour collection |
| `green_brinstar_main_shaft` | 2,806 | 0x9AD9 Green Shaft | Post-E-Tank shaft traversal |
| `bomb_torizo_exit` | 1,718 | 0x9879 Flyway | Post-boss exit to Parlor |

## Proposed future tighten cards

| Card ID | Target | One-line recipe |
|---------|--------|-----------------|
| `SM-TIGHTEN-01` | `business_to_warehouse` (2,257f) | Profile `play_business_to_warehouse`; identify standing-gate waits or missed charge-hop timings; re-record `--to kraid` to validate |
| `SM-TIGHTEN-02` | `hj_shaft_to_business` (1,885f) | Profile `play_hj_room_to_shaft_exit`; check descent ledge delays vs. morph-ball drop; re-record `--to kraid` |
| `SM-TIGHTEN-03` | `terminator_energy_tank` (4,693f) | Profile `play_terminator_to_east` E-Tank detour; check for idle settle frames or suboptimal bomb jump; re-record `--to varia` |
| `SM-TIGHTEN-04` | `green_brinstar_main_shaft` (2,806f) | Profile `play_green_shaft`; check elevator wait and platform alignment; re-record `--to varia` |

## Caveat

No frame savings claimed without re-record. These are pre-tighten ranks only.