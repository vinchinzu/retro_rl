# Status — Super Mario Bros. 3 (NES)

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M3 (isolated segment) + natural-entry 1-1 clear |
| Best verified result | World 1-1 clear from title → map → level (death-free) |
| Last verification | 2026-07-27 |
| Runtime class | Bronze |
| Intervention class | Clean |

| Field | Value |
|-------|-------|
| Status | **World 1-1 clear verified** |
| Integration | `SuperMarioBros3-Nes` |
| ROM zip | `roms/Nintendo/NES/Super Mario Bros. 3.zip` |
| Ready frame (probe) | ~439 (map) |
| Checkpoints | `Level1.state` (map), `Level1_1.state` (in-level natural entry), `AfterLevel1.state` |
| Policy | `policies/level1_1.json` (~1401 play frames to goal) |
| Evidence | [level1_clear.png](../recordings/level1_clear.png), [e2e_goal.png](../recordings/e2e_goal.png), [e2e_map.png](../recordings/e2e_map.png) |

## Done

- Directory layout and NES integration stubs
- Deterministic reset → World 1 map (`scripts/boot_probe.py`)
- Map path onto 1-1 (RIGHT → UP → A); boot pose is not the enterable node
- Scripted 1-1 clear from natural entry (`scripts/run_level1.py`)
- Platformer level registration `smb3_1_1` for optimizer reuse
- Early RAM (position, lives, auto-control / goal)

## Not done

- Broader M2 instrumentation beyond 1-1 needs
- World 1-2+ segments / continuous multi-level run
- Full-game route

## Next

1. World 1-2 natural-entry clear.
2. Stitch World 1 map path (1-1 → 1-2 → …).
