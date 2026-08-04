# Status — Mortal Kombat (SNES)

## Program gate

| Field | Value |
|-------|-------|
| Goal | ~90% full tournament clear (normal, LiuKang ladder) |
| Approach | Multimodel stage specialists + ladder PPO |
| Runtime class | Bronze (research) |

## Known results (do not re-discover)

| Run | Eval | Notes |
|-----|------|-------|
| 8M `train_speedrun.py --fresh` | ~8% overall | M1 30%, M4 10%, M7 50%; M2–M6/E/G/S 0% |
| Training log WR | ~15% | Inflated vs eval — always benchmark |
| Default tier mix | 38% boss+endurance | Starved M2–M6 |
| `mk1_shangtsung_ppo_final` | ~60% Shang | Boss specialist works |
| `mk1_goro_ppo_final` | weak | Needs more steps |
| `mk1_fresh_ppo_final` | ~40% M1 | Good ladder base for fine-tune |

## RAM (get_ram)

| Variable | Address | Hex |
|----------|---------|-----|
| health (P1) | 1209 | 0x04B9 |
| enemy_health (P2) | 1211 | 0x04BB |
| timer | 290 | 0x0122 |
| continue_timer | 999 | 0x03E7 |
| p1_character | 6514 | 0x1972 |
| p1_x / p1_y | 218 / 219 | 0x00DA / 0x00DB |
| p2_x | 372 | 0x0174 |

Max health: **161**.

## Scripts

| Script | Role |
|--------|------|
| `train_speedrun.py` | Training only |
| `speedrun_test.py` / `speedrun_multimodel.py` | Per-stage / tournament eval |
| `cheat_extractor.py`, `match_manager.py`, `validate_states.py` | States |
| `watch.py` | Visual debug |
