# Status — Mortal Kombat (SNES)

## Program gate

| Field | Value |
|-------|-------|
| Goal | Bronze/Clean Liu Kang arcade: power-on → Goro → Shang Tsung → credits |
| Approach | RAM-gated boot + per-fight v3 RAM/hitbox specialists (pixel CNN fallback) |
| Runtime class | Bronze (read-only RAM) |
| Maturity | **M3** isolated match wins from fight-ready states; boot + overnight retrain in flight |
| Bead | `rr-qpug` |

## Known results (do not re-discover)

Pixel CNN (save-state eval, historical):

| Run | Eval | Notes |
|-----|------|-------|
| 8M `train_speedrun.py --fresh` | ~8% overall | M1 30%, M4 10%, M7 50%; M2–M6/E/G/S 0% |
| `mk1_shangtsung_ppo_final` | ~30–60% Shang | Boss specialist; keep as pixel fallback |
| `mk1_goro_ppo_final` | weak | Retrain with v3 |
| `mk1_fresh_ppo_final` | ~40% M1 | Ladder base, not a full-clear model |
| `mk1_ladder_ft_ppo_final` | 0/100 clears | M1 85%, M3 0%, E1/Goro 0% |
| RAM v1 9-dim | M1 det 0%, stoch 36% | No spacing |
| RAM v2 13-dim | 0/240 det | Spacing still not enough |

v3 (20-dim hitbox RAM) is a **fresh** train per fight. Pixel zips are not
loaded into v3 (wrong obs). They remain tournament fallbacks until
`mk1_v3_<stage>_ppo_final.zip` exists.

## RAM (get_ram)

See [`ram_map.md`](ram_map.md). Max health **161**. Liu Kang id **3**.
Win = `rounds_won >= 2 AND rounds_won > rounds_lost`.

## Scripts

| Script | Role |
|--------|------|
| `scripts/boot_probe.py` | Power-on → Liu Kang fight-ready |
| `scripts/ram_probe.py` | Fighter-object / punch diffs |
| `scripts/eval_roster.py` | Per-fight eval → `models/roster.json` |
| `scripts/train_overnight.py` | 12 parallel v3 specialists |
| `scripts/run_tournament.py` | Continuous attempt, swap on fight/round |
