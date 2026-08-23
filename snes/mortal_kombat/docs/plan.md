# MK1 plan — overnight RAM specialists

## Now

M1→M2 swap is live-probed with pixel fallbacks (`round_probe.py`).
Overnight v3 **done** (12/12 at 4.0M). Save-state N=5: M5 80%, M1/M3 60%,
M4 40%, M2/M6/E1B/Shang 20%, M7/E1/E2/Goro 0%. Checkpoint sweep found
M7 best 2/20 (10%) and E1/E2/Goro 0/34 each; blind continuation is rejected.
Train non-destructive candidates, evaluate at N>=20, then `run_tournament.py`.
Pixel CNNs stay as round-loss fallbacks.

```bash
uv run python snes/mortal_kombat/scripts/setup_rom.py
uv run python snes/mortal_kombat/scripts/boot_probe.py
uv run python snes/mortal_kombat/scripts/ram_probe.py
uv run python snes/mortal_kombat/scripts/train_overnight.py --dry-run
uv run --extra ml python snes/mortal_kombat/scripts/train_overnight.py --steps 4000000 --jobs 12 --n-envs 2
```

16c/32t: 12 jobs × 2 envs. Logs: `logs/overnight_v3/<Stage>.log`.
Outputs: `models/mk1_v3_<Stage>_ppo_final.zip` + `models/roster.json`.

After training:

```bash
uv run --extra ml python snes/mortal_kombat/scripts/eval_roster.py --attempts 5
uv run --extra ml python snes/mortal_kombat/scripts/eval_roster.py --stages Match7 --checkpoints --attempts 20
uv run --extra ml python snes/mortal_kombat/scripts/eval_roster.py --compare --stages Fight,Match5,Match7 --attempts 5
uv run --extra ml python snes/mortal_kombat/scripts/run_tournament.py --ladder-model mk1_v3_Match5_ppo_final.zip
```

Continuations: `--load` plus a distinct `--output-prefix`. Wall cutoff writes
`*_ppo_{timesteps}_steps.zip` and is not the incumbent. `--promote` only at N>=20.

## Next

- N>=20 eval of `mk1_v3_Match5_ppo_final.zip` on Fight, Match2, Match7; `--promote` only if it beats the per-stage zip
- More Clean `--ladder-model mk1_v3_Match5_ppo_final.zip` attempts (furthest Match 4 so far)
- Do not retarget v3 x/y off `0x00DA` without a fresh train
- Scripted fireball: walk back after F,F,HP (F-hold closes to punch range); not a Match 1 winner yet
- Tighten attack-state / facing from live pose, not the 0x00DA object
- Weak-stage extra steps (Goro, Shang, endurance) if eval < 30%
- Credits detect if `match_counter>=12` is not enough
- Do not STATUS-promote a full clear until a continuous power-on run
  reaches credits
