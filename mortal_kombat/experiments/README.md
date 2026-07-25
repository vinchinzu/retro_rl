# MK1 Experiment Harness

Lightweight autoresearch-style runner for MK1 speedrun experiments.
Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) —
pattern only, not a full clone.

## Files

| File | Purpose |
|------|---------|
| `run_experiment.sh` | Baseline eval + results logging stub |
| `results.tsv` | Append-only experiment log (gitignored) |
| `../docs/experiments.md` | Numbered experiment backlog |
| `../docs/autoresearch_meta.md` | Research synthesis + metrics |

## Quick Start

```bash
cd mortal_kombat

# Record baseline before changes
./experiments/run_experiment.sh baseline

# Eval a candidate general model after training
./experiments/run_experiment.sh eval models/mk1_speedrun_ppo_final.zip "E003 ladder"

# View log
column -t -s $'\t' experiments/results.tsv
```

## Loop (for agents)

1. Pick next `open` experiment from `docs/experiments.md`
2. Run `./experiments/run_experiment.sh baseline` if no recent baseline
3. Execute training command from experiment entry
4. Run `./experiments/run_experiment.sh eval <model> <experiment_id>`
5. Update experiment status to `keep` or `discard` in `docs/experiments.md`
6. If `keep`, update `STAGE_MODELS` in `speedrun_multimodel.py` and registry

## External Autoresearch Clone

To experiment with alternate training loops (non-PPO), clone externally:

```bash
git clone https://github.com/karpathy/autoresearch.git ~/autoresearch-mk1
```

Port `prepare.py` → MK eval harness (`speedrun_multimodel.py` metrics).
Port `train.py` → SB3 training. Keep outside `retro_rl` until a port beats PPO.
