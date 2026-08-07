# Agent runs — Zelda I tip

## Dispatch pattern (prime-agent + deepseek-v4-flash-0731)

```bash
bd ready -l zelda_i
bd update <tip-id> --status in_progress

prime-agent \
  --provider openrouter \
  --model deepseek/deepseek-v4-flash-0731 \
  --cwd "$PWD" \
  --thinking medium \
  --goal "Complete bead <id> only. Follow prompt file. No Clean STATUS." \
  --autonomous \
  --autonomous-gate 'test -f <evidence-a>' \
  --autonomous-gate 'test -f <evidence-b>' \
  --autonomous-max-turns 24 \
  --autonomous-max-tokens 200000 \
  --autonomous-timeout-ms 3600000 \
  @nes/zelda_i/docs/tasks/agent_runs/<id>_prime_prompt.md \
  "Execute now."
```

## Monitor

- Session transcript: `~/.prime/agent/sessions/<id>.jsonl`
- Process may not show in `prime-agent list` when launched non-TUI; use `ps` + session file growth.
- Gates for rr-0fx: `Level4Entrance.state` + `recordings/l4_entry_recon.json`
- Ready queue should keep **one tip leaf** (`rr-0fx`); parallel only `rr-38p`.

## Architecture (tip spine)

```
L1 Clean → L2 assist TF → L3 assist TF+Raft → L4 live entry (tip)
parallel: early OW caps | isolated pure from checkpoints
defer: Clean combat heatmaps, later TF residual until tip
```
