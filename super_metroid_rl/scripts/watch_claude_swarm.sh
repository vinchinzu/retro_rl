#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOCKET_DIR="${OPENCLAW_TMUX_SOCKET_DIR:-${TMPDIR:-/tmp}/openclaw-tmux-sockets}"
mkdir -p "$SOCKET_DIR"
SOCKET="$SOCKET_DIR/sm_swarm.sock"
SESSION="sm_swarm"

# Fresh start
if tmux -S "$SOCKET" has-session -t "$SESSION" 2>/dev/null; then
  tmux -S "$SOCKET" kill-session -t "$SESSION"
fi

PROMPT_A=$(cat <<'EOF'
You are Worker A (planning+state audit) for Super Metroid RL in this repo. Objective: maximize probability of connecting sections up to first Torizo by morning.

Tasks:
1) Audit current training/status quickly (models, segments, logs, scripts).
2) Produce a concrete execution plan for overnight runs using two anchor save states: ZebesStart and Flyway (or closest equivalent in custom_integrations).
3) Verify required states exist and list any missing with exact creation instructions.
4) Create/update a runbook file: docs/overnight_torizo_plan.md with:
   - segment order,
   - per-segment step budgets,
   - success criteria,
   - exact commands,
   - morning validation checklist.
5) Do not spawn other agents/process managers. 3-agent cap overall.
6) Commit ONLY docs/scripts you change with message: 'plan: overnight torizo training runbook'.

When completely finished, run:
openclaw system event --text "Done Worker A: runbook+state audit complete" --mode now
EOF
)

PROMPT_B=$(cat <<'EOF'
You are Worker B (training pipeline executor) for Super Metroid RL. Goal: kick off/continue training for the route from ZebesStart through flyway_to_torizo with highest practical success.

Tasks:
1) Inspect train_curriculum.py commands and existing trained segment checkpoints.
2) Start/continue training for weak or hard segments first (parlor_descent, morph_ball_return, elevator_return, pit_room_return, climb_return), then connector segments if needed.
3) Use practical step budgets to run overnight and save checkpoints in models/ with clear names.
4) Capture logs in logs/overnight_worker_b_*.out and summarize current best segment status into logs/overnight_worker_b_summary.md.
5) Keep changes minimal and robust; avoid huge refactors.
6) Do not spawn additional agents. Keep within this worker.

At the end, commit scripts/config tweaks + summaries with message: 'train: kickoff overnight segment refinement to torizo'.

When completely finished, run:
openclaw system event --text "Done Worker B: overnight training kickoff complete" --mode now
EOF
)

PROMPT_C=$(cat <<'EOF'
You are Worker C (integration+evaluation) for Super Metroid RL.

Tasks:
1) Build an integration script (or improve existing) that chains trained segment models from ZebesStart to flyway_to_torizo with deterministic evaluation runs.
2) Run multiple evaluation episodes (headless) and output metrics to logs/overnight_worker_c_eval.json and logs/overnight_worker_c_summary.md.
3) If failures cluster in specific transitions, document targeted retrain command suggestions (exact CLI commands).
4) Optionally create a lightweight 'morning one-command check' script in scripts/ if useful.
5) Avoid broad architectural rewrites; optimize for by-morning actionable results.
6) Do not spawn extra agents.

Commit only useful integration/eval artifacts with message: 'eval: add overnight torizo integration checks'.

When completely finished, run:
openclaw system event --text "Done Worker C: integration eval complete" --mode now
EOF
)

# Create session + windows
tmux -S "$SOCKET" new-session -d -s "$SESSION" -n worker-a "cd '$ROOT' && claude --dangerously-skip-permissions \"$PROMPT_A\""
tmux -S "$SOCKET" new-window -t "$SESSION" -n worker-b "cd '$ROOT' && claude --dangerously-skip-permissions \"$PROMPT_B\""
tmux -S "$SOCKET" new-window -t "$SESSION" -n worker-c "cd '$ROOT' && claude --dangerously-skip-permissions \"$PROMPT_C\""

# Helpful status window

tmux -S "$SOCKET" new-window -t "$SESSION" -n monitor "cd '$ROOT' && watch -n 5 'ls -1t logs | head -30'"

echo "tmux socket: $SOCKET"
echo "session: $SESSION"
echo "Attach: tmux -S '$SOCKET' attach -t '$SESSION'"

# Open in Alacritty if available (non-blocking)
if command -v alacritty >/dev/null 2>&1; then
  nohup alacritty -e tmux -S "$SOCKET" attach -t "$SESSION" >/tmp/alacritty_sm_swarm.log 2>&1 &
  disown || true
  echo "Opened Alacritty attached to $SESSION"
else
  echo "alacritty not found; attach manually with tmux command above"
fi
