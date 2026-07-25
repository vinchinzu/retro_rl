#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$ROOT/custom_integrations/HarvestMoon-Snes"

state_name="${1:-latest}"
if [[ ! -f "$STATE_DIR/${state_name}.state" ]]; then
  if [[ -f "$STATE_DIR/current.state" ]]; then
    state_name="current"
  else
    echo "No ${state_name}.state or current.state found under $STATE_DIR" >&2
    exit 2
  fi
fi

exec "$ROOT/startup.sh" --background --state "$state_name" --autostart "${@:2}"
