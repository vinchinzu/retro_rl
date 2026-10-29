#!/usr/bin/env bash
# PR-006 lane kit for the Harvest perfect-rancher campaign.
#
#   snes/harvest/scripts/pr999_lane.sh <lane>
#   snes/harvest/scripts/pr999_lane.sh <lane> run -- <command...>
#   snes/harvest/scripts/pr999_lane.sh <lane> pytest [pytest args...]
#   snes/harvest/scripts/pr999_lane.sh <lane> probe [state]
#
# Creates /home/v/01_projects/11_games/rr_harvest_pr999_<lane> on branch
# harvest/pr999-<lane>. Symlinks roms, snes/harvest/.venv, pin files, and
# untracked task tapes from the main tree. Adds an info/exclude entry for
# roms. `run` is setsid nohup, then wait. pytest and run set PYTHONPATH
# to the worktree so the venv retro_rl_paths.pth cannot win.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WT_PARENT="/home/v/01_projects/11_games"
WT=""

usage() {
  sed -n '2,14p' "$0" >&2
  exit 2
}

lane_ok() {
  [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || return 1
  [[ "$1" != *..* ]] || return 1
}

python_path_for() {
  local wt="$1"
  printf '%s:%s:%s:%s\n' "$wt" "$wt/snes" "$wt/nes" "$wt/snes/harvest"
}

venv_bin() {
  local name="$1"
  local path="$MAIN/.venv/bin/$name"
  if [[ ! -x "$path" ]]; then
    echo "missing $path" >&2
    exit 1
  fi
  printf '%s\n' "$path"
}

ensure_symlink() {
  local src="$1" dst="$2"
  if [[ ! -e "$src" && ! -L "$src" ]]; then
    echo "missing symlink source: $src" >&2
    exit 1
  fi
  if [[ -L "$dst" ]]; then
    local cur want
    cur="$(readlink -f "$dst")"
    want="$(readlink -f "$src")"
    if [[ "$cur" != "$want" ]]; then
      echo "symlink $dst -> $cur, want $want" >&2
      exit 1
    fi
    return 0
  fi
  if [[ -e "$dst" ]]; then
    echo "refusing to replace $dst with a symlink" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$dst")"
  ln -s "$src" "$dst"
}

# Files present in main but absent from the checkout (ignored or untracked).
link_missing_files() {
  local src_root="$1" dst_root="$2" src dst
  [[ -d "$src_root" ]] || return 0
  while IFS= read -r -d '' src; do
    dst="$dst_root/${src#"$src_root"/}"
    if [[ -e "$dst" || -L "$dst" ]]; then
      continue
    fi
    mkdir -p "$(dirname "$dst")"
    ln -s "$src" "$dst"
  done < <(find "$src_root" -mindepth 1 \( -type f -o -type l \) -print0)
}

exclude_line() {
  local file="$1" line="$2"
  if [[ -f "$file" ]] && grep -qxF "$line" "$file"; then
    return 0
  fi
  printf '%s\n' "$line" >>"$file"
}

ensure_worktree() {
  local lane="$1"
  local wt="$WT_PARENT/rr_harvest_pr999_${lane}"
  local branch="harvest/pr999-${lane}"

  if [[ -d "$wt" ]] && git -C "$MAIN" worktree list --porcelain | grep -Fxq "worktree $wt"; then
    :
  elif [[ -e "$wt" ]]; then
    echo "path exists and is not a worktree: $wt" >&2
    exit 1
  elif git -C "$MAIN" show-ref --verify --quiet "refs/heads/$branch"; then
    git -C "$MAIN" worktree add "$wt" "$branch"
  else
    git -C "$MAIN" worktree add -b "$branch" "$wt" HEAD
  fi

  local actual
  actual="$(git -C "$wt" rev-parse --abbrev-ref HEAD)"
  if [[ "$actual" != "$branch" ]]; then
    echo "worktree $wt is on $actual, want $branch" >&2
    exit 1
  fi

  ensure_symlink "$MAIN/roms" "$wt/roms"
  ensure_symlink "$MAIN/snes/harvest/roms" "$wt/snes/harvest/roms"
  ensure_symlink "$MAIN/snes/harvest/.venv" "$wt/snes/harvest/.venv"
  link_missing_files \
    "$MAIN/snes/harvest/custom_integrations/HarvestMoon-Snes" \
    "$wt/snes/harvest/custom_integrations/HarvestMoon-Snes"
  link_missing_files \
    "$MAIN/snes/harvest/tasks" \
    "$wt/snes/harvest/tasks"

  local exclude pins tapes
  # This git reads the common info/exclude (rev-parse --git-path). A file
  # under .git/worktrees/<id>/info/exclude is not consulted.
  exclude="$(git -C "$wt" rev-parse --path-format=absolute --git-path info/exclude)"
  mkdir -p "$(dirname "$exclude")"
  if [[ ! -f "$exclude" ]]; then
    : >"$exclude"
  fi
  # gitignore `roms/` does not match a symlink. /roms is the required entry.
  exclude_line "$exclude" "/roms"
  exclude_line "$exclude" "/snes/harvest/roms"

  pins="$(find "$wt/snes/harvest/custom_integrations/HarvestMoon-Snes" -maxdepth 1 -name '*.state' -type l | wc -l)"
  tapes="$(find "$wt/snes/harvest/tasks" \( -type l \) | wc -l)"
  echo "worktree=$wt branch=$branch pin_symlinks=$pins task_symlinks=$tapes exclude=$exclude" >&2
  if [[ "$pins" -lt 1 ]]; then
    echo "no pin symlinks in $wt" >&2
    exit 1
  fi
  WT="$wt"
}

run_detached() {
  local wt="$1"
  shift
  if [[ $# -lt 1 ]]; then
    usage
  fi
  local lane log pid py status
  lane="$(basename "$wt")"
  lane="${lane#rr_harvest_pr999_}"
  mkdir -p "$wt/snes/harvest/logs"
  log="$wt/snes/harvest/logs/pr999_${lane}_$(date +%Y%m%d_%H%M%S)_$$.log"
  py="$(python_path_for "$wt")"
  (
    cd "$wt/snes/harvest"
    setsid nohup env \
      HEADLESS=1 \
      SDL_VIDEODRIVER=dummy \
      SDL_AUDIODRIVER=dummy \
      QT_QPA_PLATFORM=offscreen \
      PYGAME_HIDE_SUPPORT_PROMPT=1 \
      PYTHONPATH="$py" \
      "$@" >>"$log" 2>&1 </dev/null &
    pid=$!
    echo "pid=$pid log=$log" >&2
    wait "$pid"
    status=$?
    echo "exit=$status log=$log" >&2
    exit "$status"
  )
}

pytest_wt() {
  local wt="$1"
  shift
  local py
  py="$(python_path_for "$wt")"
  if [[ $# -eq 0 ]]; then
    set -- snes/harvest/tests -q -m "not rom and not ml"
  fi
  (
    cd "$wt"
    export PYTHONPATH="$py"
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
    export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
    export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-dummy}"
    export HEADLESS=1
    export PYGAME_HIDE_SUPPORT_PROMPT=1
    exec "$(venv_bin pytest)" --rootdir "$wt" --confcutdir "$wt" "$@"
  )
}

probe_wt() {
  local wt="$1"
  local state="${2:-Y1_Inside_House}"
  PIN_PROBE_STATE="$state" \
    PIN_PROBE_WT="$wt" \
    PIN_PROBE_MAIN="$MAIN" \
    run_detached "$wt" "$(venv_bin python)" -c '
import os
from pathlib import Path

import numpy as np

import harvest
from harvest.core.ram_catalog import read_ram_value
from harvest.runtime.save_state_io import parse_save_state, resolve_state_path

wt = Path(os.environ["PIN_PROBE_WT"]).resolve()
main = Path(os.environ["PIN_PROBE_MAIN"]).resolve()
src = Path(harvest.__file__).resolve()
if wt not in src.parents:
    raise SystemExit(f"harvest imported from {src}, not under {wt}")
state = os.environ["PIN_PROBE_STATE"]
path = Path(resolve_state_path(state))
if not path.is_symlink():
    raise SystemExit(f"pin is not a symlink: {path}")
real = path.resolve()
if main not in real.parents:
    raise SystemExit(f"pin does not resolve into the main tree: {real}")
ram = np.frombuffer(parse_save_state(path).ram, dtype=np.uint8).copy()
print(f"harvest={src}")
print(f"pin={path}")
print(f"pin_realpath={real}")
print("emulator_frames=0")
for key in ("tilemap", "year", "season", "day", "hour", "minute"):
    print(f"{key}={read_ram_value(ram, key, raw=True)}")
tape = wt / "snes/harvest/tasks/horse_barn_edges.json"
roms = wt / "roms"
print(f"tape_symlink={tape.is_symlink()}")
print(f"roms_symlink={roms.is_symlink()}")
'
}

if [[ $# -lt 1 ]]; then
  usage
fi
LANE="$1"
shift
lane_ok "$LANE" || usage
CMD="${1:-setup}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$CMD" in
  setup)
    ensure_worktree "$LANE"
    ;;
  run)
    if [[ "${1:-}" == "--" ]]; then
      shift
    fi
    ensure_worktree "$LANE"
    run_detached "$WT" "$@"
    ;;
  pytest)
    ensure_worktree "$LANE"
    pytest_wt "$WT" "$@"
    ;;
  probe)
    ensure_worktree "$LANE"
    probe_wt "$WT" "${1:-Y1_Inside_House}"
    ;;
  *)
    usage
    ;;
esac
