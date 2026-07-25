#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GAME_DIR="${SCRIPT_DIR}/custom_integrations/SuperMarioWorld-Snes-v0"
GAME_ROM="${GAME_DIR}/rom.sfc"
EXPECTED_SHA="$(tr -d '[:space:]' < "${GAME_DIR}/rom.sha")"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv not found on PATH." >&2
  exit 1
fi

install_candidate_rom() {
  local candidate="$1"
  local actual_sha

  actual_sha="$(sha1sum "$candidate" | awk '{print $1}')"
  if [[ "$actual_sha" != "$EXPECTED_SHA" ]]; then
    echo "Error: SMW ROM hash mismatch: $candidate" >&2
    echo "Expected: $EXPECTED_SHA" >&2
    echo "Actual:   $actual_sha" >&2
    exit 1
  fi

  mkdir -p "$GAME_DIR"
  ln -sf "$candidate" "$GAME_ROM"
}

if [[ -f "$GAME_ROM" ]]; then
  actual_sha="$(sha1sum "$GAME_ROM" | awk '{print $1}')"
  if [[ "$actual_sha" != "$EXPECTED_SHA" ]]; then
    echo "Error: SMW ROM hash mismatch: $GAME_ROM" >&2
    echo "Expected: $EXPECTED_SHA" >&2
    echo "Actual:   $actual_sha" >&2
    exit 1
  fi
else
  for candidate in \
    "${SCRIPT_DIR}/roms/smw.sfc" \
    "${REPO_ROOT}/roms/smw.sfc" \
    "${REPO_ROOT}/roms/Super Mario World.sfc" \
    "${REPO_ROOT}/roms/Super Mario World (USA).sfc" \
    "${REPO_ROOT}/roms/Super Mario World.smc" \
    "${REPO_ROOT}/roms/Super Mario World (USA).smc" \
    "${SCRIPT_DIR}/Super Mario World.sfc" \
    "${SCRIPT_DIR}/Super Mario World.smc"; do
    if [[ -f "$candidate" ]]; then
      install_candidate_rom "$candidate"
      break
    fi
  done
fi

if [[ ! -f "$GAME_ROM" ]]; then
  echo "Error: SMW ROM not found." >&2
  echo "Place it at: ${SCRIPT_DIR}/roms/smw.sfc" >&2
  echo "Or at:       ${REPO_ROOT}/roms/smw.sfc" >&2
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH=.
exec uv run python -m SMW speedrun "$@"
