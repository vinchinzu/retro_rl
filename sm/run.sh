#!/usr/bin/env bash
# Native lsnes rr2-β25 ELF + Clean Lua spine through Gravity. No Python.
#
#   ./sm/run.sh                 # Gravity (default)
#   TIP=ceres ./sm/run.sh       # Ceres prefix → Landing
#   TURBO=1 ./sm/run.sh
#
# β25 ELF argv is --rom-a= (not --rom=; β23 type-check bug). Not the Wine PE.
set -euo pipefail

SM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SM_DIR}/.." && pwd)"
LSNES="${LSNES:-${HOME}/.local/opt/lsnes-rr2-beta25/lsnes}"
ROM="${ROM:-${REPO}/roms/SuperMetroid.sfc}"
if [[ -z "${LUA:-}" ]]; then
  if [[ "${TIP:-gravity}" == "ceres" ]]; then
    LUA="${SM_DIR}/lua/run_ceres.lua"
  else
    LUA="${SM_DIR}/lua/run_gravity.lua"
  fi
fi
EXPECT_SHA256="12b77c4bc9c1832cee8881244659065ee1d84c70c3d29e6eaf92e6798cc2ca72"

if [[ ! -e "${LSNES}" ]]; then
  echo "error: lsnes missing: ${LSNES}" >&2
  echo "  native ELF (not Wine PE) at ~/.local/opt/lsnes-rr2-beta25/lsnes" >&2
  exit 1
fi
if [[ ! -x "${LSNES}" ]]; then
  echo "error: lsnes not executable: ${LSNES}" >&2
  exit 1
fi
sig="$(od -An -N4 -tx1 "${LSNES}" | tr -d ' \n')"
if [[ "${sig}" != "7f454c46" ]]; then
  echo "error: LSNES is not a native ELF (magic ${sig}); not the Wine PE" >&2
  echo "  want ~/.local/opt/lsnes-rr2-beta25/lsnes" >&2
  exit 1
fi
if [[ ! -f "${ROM}" ]]; then
  echo "error: ROM missing: ${ROM}" >&2
  exit 1
fi
if [[ ! -f "${LUA}" ]]; then
  echo "error: lua missing: ${LUA}" >&2
  exit 1
fi

got_sha256="$(sha256sum "${ROM}" | awk '{print $1}')"
if [[ "${got_sha256}" != "${EXPECT_SHA256}" ]]; then
  echo "error: ROM SHA-256 ${got_sha256} != ${EXPECT_SHA256}" >&2
  exit 1
fi

mkdir -p "${SM_DIR}/recordings"

echo "lsnes ${LSNES}"
echo "ROM   ${ROM}"
echo "SHA   ${got_sha256}"
echo "LUA   ${LUA}"
if [[ "${TURBO:-}" == "1" ]]; then
  echo "TURBO=1 (Lua may exec set-speed turbo)"
  export TURBO=1
fi

exec "${LSNES}" --rom-a="${ROM}" --lua="${LUA}" "$@"
