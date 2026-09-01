#!/usr/bin/env bash
# Replay Sniq 100% #4010M under lsnes rr2-β23 + bsnes v085 (authoring core).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SM_ROOT="${REPO_ROOT}/snes/super_metroid"
ORACLE_DIR="${SM_ROOT}/tas/oracle"
MOVIE="${MOVIE:-${SM_ROOT}/tas/ref/sniq_100_4010M.lsmv}"
ROM="${ROM:-${REPO_ROOT}/roms/SuperMetroid.sfc}"
LUA="${ORACLE_DIR}/lsnes_dump_sm.lua"
OUT_DIR_RAW="${1:-${SM_ROOT}/recordings/tas_oracle/sniq_100_lsnes}"
EARLY_EXIT="${EARLY_EXIT:-1}"
MAX_FRAMES="${MAX_FRAMES:-60000}"
SERIES_STRIDE="${SERIES_STRIDE:-0}"
PLAYBACK_SPEED="${PLAYBACK_SPEED:-turbo}"

LSNES_EXE="${LSNES:-${HOME}/.local/opt/lsnes-rr2-beta23/lsnes-bsnes.exe}"
WINE_BIN="${WINE:-${HOME}/.local/opt/wine/bin/wine}"
WINEPREFIX="${WINEPREFIX:-${HOME}/.local/share/lsnes-wine}"
export WINEPREFIX WINEDEBUG="${WINEDEBUG:--all}"
export WINEDLLOVERRIDES="${WINEDLLOVERRIDES:-winemenubuilder.exe=d}"
export DISPLAY="${DISPLAY:-:0}"

if [[ ! -x "${WINE_BIN}" ]]; then
  echo "error: wine not found at ${WINE_BIN}" >&2
  echo "  install wine, or unpack Kron4ek wine-*-amd64-wow64 to ~/.local/opt/wine" >&2
  exit 1
fi
if [[ ! -f "${LSNES_EXE}" ]]; then
  echo "error: lsnes-bsnes.exe missing: ${LSNES_EXE}" >&2
  echo "  extract lsnes-rr2-beta23.7z (TASVideos) to ~/.local/opt/lsnes-rr2-beta23" >&2
  exit 1
fi
if [[ ! -f "${MOVIE}" ]]; then
  echo "error: movie missing: ${MOVIE}" >&2
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

rom_sha1="$(sha1sum "${ROM}" | awk '{print toupper($1)}')"
rom_sha256="$(sha256sum "${ROM}" | awk '{print $1}')"
expect_sha1="DA957F0D63D14CB441D215462904C4FA8519C613"
expect_sha256="12b77c4bc9c1832cee8881244659065ee1d84c70c3d29e6eaf92e6798cc2ca72"
if [[ "${rom_sha1}" != "${expect_sha1}" ]]; then
  echo "error: ROM SHA1 ${rom_sha1} != ${expect_sha1}" >&2
  exit 1
fi
if [[ "${rom_sha256}" != "${expect_sha256}" ]]; then
  echo "error: ROM SHA256 ${rom_sha256} != movie rom.sha256" >&2
  exit 1
fi

mkdir -p "${OUT_DIR_RAW}"
OUT_DIR="$(realpath "${OUT_DIR_RAW}")"
MOVIE="$(realpath "${MOVIE}")"
ROM="$(realpath "${ROM}")"
LUA="$(realpath "${LUA}")"
LSNES_EXE="$(realpath "${LSNES_EXE}")"

# First-run prefix (quiet). Lua inside Wine needs Windows paths.
if [[ ! -d "${WINEPREFIX}/drive_c" ]]; then
  echo "wineboot prefix ${WINEPREFIX}"
  "${WINE_BIN}" wineboot --init >/dev/null 2>&1 || true
fi
winpath() {
  "${WINE_BIN}" winepath -w "$1"
}
# Copy inputs into the prefix so lsnes sees short 8.3-safe Windows paths
# (Z:\home\v\... is easy for argv/Lua to mangle).
STAGE="${WINEPREFIX}/drive_c/lsnes"
mkdir -p "${STAGE}"
cp -f "${ROM}" "${STAGE}/SuperMetroid.sfc"
cp -f "${MOVIE}" "${STAGE}/sniq_100_4010M.lsmv"
cp -f "${LUA}" "${STAGE}/lsnes_dump_sm.lua"
OUT_WIN="$(winpath "${OUT_DIR}")"
MOVIE_WIN="C:\\lsnes\\sniq_100_4010M.lsmv"
ROM_WIN="C:\\lsnes\\SuperMetroid.sfc"
LUA_WIN="C:\\lsnes\\lsnes_dump_sm.lua"
# Lua resolves oracle_flags.txt next to the script (now C:\lsnes).
{
  echo "out_dir=${OUT_WIN}"
  echo "early_exit=${EARLY_EXIT}"
  echo "max_frames=${MAX_FRAMES}"
  echo "series_stride=${SERIES_STRIDE}"
  echo "playback_speed=${PLAYBACK_SPEED}"
  echo "rom=C:/lsnes/SuperMetroid.sfc"
  echo "movie=C:/lsnes/sniq_100_4010M.lsmv"
} > "${STAGE}/oracle_flags.txt"
printf '%s\n' "${OUT_WIN}" > "${STAGE}/oracle_out_dir.txt"

{
  echo "out_dir=${OUT_WIN}"
  echo "early_exit=${EARLY_EXIT}"
  echo "max_frames=${MAX_FRAMES}"
  echo "series_stride=${SERIES_STRIDE}"
  echo "playback_speed=${PLAYBACK_SPEED}"
} > "${ORACLE_DIR}/oracle_flags.txt"
cp "${ORACLE_DIR}/oracle_flags.txt" "${OUT_DIR}/oracle_flags.txt"
printf '%s\n' "${OUT_WIN}" > "${ORACLE_DIR}/oracle_out_dir.txt"
printf '%s\n' "${OUT_WIN}" > "${OUT_DIR}/out_dir.txt"

python3 - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path
meta = {
    "launched_at": datetime.now(timezone.utc).isoformat(),
    "emulator": "lsnes rr2-beta23",
    "core": "bsnes v085 (Compatibility core)",
    "lsnes_exe": r"""${LSNES_EXE}""",
    "wine": r"""${WINE_BIN}""",
    "wineprefix": r"""${WINEPREFIX}""",
    "rom": r"""${ROM}""",
    "rom_sha1": "${rom_sha1}",
    "rom_sha256": "${rom_sha256}",
    "movie": r"""${MOVIE}""",
    "movie_publication": "https://tasvideos.org/4010M",
    "lua": r"""${LUA}""",
    "out_dir": r"""${OUT_DIR}""",
    "out_dir_win": r"""${OUT_WIN}""",
    "early_exit": "${EARLY_EXIT}",
    "max_frames": int("${MAX_FRAMES}"),
    "series_stride": int("${SERIES_STRIDE}"),
    "playback_speed": "${PLAYBACK_SPEED}",
}
Path(r"""${OUT_DIR}""" + "/meta_launch.json").write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps(meta, indent=2))
PY

echo "Launching lsnes 100% #4010M → ${OUT_DIR}"
echo "  ROM SHA256 ${rom_sha256}"
echo "  early_exit=${EARLY_EXIT} max_frames=${MAX_FRAMES}"
echo "  playback_speed=${PLAYBACK_SPEED}"

cd "$(dirname "${LSNES_EXE}")"
# Boot the ROM and movie together.  This is lsnes' native startup path: it
# constructs the core with the movie's settings and RTC before the first
# emulated frame.  Loading a blank ROM first and replacing its movie later is
# not equivalent for a frame-perfect TAS.
exec "${WINE_BIN}" "${LSNES_EXE}" \
  --rom-a="C:/lsnes/SuperMetroid.sfc" \
  --lua="C:/lsnes/lsnes_dump_sm.lua" \
  "C:/lsnes/sniq_100_4010M.lsmv"
