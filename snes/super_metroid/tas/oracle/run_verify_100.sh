#!/usr/bin/env bash
# Phase 1: play tas/ref/sniq_100p.bk2 under BizHawk BSNES and dump verify proof.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SM_ROOT="${REPO_ROOT}/snes/super_metroid"
ORACLE_DIR="${SM_ROOT}/tas/oracle"
MOVIE="${SM_ROOT}/tas/ref/sniq_100p.bk2"
ROM="${REPO_ROOT}/roms/SuperMetroid.sfc"
LUA="${ORACLE_DIR}/verify_movie_sync.lua"
BIZHAWK="${BIZHAWK:-${HOME}/.local/bin/bizhawk}"
BH_HOME="${BH_HOME:-${HOME}/.bizhawk}"
OUT_DIR_RAW="${1:-${SM_ROOT}/recordings/tas_oracle/sniq_100_bsnes_verify}"
EARLY_EXIT="${EARLY_EXIT:-1}"
# SM intro/menus alone are ~3–5 min (~11–18k frames). Default past early Zebes.
MAX_FRAMES="${MAX_FRAMES:-60000}"
# LUA_SCRIPT=long_count.lua for trusted long intro wait; default verify_movie_sync.lua
LUA_SCRIPT="${LUA_SCRIPT:-verify_movie_sync.lua}"
LUA="${ORACLE_DIR}/${LUA_SCRIPT}"

if [[ ! -x "${BIZHAWK}" ]]; then
  echo "error: bizhawk not found at ${BIZHAWK}" >&2
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

# BizHawk wrapper cds into ~/.bizhawk — all CLI paths must be absolute.
MOVIE="$(realpath "${MOVIE}")"
ROM="$(realpath "${ROM}")"
LUA="$(realpath "${LUA}")"
mkdir -p "${OUT_DIR_RAW}"
OUT_DIR="$(realpath "${OUT_DIR_RAW}")"

rom_sha="$(sha1sum "${ROM}" | awk '{print toupper($1)}')"
expect_sha="DA957F0D63D14CB441D215462904C4FA8519C613"
if [[ "${rom_sha}" != "${expect_sha}" ]]; then
  echo "error: ROM SHA1 ${rom_sha} != ${expect_sha}" >&2
  exit 1
fi

CFG="${OUT_DIR}/bizhawk_config.ini"
python3 - "${BH_HOME}/config.ini" "${CFG}" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
cfg = json.loads(open(src).read())
cfg.setdefault("PreferredCores", {})["SNES"] = "BSNES"
# Turbo without frame-skip: skipping frames can desync movies / crash waterbox.
cfg["SpeedPercent"] = 400
cfg["SpeedPercentAlternate"] = 400
cfg["FrameSkip"] = 0
cfg["Unthrottled"] = True
cfg["ClockThrottle"] = False
cfg["SoundThrottle"] = False
cfg["VSync"] = False
cfg["VSyncThrottle"] = False
cfg["StartPaused"] = False
cfg["RunLuaDuringTurbo"] = True
cfg["SoundEnabled"] = False
cfg["DispMethod"] = 1  # GDI+ on Linux
cfg.setdefault("Movies", {})["MovieEndAction"] = 0  # Finish
cfg["PlayMovieMatchHash"] = True
open(dst, "w").write(json.dumps(cfg, indent=2))
print(f"wrote {dst}")
PY

# Lua resolves out_dir from a sidecar next to the .lua (BizHawk often sets cwd
# to the script directory when loading --lua).
printf '%s\n' "${OUT_DIR}" > "${ORACLE_DIR}/oracle_out_dir.txt"
printf '%s\n' "${OUT_DIR}" > "${BH_HOME}/oracle_out_dir.txt"
printf '%s\n' "${OUT_DIR}" > "${OUT_DIR}/out_dir.txt"
{
  echo "out_dir=${OUT_DIR}"
  echo "early_exit=${EARLY_EXIT}"
  echo "max_frames=${MAX_FRAMES}"
} > "${ORACLE_DIR}/oracle_flags.txt"
cp "${ORACLE_DIR}/oracle_flags.txt" "${OUT_DIR}/oracle_flags.txt"

META="${OUT_DIR}/meta_launch.json"
python3 - <<PY
import json, subprocess, pathlib
from datetime import datetime, timezone
bh = pathlib.Path("${BIZHAWK}").expanduser()
ver = subprocess.check_output([str(bh), "--version"], text=True, stderr=subprocess.STDOUT)
# version line often mixed with mono noise
version_line = next((ln.strip() for ln in ver.splitlines() if "Version" in ln or ln.strip().startswith("2.")), ver.strip().splitlines()[-1])
meta = {
    "launched_at": datetime.now(timezone.utc).isoformat(),
    "bizhawk_bin": str(bh),
    "bizhawk_home": "${BH_HOME}",
    "bizhawk_version_raw": ver.strip(),
    "bizhawk_version_line": version_line,
    "rom": "${ROM}",
    "rom_sha1": "${rom_sha}",
    "movie": "${MOVIE}",
    "lua": "${LUA}",
    "out_dir": "${OUT_DIR}",
    "early_exit": "${EARLY_EXIT}",
    "max_frames": int("${MAX_FRAMES}"),
    "preferred_core": "BSNES",
    "movie_header_core": "BSNES",
    "movie_sync_profile": "Compatibility",
}
pathlib.Path("${META}").write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps(meta, indent=2))
PY

echo "Launching BizHawk movie verify → ${OUT_DIR}"
echo "  ROM SHA1 ${rom_sha}"
echo "  early_exit=${EARLY_EXIT} max_frames=${MAX_FRAMES}"

# Movie + lua. ROM last per BizHawk help.
# Avoid --chromeless on Linux/Mono (X11 BadMatch flakiness observed).
# Sidecar oracle_out_dir.txt / oracle_flags.txt are authoritative for Lua.
exec env SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-dummy}" \
  "${BIZHAWK}" \
  --config="${CFG}" \
  --movie="${MOVIE}" \
  --lua="${LUA}" \
  --userdata="out_dir:${OUT_DIR};early_exit:${EARLY_EXIT};max_frames:${MAX_FRAMES}" \
  "${ROM}"
