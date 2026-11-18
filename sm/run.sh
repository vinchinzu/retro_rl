#!/usr/bin/env bash
# Native lsnes rr2-β25 ELF + Clean Lua spine through Gravity. No Python.
#
#   ./sm/run.sh                 # Gravity (default), headless
#   TIP=ceres ./sm/run.sh       # Ceres prefix → Landing
#   TIP=credits ./sm/run.sh     # sm/lua/run_credits.lua when that file exists
#   TURBO=1 ./sm/run.sh
#   HEADED=1 ./sm/run.sh        # real display (no dummy SDL)
#
# β25 needs --rom-type=snes and --rom-a=. --rom-a= alone is a null core,
# and input.joyset then throws "Invalid controller". Not the Wine PE.
# Headless default: Xvfb + GDK_BACKEND=x11. wxGTK ignores dummy SDL and
# would otherwise map on the Hyprland session. HEADED=1 shows the windows.
set -euo pipefail

SM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SM_DIR}/.." && pwd)"
LSNES="${LSNES:-${HOME}/.local/opt/lsnes-rr2-beta25/lsnes}"
ROM="${ROM:-${REPO}/roms/SuperMetroid.sfc}"
if [[ -z "${LUA:-}" ]]; then
  tip="${TIP:-gravity}"
  if [[ "${tip}" == "ceres" ]]; then
    LUA="${SM_DIR}/lua/run_ceres.lua"
  elif [[ "${tip}" == "credits" ]]; then
    LUA="${SM_DIR}/lua/run_credits.lua"
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

# wx html needs libmspack. lsnes DT_RUNPATH is not used for that dependency.
ARCH_LIB="${HOME}/.local/opt/arch-prefix/usr/lib"
if [[ -r "${ARCH_LIB}/libmspack.so.0" ]]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${ARCH_LIB}:"*) ;;
    *)
      MSPACK_DIR="${TMPDIR:-/tmp}/lsnes-rr2-mspack"
      mkdir -p "${MSPACK_DIR}"
      ln -sfn "${ARCH_LIB}/libmspack.so.0" "${MSPACK_DIR}/libmspack.so.0"
      case ":${LD_LIBRARY_PATH:-}:" in
        *":${MSPACK_DIR}:"*) ;;
        *) export LD_LIBRARY_PATH="${MSPACK_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
      esac
      ;;
  esac
fi

XVFB_PID=""
LSNES_PID=""
cleanup_children() {
  if [[ -n "${LSNES_PID}" ]] && kill -0 "${LSNES_PID}" 2>/dev/null; then
    kill "${LSNES_PID}" 2>/dev/null || true
    wait "${LSNES_PID}" 2>/dev/null || true
  fi
  if [[ -n "${XVFB_PID}" ]] && kill -0 "${XVFB_PID}" 2>/dev/null; then
    kill "${XVFB_PID}" 2>/dev/null || true
    wait "${XVFB_PID}" 2>/dev/null || true
  fi
}
trap cleanup_children EXIT
trap 'cleanup_children; exit 143' INT TERM

if [[ "${HEADED:-}" == "1" ]]; then
  if [[ "${SDL_VIDEODRIVER:-}" == "dummy" ]]; then
    unset SDL_VIDEODRIVER
  fi
  if [[ "${SDL_AUDIODRIVER:-}" == "dummy" ]]; then
    unset SDL_AUDIODRIVER
  fi
  echo "HEADED=1 (real display)"
else
  xvfb_bin=""
  if [[ -x "${HOME}/.local/bin/Xvfb" ]]; then
    xvfb_bin="${HOME}/.local/bin/Xvfb"
  elif command -v Xvfb >/dev/null 2>&1; then
    xvfb_bin="$(command -v Xvfb)"
  fi
  if [[ -z "${xvfb_bin}" ]]; then
    echo "error: Xvfb missing (wxGTK ignores SDL_VIDEODRIVER=dummy)" >&2
    echo "  user binary: ~/.local/bin/Xvfb" >&2
    exit 1
  fi
  xvfb_display=""
  for n in $(seq 30 39); do
    if [[ ! -e "/tmp/.X${n}-lock" ]]; then
      xvfb_display=":${n}"
      break
    fi
  done
  if [[ -z "${xvfb_display}" ]]; then
    echo "error: no free X display in :30-:39" >&2
    exit 1
  fi
  "${xvfb_bin}" "${xvfb_display}" -screen 0 1280x720x24 -nolisten tcp -ac \
    >/tmp/xvfb-lsnes.log 2>&1 &
  XVFB_PID=$!
  xvfb_sock="/tmp/.X11-unix/X${xvfb_display#:}"
  ready=0
  for _ in $(seq 1 50); do
    if [[ -S "${xvfb_sock}" ]]; then
      ready=1
      break
    fi
    if ! kill -0 "${XVFB_PID}" 2>/dev/null; then
      break
    fi
    sleep 0.05
  done
  if [[ "${ready}" != "1" ]]; then
    echo "error: Xvfb did not open ${xvfb_sock}" >&2
    tail -20 /tmp/xvfb-lsnes.log >&2 || true
    exit 1
  fi
  export DISPLAY="${xvfb_display}"
  unset WAYLAND_DISPLAY
  export GDK_BACKEND=x11
  export SDL_VIDEODRIVER=dummy
  export SDL_AUDIODRIVER=dummy
  echo "headless DISPLAY=${DISPLAY} GDK_BACKEND=${GDK_BACKEND} Xvfb=${XVFB_PID}"
fi

"${LSNES}" --rom-type=snes --rom-a="${ROM}" --lua="${LUA}" "$@" &
LSNES_PID=$!
set +e
wait "${LSNES_PID}"
status=$?
set -e
exit "${status}"
