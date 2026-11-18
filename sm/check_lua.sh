#!/usr/bin/env bash
# Syntax-check every sm/lua/**/*.lua (parse only; do not load or run).
# PUC-Rio /usr/bin/lua5.1 has no -p — that flag is /usr/bin/luac5.1 -p (Lua 5.1.5).
set -euo pipefail

SM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SM_DIR}/.." && pwd)"
LUAC="/usr/bin/luac5.1"

if [[ ! -x "${LUAC}" ]]; then
  echo "error: luac5.1 missing: ${LUAC}" >&2
  exit 1
fi

cd "${REPO}"

ok=0
fail=0

while IFS= read -r -d '' f; do
  if err="$("${LUAC}" -p -- "${f}" 2>&1)"; then
    ok=$((ok + 1))
  else
    fail=$((fail + 1))
    printf '%s: %s\n' "${f}" "${err//$'\n'/ }"
  fi
done < <(find sm/lua -name '*.lua' -print0 | sort -z)

printf 'ok=%d fail=%d\n' "${ok}" "${fail}"
if [[ "${fail}" -ne 0 ]]; then
  exit 1
fi
