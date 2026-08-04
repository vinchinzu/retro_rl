#!/usr/bin/env bash
# Evidence parsing for the room-farm rollup.  Kept separate so it can be
# unit-tested without launching a farm or touching its log directory.

# Print the declared status from a schema-compliant residual/note, if present.
# A result heading alone is not enough: the first outcome line below it must
# explicitly declare GREEN, RED, BLOCKED, or PARTIAL.
farm_residual_result() {
  local card_id="$1"
  local tasks_dir="$2"
  local suffix file result

  for suffix in -residual.md -note.md; do
    file="$tasks_dir/${card_id}${suffix}"
    [[ -f "$file" ]] || continue
    result="$(awk '
      /^[[:space:]]*###[[:space:]]+[Rr]esult[[:space:]]*$/ {
        in_result = 1
        next
      }
      in_result && /^[[:space:]]*#{1,3}[[:space:]]/ { exit }
      in_result {
        line = toupper($0)
        gsub(/^[[:space:]]+/, "", line)
        gsub(/^[`*_]+/, "", line)
        if (line ~ /^GREEN([[:space:][:punct:]]|$)/) { print "GREEN"; exit }
        if (line ~ /^RED([[:space:][:punct:]]|$)/) { print "RED"; exit }
        if (line ~ /^BLOCKED([[:space:][:punct:]]|$)/) { print "BLOCKED"; exit }
        if (line ~ /^PARTIAL([[:space:][:punct:]]|$)/) { print "PARTIAL"; exit }
      }
    ' "$file")"
    if [[ -n "$result" ]]; then
      printf '%s\n' "$result"
      return 0
    fi
  done
  return 1
}

# JSON evidence comes from the runner, not a worker's prose.  Do not accept
# shell-style `success=true`, which is easy for a final message to imitate.
farm_log_has_json_success() {
  local log="$1"
  rg -q '"success"[[:space:]]*:[[:space:]]*true([[:space:],}]|$)' "$log"
}

# A GREEN classification requires either an explicit residual result or a
# runner JSON success signal.  Explicit residual outcomes always win over an
# earlier runner success (for example, when a later promote/recheck failed).
farm_card_result() {
  local card_id="$1"
  local log="$2"
  local tasks_dir="$3"
  local residual_result

  residual_result="$(farm_residual_result "$card_id" "$tasks_dir" || true)"
  if [[ -n "$residual_result" ]]; then
    printf '%s\n' "$residual_result"
  elif farm_log_has_json_success "$log"; then
    printf 'GREEN\n'
  else
    printf 'NO_EVIDENCE\n'
  fi
}
