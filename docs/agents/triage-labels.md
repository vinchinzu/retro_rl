# Triage roles → beads

Canonical Matt Pocock triage roles, mapped onto beads status. Do not create
duplicate GitHub-style labels. Game/kind labels stay as in `docs/BEADS.md`.

| Role | Beads |
|---|---|
| `needs-triage` | `open` and not yet inspected (`bd show`, then classify) |
| `needs-info` | `blocked` (or leave open with a comment on what is missing) |
| `ready-for-agent` | `open` and unblocked — this is `bd ready` |
| `ready-for-human` | `open` plus a comment that a human must act; do not claim it |
| `wontfix` | `bd close <id> --reason "wontfix: …"` |

`/to-tickets` publishes slices already in `ready-for-agent` shape: unblocked
open beads. Do not run `/triage` on those.

`/triage` is only for issues you did not create (incoming bugs, raw requests).
