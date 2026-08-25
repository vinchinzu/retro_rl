# Issue tracking — bd (beads)

This monorepo uses **[bd](https://github.com/steveyegge/beads)** (Steve Yegge)
for agent work tracking. Game-local STATUS / plan / MILESTONES stay the
**verified product board**; beads owns **ready / in-flight / blocked** work
with first-class dependencies.

```bash
bd ready                 # unblocked open work
bd show <id>             # details + deps
bd update <id> --status in_progress
bd close <id> --reason "…"
bd create "Title" -t task -p 0 -l super_metroid --json
bd dep add <blocked> <blocker>   # blocker must finish first
bd export -o .beads/issues.jsonl  # then commit with the code
```

## Rules

1. Start sessions with `bd ready` (SM: `bd ready -l super_metroid -l spine`;
   Zelda I: `bd ready -l zelda_i -l spine`; Harvest:
   `bd ready -l harvest -l spine`).
2. Claim one issue before coding; do not invent parallel trackers.
3. Discovered work → `bd create … --deps discovered-from:<parent>`.
4. Product evidence still lives under `<console>/<game>/docs/` (STATUS,
   pure-first, natural-entry). Closing a bead is not a STATUS promote.
5. Commit `.beads/issues.jsonl` with the code that matches it.
6. Session end: close/update issues → `bd export -o .beads/issues.jsonl` →
   commit. Push only when the human asked for a push (do not force-push
   beads/history).

Prefix: `rr-`. Labels: game name (`super_metroid`, `smb`, …), kind
(`pure`, `graph`, `compose`, `stabilize`, `status`, `meta`). Super Metroid,
Zelda I, and Harvest product sessions use **`spine`** as well:
`bd ready -l super_metroid -l spine`, `bd ready -l zelda_i -l spine`,
`bd ready -l harvest -l spine` (TAS/oracle/library and non-serial buffet
stay off this filter).

Game process still applies (e.g. Super Metroid pure-first in
`snes/super_metroid/AGENTS.md`).

## Landing the plane (session end)

1. File or update beads for remaining work
2. Run the narrowest tests for files you changed
3. Close finished issues; leave in_progress honest
4. `bd export -o .beads/issues.jsonl` and commit code + that file together
5. Push only if the user requested it
6. Hand off: `bd ready` + one-line next action
