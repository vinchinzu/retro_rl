# Issue tracker: bd (beads)

Issues and specs for this repo live in **[bd](https://github.com/steveyegge/beads)**.
Game-local STATUS / plan / MILESTONES stay the verified product board; beads
owns ready / in-flight / blocked work. Full rules: [`docs/BEADS.md`](../BEADS.md).

Prefix: `rr-`. Labels: game name (`super_metroid`, `smb`, …), kind
(`pure`, `graph`, `compose`, `stabilize`, `status`, `meta`). Super Metroid
product sessions also use `spine`.

## Conventions

```bash
bd ready                 # unblocked open work
bd show <id>             # details + deps
bd update <id> --status in_progress
bd close <id> --reason "…"
bd create "Title" -t task -p 0 -l <game> --json
bd dep add <blocked> <blocker>   # blocker must finish first
bd export -o .beads/issues.jsonl
```

- One bead per tracer-bullet slice. Do not invent a parallel tracker
  (no `.scratch/` issues, no GitHub Issues) unless the user asks.
- Blocking edges are `bd dep add <blocked> <blocker>`.
- Commit `.beads/issues.jsonl` with the matching code. Push only if asked.
- Closing a bead is not a STATUS promote.

## When a skill says "publish to the issue tracker"

```bash
bd create "<title>" -t task -p 0 -l <game> -l <kind> --json
```

Then set deps with `bd dep add`. Apply no extra triage label: an unblocked
open bead is already `bd ready` (ready-for-agent).

## When a skill says "fetch the relevant ticket"

`bd show <id>` (for example `rr-abc`). The user will normally pass the id.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a parent bead; each **decision ticket**
is a child.

- **Map**: `bd create "wayfinder: <effort>" -t task -l meta --json`
- **Child ticket**: `bd create "<question>" -t task --deps discovered-from:<map-id> --json`
- **Blocking**: `bd dep add <blocked> <blocker>`
- **Frontier**: `bd ready` (optionally `-l <game>`)
- **Claim**: `bd update <id> --status in_progress`
- **Resolve**: `bd close <id> --reason "<decision gist>"` and point at the
  answer from the parent bead body if needed
