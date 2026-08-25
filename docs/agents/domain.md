# Domain docs

How the engineering-hygiene skills should consume this repo's domain
documentation. This is a multi-game monorepo, not a single `CONTEXT.md`.

## Before exploring, read these

- Root [`AGENTS.md`](../../AGENTS.md) — repo-wide layout, import names, bd
- [`docs/BEADS.md`](../BEADS.md) — issue tracker
- [`docs/REPO_HYGIENE.md`](../REPO_HYGIENE.md) — when shared docs may grow
- Nearest game `AGENTS.md` under `snes/<game>/` or `nes/<game>/` for the
  files you are about to touch
- Game `STATUS.md` / `plan.md` in that same tree (verified vs future)

If `CONTEXT.md` or `docs/adr/` do not exist, **proceed silently**. Do not
flag their absence; `/domain-modeling` and `/grill-with-docs` create them
lazily when a term or decision actually gets resolved.

## File structure

```
/
├── AGENTS.md
├── docs/
│   ├── BEADS.md
│   ├── REPO_HYGIENE.md
│   └── agents/          ← this folder
├── snes/<game>/AGENTS.md
└── nes/<game>/AGENTS.md
```

Do not grow root `AGENTS.md` with game-specific workflow. Prefer the
nearest local `AGENTS.md`.

## Use the glossary's vocabulary

When your output names a domain concept, use the term as defined in the
nearest `AGENTS.md` and in `docs/GAME_MATRIX.md` / `docs/manifests/*.yaml`.
Short import names (`import alttp`, `import smb`) and tree paths
(`snes/super_metroid`, not `super_metroid_rl/`) are load-bearing.

If the concept you need isn't in those files, either you are inventing
language the project doesn't use (reconsider) or there is a real gap
(note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts a recorded decision in a game `AGENTS.md` or an
ADR under `docs/adr/`, surface it rather than silently overriding.
