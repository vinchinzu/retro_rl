# ALTTP Gauntlet

Ownership boundary for **combat / arena experiments** (GT-style fights,
gauntlet envs, RL training).

## Rules

- Do **not** import opening-route continuous evidence from this package.
- Do **not** put Sanctuary / Zelda continuous segments here.
- Core combat helpers used by the opening route stay in
  `alttp.primitives` (`fight_nearby`, etc.).
- Historical plan references (`gauntlet_env.py`, etc.) lived at the
  `alttp/` root; new work belongs under this folder.

## Status

Empty shell as of the 2026 package split. Opening continuous path takes
priority until Sanctuary is truthful.
