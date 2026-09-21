# Regenerate the game matrix

The game matrix is the generated board at `docs/GAME_MATRIX.md`. Users edit `docs/manifests/*.yaml` and run the generator; they do not hand-edit the tables.

## Sub-features

- `matrix-write` writes `docs/GAME_MATRIX.md` and prints `Wrote docs/GAME_MATRIX.md`.
- `matrix-count` keeps the `Manifest count: **N**` line in sync with the number of `docs/manifests/*.yaml` files.
- `matrix-slugs` includes in-repo slugs such as `tmnt_iv`, `zelda_i`, and `smb`.

## How to get to it (user POV)

- Run `uv run python docs/generate_game_matrix.py` from the repo root (README / `docs/README.md`).
- Confirm with `uv run pytest tests/test_docs.py -q` after a manifest edit.

## Driving it with verify-retro-rl

Preconditions:

- Doctor reports `ok=true` and a `manifest_count` of 1 or more.
- No other process is running the generator against this working tree.
- Snapshot `docs/GAME_MATRIX.md` into `$RUN/GAME_MATRIX.before.md` before writing.

- **Generate.** Run `.cursor/skills/verify-retro-rl/scripts/cli --out $RUN/game-matrix -- uv run python docs/generate_game_matrix.py`. `exit.txt` is `0`. `stdout.txt` contains `Wrote docs/GAME_MATRIX.md`.
- **Read the file.** Copy `docs/GAME_MATRIX.md` to `$RUN/GAME_MATRIX.after.md`. The after file starts with `# Game Matrix`, contains `Generated from \`docs/manifests/*.yaml\``, `Manifest count: **N**` matching doctor `manifest_count`, and the slugs `tmnt_iv`, `zelda_i`, and `smb`. It does not contain `Ladder rank`.
- **Diff.** `diff -u $RUN/GAME_MATRIX.before.md $RUN/GAME_MATRIX.after.md` captured to `$RUN/GAME_MATRIX.diff`. An empty diff means the committed board was already in sync. A nonempty diff means the generator ran and the committed file was stale or a manifest changed — keep the diff as evidence.
- **Restore (run hygiene).** Copy `$RUN/GAME_MATRIX.before.md` back to `docs/GAME_MATRIX.md` unless the user asked to keep the regenerated board. Evidence retains before/after/diff.
- **Proof.** `exit.txt` is `0`, stdout has the Wrote line, and the after snapshot has the header, count, and slugs.

## Gotchas

- The generator always overwrites the tracked file. Two concurrent runs will clobber each other; refuse the second.
- Hand-edits to `docs/GAME_MATRIX.md` are wiped on the next generate. Edit manifests instead.
- `tests/test_docs.py::test_game_matrix_is_generated` reads the file on disk; it does not run the generator. A green docs test after a skipped generate is not this feature.
- Manifest count is the number of `docs/manifests/*.yaml` files, including planned games without a `directory` key.
