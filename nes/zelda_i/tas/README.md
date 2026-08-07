# Legend of Zelda (NES) TAS import

Button-press movies under `tas/ref/` (gitignored; re-fetch).

**Default = non-glitch.** TASVideos has no formal glitchless any%. We vendor
publications **without** the [Heavy glitch abuse](https://tasvideos.org/Movies-bugs)
tag — i.e. the **all-items** branch.

## Ref movies (default)

| File | Source | Frames | Format | ROM | Glitch tag |
|------|--------|--------|--------|-----|------------|
| `ref/chatterbox_allitems_4767M.fm2` | [4767M](https://tasvideos.org/4767M) chatterbox all-items | **114 913** / 31:52.07 | FCEUX FM2 | USA **PRG1** ✓ | none (soft reset / damage only) |
| `ref/taseditor_allitems_2508M.fm2` | [2508M](https://tasvideos.org/2508M) TASeditor all-items | ~32:17 | FCEUX FM2 | USA PRG0 | none |

Author notes: [submission #7565](https://tasvideos.org/7565S) (route, item
definition, deliberate skip of recorder-warp).

## Glitched (optional)

```bash
uv run python -m zelda_i.tas.fetch_refs --include-glitched
```

Fetches Lord Tom any% / swordless, 2nd-quest any%, FDS game-end glitch.
Not for Clean STATUS adaptation.

FM2 input: `|cmd|RLDUTSBA|` → NES-9
`[B, null, SELECT, START, UP, DOWN, LEFT, RIGHT, A]`.

## Commands

```bash
uv run python -m zelda_i.tas.fetch_refs
uv run python -m zelda_i.tas.import_fm2 --summary-only

uv run python -m zelda_i.tas.import_fm2 \
  nes/zelda_i/tas/ref/chatterbox_allitems_4767M.fm2 \
  --out nes/zelda_i/models/zelda_allitems_raw.json \
  --route-id zelda_chatterbox_allitems

uv run pytest nes/zelda_i/tests/test_fm2.py -q
```

## Layout

```text
tas/
  ref/           # vendored .fm2 (gitignored)
  fm2.py
  fetch_refs.py  # --include-glitched for heavy-glitch movies
  import_fm2.py
  README.md
```
