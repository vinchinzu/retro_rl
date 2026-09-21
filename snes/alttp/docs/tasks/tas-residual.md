# Living residual — TAS import (`rr-ejn`)

Planner owns `docs/STATUS.md`. This sitting: parser pipeline + catalog +
tiny refs. Did not STATUS-promote. Did not edit `opening_route`. Did not
claim a slice plays.

## Done this sitting

- Catalog of TASVideos [game 191](https://tasvideos.org/191G) published
  movies + useful userfiles (`.lsmv` / `.bk2` / `.smv`). GitHub had no
  extra machine-readable any% button logs.
- Thin parsers wrapping `retro_harness.bk2` + `super_metroid.tas.{lsmv,smv,rle}`.
- `snes12_rle` export (`game_name=Zelda3-Snes`) and stub slice ids
  `castle` / `sewers` / `sanctuary` (unverified; export refused).
- Vendored small unwrapped refs under `tas/ref/` (gitignored `*.lsmv`/`*.bk2`):
  - `fmp_geg_3898M.lsmv` (#3898M, 3277f, sha256 `58a36cf1…`)
  - `m_riss_100_wip_10k.bk2` (USA 10k WIP, sha256 `e3ddf701…`)
  - `taseditor_glitched.lsmv` (8127f, sha256 `6f73361a…`)
- Fetch-only (URL + sha in `tas/catalog.py`, not copied as blobs):
  Tompa #1269M SMV 274 264f, fmp NMG BK2 113 135f, full-inventory #3874M
  190 660f.

## Leftover

1. **gitignore** — `**/tas/` + `*.lsmv`/`*.bk2`/`*.smv` hide this package.
   Needs the same un-ignore block SM/SMB have (`!snes/alttp/tas/`,
   `!snes/alttp/tas/**/*.py`, `!snes/alttp/tas/README.md`). Out of bead
   file-ownership this sitting.
2. **Measure opening windows** on Tompa #1269M (USA) or fmp NMG (JP) for
   castle / sewers / sanctuary. Stubs stay unverified until then.
3. **Replay / annotate** under stable-retro is not built. JP vs USA ROM
   mismatch: GEG/NMG/full-inv are JP 1.0; workspace is USA.
4. Fetch Tompa SMV (~549 KB unwrapped) and NMG/full-inv when measuring
   windows: `uv run python -m alttp.tas.fetch_refs`.

Leave proof (no ROM): `uv run pytest snes/alttp/tests/test_tas.py -q`
(from `snes/`: `uv run pytest alttp/tests/test_tas.py -q`).
