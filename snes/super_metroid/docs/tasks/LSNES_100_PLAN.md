# Plan — Sniq 100% (#4010M) as lsnes tooling

YouTube / the TASVideos encode is not the oracle. Tooling is: **same emulator as the publication**, replay the LSMV, dump RAM, then parse hops. Do not STATUS from this.

Publication: [TASVideos #4010M](https://tasvideos.org/4010M) · Sniq SM 100% · **lsnes rr2-β23** · core `bsnes v085 (Compatibility core)`.

## Recheck (2026-08-31, corrected native boot)

| Claim | Result |
|-------|--------|
| `~/Downloads/sniq-sm-100.lsmv.zip` unwraps to vendored `tas/ref/sniq_100_4010M.lsmv` | **same SHA256** `1bd065d8…` |
| ROM `roms/SuperMetroid.sfc` matches movie `rom.sha256` | **yes** `12b77c4b…` |
| Parse to SNES-12 (`tas/lsmv.py`) | **222 788f**, Start/A mash, no resets. Tests: `test_parse_sniq_100_lsmv_4010` |
| Replay under **snes9x** harness | documented **dead-end** (Ceres thrash, items `0`) |
| Replay under BizHawk `sniq_100p.bk2` | converter copy; Linux libsnes **SEGV** |
| Replay under **lsnes-bsnes.exe** + Wine wow64 | **GREEN** with the movie present before core boot; 222 788f readonly |
| First Ceres elev `0xDF45` | **f8319** energy 99 |
| first_control `gs=8` pose 0 @(128,0) | **f8538** |
| Ceres Ridley `0xE0B5` | **f9979**, energy **99→24** during the fight |
| Linear escape | `E06B` f11821 → `E021` f12079 → `DFD7` f12342 → `DF8D` f12671 → `DF45` f12952 |
| Landing `0x91F8` / Zebes | **GREEN f15198**, energy 99, area 0 |
| Encode / “watch the video” | **out of scope** |

Authoritative artifact (gitignored): `recordings/tas_oracle/sniq_100_lsnes/`.
`proof.json` and every `events.jsonl` row parse. `proof.json` reports
`status=GREEN`, `landing_frame=15198`, seven unique rooms, six Ceres rooms,
and one Zebes room. `series.jsonl` contains per-frame RAM through Landing.

### Root cause

The old wrapper booted an empty ROM/movie first, then issued asynchronous
`load-rom` / `load-movie` / `load-readonly` commands from Lua. That is not the
lsnes publication startup path and was already visibly desynced at first
control. The old f11182/f37269/70k observations are invalid sync evidence.

Correct startup passes the LSMV as the positional movie and the ROM as
`--rom-a=...`, so lsnes constructs the core with the movie settings and RTC
before frame zero. (`--rom=...` hits an rr2-β23 single-file type-check bug;
`--load=...` is documented but is not consumed as the startup movie by this
build.) Lua now sees `movie.framecount=222788` at startup and only dumps RAM.

## Goal

One command produces a **native-core** dump `recordings/tas_oracle/sniq_100_lsnes/`: rooms past Ceres, item/beam gains, Landing/morph GREEN. Button parse, Landing GREEN, and `extract_hops` ingestion are done.

## Do next (in order)

1. **Import** — done. `tas/extract_hops.py` accepts `events.jsonl` + `proof.json`
   without `trace.json` / `summary.json`; `source=lsnes_oracle`. Fixture:
   `test_extract_run_lsnes_oracle_without_trace`. Real dump: 12 hops, 12 usable,
   0 desync/thrash. Compact hop table: `tas/bodies/sniq_100_ceres_lsnes_hops.json`.
2. Optional later: continue the same native replay to morph/item gains, or run
   any% `sniq_any_3653M.lsmv` through a parameterized wrapper.

## Do not

- Watch or encode the TASVideos video as evidence.
- Power-on the LSMV/BK2 under snes9x and “annotate harder.”
- Treat BizHawk `sniq_100p.bk2` as the authoring movie.
- STATUS-promote movie frames or oracle dumps.
- Grid-search Climb under harness.

## Commands

```bash
# Parse only (no emulator)
uv run pytest snes/super_metroid/tests/test_tas_movies.py::test_parse_sniq_100_lsmv_4010 \
  snes/super_metroid/tests/test_tas_catalog.py::test_existing_sniq_slices_unchanged -q

# Native replay (lsnes rr2-β23 via Wine). Landing is f15198.
MAX_FRAMES=20000 EARLY_EXIT=1 SERIES_STRIDE=1 \
  ./snes/super_metroid/tas/oracle/run_lsnes_100.sh \
  snes/super_metroid/recordings/tas_oracle/sniq_100_lsnes

# Offline extract (no emulator)
uv run python -m super_metroid.tas.extract_hops \
  snes/super_metroid/recordings/tas_oracle/sniq_100_lsnes
```

Env: `tas/ref/ORACLE_ENV.md`. Wrapper: `tas/oracle/run_lsnes_100.sh`. Lua: `tas/oracle/lsnes_dump_sm.lua`. Host binaries stay out of git (`~/.local/opt/lsnes-rr2-beta23/`, `~/.local/opt/wine`).

## Done when

Completed: `proof.json` parses, status=GREEN at Landing, unique rooms include
Zebes, and `extract_hops` emits a usable board (`source=lsnes_oracle`, 12/12
usable Ceres+Landing hops, no desync/thrash). Continuous tip stays product
pure. Optional: dump past Landing for item/beam gains.
