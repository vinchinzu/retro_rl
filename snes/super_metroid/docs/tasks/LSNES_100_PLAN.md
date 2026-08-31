# Plan — Sniq 100% (#4010M) as lsnes tooling

YouTube / the TASVideos encode is not the oracle. Tooling is: **same emulator as the publication**, replay the LSMV, dump RAM, then parse hops. Do not STATUS from this.

Publication: [TASVideos #4010M](https://tasvideos.org/4010M) · Sniq SM 100% · **lsnes rr2-β23** · core `bsnes v085 (Compatibility core)`.

## Recheck (2026-08-31)

| Claim | Result |
|-------|--------|
| `~/Downloads/sniq-sm-100.lsmv.zip` unwraps to vendored `tas/ref/sniq_100_4010M.lsmv` | **same SHA256** `1bd065d8…` |
| ROM `roms/SuperMetroid.sfc` matches movie `rom.sha256` | **yes** `12b77c4b…` |
| Parse to SNES-12 (`tas/lsmv.py`) | **222 788f**, Start/A mash, no resets. Tests: `test_parse_sniq_100_lsmv_4010` |
| Replay under **snes9x** harness | documented **dead-end** (Ceres thrash, items `0`) |
| Replay under BizHawk `sniq_100p.bk2` | converter copy; Linux libsnes **SEGV** |
| Replay under **lsnes-bsnes.exe** + Wine wow64 | **movie loads** 222 788f readonly |
| First Ceres elev `0xDF45` | **f10962** energy 99 |
| first_control `gs=8` pose 0 @(128,0) | **f11182** (matches harness any% first_control) |
| Ceres Ridley `0xE0B5` | **f37269**, energy **99→25** (fight, not garbage RAM) |
| Landing `0x91F8` / morph / Zebes | **not reached by f70000** — still the six Ceres rooms only |
| Encode / “watch the video” | **out of scope** |

Artifacts (gitignored): `recordings/tas_oracle/sniq_100_lsnes_ceres/` (to first elev) and `…/sniq_100_lsnes/` (70k soak). `dump_log.txt` / `room_timeline.csv` are the truth files; `proof.json` currently has illegal `\h` escapes from Wine `Z:\` paths — dump Lua must JSON-escape.

**Post-Ridley is unverified.** f45663 elev with energy 99, then Falling again, then Ridley **again** at f62639 with energy 99, looks like a Ceres restart, not Zebes. Do not treat the 70k soak as a full-movie sync.

## Goal

One command produces a **native-core** dump `recordings/tas_oracle/sniq_100_lsnes/` that `extract_hops` can read: rooms past Ceres, item/beam gains, Landing/morph GREEN. Button parse is already done.

## Do next (in order)

1. **Fix dump JSON** — escape `\` in Lua `proof.json` / `events.jsonl`. Sidecar is unreadable until this.
2. **Diagnose post-Ridley** (one change): after f37269, did we desync? Suspects: `load-rom`+`load-movie`+`load-readonly` stack, turbo, Wine. Compare a **no-turbo** soak to f45000 against the 70k log. Halt at first unexpected room (second Ridley @99 is the miss class).
3. **Landing GREEN** — `0x91F8` or morph bit, then early-exit. Only after (2) is clean.
4. **Import** — oracle dir → `pins.json` / `extract_hops` with `source=lsnes_oracle`. Prefer this over `sniq_100_full` thrash boards.
5. Optional later: any% `sniq_any_3653M.lsmv` same wrapper.

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

# Native replay (lsnes rr2-β23 via Wine). Intro is ~11k frames.
MAX_FRAMES=15000 EARLY_EXIT=1 \
  ./snes/super_metroid/tas/oracle/run_lsnes_100.sh \
  snes/super_metroid/recordings/tas_oracle/sniq_100_lsnes
```

Env: `tas/ref/ORACLE_ENV.md`. Wrapper: `tas/oracle/run_lsnes_100.sh`. Lua: `tas/oracle/lsnes_dump_sm.lua`. Host binaries stay out of git (`~/.local/opt/lsnes-rr2-beta23/`, `~/.local/opt/wine`).

## Done when

`proof.json` parses, status=GREEN (Landing or morph), unique rooms include Zebes, `extract_hops` on that dir is usable. Continuous tip stays product pure.
