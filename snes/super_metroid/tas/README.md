# Super Metroid TAS import

HappyLee-style **button-press movies** for Super Metroid, vendored under
`ref/` and sliced into `snes12_rle` JSON under `slices/`.

## Ref movies

| File | Source | Frames | Format |
|------|--------|-------:|--------|
| `ref/sniq_any_3653M.lsmv` | [TASVideos #3653M](https://tasvideos.org/3653M) Sniq any% | 129 712 | lsnes LSMV |
| `ref/sniq_100p.bk2` | [Userfile](https://tasvideos.org/UserFiles/Info/55928342467251616) Sniq 100% | 222 789 | BizHawk BK2 |
| `ref/sniq_any_wip.lsmv` | Sniq WIP → Red Brinstar | 55 037 | LSMV |
| `ref/moozooh_smtc4.bk2` | SM TAS Contest R4 (short) | 5 384 | BK2 |

Author notes (tech + multi-frame chords): [submission #5833](https://tasvideos.org/5833S).

Formats:

- LSMV input: `F.|BYsSudlrAXLR` — [spec](https://tasvideos.org/EmulatorResources/Lsnes/LSMV)
- BK2 `Input Log.txt` + LogKey — [spec](https://tasvideos.org/Bizhawk/BK2Format)

## Commands

```bash
# Movies are gitignored (*.lsmv / *.bk2) — fetch then slice
uv run python -m super_metroid.tas.fetch_refs
uv run python -m super_metroid.tas.export_slices --finish

# List catalog / full RLE export (large JSON under slices/)
uv run python -m super_metroid.tas.export_slices --list
uv run python -m super_metroid.tas.export_slices --all

# Harness replay + WRAM annotate (pose / x,y / vel / rooms / items)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m super_metroid.tas.replay --slice sniq_any_menu --annotate
uv run python -m super_metroid.tas.replay --list-slices
uv run python -m super_metroid.tas.replay --slice sniq_any_full \
  --annotate --series-stride 8 \
  --states-on room_enter,control,item_gain,beam_gain,capacity_gain

# Zebes resync: product pure → Landing, then movie body (search or fixed index)
uv run python -m super_metroid.tas.resync --to landing --movie-start 15000 --body 12000
uv run python -m super_metroid.tas.resync --to landing --search \
  --search-lo 15000 --search-hi 25000 --search-step 500
# Product through Climb → Pit (movie body optional; Pit prefers product first-jump)
uv run python -m super_metroid.tas.resync --to pit --movie-start 17000 --body 2000

# 100% power-on annotate (long; expect Ceres-only desync — pins still useful)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m super_metroid.tas.replay --slice sniq_100_full \
    --annotate --series-stride 8 \
    --states-on room_enter,control,item_gain,beam_gain,capacity_gain \
    --out snes/super_metroid/recordings/tas_import/sniq_100_full

# Hop inventory + skills/graph extraction board (offline; no emulator)
uv run python -m super_metroid.tas.extract_hops \
  snes/super_metroid/recordings/tas_import/sniq_100_full
uv run python -m super_metroid.tas.extract_hops --list-stages

# Materialize control-relative body seeds (movie parse only; status=materialized_unproven)
uv run python -m super_metroid.tas.materialize --stage landing_to_parlor
uv run python -m super_metroid.tas.materialize \
  --from-board snes/super_metroid/recordings/tas_import/resync_zebes_rooms \
  --zebes-only
# → tas/bodies/<stage>.json (or board/bodies/ for --from-board)

# Tests (no emulator for parse; skip if refs missing)
uv run pytest snes/super_metroid/tests/test_tas_movies.py \
  snes/super_metroid/tests/test_tas_trace.py \
  snes/super_metroid/tests/test_tas_stages_extract.py \
  snes/super_metroid/tests/test_tas_materialize.py -q
```

Playbooks:

- Harness hybrid (snes9x re-anchor / Landing→Parlor): [`docs/TAS_ADAPT.md`](../docs/TAS_ADAPT.md)
- **Long path oracle (BizHawk BSNES / lsnes truth dumps):** [`docs/TAS_BSNES_ORACLE.md`](../docs/TAS_BSNES_ORACLE.md) — epic `rr-0lz6`
- **Oracle env (BizHawk version, ROM SHA1, SEGV notes):** [`tas/ref/ORACLE_ENV.md`](ref/ORACLE_ENV.md)
- **Oracle tooling:** [`tas/oracle/`](oracle/) (`run_verify_100.sh`, Phase 1 verify Lua)
- Next-session prompt: [`docs/tasks/NEXT_SESSION_TAS_ORACLE.md`](../docs/tasks/NEXT_SESSION_TAS_ORACLE.md)

Artifacts (harness thrash / hybrid):
`recordings/tas_import/<run_id>/` (`trace.json`, `pins.json`, `series.jsonl`,
optional `states/`, `extraction_board.json`, `hop_inventory.csv`).

Oracle artifacts (target): `recordings/tas_oracle/<run_id>/` (native-core pins only).

## Finish-oriented slices

Tagged `finish` in `slice.SLICE_CATALOG`:

- `sniq_any_full` / `sniq_100_full` — full movies
- `sniq_any_late` / `sniq_any_tourian_escape` / `sniq_any_final_10k`
- `sniq_100_late` / `sniq_100_final_15k`

Windows are **movie frame indices**, not room-ID control points. Re-anchor
under the harness (same class of desync as HappyLee FM2 on fceumm) before
STATUS claims. Prefer late/escape tails for endgame research while pure
KPDR remains the continuous tip.

## Layout

```
tas/
  ref/           # vendored .lsmv / .bk2
  slices/        # snes12_rle JSON + manifest.json
  lsmv.py        # lsnes parser
  bk2.py         # BizHawk parser (LogKey-aware)
  rle.py         # compress / expand
  slice.py       # catalog + export
  stages.py      # RoomStageSpec table (control settle → goal)
  extract_hops.py # hop inventory + skills/graph extraction board
  materialize.py # stage window → snes12_rle body seeds (unproven)
  bodies/        # materialized stage seeds (not STATUS)
  annotate.py    # event detectors (rooms/items/speed/shine/stall)
  trace.py       # emulator replay + series / state dumps
  replay.py      # CLI power-on / slice replay
  resync.py      # product re-anchor + movie splice search
  export_slices.py
```
