# Super Metroid TAS import

HappyLee-style **button-press movies** for Super Metroid, vendored under
`ref/` and sliced into `snes12_rle` JSON under `slices/`. Catalog:
[`catalog.py`](catalog.py) (TASVideos [game 121](https://tasvideos.org/121G) +
[userfiles](https://tasvideos.org/UserFiles/Game/121)). ROM hacks and RAM
watches are listed as skips. Another agent owns moonfall policy — this
tree only vendors inputs for later skill parse.

## Ref movies

| File | Source | Frames | Format |
|------|--------|-------:|--------|
| `ref/sniq_any_3653M.lsmv` | [TASVideos #3653M](https://tasvideos.org/3653M) Sniq any% | 129 712 | lsnes LSMV |
| `ref/sniq_100p.bk2` | [Userfile](https://tasvideos.org/UserFiles/Info/55928342467251616) Sniq 100% | 222 789 | BizHawk BK2 |
| `ref/sniq_any_wip.lsmv` | Sniq WIP → Red Brinstar | 55 037 | LSMV |
| `ref/moozooh_smtc4.bk2` | SM TAS Contest R4 (short, **not vanilla**) | 5 384 | BK2 |
| `ref/sniq_low_3273M.lsmv` | [TASVideos #3273M](https://tasvideos.org/3273M) Sniq 13% | 167 797 | LSMV |
| `ref/sniq_100_4010M.lsmv` | [TASVideos #4010M](https://tasvideos.org/4010M) Sniq 100% native | 222 788 | LSMV |
| `ref/sniq_geg_5238M.lsmv` | [TASVideos #5238M](https://tasvideos.org/5238M) game-end-glitch | 18 640 | LSMV |
| `ref/sniq_any_3362M.lsmv` | [TASVideos #3362M](https://tasvideos.org/3362M) prior any% | 135 769 | LSMV |
| `ref/total_13pct_charge_speed.lsmv` | [Userfile](https://tasvideos.org/UserFiles/Info/30904919119106655) 13% Charge/Speed | 182 797 | LSMV |
| `ref/saturn_rbo_2078M.smv` | [TASVideos #2078M](https://tasvideos.org/2078M) RBO | 168 144 | SMV → sidecar BK2 |
| `ref/taco_kriole_any_1368M.smv` | [TASVideos #1368M](https://tasvideos.org/1368M) any% | 139 292 | SMV → sidecar BK2 |
| `ref/cpadolf_xray_1978M.smv` | [TASVideos #1978M](https://tasvideos.org/1978M) X-ray climb | 77 108 | SMV → sidecar BK2 |
| `ref/hero_bubbleroom.smv` | Isolated Norfair bubble-room | 407 | SMV → sidecar BK2 |
| `ref/hero_kraid_entry.smv` | Isolated Kraid entry | 422 | SMV → sidecar BK2 |
| `ref/cpadolf_gt_2558M.lsmv` | [TASVideos #2558M](https://tasvideos.org/2558M) GT-code GEG | 53 661 | LSMV |
| `ref/nymx_sniq_sporespawn_4481S.lsmv` | [Playground #4481S](https://tasvideos.org/4481S) Spore Spawn | 33 342 | LSMV |
| `ref/nymx_ed_100map_5110S.bk2` | [Playground #5110S](https://tasvideos.org/5110S) 100% map | 261 130 | BKM-in-zip |
| `ref/sniq_geg_3768M.lsmv` | [TASVideos #3768M](https://tasvideos.org/3768M) GEG NTSC | 24 192 | LSMV |
| `ref/saturn_low_ice_2202M.smv` | [TASVideos #2202M](https://tasvideos.org/2202M) 14% Ice | 153 429 | SMV → sidecar BK2 |
| `ref/namespoofer_low_speed_2220M.smv` | [TASVideos #2220M](https://tasvideos.org/2220M) 14% Speed | 159 518 | SMV → sidecar BK2 |

Author notes (tech + multi-frame chords): [submission #5833](https://tasvideos.org/5833S).

Formats:

- LSMV input: `F.|BYsSudlrAXLR` — [spec](https://tasvideos.org/EmulatorResources/Lsnes/LSMV)
- BK2 `Input Log.txt` + LogKey — [spec](https://tasvideos.org/Bizhawk/BK2Format)
- SMV snes9x — parsed via `tas/smv.py` (BizHawk SmvImport mapping) and written
  as a sidecar `.bk2` for BizHawk replay. Sync on snes9x vs BSNES is unverified.
- BKM (BizHawk 1.x) — TASVideos may wrap a `.bkm` log in a zip named `.bk2`;
  `tas/bk2.py` reads that log with mnemonic button order.

## Commands

```bash
# Movies are gitignored (*.lsmv / *.bk2 / *.smv) — fetch then slice
uv run python -m super_metroid.tas.fetch_refs --list --skipped
uv run python -m super_metroid.tas.fetch_refs
uv run python -m super_metroid.tas.export_slices --finish
uv run python -m super_metroid.tas.export_slices --catalog

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

# Named skill windows from snes12_rle (offline; no emulator)
uv run python -m super_metroid.tas.skills_extract --slice hero_bubbleroom_full
uv run python -m super_metroid.tas.skills_extract --slice hero_kraid_entry_full
# Live pin, assist off, halt at first RAM miss (scratch JSON; not STATUS)
uv run python snes/super_metroid/scripts/probe/tas_skill_window.py \
  --slice hero_bubbleroom_full --skill arm_pump \
  --state-path snes/super_metroid/custom_integrations/SuperMetroid-Snes/room_acb3_from_b07a.state

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
  snes/super_metroid/tests/test_tas_catalog.py \
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
  ref/           # vendored .lsmv / .bk2 / .smv (+ SMV sidecar .bk2)
  slices/        # snes12_rle JSON + manifest.json
  catalog.py     # TASVideos game 121 fetch list (vanilla vs skip)
  lsmv.py        # lsnes parser
  bk2.py         # BizHawk parser (LogKey-aware)
  smv.py         # snes9x SMV → SNES-12 + BizHawk BK2 sidecar
  rle.py         # compress / expand
  slice.py       # named windows + catalog full-movie slices
  stages.py      # RoomStageSpec table (control settle → goal)
  extract_hops.py # hop inventory + skills/graph extraction board
  skills_extract.py # named button-pattern windows from snes12_rle
  materialize.py # stage window → snes12_rle body seeds (unproven)
  bodies/        # materialized stage seeds (not STATUS)
  annotate.py    # event detectors (rooms/items/speed/shine/stall)
  trace.py       # emulator replay + series / state dumps
  replay.py      # CLI power-on / slice replay
  resync.py      # product re-anchor + movie splice search
  export_slices.py
```
