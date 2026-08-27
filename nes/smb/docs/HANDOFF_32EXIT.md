# Handoff — SMB 32-exit TAS: 2-2…8-4

Movie: HappyLee & Mars608 warpless
[3728M](https://tasvideos.org/3728M) (`happylee_mars608_warpless_3728M.fm2`,
67,117f). **Every body is a cut of this file.** `#1715M` warp 1-1 / W4 1-2
and `smb_1_2_flag.json` are a different phase — `require_warpless_slice`
must reject them.

1-2 flag / isolated 1-3 archive: [`HANDOFF_32EXIT_1_3.md`](HANDOFF_32EXIT_1_3.md).
Warp any% is a different track (`record_happylee`, `natural_82`).

## TL;DR

**1-4 and 2-1 are green.** Next knob is **2-2** from the TAS 2-1 leave
(2-2 drop-in, y=0 then land y=176). Same loop through **2-3…8-4**.
Do not start 2-3 until 2-2 plays from the 2-1 predecessor.

Search / export / verify is one table (`WARPLESS_LEGS` +
`smb.tas.warpless_extract`). Do not clone per-stage search functions.

```bash
# Predecessor (already green): Level1_1 → 2-2 drop-in
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.record_warpless --to 2-1

SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.annotate_fm2 --search 2-2 --export
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.annotate_fm2 --verify 2-2
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.record_warpless --to 2-2
```

**2-2 start pin (verified 2-1 leave):** `world=1`, `dash_level=1`, x=40,
y=0 dropping in (lands y=176 ~80f later), timer=401, ps=7, lives=2.
Movie hint **10451** (= 8095 + 2356). First ±80 searches missed (wait=0
at y=0 dies ~x=1315; wait=61 at y=162 stall ~x=700). Overworld
`WarplessLeg.control` now requires y≥160 so 2-2 reach can idle to land
(leave still uses dash-only, so 2-1 stays 2356f @8095). Next: wider
window / lead idle, not another clone. x-2 is a flag pipe, not a warp.

Leave success = **2-3 control** (world 1, dash 2, x≤80, timer>0, ps 7/8).

## Same-file chain (locked)

| Stage | FM2 start | Body | Wait after pred | Leave | Seed |
|-------|----------:|-----:|----------------:|-------|------|
| 1-1 | 190 | 1754 | settle=2 | 1-2 surface | `smb_1_1_warpless_slice.json` |
| 1-2 flag | 2109 | 2544 | 165 | 1-3 control | `smb_1_2_warpless_flag_slice.json` |
| 1-3 | 4653 | 1740 | 0 | 1-4 control | `smb_1_3_warpless_slice.json` |
| 1-4 | 6393 | 1702 | 0 | 2-1 control | `smb_1_4_warpless_slice.json` |
| 2-1 | 8095 | 2356 | 0 | 2-2 control | `smb_2_1_warpless_slice.json` |
| **2-2** | **~10451+wait** | **search** | **measure** | **2-3 control** | `smb_2_2_warpless_slice.json` |

Play path: `smb.tas.warpless.play_warpless_to` /
`smb.scripts.record_warpless`. 10263f / 2:50.850 to 2-2 drop-in
(`record_warpless --to 2-1`, 2:50.769). 1-4 leave was 7907f / 2:11.567.

Identity is `$075C` LevelNumber (`snap.dash_level`), never `$0760`
AreaNumber. 1-2 / 4-2 / 6-2 / 8-2 underground flips AreaNumber; the
32-exit clock must not treat that as the next stage.

## 1-4 extract (done)

Generalized: `smb.tas.warpless.WarplessLeg` (32-exit table) +
`smb.tas.warpless_extract` (reach / search / export / verify) +
table-driven `play_warpless_to`. `annotate_fm2.py` is the CLI.

1-4 castle **1702f @ FM2 6393** (wait=0 after 1-3). 7 clears in ±80;
shortest was the movie-aligned center. Leave = 2-1 overworld (world=1,
dash=0, x=40, y=176). Seed `smb_1_4_warpless_slice.json`.
`--verify-1-4` 1/1 from TAS 1-3 leave. `record_warpless --to 1-4` →
7907f / 2:11.567.

Human `all_exits_v1` 1-4 pin is still a **1-3 castle tally**. Isolated
`Level1_4` is not a TAS pin. Isolated `Level1_3` remains rr-tb15.

## Rest of game (2-1…8-4)

27 stages after 2-1. Each leg:

```text
predecessor leave pin
  → idle_until(is_<this>_control)     # measure wait
  → center = prev_fm2_start + prev_body + wait
  → grid-search FM2 starts
  → success = is_<next>_control (or ending on 8-4)
  → export smb_<w>_<l>_warpless_slice.json
  → verify from the chained predecessor (exported seeds, same movie)
  → record_warpless --to <this>  (play + record + test)
  → only then the next stage
```

Next-control map (1-indexed world-level → RAM `world`, `dash_level`):

| This | Next control |
|------|----------------|
| n-1, n-2, n-3 | same world, dash+1 |
| 1-4 … 7-4 | world+1, dash 0 (overworld spawn, y≈176) |
| 8-4 | `reached_ending` (`world=7`, dash=3, `oper_mode=2`) — not a 9-1 |

x-2 in this movie is the **flag pipe**, not a warp room (1-2 already
proved this). x-3 is athletic. x-4 is castle. 8-4 is maze + Bowser + axe;
hold through Peach if recording.

Do not search from power-on FM2 (fceumm desyncs the title). Always
Level1_1 + exported #3728M bodies + idle gates.

After each world-4 leave, extend `CHAIN_TARGETS` / `WARPLESS_SEEDS` and
the provenance test (`fm2_start + body + wait == next_start`).

## Commands already green

```bash
uv run python -m smb.tas.fetch_refs
uv run python -m smb.scripts.convert_fm2
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.record_warpless --to 2-1
./play smb --list    # human 32-exit; 1-4 pin is still the bogus tally
```

## Traps

- L+R stays in the movie. Replay via `to_action9`, never `sanitize_action`.
- One fceumm instance per process. `set_state` → `reset()` → `set_state`
  when booting a pin.
- RAM y is head/top (floor stand y=176; castle stand y=80).
- Cap every walk/search loop. 1-3 search was 161 starts; 2 clears
  (4653/1740 shorter than 4589/1803). 1-4 search 161 starts; 7 clears
  (6393/1702 shortest, movie-aligned).
- Warp any% seeds (`smb_*_happylee_slice.json`, `smb_1_2_reactive_fragments.json`,
  `natural_82`) stay untouched.
- Isolated Level1_3 TAS body still misses (rr-tb15). That is not this knob.

## Beads

- **rr-g2ht** — remaining #3728M extract (2-2…8-4). Next: 2-2 @~10451.
- **rr-z01b** — 1-4 slice; done.
- **rr-tb15** — isolated Level1_3; leave open.
- **rr-n6sz** — human `all_exits_v1` tape; not this track.
- rr-xpeq done (1-1…1-3 same-file play/record).
