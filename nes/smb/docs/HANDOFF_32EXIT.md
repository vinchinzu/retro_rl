# Handoff — SMB 32-exit TAS: 1-4 then 2-1…8-4

Movie: HappyLee & Mars608 warpless
[3728M](https://tasvideos.org/3728M) (`happylee_mars608_warpless_3728M.fm2`,
67,117f). **Every body is a cut of this file.** `#1715M` warp 1-1 / W4 1-2
and `smb_1_2_flag.json` are a different phase — `require_warpless_slice`
must reject them.

1-2 flag / isolated 1-3 archive: [`HANDOFF_32EXIT_1_3.md`](HANDOFF_32EXIT_1_3.md).
Warp any% is a different track (`record_happylee`, `natural_82`).

## TL;DR

Next knob is **1-4** from the already-green TAS 1-3 leave. Then the same
loop through **2-1…8-4**. Do not start 2-1 until 1-4 plays from the 1-3
predecessor (same gate as 1-3: play / record / test before continuing).

```bash
# Predecessor (already green): Level1_1 → 1-4 control
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.record_warpless --to 1-3

# 1-4 search / export / verify (add these flags; do not clone search_1_3)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.annotate_fm2 --search-1-4 --export-1-4
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.annotate_fm2 --verify-1-4
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.record_warpless --to 1-4
```

**1-4 start pin (verified):** `dash_level=3`, x=40, y=80 (castle),
timer=301, ps=7, lives=2. Movie hint **6393** (= 4653 + 1740). Ctrl wait
after 1-3 is unknown — measure with `idle_until(is_1_4_control)` on the
leave; 1-3's wait was 0.

Leave success = **2-1 control** (`world=1`, `dash_level=0`, x≤80, timer>0,
ps 7/8). 2-1 is overworld (y≈176), not castle y=80.

## Same-file chain (locked)

| Stage | FM2 start | Body | Wait after pred | Leave | Seed |
|-------|----------:|-----:|----------------:|-------|------|
| 1-1 | 190 | 1754 | settle=2 | 1-2 surface | `smb_1_1_warpless_slice.json` |
| 1-2 flag | 2109 | 2544 | 165 | 1-3 control | `smb_1_2_warpless_flag_slice.json` |
| 1-3 | 4653 | 1740 | 0 | 1-4 control | `smb_1_3_warpless_slice.json` |
| **1-4** | **~6393+wait** | **search** | **measure** | **2-1 control** | `smb_1_4_warpless_slice.json` |

Play path: `smb.tas.warpless.play_warpless_to` /
`smb.scripts.record_warpless`. 6205f / 1:43.247 to 1-4 control
(`recordings/tas_import/warpless_3728M/warpless_1_3_play.json`).

Identity is `$075C` LevelNumber (`snap.dash_level`), never `$0760`
AreaNumber. 1-2 / 4-2 / 6-2 / 8-2 underground flips AreaNumber; the
32-exit clock must not treat that as the next stage.

## 1-4 extract (one knob)

`annotate_fm2.py` is already 833 LOC. **Generalize** `reach_*` / `search_*`
/ `export_*` / `verify_*` (one StageSpec-like row: control, leave, hint,
seed name) before a fourth copy. `play_warpless_to` must grow from a
target table (`CHAIN_TARGETS` + leave predicate), not another `if target ==`.

Completion — all of these, in order:

1. **Control sync.** `play_warpless_to(env, to="1-3")` lands
   `is_1_4_control`. Fingerprint matches the table above. Add
   `is_2_1_control` in `smb.tas.stages` (world 1, dash 0, same x/timer/ps
   shape as `is_1_4_control`).
2. **Wait.** Idle on that pin until `is_1_4_control` still holds; record
   `ctrl_wait_1_4`. Search center = `WL_1_4_FM2_HINT + wait` (6393 + wait).
3. **Search.** FM2 start grid ±80 (step 1) from center. Success = first
   `is_2_1_control` without death. Castle is short; `max_play≈2500`. Keep
   the shortest clear. Print `si / max_x / leave / death`.
4. **Export.** `models/smb_1_4_warpless_slice.json`: `route_id=smb_all_exits`,
   `stage_id=1-4`, `target=2_1_control`, `source` contains `3728M` and
   `warpless`, `fm2_start_index` = winning si, `num_frames` = leave.
   `require_warpless_slice` must accept it and still reject `#1715M`.
5. **Verify.** Replay that body from the TAS 1-3 leave (not `Level1_4`,
   not the human 1-4 pin). 1/1 → 2-1 control.
6. **Play / record.** `record_warpless --to 1-4` then `--record`. Same
   gate as 1-3: play + record + test **before** 2-1.

Human `all_exits_v1` 1-4 pin is a **1-3 castle tally** (HUD WORLD 1-3,
`$075C=2`). It is not 1-4 control and is not extractable. Isolated
`Level1_4` is not a TAS pin — do not fold the body onto it if phase
differs (same trap as `Level1_3` vs TAS 1-3: rr-tb15).

## Rest of game (2-1…8-4)

29 stages after 1-3. Each leg:

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
  uv run python -m smb.scripts.record_warpless --to 1-3
./play smb --list    # human 32-exit; 1-4 pin is still the bogus tally
```

## Traps

- L+R stays in the movie. Replay via `to_action9`, never `sanitize_action`.
- One fceumm instance per process. `set_state` → `reset()` → `set_state`
  when booting a pin.
- RAM y is head/top (floor stand y=176; castle stand y=80).
- Cap every walk/search loop. 1-3 search was 161 starts; 2 clears
  (4653/1740 shorter than 4589/1803).
- Warp any% seeds (`smb_*_happylee_slice.json`, `smb_1_2_reactive_fragments.json`,
  `natural_82`) stay untouched.
- Isolated Level1_3 TAS body still misses (rr-tb15). That is not this knob.

## Beads

- **rr-z01b** — 1-4 slice from TAS 1-3 leave → 2-1. Claim this first.
- **rr-g2ht** — remaining #3728M extract after 1-4 (2-1…8-4).
- **rr-tb15** — isolated Level1_3; leave open.
- **rr-n6sz** — human `all_exits_v1` tape; not this track.
- rr-xpeq done (1-1…1-3 same-file play/record).
