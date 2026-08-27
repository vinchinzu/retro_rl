# Handoff — SMB 32-exit TAS: 2-2…8-4

Movie: HappyLee & Mars608 warpless
[3728M](https://tasvideos.org/3728M) (`happylee_mars608_warpless_3728M.fm2`,
67,117f). **Every body is a cut of this file.** `#1715M` warp 1-1 / W4 1-2
and `smb_1_2_flag.json` are a different phase — `require_warpless_slice`
must reject them.

1-2 flag / isolated 1-3 archive: [`HANDOFF_32EXIT_1_3.md`](HANDOFF_32EXIT_1_3.md).
Warp any% is a different track (`record_happylee`, `natural_82`).

## TL;DR

**1-4 and 2-1 are green.** 2-1 is the clip-phase recut **2440f @7999**
(+84f vs movie-aligned 8095/2356). Next knob is **2-2** from that drop.
Do not start 2-3 until 2-2 plays from the 2-1 predecessor.

Search / export / verify is one table (`WARPLESS_LEGS` +
`smb.tas.warpless_extract`). Do not clone per-stage search functions.
Knobs: `--window`, `--lead-max`, `--from-pred` (pin at predecessor
leave), stall abort at 180f with no x progress before the flag zone.
2-2 `control` accepts the y=0 drop so play does not idle 61f.

```bash
# Predecessor (already green): Level1_1 → 2-2 drop-in
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.record_warpless --to 2-1

# 2-2 TAS plays during the drop. Idle-to-land stalls.
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.annotate_fm2 --search 2-2 --from-pred --export
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.annotate_fm2 --verify 2-2
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.record_warpless --to 2-2
```

**2-2 start pin (verified 2-1 leave):** `world=1`, `dash_level=1`, x=40,
y=0 dropping in, timer=401, ps=7, power=2, lives=2. Chain hint **10439**
(= 7999 + 2440). Movie TAS 2-2 is still **10451**. x-2 is a flag pipe /
water corner-clip, not a warp. `smb.approx` does not model swim — do not
approx-heal 2-2 clips.

Leave success = **2-3 control** (world 1, dash 2, x≤80, timer>0, ps 7/8).

## 2-2 search (open)

Movie-aligned 2-1 (8095/2356) never left 2-3. Recut 2-1 to **7999/2440**
and searched from that drop (`--from-pred`, wait=0, play 10347f):

| Pin | Search | Best |
|-----|--------|------|
| 8095/2356 drop ±160 @10451 | 10291–10611 | unique peak **10451 / x=1315** stall |
| 8095/2356 drop + lead 0..21 | 10445–10457 | `si+lead=10451` only; same 1315 miss |
| 7999/2440 drop ±80 @10439 | 10359–10519 | unique peak **10451 / x=2225** stall @1769 |
| 7999/2440 drop lead 0..16 @10451 | 17 trials | lead≥1 loses the 1315 clip |

7999–8019 / 2440 leaves are identical at TAS @10451 (all 2225 @1769).
7979/2461 and ≥8045 stay stuck at 1315. 2-2 `control` now accepts y=0.

**2225 is the coral tower**, not the exit pipe. TAS @10451 from the recut
drop clips the first wall then bonks the tower face (y≈61, xs→0). Extra
A at body 1480–1511 enters the tower to **x=2383–2385** (xs=24 at x=2301
y=75) then wrap-death at the bottom (y 0→255, ps=6). Holding R/RA/RLA
from inside does not pop out. Evidence:
`recordings/tas_import/warpless_3728M/2_2_x2225.png`.

Next: finish the tower clip from 7999/2440 + TAS @10451 (keep a pure
#3728M cut; do not bake a patched body). Isolated 1-3 remains rr-tb15.

## Same-file chain (locked)

| Stage | FM2 start | Body | Wait after pred | Leave | Seed |
|-------|----------:|-----:|----------------:|-------|------|
| 1-1 | 190 | 1754 | settle=2 | 1-2 surface | `smb_1_1_warpless_slice.json` |
| 1-2 flag | 2109 | 2544 | 165 | 1-3 control | `smb_1_2_warpless_flag_slice.json` |
| 1-3 | 4653 | 1740 | 0 | 1-4 control | `smb_1_3_warpless_slice.json` |
| 1-4 | 6393 | 1702 | 0 | 2-1 control | `smb_1_4_warpless_slice.json` |
| 2-1 | 7999 | 2440 | 0 | 2-2 control | `smb_2_1_warpless_slice.json` |
| **2-2** | **~10439 / TAS 10451** | **search** | **0 (drop)** | **2-3 control** | `smb_2_2_warpless_slice.json` |

Play path: `smb.tas.warpless.play_warpless_to` /
`smb.scripts.record_warpless`. 10347f / 2:52.166 to 2-2 drop-in
(`record_warpless --to 2-1` after the recut). Movie-aligned 2-1 was
10263f / 2:50.769. 1-4 leave was 7907f / 2:11.567. 2-1 is **not**
consecutive with 1-4 (hint 8095 vs start 7999).

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
  (6393/1702 shortest, movie-aligned). 2-2 from 7999/2440 unique-peaks
  TAS @10451 at the coral tower (x=2225); 8095/2356 unique-peaks at 1315.
- `--from-pred` pins at predecessor leave (2-2 drop). 2-2 `control`
  accepts y=0, so default reach is also wait=0. Landed y≥160 is the
  x≈700 stall path.
- Warp any% seeds (`smb_*_happylee_slice.json`, `smb_1_2_reactive_fragments.json`,
  `natural_82`) stay untouched.
- Isolated Level1_3 TAS body still misses (rr-tb15). That is not this knob.

## Beads

- **rr-g2ht** — remaining #3728M extract (2-2…8-4). 2-1 recut 2440f
  @7999. Next: heal 2-2 coral tower @x=2225 from TAS 10451.
- **rr-z01b** — 1-4 slice; done.
- **rr-tb15** — isolated Level1_3; leave open.
- **rr-n6sz** — human `all_exits_v1` tape; not this track.
- rr-xpeq done (1-1…1-3 same-file play/record).
