# TAS adaptation — Super Metroid

HappyLee-style **button-stream import → harness replay → annotate → re-anchor**.
Movies are vendored under `tas/ref/`; slices under `tas/slices/` (`snes12_rle`).

Primary refs: Sniq any% (lsnes LSMV #3653M), Sniq 100% (BK2). See
[`tas/README.md`](../tas/README.md).

## Gap

Our continuous KPDR tip is **product pure** (wave ~136k). TAS any% is much
shorter and encodes frame-perfect tech (arm-pump, mockball, door speed, shine).
Raw movie frame indices are **not** room-ID control points under stable-retro —
same desync class as HappyLee FM2 on fceumm.

## Pipeline

```text
1. Fetch refs + export RLE slices     (done: fetch_refs / export_slices)
2. Power-on replay under harness      (tas.replay)
3. Annotate: rooms, items, pose, speed, shine, stalls
4. Dump .state at room_enter / control / item_gain
5. Re-anchor: control-relative bodies from harness states
6. Compare residual pins to pure hops; adopt tech only after pure-first proof
```

### Commands

```bash
# Movies gitignored — re-fetch if missing
uv run python -m super_metroid.tas.fetch_refs
uv run python -m super_metroid.tas.export_slices --finish

# List catalog
uv run python -m super_metroid.tas.replay --list-slices

# Menu smoke (~600f)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m super_metroid.tas.replay --slice sniq_any_menu \
  --annotate --series-stride 1

# Short contest (Wrecked Ship start BK2)
uv run python -m super_metroid.tas.replay --slice moozooh_smtc4_full \
  --annotate --series-stride 4 --states-on room_enter,control

# Full any% power-on (long; annotate milestones — expect core desync)
uv run python -m super_metroid.tas.replay --slice sniq_any_full \
  --annotate --series-stride 8 \
  --states-on room_enter,control,item_gain,beam_gain,capacity_gain \
  --out snes/super_metroid/recordings/tas_import/sniq_any_full
```

Artifacts land under `recordings/tas_import/<run_id>/`:

| File | Role |
|------|------|
| `trace.json` | Events, room hops, final pin, meta |
| `summary.json` | Compact agent/human report |
| `pins.json` | room_enter / control / item pins |
| `series.jsonl` | Strided pose/x/y/vel/buttons |
| `states/*.state` | Optional checkpoints |

## Hard rules

1. **Never sanitize L+R** on TAS replay (`trace.action_array` is raw SNES-12).
2. **Do not STATUS-claim** from movie indices alone — re-anchor first.
3. **Assist off** during TAS replay (real energy / death).
4. Prefer **state-anchored** bodies for room tech; power-on for boot research.
5. Room settle = `gs==8`, `door_transition==0`, `room_id!=0` (same as RoomTimer).
6. Product pure-first process still applies; TAS is a **reference**, not a warp.

## Annotation event kinds

| Kind | Meaning |
|------|---------|
| `control` | First ordinary gameplay settle |
| `room_enter` / `room_leave` | Settled room boundaries |
| `item_gain` / `beam_gain` | Collected mask bits |
| `capacity_gain` | Max missiles / supers / PBs / energy |
| `speed_echo` / `speed_echo_drop` | Speed-booster charge level |
| `shine_arm` / `shine_clear` | `$0A68` timer |
| `pose_cluster` | morph / shinespark / walljump / … |
| `desync_suspect` | Frozen pose+xy while buttons active |
| `death` / `ending` / `pause` | Phase edges |

## Verified harness runs (2026-08-07)

| Run | Frames | first_control | Notes |
|-----|-------:|--------------:|-------|
| `sniq_any_menu` | 600 | — | Still boot/menu (expected; Ceres open ~f8639 movie idx) |
| `sniq_any_open_15k` | 15 000 | **11 182** | Room `0xDF45` (Ceres elev shaft); pose series + 1 state dump |
| `sniq_any_full` | 129 712 | **11 182** | ~180s @ ~720 f/s; **54** `room_enter`, **50** RoomTimer visits, **14** `desync_suspect`, **3** deaths; final items still `0x0000` → **core desync after early Ceres** (lsnes vs stable-retro). Artifacts under `recordings/tas_import/sniq_any_full/`. |
| `moozooh_smtc4_full` | 5 384 | — | Mid-game contest inputs; no usable BizHawk savestate in archive — power-on meaningless |

Artifacts: `recordings/tas_import/<run_id>/{trace,summary,pins,series,states}`.

**Takeaway:** power-on any% is useful through first Ceres control (~11.2k). Past
that, re-anchor from dumped states / pure product prefixes before adopting
button bodies. Do not STATUS-claim full-movie sync.

## Zebes resync (product re-anchor + movie splice)

lsnes is **not installed** here; LSMV P1 field order already matches harness
SNES-12 (`BYsSudlrAXLR`). “lsnes wiring” = parse that order raw (no sanitize).
Bit-exact lsnes playback would need the lsnes core; we re-anchor instead.

```bash
# Product pure → Landing, search movie index, annotate movie body
uv run python -m super_metroid.tas.resync --to landing --search \
  --search-lo 15000 --search-hi 25000 --search-step 500 --body 8000

# Known good splice (Landing → Parlor under movie)
uv run python -m super_metroid.tas.resync --to landing --movie-start 15000 --body 12000 \
  --out snes/super_metroid/recordings/tas_import/resync_zebes_rooms
```

| Mode | Result (2026-08-07) |
|------|---------------------|
| **Product morph spine** | **GREEN multi-room Zebes:** Landing `0x91F8` → Parlor `0x92FD` → Climb `0x96BA` → Pit `0x975C` → elev `0x97B5` → Morph `0x9E9F` + morph bit `0x0004` @ **26 824f**. Timeline: `recordings/tas_import/product_morph_annotate/`. |
| **Movie splice @15000** | Product Landing pin f21548 + Sniq movie body → **enters Parlor** f23740 (pose 18 @ 1240,139); later thrash Landing↔Parlor (no Climb). Artifacts: `resync_zebes_rooms/`. |
| **Product → Pit + movie** | Open-loop Sniq body from product Pit pin never exits (`pit_max_x≈555`, falls to lower pit). Skip full-movie Pit splice. |
| TAS boot + movie@8639 | Reaches Ceres rooms but thrash; never Landing (core phase desync). |

```bash
# Product pure prefix through Pit (skips Climb under movie)
uv run python -m super_metroid.tas.resync --to pit --movie-start 17000 --body 2000
```

**Policy:** product pure owns Ceres→Morph multi-room continuity. TAS movie is
for **button/tech reference** and **single-hop** splice research (Landing door
works at `movie_start=15000`). Climb+ under open-loop Sniq needs a tighter pin;
**Pit uses product first-jump + seed tail** (below).

### Pit Room — first jump, then seed tail

Climb is a mostly-fall seed (1 A-edge); **skip TAS for Climb**. Pit is the
interesting horizontal dash-jump room (`policies/morph/seg03_pit_room.json`,
810f, 11 A-edges).

Verified model (product natural entry → elev):

| Phase | Seed frames | Pin / action |
|-------|------------:|--------------|
| Entry | — | `0x975C` pose 9 @ (13,139) (often mid door_transition) |
| Approach | 0–149 | Settle + RIGHT runup + `B+RIGHT` dash → pre-J0 **(126,139)** pose 9 mom 2 |
| **First jump** | 150–164 | `B+RIGHT+A` **hold 15** |
| Coast | 165–197 | `B+RIGHT` ~33f → **land (195,123) pose 9** |
| Seed tail | 198–809 | Remaining jumps → door **(748,139)** gs=11 |
| Elev | +RIGHT | BB elev `0x97B5` |

**Rule:** land the first jump on **(195, 123) pose 9**; the stock seed tail from
frame 198 then carries to the elev door. First-jump hold is robust (holds
8–22 with matching coast still clear elev in a 177/195 grid from pre-J0 state).

Artifacts: `recordings/tas_import/resync_pit/pit_first_jump_pin.json`,
`first_jump_search.json`.

Splice strategy going forward:

1. Product pure through Climb → Pit entry.
2. Product approach + first jump (or TAS-timed A-hold **only if** land pin matches).
3. Product seed tail from f198 — do not open-loop full movie body in Pit.
4. Door settle into elev; product elev/morph seeds continue.

## Follow-ups

- [x] Replay + annotate CLI (`tas.replay` / `tas.trace` / `tas.annotate`)
- [x] Full any% power-on annotate pass (desync documented)
- [x] Product → Landing re-anchor + movie search (`tas.resync`)
- [x] Multi-room Zebes via product morph spine annotate
- [x] Landing→Parlor TAS movie splice (`movie_start=15000`)
- [x] Pit first-jump land pin + seed-tail strategy (skip Climb movie)
- [ ] Optional: retune first-jump A-hold only (land pin gate) for speed
- [ ] Finer parlor-exit pin search for Climb under movie body (low priority)
- [ ] Boot pad/skip alignment search (menu length vs lsnes)
- [ ] Control-relative re-slice export from annotated room hops
- [ ] Ceres TAS boot residual re-pin (`play_boot_to_ceres_tas`) using f11182 pin
- [ ] Video HUD replay (optional; continuous video path)
- [ ] 100% full movie annotate pass
