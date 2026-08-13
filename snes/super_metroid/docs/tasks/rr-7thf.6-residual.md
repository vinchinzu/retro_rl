## Residual — rr-7thf.6 Wave C pure green (G4 statues → Metroids → Big Boy)

### Result
**PARTIAL** — dual hop-replay (boot-settle **0**) on
`g4_tourian_human`: **15/16 GREEN**, **1 RED** (hop 12 Metroid Room 4).
Safe trim seeds exported for all GREEN hops. Combat trim export for
metroids + Big Boy (edit / thrash inventory). No STATUS promote.

### GREEN/RED table

| Hop | Room | Leave / goal | Dual | End pin (both runs) | Notes |
|----:|------|--------------|:----:|---------------------|-------|
| 0 | Landing Site `0x91F8` | → Parlor `0x92FD` | **GREEN** | room=`0x92FD` xy=`[17, 1163]` pose=12 `ROOM_TRANSITION` | replay f1..551; boot anchor |
| 1 | Parlor `0x92FD` | → Terminator `0x990D` | **GREEN** | room=`0x990D` xy=`[19, 139]` pose=12 `ROOM_TRANSITION` | f735..1367; enter f734 |
| 2 | Terminator `0x990D` | → GPS `0x99BD` | **GREEN** | room=`0x99BD` xy=`[17, 651]` pose=12 `ROOM_TRANSITION` | f1494..1819 |
| 3 | Green Pirates Shaft `0x99BD` | → Statues hall `0xA5ED` | **GREEN** | room=`0xA5ED` xy=`[236, 1675]` pose=9 `ROOM_TRANSITION` | f1964..2919 |
| 4 | Statues Hallway `0xA5ED` | → Statues `0xA66A` | **GREEN** | room=`0xA66A` xy=`[1260, 139]` pose=9 `ROOM_TRANSITION` | f3041..3335 |
| 5 | Statues Room `0xA66A` | → Elev `0xDAAE` | **GREEN** | room=`0xDAAE` xy=`[128, 547]` pose=155 `ROOM_TRANSITION` | f3466..6648 (3313f dwell) |
| 6 | Tourian Elevator `0xDAAE` | → Save `0xDF1B` | **GREEN** | room=`0xDF1B` xy=`[237, 907]` pose=9 `ROOM_TRANSITION` | f6818..7585; enter f6817 |
| 7 | Upper Tourian Save `0xDF1B` | → Elev `0xDAAE` | **GREEN** | room=`0xDAAE` xy=`[20, 139]` pose=10 `ROOM_TRANSITION` | f7704..8332 |
| 8 | Tourian Elevator `0xDAAE` | → Metroid 1 `0xDAE1` | **GREEN** | room=`0xDAE1` xy=`[20, 907]` pose=10 `ROOM_TRANSITION` | f8453..8636; 2nd elev enter f8452 |
| 9 | Metroid Room 1 `0xDAE1` | → Metroid 2 `0xDB31` | **GREEN** | room=`0xDB31` xy=`[20, 139]` pose=10 `ROOM_TRANSITION` | f8761..15096 (6460f combat) |
| 10 | Metroid Room 2 `0xDB31` | → Metroid 3 `0xDB7D` | **GREEN** | room=`0xDB7D` xy=`[238, 395]` pose=9 `ROOM_TRANSITION` | f15216..17322 |
| 11 | Metroid Room 3 `0xDB7D` | → Metroid 4 `0xDBCD` | **GREEN** | room=`0xDBCD` xy=`[1518, 139]` pose=9 `ROOM_TRANSITION` | f17448..19322 |
| 12 | Metroid Room 4 `0xDBCD` | → Hopper `0xDC19` | **RED** | stuck room=`0xDBCD` xy=`[135, 347]` pose=2 `ORDINARY_GAMEPLAY` | desync mid-combat; see below |
| 13 | Tourian Hopper `0xDC19` | → Dust Torizo `0xDC65` | **GREEN** | room=`0xDC65` xy=`[20, 118]` pose=82 `ROOM_TRANSITION` | f23072..23736 |
| 14 | Dust Torizo `0xDC65` | → Big Boy `0xDCB1` | **GREEN** | room=`0xDCB1` xy=`[19, 109]` pose=130 `ROOM_TRANSITION` | f23855..24327 |
| 15 | Big Boy Room `0xDCB1` | end (no leave) | **GREEN** | room=`0xDCB1` xy=`[146, 187]` pose=1 `ORDINARY_GAMEPLAY` | f24448..27454; end_xy tape `[136, 187]` (tol 24) |

**Score: 15 GREEN / 1 RED / 16 hops.**

### Hop 12 — Metroid Room 4 (partial green via lockstep mid)

- **Full leave slice** (enter → Hopper): still **RED** dual open-loop.
  Both runs end in `0xDBCD` at `[135, 347]` — never door-transition.
- **Lockstep materialize** (`materialize_hop_mid.py --materialize`):
  - `contiguous_last_match` = **20360** (first mismatch f20361 xy drift).
  - Dumped live mid: `f020360_mid_lockstep_0xDBCD.state` kind=`mid_lockstep`
    xy=`[172, 459]` pose=2 (bottom floor combat).
  - **Enter → mid** (f19442..20360, 919f): dual **GREEN** (geometry;
    leave-room check N/A — still in `0xDBCD`).
  - **Mid → leave** (f20361..22906 from mid pin): still **RED** (same
    stuck band as full hop). Combat RNG desyncs on the first post-pin
    frames; open-loop cannot recover leave.
- **Offline candidates** (propose): floor_land @19524, combat poses @19574 /
  20262 / 20358 / **20438 (pose 138)** / 20594 / 22437, pre_leave @22905.
  Only indices ≤20360 are lockstep-safe dump targets without re-record.
- **Seed:** `tasks/g4_tourian_human_seeds/metroid4_enter_to_mid_safe.json`
  (919 frames, dual-green prefix). Do **not** open-loop `metroid4_combat.json`
  as a leave skill.
- **Remaining for full hop GREEN:**
  1. Guided re-take of Metroid 4 with F6 after last freeze / at door, **or**
  2. Pure controller rewrite for mid→Hopper (edit hints from traversal trim).

### Big Boy acceptance (hop 15)

Leave room is **None** (tape end in Big Boy). Green via **end_xy** band
`[136, 187]` with default `xy_tol=24`. Both runs landed `[146, 187]`
(Δx=10) — geometry green. No phase accept required.

### Commands (re-verify)

```bash
# Critical: --boot-settle 0 (default).

# All hops list / dry resolve
uv run python snes/super_metroid/scripts/tools/replay_human_hop.py \
  snes/super_metroid/tasks/g4_tourian_human.json --list-hops
uv run python snes/super_metroid/scripts/tools/extract_human_tape.py \
  snes/super_metroid/tasks/g4_tourian_human.json --summary --list-anchors

# Dual green examples
uv run python snes/super_metroid/scripts/tools/replay_human_hop.py \
  snes/super_metroid/tasks/g4_tourian_human.json --hop 5 --dual
uv run python snes/super_metroid/scripts/tools/replay_human_hop.py \
  snes/super_metroid/tasks/g4_tourian_human.json --hop 9 --dual
uv run python snes/super_metroid/scripts/tools/replay_human_hop.py \
  snes/super_metroid/tasks/g4_tourian_human.json --hop 15 --dual

# Known RED
uv run python snes/super_metroid/scripts/tools/replay_human_hop.py \
  snes/super_metroid/tasks/g4_tourian_human.json --hop 12 --dual

# Safe seeds (GREEN hops only as open-loop candidates)
uv run python snes/super_metroid/scripts/tools/trim_human_hop.py \
  snes/super_metroid/tasks/g4_tourian_human.json --hop N --mode safe \
  -o snes/super_metroid/tasks/g4_tourian_human_seeds/<name>_safe.json
```

Session dual log: `docs/tasks/logs/rr-7thf.6-dual-replay.txt`  
Trim log: `docs/tasks/logs/rr-7thf.6-safe-trim.txt`

### Seeds written

Under `snes/super_metroid/tasks/g4_tourian_human_seeds/`:

| File | Hop | Mode | Frames after trim | Dual raw hop |
|------|----:|------|------------------:|:------------:|
| `landing_site_safe.json` | 0 | safe | 491 | GREEN |
| `parlor_safe.json` | 1 | safe | 825 | GREEN |
| `terminator_safe.json` | 2 | safe | 461 | GREEN |
| `green_pirates_shaft_safe.json` | 3 | safe | 1109 | GREEN |
| `statues_hallway_safe.json` | 4 | safe | 297 | GREEN |
| `statues_safe.json` | 5 | safe | 3322 | GREEN |
| `tourian_elevator_in_safe.json` | 6 | safe | 337 | GREEN |
| `upper_tourian_save_safe.json` | 7 | safe | 668 | GREEN |
| `tourian_elevator_to_metroids_safe.json` | 8 | safe | 313 | GREEN |
| `metroid1_safe.json` | 9 | safe | 6469 | GREEN |
| `metroid2_safe.json` | 10 | safe | 2096 | GREEN |
| `metroid3_safe.json` | 11 | safe | 1883 | GREEN |
| `hopper_safe.json` | 13 | safe | 670 | GREEN |
| `dust_torizo_safe.json` | 14 | safe | 554 | GREEN |
| `big_boy_safe.json` | 15 | safe | 2952 | GREEN |
| `metroid1_combat.json` | 9 | combat | 6469 | GREEN (raw) |
| `metroid2_combat.json` | 10 | combat | 2096 | GREEN (raw) |
| `metroid3_combat.json` | 11 | combat | 1883 | GREEN (raw) |
| `metroid4_combat.json` | 12 | combat | 3473 | **RED** leave (edit only) |
| `metroid4_enter_to_mid_safe.json` | 12 prefix | lockstep | 919 | **GREEN** enter→mid |
| `big_boy_combat.json` | 15 | combat | 2952 | GREEN (raw) |

**Open-loop:** use `*_safe.json` for dual-green hops only. Combat copies for
9–11/15 match safe (leading/trailing only). **Do not** open-loop
`metroid4_combat.json` as a leave-room skill seed.

Did **not** dual-green re-validate trimmed seeds (only raw hop-replay), same
as Wave A/B residual policy.

### Acceptance

- [x] List 16 hops + live anchors for `g4_tourian_human`
- [x] Dual hop-replay all hops with `--boot-settle 0`
- [x] GREEN/RED table
- [x] Combat rooms attempted; hop 12 leave-extra probe documented
- [x] Safe seeds for GREEN hops under `tasks/g4_tourian_human_seeds/`
- [x] Residual with commands + results
- [ ] All hops dual GREEN — **blocked on hop 12**
- [ ] STATUS promote — **not done** (per card)
- [ ] Close `rr-7thf.6` — leave **in_progress** until Metroid 4 green or
      explicit accept of residual RED

### Residual risks

1. Hop-replay greens are **anchor + frame slice**, not continuous power-on →
   Big Boy / credits.
2. Long metroid rooms (esp. M1 6460f) greened this session but remain RNG-
   sensitive; re-verify after any assist / harness change.
3. Safe seeds not dual-green re-run as frame lists.
4. Hop 12 needs mid-pin re-record or pure rewrite — leave-extra cannot fix.
5. Elevator hop 6 has large leading idle cut (609f) in safe seed — re-check
   if used as skill start without matching elevator settle.

### Next action

- Keep `rr-7thf.6` **in_progress** — hop 12 leave still RED; enter→mid GREEN.
- Prefer pure rewrite or F6 re-take for mid→Hopper (not more leave-extra).
- Lockstep mid tooling landed: `scripts/tools/materialize_hop_mid.py`.
- Optional: dual-green validate other `*_safe.json` frame lists.

### Non-claims

- Did not STATUS-promote continuous G4→Big Boy
- Did not package pure controllers under `routes/kpdr`
- Did not invent mid anchors via full open-loop
- Did not force-push / commit (not requested)
- Did not dual-green re-run trimmed seeds

---

## Wave2 residual probe (2026-08-10)

Deepen hop-12 leave RED without multi-minute invent. Assist ON, boot-settle **0**.

### Re-verify enter→mid

```bash
uv run python snes/super_metroid/scripts/tools/replay_human_hop.py \
  snes/super_metroid/tasks/g4_tourian_human.json \
  --from-frame 19442 --to-frame 20360 --dual --boot-settle 0
```

**GREEN dual** — both runs `0xDBCD` xy=`[172, 459]` pose=2 (mid pin band). Anchor
`f019441_enter_0xDBCD`. Confirms Wave1 enter→mid.

### Post-mid short windows (from mid_lockstep pin)

Boot `f020360_mid_lockstep_0xDBCD.state`, step f20361..end, dual end-geometry
(xy_tol=24):

| Window | Dual end-band | Notes |
|--------|:-------------:|-------|
| +1 / +5 / +10 / +12f | **GREEN** | both runs match; still near mid floor |
| +15 / +18 / +20 / +25 / +50 / +100 / +200f | **RED** | identical dual drift (not flaky); e.g. +15 got `[191,436]` want `[153,449]` |

**Lockstep from mid** (xy_tol=12): `last_match=None`, first mismatch **f20361**
got xy=`[172,459]` pose=2 want `[163,445]` pose=84 — dies on the **first**
post-pin frame (same cliff as Wave1 enter scan at f20361).

**Conclusion:** no useful open-loop extend past mid. Soft end-band only
~**+12f**; lockstep-safe extension = **0f**. Cannot dump a later mid_lockstep
from this tape.

### Offline propose — leave half only (f20361..22905)

| i | kind | xy | pose | note |
|--:|------|----|-----:|------|
| 20438 | combat_pose | [37, 459] | 138 | first post-mid combat pin |
| 20594 | combat_pose | [52, 459] | 138 | |
| 22051 | combat_pose | [146, 347] | 84 | mid-room |
| 22437 | combat_pose | [37, 459] | 138 | |
| 22905 | pre_leave | [140, 489] | 24 | door band |

Edit / F6 targets only — none are lockstep-materializable from enter or mid.

### Safe / traversal trim notes

- **Safe full hop 12:** 3593 → 3473 (leading_idle=120). Re-exported
  `metroid4_combat.json` (edit only; leave still RED).
- **Safe enter→mid:** 919 → 919 (no cuts). Seed
  `tasks/g4_tourian_human_seeds/metroid4_enter_to_mid_safe.json` dual-green
  prefix only.
- **Traversal thrash (edit hints, not open-loop safe):** 3593 → 1255; cut **4
  retry loops / 2218f**. Kept sparse ranges
  `[(19914,19925), (20429,20466), (21005,21006), (21709,22915)]` — late leave
  approach is short; most dwell is retry thrash. Do **not** open-loop the
  thrash-trimmed list as a leave skill.

### Recommendation (hop 12 Metroid Room 4)

| Option | Fit | Why |
|--------|-----|-----|
| More mid_lockstep from this tape | **No** | lockstep dies f20361; +12f soft only |
| Pure rewrite mid→Hopper | **Yes** | thrash trim shows huge retry mass; leave approach is short |
| F6 re-take mid→door | **Preferred** | dump natural pins at freeze/door (propose 20438 / 22437 / 22905) after guided play from mid or enter |

**Exact next command (guided re-take from mid pin — human/F6 path):**

```bash
# After F6 re-record mid→Hopper (or full hop 12), re-scan lockstep + dual leave:
uv run python snes/super_metroid/scripts/tools/materialize_hop_mid.py \
  snes/super_metroid/tasks/g4_tourian_human.json --hop 12 --scan --boot-settle 0
uv run python snes/super_metroid/scripts/tools/replay_human_hop.py \
  snes/super_metroid/tasks/g4_tourian_human.json --hop 12 --dual --boot-settle 0
```

Until re-record: keep enter→mid as dual-green prefix skill only; do not claim
hop-12 leave pure green.
