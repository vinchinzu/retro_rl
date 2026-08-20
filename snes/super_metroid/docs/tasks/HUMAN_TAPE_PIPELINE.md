# Human tape → green room skills

**Epic:** [rr-7thf](../../../.beads/issues.jsonl) — late spine (WS→credits,
esp. G4/MB) converted into per-room **pure dual-green** skills.

**Board:** [`tasks/LATE_SPINE_HOP_BOARD.json`](../../tasks/LATE_SPINE_HOP_BOARD.json)  
**Generator:** `scripts/tools/build_late_spine_board.py`  
**Extract:** `scripts/tools/extract_human_tape.py` + package `super_metroid.human_tape`

## Product path (record → skills → autopilot)

```text
./play (full buttons + live anchors)
  → F5: archive prior take (tape + hop bodies) if name reuse
  → materialize: settled hops + <name>_hops/ bodies + run_timing
  → optional --bank → recordings/skill_bank/bank.json (dual_green=False)
  → hop-replay --dual --promote-bank  (or compose --promote-bank)
  → compose_human_hops (re-pin dual-verify per hop)
  → trim / hill-climb one hop (HopOptimizeJob)
  → export skill / pure controller → continuous spine
```

Full **button recording is first-class** (not forbidden). Seams replay as
**hop-compose from live pins**, not frame-append of multi-session tapes.
Product continuous early tip (ice / KPDR) stays a separate scripted chain;
late chain is **human → hop inventory → hop-replay → trim → pure green**.

## Library package

Library lives under `snes/super_metroid/human_tape/`:

| Submodule | Role |
|-----------|------|
| `anchors` | gzip pins, fingerprints, `AnchorRecorder`, index match |
| `hops` | room hops, skill groups, `resolve_hop_slice`, `extract_tape` |
| `replay` | step loop, `replay_hop`, `check_hop_green`, `run_hop_replay` |
| `bodies` | per-hop SNES-12 export (`<stem>_hops/`) for bank / hill-climb |
| `compose` | multi-hop pin→body chain (seam-safe open-loop) |
| `segment_archive` | immutable prior take when reusing `--name` |
| `midpoints` | offline mid candidates + lockstep materialize |
| `trim` | offline idle/retry hop trim |
| `stitch` | multi-session **timing** stitch (RTA clock; not button concat) |

Prefer `from super_metroid.human_tape import …` (or submodules).

## Glossary

| Term | Meaning |
|------|---------|
| **Hop** | One **settled room visit**: live enter pin → leave next room (or `end_xy` band on the final hop). Built from trace room-id changes, then **settled** to first ordinary non-door frame (matches `room_enter` / RoomTimer). Identity for bank / PB: `hop_key = {room}:{from}→{to}:{items}` (direction + inventory matter). Unit of dual-green, body export, hill-climb, and open-loop replay. |
| **Open-loop** | Replay a **fixed** SNES-12 button body from a pin with **no mid-body branching** (no “if enemy, jump”). Safe unit = **one hop from its live `entry_anchor`**. Unsafe = multi-minute mega-tape from power-on to invent mid pins (desync / pin overwrite). A **reactive route script** may *choose which* open-loop skill (or pure controller) to run; the open-loop body itself is not the decision layer. |
| **Pin / anchor** | Gzip savestate + fingerprint (room, xy, items, …) dumped live (`room_enter`, F6, item_delta, end). Hop open-loop boots these. |
| **Dual-green** | Hop-replay leave/goal check passes twice from the same pin. Bank `dual_green=True` only after promote (`--promote-bank`). |
| **Natural-entry compose** | After hop A, leave kinematics seed hop B (or re-pin if mismatch). Not frame-append of multi-session tapes; not only parallel re-boot of archived pins. |
| **Reactive script** | Policy/FSM/pure controllers that **decide** next skill, recovery, human/bot handoff. Consumes bank dual-green bodies and pure segments as options. |

**Product layers:** library of takes → skill bank (mix) → reactive route script. Epic: `rr-nzrg`.

## Open-loop units (what is safe)

| Unit | Use |
|------|-----|
| **Hop from live pin** | Default open-loop: boot `entry_anchor`, play body, check leave |
| **Hop-compose chain** | Multi-room: re-pin each hop (compose_human_hops) for dual-verify; natural-entry compose is the stronger mix goal (`rr-nzrg.2`) |
| **Full-tape multi-minute from boot** | High desync risk for **pin recovery**; do not invent mid pins this way |

Long open-loop of multi-minute guided takes **desyncs** when used as a pin
recovery tool:

- Assist timing and enemy RNG diverge from the original session.
- Accumulated subpixel / door-entry error compounds across rooms.
- Checkpoint reloads renumber some `frame` fields (hop **dwell uses trace
  indices**, not raw frame alone).

A full-tape open-loop once **overwrote** `maridia_grapple_human` end state
(see `tasks/maridia_grapple_human_end_LOST.json`). Lesson: recover pins with
**live anchors / F6 / mid_lockstep**, not tens of thousands of frames from boot.
Recording and replaying **button bodies hop-by-hop from those pins** is the
supported path to skills and autopilot.

## Live anchors

During `guided_human` (anchors ON by default), `AnchorRecorder` dumps gzip
states on:

| kind | when |
|------|------|
| `boot` | first settled ordinary frame |
| `room_enter` | first settled ordinary frame in a new room |
| `item_delta` | collected_items change |
| `manual` | F6 pin |
| `mid_lockstep` | materialize dump (enter→last contiguous lockstep match) |

Index: `tasks/<name>_anchors.json` → files under `tasks/<name>_anchors/`.

Live anchors are **authoritative** for hop start. Hop inventory still comes
from the per-frame `trace` in the task JSON (offline, no emulator).
`mid_lockstep` ranks with `manual` for sub-hop boots (not F6; recovered via
lockstep from a live enter/boot pin).

## Pipeline

```
record → materialize (bodies) → hop-replay → compose → trim/hill-climb → skill
```

1. **Record** — `./play` / `guided_human`: full SNES-12 frames + trace + live
   anchors. Reusing `--name` archives prior tape to
   `tasks/<name>_segments/sN/` (immutable button body for that seam).
2. **Materialize** — settled hops, `*_run_timing.json`, per-hop bodies under
   `tasks/<name>_hops/`, optional `--bank`. Offline (no emulator).
3. **Hop-replay** — load the hop’s `anchor_path` (enter/boot), play only the
   frame slice, check leave room / end_xy.  
   - **Default `--boot-settle 0`**: live room_enter anchors are already settled;
     settle=5 desyncs long hops (Escape 4).  
   - **Default assist ON** (matches guided_human contract energy/ammo). Use
     `--no-assist` for pure stress later; long combat hops die without it.  
   - **No leave (final hop):** green via **end_xy** band (e.g. Landing Site
     ship pin).  
   CLI: `scripts/tools/replay_human_hop.py` (rr-7thf.1).
4. **Compose** — multi-hop chain: re-pin each hop, open-loop body only.
   CLI: `scripts/tools/compose_human_hops.py` or `./play --compose <name>`.
5. **Trim** — two layers:
   - **`safe` (default, open-loop):** leading + trailing idle only → **one
     contiguous** `kept_ranges`. Dual-green hop-replayable.
   - **`traversal` heuristics:** mid-idle + retry HWM cuts (progress drop +
     recover). These **skip frames that still tick enemy RNG** — dual-green
     validate before bank seed. Use as *edit hints* for pure rewrites.
   - **`combat`:** leading + trailing only (bosses / metroids).  
   CLI: `scripts/tools/trim_human_hop.py` (rr-7thf.2).
6. **Midpoints (old tapes / long RED hops)** — do not invent mid anchors via
   multi-minute full-tape open-loop. Two safe layers:
   - **Offline propose** (`--propose`): parse hop `trace` for floor lands,
     combat poses, energy cliffs, pre-leave. Works on *all* tapes (including
     ws/gravity/grapple with no live anchors). Edit / re-record hints only.
   - **Lockstep materialize** (`--materialize`): boot the hop's **live**
     enter/boot anchor, step while matching `trace` in a **single contiguous
     pass**, stop at the **first** xy/room mismatch, dump gzip at
     `contiguous_last_match`. Kind **`mid_lockstep`**. Dual-verify enter→mid.
   CLI: `scripts/tools/materialize_hop_mid.py`.
7. **Dual pure green + bank promote** — hop-replay `--dual --promote-bank`
   (or compose `--promote-bank`) sets `dual_green=True` on the matching bank
   record. Without promote, bank rows stay candidates only.
8. **Hill-climb / skill** — `HopOptimizeJob` + hop body seed; export RLE /
   controller under `routes/skills` or `routes/kpdr`; board hop remains the
   inventory source of truth.

Settled hop bounds are the default for resolve / replay / compose / bodies
so bank dwell matches open-loop slices. Start presets live in
`super_metroid.start_presets` (not in the recorder CLI).

Aggregate thrash ranking + per-hop `mode` / `priority` / `leave_room`:

```bash
uv run python snes/super_metroid/scripts/tools/build_late_spine_board.py --summary
```

## Product-chain hop-compose (living run → AP)

Epic **`rr-4nli`**. Work list: `tasks/PRODUCT_CHAIN_HOP_BOARD.json`.

This is the **autopilot** path for `full_start_v1`, not a concatenated movie
and not the G4 late-spine waves above.

```text
board row
  → hop-replay --dual from archived sN live pin
  → --promote-bank
  → optimize_room_policy.py --takeover-sweep  (AP can join mid-room)
  → RoomAutopilot + room_adapter.search_live_adapter
```

**Subpixel / door / enemy RNG** are not edited out of tapes. The adapter
starts from exact live RAM (subpixels, velocity, door kinematics, enemy
phase) and pulse-searches onto the compiled trajectory. Door bands live in
`door_kinematics`.

```bash
uv run python snes/super_metroid/scripts/tools/build_product_chain_board.py --summary
# Template (GREEN): Climb s2 hop 9 from archived pin
uv run python snes/super_metroid/scripts/tools/replay_human_hop.py \
  snes/super_metroid/tasks/full_start_v1_segments/s2/tape.json --hop 9 --dual
```

## Wave order

| Wave | Bead | Focus | Tape |
|------|------|-------|------|
| **A** | rr-7thf.4 | Escape 1–4 | `g4_tourian_human_mb` |
| **B** | rr-7thf.5 | Climb / Parlor / Landing Site | `g4_tourian_human_mb` |
| **C** | rr-7thf.6 | G4 statues → Metroids → Big Boy | `g4_tourian_human` |
| **D** | rr-7thf.7 | MB approach + Mother Brain fight | `g4_tourian_human_bb` + `_mb` |
| thrash queue | rr-7thf.9 | Long dwells (Ridley, Worst, Metal Pirates, Draygon, …) | board `thrash_ranking` |

Within a wave, prefer **priority 1** hops (dwell > 3000 or G4/MB path) only
after shorter traversal hops prove the leave door.

## Wave status (2026-08-10)

| Wave | Result | Notes |
|------|--------|-------|
| A Escape 1–4 | **GREEN** ×4 dual | seeds `g4_tourian_human_mb_seeds/escape*_safe.json` |
| B Climb/Parlor/LS | **GREEN** ×3 dual | LS → `ENDING_OR_CREDITS` observed; green via end_xy; residual `rr-7thf.5-residual.md` |
| C G4→Big Boy | **15/16 GREEN** | **RED** Metroid 4 hop12 — mid-pin or pure rewrite (`rr-7thf.6`) |
| D MB approach | **GREEN** approach 0–7; leave-to-escape GREEN | **RED** bb hop8 full mid-stun dwell; phase-split residual `rr-7thf.7` |

### Known open-loop RED (need more mid F6 pins or pure rewrite)

1. **Metroid Room 4 leave** (`g4_tourian_human` hop 12) — enter→mid
   **dual GREEN** (lockstep mid `f020360_mid_lockstep_0xDBCD.state`); mid→Hopper
   still RED (desync at f20361). Seed:
   `tasks/g4_tourian_human_seeds/metroid4_enter_to_mid_safe.json`
2. **MB full fight dwell** (`g4_tourian_human_bb` hop 8) — enter→mid
   lockstep pin `f008135_mid_lockstep_0xDD58.state` (dual enter→mid); end
   stun + escape leave still from mb boot (GREEN). Mid→stun open-loop still RED.

## Commands

```bash
# Board
uv run python snes/super_metroid/scripts/tools/build_late_spine_board.py --summary

# Hop inventory for one tape (no full open-loop)
uv run python snes/super_metroid/scripts/tools/extract_human_tape.py \
  snes/super_metroid/tasks/g4_tourian_human_mb.json \
  --out snes/super_metroid/tasks/g4_tourian_human_mb_extract.json \
  --summary

# List live anchors
uv run python snes/super_metroid/scripts/tools/extract_human_tape.py \
  snes/super_metroid/tasks/g4_tourian_human_mb.json --list-anchors

# Dual green a hop
uv run python snes/super_metroid/scripts/tools/replay_human_hop.py \
  snes/super_metroid/tasks/g4_tourian_human_mb.json --hop 1 --dual

# Multi-hop compose from live pins
uv run python snes/super_metroid/scripts/tools/compose_human_hops.py \
  snes/super_metroid/tasks/g4_tourian_human_mb.json --dual
# or:  ./play --compose g4_tourian_human_mb

# Open-loop-safe trim
uv run python snes/super_metroid/scripts/tools/trim_human_hop.py \
  snes/super_metroid/tasks/g4_tourian_human_mb.json --hop 1 --mode safe -o out.json

# Thrash edit hints (not open-loop-safe alone)
uv run python snes/super_metroid/scripts/tools/trim_human_hop.py \
  snes/super_metroid/tasks/g4_tourian_human_mb.json --hop 2 --mode traversal -o hint.json

# Offline midpoints (any tape, including no-anchor takes)
uv run python snes/super_metroid/scripts/tools/materialize_hop_mid.py \
  snes/super_metroid/tasks/ws_ship_human.json --all-hops --propose

# Lockstep mid dump from live enter (dual-verify enter→mid; kind=mid_lockstep)
uv run python snes/super_metroid/scripts/tools/materialize_hop_mid.py \
  snes/super_metroid/tasks/g4_tourian_human.json --hop 12 --materialize

# Refresh extracts then rebuild board
uv run python snes/super_metroid/scripts/tools/build_late_spine_board.py --refresh --summary
```

G4 free-record entry points (see also `AGENTS.md`):

```bash
uv run python snes/super_metroid/scripts/record/guided_human.py \
  --from post-bosses --name g4_tourian_human --no-guide
# Continue from g4_tourian_human_end → _bb → _mb as separate takes
```

## Board schema (sketch)

`tasks/LATE_SPINE_HOP_BOARD.json`:

- `pipeline`: `"hop-replay → trim → pure dual green"`
- `tapes[]`: name, task path, frames, anchors flag, end_fingerprint, `hops[]`
- each hop: `index`, `room`, `name`, `start_index`/`end_index`, `dwell`,
  `mode` (`combat`|`traversal`), `leave_room`, `end_xy`, `anchor_path`, `priority`
- `thrash_ranking[]`: sorted by dwell desc
- `gaps[]`: e.g. `ws_ship_human` no_anchors, `maridia_grapple_human` end_state_lost
- `wave_order[]`: A–D + thrash queue ↔ beads

**mode heuristic:** combat if room id is a boss/Metroid/Big Boy/Metal Pirates
set (see generator `COMBAT_ROOM_IDS`); else traversal.

**anchor_path:** prefer `boot` / `room_enter` in the hop dwell window (live
dumps settle a few frames after room change), else same-room enter/boot with
frame ≤ hop start; paths relative to `snes/super_metroid/`.

**priority:** `1` if dwell > 3000 or tape is G4 triple; `2` combat / G4-path
rooms / dwell ≥ 1500; else `3`.

## Gaps / residuals

| Tape | Issue | Follow-up |
|------|-------|-----------|
| `ws_ship_human` | no live anchors | short re-record (rr-7thf.8) |
| `maridia_grapple_human` | end state lost (open-loop) | use `--from post-grapple`; hops still in extract |
| `gravity_path_human` | legacy snapshot extract | not a hop board |
| thrash rooms outside G4 | long dwells on post-main-hall / SJ / Botwoon | rr-7thf.9 queue |

Closing this inventory bead does **not** STATUS-promote pure greens; each wave
bead owns dual-green evidence under game process docs.

## Related

- Per-take notes: `SM-POST-MAIN-HALL-HUMAN.md`, `SM-POST-SJ-EXIT-HUMAN.md`,
  `SM-MARIDIA-BOTWOON-HUMAN.md`, `SM-MARIDIA-GRAPPLE-HUMAN.md`
- Process: `AGENTS.md` (pure-first)
- Package: `super_metroid.human_tape` (`anchors` / `hops` / `replay` /
  `midpoints` / `trim`)
