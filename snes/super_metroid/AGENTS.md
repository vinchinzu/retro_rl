# Agent Instructions: Super Metroid

Package `super_metroid` (disk: `snes/super_metroid/`). Goal: chain reusable
reactive skills into a sub-hour Clean speedrun. The same skills should serve
Map Rando, SMZ3, and talk-and-play. Verified facts:
`docs/STATUS.md`. Future work and the red-room list: `docs/plan.md`.
Words: `CONTEXT.md`. What counts as Clean: `docs/CLEAN_TRACK.md`.

## How work gets done here

The machine searches. The agent writes what "cleared" means and what to try.

The emulator restores a saved frame bit-exact and steps 1,600 frames a
second per core (32 threads on this box). So a question like "does this
jump land" or "does this shot hit without contact" is answered by saving,
trying the candidates, and restoring. Do not answer it by editing a
constant and re-running a room, and do not model enemy motion by hand.
The ROM is the model.

Use TAS and human tapes to find techniques and establish an A/B time.
Implement the movement or fight as a skill driven by live pose, geometry,
inventory, and enemies. Test from the predecessor's real leave and vary the
entry when checking reuse. A faster skill is ready when the successor
settles with the resources it needs and the chain shows a net gain.

Splice faster skills at settled room boundaries and run the successors from
the new leave. Treat a shifted successor as a candidate for a reactive
skill. Record the generated buttons for the power-on proof with zero loads
and zero RAM writes.

## Commands

```bash
# Which rooms the machine already clears from the human tapes (2 min)
uv run python snes/super_metroid/scripts/tools/tape_census.py
uv run python snes/super_metroid/scripts/tools/tape_census.py --rows 40-60

# One room from a save state, headed, for looking at a red row
uv run python snes/super_metroid/scripts/probe/kpdr.py pure <hop> --source <pin> --headed
./snes/super_metroid/play <pin> --headed

# Assisted run to Phantoon (the verified tip in STATUS; not the Clean goal)
uv run python snes/super_metroid/scripts/record/continuous.py --to phantoon --no-video

# Speed benchmark vs standard 0:56:28.88 KPDR splits + swappable route policies
uv run python -m super_metroid.splice benchmark
uv run python -m super_metroid.splice policy list
uv run python -m super_metroid.splice policy ab <policy> <rule_id>
```

Prepend `QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy`
to anything headless.

## Rules

- One emulator: stable-retro snes9x. `sm/` (lsnes Lua) is parked. TAS
  movies do not sync on snes9x; use them for route and technique, and the
  human tapes for buttons.
- Run the census before replacing a row. Use `as_recorded` and `retimed`
  as baselines for faster skills.
- A red room gets a search or a rollout fight, in the existing owner
  module. Not a new runner, not a new board, not a new glossary word.
- Count progress in rooms on the master tape from power-on. A room cleared
  from a human start state is a candidate, not progress.
- Start speed work from the latest Clean master. Carry each improvement
  forward on its candidate stem and repair joins from the candidate's actual
  leave. Promote after the full power-on replay and independent audit pass.
  Preserve prior tape pairs for explicit replay with `--stem`.
- No energy, ammo, item, capacity, boss, door, pose, or timer writes on
  the Clean path — including Clean development probes (read-only RAM and
  save/restore lookahead only). The assisted Phantoon run keeps its own
  contract (`docs/ASSIST_CONTRACT.md`) and its artifacts are never overwritten.
- Each session's gate is `uv run python -m super_metroid.splice chain --replay`.
  Run the independent ROM audit at the Tourian entrance, boss kills, the
  finish, and before publishing. Use the candidate stem and report the
  audited rows. [Replay and audit commands](docs/SPEED_SPLITS.md#v5-through-credits)
- STATUS changes only from a power-on replay.

## Traps

- A tape slice ends on the frame the room id changes. Hold the last press
  through the door or the replay stops one frame short.
- Door settle: wait for game state 8. State 11 can last 50-100+ frames.
- High WRAM (`$7E:D820+`): `read_bank7e_wram` / `write_wram_u8`. Raw
  `get_ram()[0xD820]` is open-bus garbage.
- Morph bombs are X while morphed, not A. Hop `side` is D-pad LEFT/RIGHT.
  Shoulders are L/R.
- RNG is `$05E5`; boot writes `$0061` after clearing WRAM. A restore puts
  it back, so a rollout sees the real future. See `docs/RNG.md`.
- The human tapes were recorded with energy and ammo refills on. A room
  that clears with a full tank on tape may not on the Clean chain. The
  census prints the energy each room ends with.
- Human slices assume the beam is selected. Clean often arrives with Super
  or Grapple selected, so the slice's first X fires the wrong weapon (rows
  299, 313). OR Item Cancel (Y, index 1) into one frame; timing stays
  identical. It does nothing while input is ignored (door, elevator ride:
  row 313 needs frame 460, not 0).
- Rows 307+ come from other recordings that start mid-room. Row 307 (Landing
  Site) enters at the far door, not where the slice starts; bridge with a
  short run-left prefix, then join the slice at the matching offset.
- The board's `live` segment rows point at the wrong tape slice. Rebuild
  the board before trusting those 28 rows.
- `bd` (beads) is gone from this clone. Open work lives in `docs/plan.md`.
- Soft max ~1000 LOC per file: merge into the owner or delete. Where
  files go: `docs/ARCHITECTURE.md`.
