# Agent Instructions: Super Metroid

Package `super_metroid` (disk: `snes/super_metroid/`). Goal: a Clean
power-on run through the end of credits, then 100%. Verified facts:
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

One room is cleared when, from the previous room's real leave state, some
button sequence reaches the next room settled (game state 8, no door
transition) without dying. The human tape for that room is the first
candidate. A retimed tape is the second. A search is the third. A fight
picks each next action by trying the options against the ROM and dropping
any that take a hit.

The result of a cleared room is its buttons, appended to the master tape,
plus the leave state the next room starts from. Loads and restores happen
while building. The proof has none: replay the master tape from power-on
with zero loads and zero RAM writes.

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
```

Prepend `QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy`
to anything headless.

## Rules

- One emulator: stable-retro snes9x. `sm/` (lsnes Lua) is parked. TAS
  movies do not sync on snes9x; use them for route and technique, and the
  human tapes for buttons.
- Before tuning a room by hand, run the census row for it. If the row is
  `as_recorded` or `retimed`, there is nothing to tune.
- A red room gets a search or a rollout fight, in the existing owner
  module. Not a new runner, not a new board, not a new glossary word.
- Count progress in rooms on the master tape from power-on. A room cleared
  from a human start state is a candidate, not progress.
- No energy, ammo, item, capacity, boss, door, pose, or timer writes on
  the Clean path — including Clean development probes (read-only RAM and
  save/restore lookahead only). The assisted Phantoon run keeps its own
  contract (`docs/ASSIST_CONTRACT.md`) and its artifacts are never overwritten.
- Verify every Clean master-tape milestone with **both**
  `uv run python -m super_metroid.splice chain --replay` and the
  zero-harness-import audit
  `uv run python snes/super_metroid/scripts/tools/audit_master_tape.py`.
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
- The board's `live` segment rows point at the wrong tape slice. Rebuild
  the board before trusting those 28 rows.
- `bd` (beads) is gone from this clone. Open work lives in `docs/plan.md`.
- Soft max ~1000 LOC per file: merge into the owner or delete. Where
  files go: `docs/ARCHITECTURE.md`.
