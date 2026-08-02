## Residual — SM-BOSS-NATURAL-ENTRY-CLI

### Result
GREEN

### Files changed
- `super_metroid/combat/natural_entry.py` — multi-boss capture dispatch, settle fingerprint (room/pose/door), source-state idle capture, argparse surface (`build_cli_parser` / `run_capture_natural` / `list` / `describe`)
- `super_metroid/scripts/probe/natural_entry_cli.py` — thin shared CLI wrapper (one entry, not per-boss sprawl)
- `super_metroid/docs/tasks/SM-BOSS-NATURAL-ENTRY-CLI-residual.md` — this residual

### Verify paste
```text
$ uv run python super_metroid/scripts/probe/bomb_torizo_combat.py --help
usage: bomb_torizo_combat.py [-h]
                             {strategy,capture-natural,prove-natural,eval,train}
                             ...
exit=0

$ uv run python super_metroid/scripts/probe/natural_entry_cli.py capture-natural --help
usage: natural_entry_cli.py capture-natural [-h] [--from-state FROM_STATE]
                                            [--mode {active,full_hp,room_entry,statue}]
                                            ...
                                            boss
exit=0

$ uv run python super_metroid/scripts/probe/natural_entry_cli.py capture-natural phantoon --plan-only
{"bossId": "phantoon", "roomIdHex": "0xCD13", "requiresFromState": true,
 "defaultMode": "room_entry", "planOnly": true, ...}
exit=0

$ uv run python super_metroid/scripts/probe/natural_entry_cli.py capture-natural kraid --from-state entry --mode room_entry
{"success": true, "bossId": "kraid", "mode": "room_entry",
 "settle": {"roomIdHex": "0xA59F", "pose": 1, "doorTransition": 0, "gameState": 8, ...},
 "progressionWrites": 0, "developmentOnly": true, ...}
exit=0

$ uv run pytest super_metroid/tests/test_boss_pipeline.py -q
14 passed in 0.18s
```

### Acceptance
- [x] CLI or documented entry path works for at least one non-BT boss id (`kraid` via `--from-state entry --mode room_entry`)
- [x] No progression / boss-bit forges in the capture path (`progressionWrites: 0`, provenance flags `forgedBossBits: false`)
- [x] Residual next card ID + one change

### Residual risks
- Only Bomb Torizo has a continuous power-on prefix; Phantoon / Botwoon / etc. require a doorway or predecessor save (`--from-state`)
- Phantoon / Botwoon named `entry` aliases are documented but scratch files may be missing until those sources are captured
- Source-state settle capture is **not** continuous natural entry on the power-on chain
- Bomb Torizo back-compat path still lives on `bomb_torizo_combat.py` (unchanged)

### Usage (Phantoon / Botwoon / etc.)

```bash
# List catalog + which bosses need --from-state
uv run python super_metroid/scripts/probe/natural_entry_cli.py list
uv run python super_metroid/scripts/probe/natural_entry_cli.py describe phantoon

# Kraid (known doorway alias)
uv run python super_metroid/scripts/probe/natural_entry_cli.py capture-natural \
  kraid --from-state entry --mode room_entry

# Phantoon / Botwoon: pass a real doorway/predecessor .state once captured
uv run python super_metroid/scripts/probe/natural_entry_cli.py capture-natural \
  phantoon --from-state path/to/phantoon_entry.state --mode room_entry
uv run python super_metroid/scripts/probe/natural_entry_cli.py capture-natural \
  botwoon --from-state path/to/botwoon_entry.state --mode room_entry

# Plan only (no emulator)
uv run python super_metroid/scripts/probe/natural_entry_cli.py capture-natural \
  phantoon --plan-only

# Bomb Torizo continuous prefix (slow; also via bomb_torizo_combat.py)
uv run python super_metroid/scripts/probe/natural_entry_cli.py capture-natural bomb_torizo
```

Modes: `room_entry` (settled ordinary in boss room), `active` / `full_hp` (fight start), `statue` (BT idle).

### Next action (required)
- **Next card ID:** SM-BOSS-UNIT-MATRIX
- **One change:** Add parametrized catalog×strategy unit matrix tests (`tests/test_boss_catalog_matrix.py`) over every `BOSS_CATALOG` id + soft wrap_* presence
- **Source state:** N/A for matrix card; Phantoon/Botwoon capture still needs doorway source cards when those rooms are on the continuous path

### Non-claims
- Did not STATUS-promote
- Did not forge progression/capacity/door/event/boss RAM
- Not continuous evidence
- Did not claim natural entry on the continuous chain
- Did not edit `continuous.py`, `STATUS.md`, Kraid fight, spine controllers, or `combat/primitives.py`

### Probe pin
room=0xA59F pose=1 x=39 y=395 door_transition=0 game_state=8 (kraid `room_entry` settle from `eye_hj_kraid_entry`)
