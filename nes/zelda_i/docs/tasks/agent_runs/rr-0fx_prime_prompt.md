# rr-0fx — Z4.1 Live recon (prime-agent / deepseek-v4-flash-0731)

Repo: `/home/v/01_projects/11_games/retro_rl`
Game package: `zelda_i` under `nes/zelda_i/` (import name `zelda_i`).
Tracker: bead **rr-0fx** (already `in_progress`). Label `zelda_i`.

## Goal (one bead only)

Assisted first-pass: from checkpoint **`Level3Complete`** (Raft already owned,
`ADDR_RAFT=0x0660` == 1) walk overworld to Level 4 island door, enter dungeon,
save **`Level4Entrance`**, record live screen/room ids. **2/2** trials.

Do **not** claim Clean STATUS. Do **not** poke Raft RAM. Do **not** work other beads.

## Architecture (follow existing patterns)

| Piece | Pattern |
|-------|---------|
| Shared OW hop engine | `zelda_i/ow_path.py` → `OverworldPathController` |
| L5 reference | `level5_overworld.py` + `scripts/probe_level5_entry.py` |
| L3 reference | `level3_overworld.py` + `scripts/run_l2_to_l3.py` |
| L4 scaffold | `level4_overworld.py` (SOURCE_HYPOTHESIS dock `0x55` island `0x45`) |
| Plan-only | `scripts/run_level4_entry.py --plan-only` |
| Assist | `assist.UnlimitedHealthAssist` / `--infinite-life` |
| Evidence | `recordings/*.json` + `.state` under `custom_integrations/LegendOfZelda-Nes/` |

## Steps (tight)

1. Read `AGENTS.md`, `docs/LEVEL4_ROUTE.md`, `level4_overworld.py`, `ow_path.py`, L5 probe pattern.
2. Boot probe: load `Level3Complete`, print mode/level/screen/x/y/raft/triforce. If still in dungeon/fanfare, idle-settle to OW play mode and note return screen.
3. Recon walk (script or short probe): find live dock screen + island/door screen with real Raft. Override hyp ids only when observed.
4. Implement `OverworldToLevel4Controller` (or equivalent) in `level4_overworld.py` using hop table + dock walk + door hunt; live constants for door screen + entry room.
5. Implement/extend runner:
   `uv run python nes/zelda_i/scripts/run_level4_entry.py --infinite-life --from-state Level3Complete --trials 2 --save-state`
6. On success: save `OW_L4Dock` + `Level4Entrance`, write `recordings/l4_entry_recon.json` with screens, entry room, frames, 2/2.
7. Update `docs/LEVEL4_ROUTE.md` live table; unit smoke if you add pure functions.
8. Tests: `uv run pytest nes/zelda_i/tests -q` (or narrow new tests).
9. `bd close rr-0fx --reason "…evidence paths…"` only if 2/2 enter green. Else leave open with PARTIAL notes + residual child via `bd create … --deps discovered-from:rr-0fx`.
10. `bd sync` — do **not** git commit/push unless asked.

## Done gates (must all pass)

- `Level4Entrance.state` exists under `nes/zelda_i/custom_integrations/LegendOfZelda-Nes/`
- `nes/zelda_i/recordings/l4_entry_recon.json` (or equivalent rollup) shows enter success ≥2/2
- Live door screen + entry room written into `level4_overworld.py` (not only docs)
- No Clean STATUS invent; no Raft poke

## Failure mode

If stuck: document live screens tried, save best checkpoint, leave bead open with clear residual, do not force-pass predicates.
