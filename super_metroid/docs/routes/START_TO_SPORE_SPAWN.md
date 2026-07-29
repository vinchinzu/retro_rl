# Continuous start-to-Spore-Spawn baseline

## Reproduce and verify

From the repository root:

```bash
uv run python super_metroid/scripts/setup_rom.py
uv run python super_metroid/scripts/export/spore_spawn_plan.py
uv run python super_metroid/scripts/export/progression_map.py
uv run python super_metroid/scripts/record/start_to_spore_spawn.py
uv run python super_metroid/scripts/verify/start_to_spore_spawn.py
```

Use `--no-video` for a dry machine-report run. Current-energy and
naturally-unlocked current-ammo assists are enabled by default. Disable either
with `--no-unlimited-energy` or `--no-unlimited-ammo`.

The run starts at emulator power-on (`retro.State.NONE`) and never loads a
save state. Energy refill is suspended on Ceres so Ridley's damage can trigger
the natural countdown. On Zebes it restores current energy only to the
naturally acquired maximum. Ammo refill likewise writes current ammo only
after a nonzero capacity is collected.

## Planning from editor map data

`maps/post_torizo_to_spore_spawn_plan.json` is pre-calculated from:

- the `super_metroid_editor` SMEDIT navigation export, SHA-256
  `8bcd93715dc5ef386ac4be5f62f8774cf7564fd09dea53445fbc33b3bf2c59e0`;
- the editor practice-route reference, SHA-256
  `2b3629503f4257ba6641750968b51aab56ada66ccedca4fcd8839da862e92849`;
- one explicit directed patch from Main Shaft to Dachora, supported by the
  exported inverse door and the practice-route door record.

The capability-aware planned room path is:

```text
Parlor (0x92FD)
  → Terminator (0x990D)
  → Green Pirates Shaft (0x99BD)
  → Lower Mushrooms (0x9969)
  → Green Brinstar elevator (0x9938)
  → Main Shaft (0x9AD9)
  → Dachora (0x9CB3)
  → Big Pink (0x9D19)
  → Spore Kihunters (0x9D9C)
  → Spore Spawn (0x9DC7)
  → Spore Super room (0x9B5B)
```

Early Supers are not required. The reference route's Early Supers notes are
advisory legacy context; the editor graph, capability planner, and continuous
trace establish the route above with Morph Ball, Bombs, and Missiles.

Planning and acceptance are intentionally separate. The plan is labeled
`planned_not_continuous`; it never claims that an editor edge was traversed.
The acceptance report independently records all ten suffix door transitions,
then the verifier checks that each planned pair was observed in order and
matches the typed progression graph.

## Controller policy

`spore_spawn_controller.py` emits controller input only. It checks natural
room, capability, coordinate, Energy Tank, enemy-clear, boss-activation, HP,
and exit boundaries. Big Pink and the post-boss shaft use finite map-guided
controller sequences found in the development emulator. Spore combat reads
typed enemy position, HP, and mouth-open spritemap state to aim and fire.

The development driver may start editor save states for route search, but it
is explicitly excluded from acceptance. The accepted runner composes the
controller only after the verified power-on-through-Bomb-Torizo prefix.

## Acceptance evidence

`recordings/start_to_spore_spawn.mp4`:

- H.264, 512×448, 60 fps;
- 91,220 encoded frames / 1,520.333 seconds;
- SHA-256
  `0c7fb620aa272044250f3b5e1df38af34da9bef52b3a54a83180f73b9046dd9e`;
- visibly shows power-on/title, the Terminator Energy Tank, Green Brinstar,
  Spore Spawn activation and combat, the death animation, natural exit, and
  settled post-boss room.

`recordings/start_to_spore_spawn.json` records:

- the accepted Missile capacities and Morph Ball/Bombs prefix;
- Terminator max energy changing naturally `99 → 199` at frame 51,142;
- all 40 observed room transitions matching typed graph edges;
- all ten editor-planned suffix transitions observed;
- Spore Spawn activating at 960 HP at frame 66,130;
- HP history `960 → 860 → … → 60 → 0`, reaching zero at frame 89,303;
- natural `0x9DC7 → 0x9B5B` transition at frame 90,802;
- zero save-state loads, deaths, progression writes, and capacity writes;
- 252 current-energy refills and 497 current-Missile refills within the assist
  contract.

`recordings/start_to_spore_spawn.verify.json` independently re-hashes the ROM,
report, video, accepted prefix policies, available policy sources, controller,
planner artifact, editor graph, and reference route. It re-counts the MP4 with
ffprobe and revalidates split order, capacities, boss HP, every typed edge, and
the planned-versus-observed route relationship.

Visual spot checks are in `recordings/verification_spore_spawn/`; the contact
sheet includes the title, Energy Tank, Green Brinstar, Kihunters, Spore fight,
zero-HP/death frames, natural exit, and terminal room.
