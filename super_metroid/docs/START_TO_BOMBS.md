# Continuous start-to-Bomb-Torizo baseline

## Reproduce and verify

From the repository root:

```bash
uv run python super_metroid/scripts/setup_rom.py
uv run python super_metroid/scripts/import_legacy_assets.py
uv run python super_metroid/scripts/export_progression_map.py
uv run python super_metroid/scripts/record_start_to_bombs.py
uv run python super_metroid/scripts/verify_start_to_bombs.py
```

Use `--no-video` on the recorder for a dry machine-report run. Unlimited ammo
is enabled by default and may be disabled with `--no-unlimited-ammo`.

The accepted run starts at emulator power-on (`retro.State.NONE`) and never
loads a save state. The title, intro, Ceres, and Morph prefix is the previous
accepted policy. Four hash-pinned continuation policies collect both early
Missile expansions, return through Morph Ball Room/elevator, climb to Parlor,
collect Bombs, defeat Bomb Torizo, and exit to a settled Parlor state.

## Route

```text
power-on → Ceres → Landing Site → Parlor ↓ Climb ↓ Pit
  → Blue Brinstar elevator → Morph Ball
  → Construction Zone → First Missile (capacity 5)
  → Construction Zone → Blue Brinstar Missile (capacity 10)
  → Construction Zone → Morph Ball → elevator → Pit
  → Climb ↑ → Parlor → Flyway → Bombs/Bomb Torizo
  → Flyway → Parlor
```

`policy.py` validates every replay file, entry predicate, exit predicate,
button vector, provenance hash, and action count. Segment evidence records
entry/exit fingerprints, policy/source hashes, action frames, opposite-input
counts, Start-button counts, and maximum identical navigation intervals.

The Pit boundary demonstrates why these checks matter. Position, pose, health,
and timing matched the legacy climb replay, but the two-Missile detour left
Missiles selected. A normal Select-button input during the existing ten-frame
settle restores the replay's beam-selected entry state. No selected-item RAM
write is used.

## Acceptance evidence

`recordings/start_to_bomb_torizo.mp4`:

- H.264, 512×448, 60 fps
- 47,133 encoded frames / 785.55 seconds
- SHA-256
  `67653e011e1da0407b1e95f2749310f72f7d421df99f97a2f24b6e61b3e7a8b5`
- visibly shows two distinct `MISSILE` banners, the `BOMB` banner, Bomb
  Torizo's fight/explosion, and the unlocked exit through Flyway

`recordings/start_to_bomb_torizo.json` records:

- Missile capacities `0 → 5` at frame 27,928 and `5 → 10` at 29,690
- Bombs item bit `0x1000` at frame 42,672
- Bomb Torizo activation at 800 HP and zero HP at frame 44,524
- natural Bomb Torizo Room → Flyway exit at frame 46,449
- 30/30 typed room transitions
- zero save-state loads, capacity writes, and progression writes
- 116 current-Missile refill writes after the natural unlock

`recordings/start_to_bomb_torizo.verify.json` is produced independently from
the recorder. It re-hashes the ROM, report, video, policy files, and available
legacy source files; re-counts MP4 frames with ffprobe; checks every graph
edge; and reasserts split order, inventory, and assist invariants.

Visual spot checks and contact sheets are under
`recordings/verification_bomb_torizo/`.
