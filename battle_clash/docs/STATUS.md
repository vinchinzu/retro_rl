# Status — Battle Clash


## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M1 |
| Best verified result | Title boots; Super Scope input blocked |
| Last verification | 2026-07-22 |
| Runtime class | Bronze |
| Intervention class | Clean |

| Field | Value |
|-------|-------|
| Status | **input blocked** |
| Integration | `BattleClash-Snes` |
| ROM zip | `Battle Clash.zip` |

## Done

- Directory layout and integration stubs (`data.json` / `metadata.json` /
  `scenario.json`)
- `scripts/setup_rom.py` wiring via `snes_oneshot.rom_setup`
- Plan notes for control style and first segment milestone
- Emulator/title boot smoke test
- Input diagnostic: stable-retro exposes the standard 12-button SNES joypad,
  while its emulator object exposes no cursor, light-gun, or mouse injection
- Diagnostic screenshot: [boot_title_input_blocked.png](../recordings/boot_title_input_blocked.png)

## Not done

- Development save states
- RAM map discovery
- Segment policies / behavior tree
- Continuous multi-segment or full-game runs

## Next

Add Super Scope cursor/trigger injection to the emulator bridge. The game then
needs a first-boss checkpoint and a cursor-track/fire policy.
