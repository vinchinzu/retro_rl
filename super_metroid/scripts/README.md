# Super Metroid scripts

| Dir | Purpose |
|-----|---------|
| `record/` | Continuous power-on recordings (baselines) |
| `verify/` | Offline report / graph checks on recordings |
| `probe/` | Development probes (post-Super, bosses, route warps, room timer) |
| `export/` | Regenerate maps / path board / plans |
| `room/` | Room-problem practice runner |
| `setup_rom.py` | Install shared ROM into integration |
| `import_legacy_assets.py` | Pull legacy map assets |

Invoke from repo root, e.g.:

```bash
uv run python super_metroid/scripts/record/continuous.py --no-video
uv run python super_metroid/scripts/record/continuous.py --to supers --no-video --room-timing
uv run python super_metroid/scripts/probe/post_spore_pb.py --to main
uv run python super_metroid/scripts/export/path_room_board.py

# Room timing (emulator frames; stock ROM — see docs/ROOM_TIMER.md)
uv run python super_metroid/scripts/probe/room_timer.py self-check
uv run python super_metroid/scripts/probe/room_timer.py offline -i samples.json
```
