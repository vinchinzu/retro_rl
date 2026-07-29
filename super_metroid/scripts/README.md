# Super Metroid scripts

| Dir | Purpose |
|-----|---------|
| `record/` | Continuous power-on recordings (baselines) |
| `verify/` | Offline report / graph checks on recordings |
| `probe/` | Development probes (post-Super, bosses, route warps) |
| `export/` | Regenerate maps / path board / plans |
| `room/` | Room-problem practice runner |
| `setup_rom.py` | Install shared ROM into integration |
| `import_legacy_assets.py` | Pull legacy map assets |

Invoke from repo root, e.g.:

```bash
uv run python super_metroid/scripts/record/start_to_supers.py --no-video
uv run python super_metroid/scripts/probe/post_spore_pb.py --to main
uv run python super_metroid/scripts/export/path_room_board.py
```
