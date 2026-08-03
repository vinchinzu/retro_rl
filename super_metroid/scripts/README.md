# Super Metroid scripts

| Dir | Purpose |
|-----|---------|
| `record/` | Continuous power-on recordings (baselines) + guided human demos |
| `verify/` | Offline report / graph checks on recordings |
| `probe/` | Development probes (post-Super, bosses, route warps, room timer) |
| `export/` | Regenerate maps / path board / plans |
| `room/` | Room-problem practice runner |
| `scaffold_tip.py` | Pure-first tip extension scaffold (controller stub + residual + checklist) |
| `setup_rom.py` | Install shared ROM into integration |
| `import_legacy_assets.py` | Pull legacy map assets |

Invoke from repo root, e.g.:

```bash
uv run python super_metroid/scripts/record/continuous.py --no-video
uv run python super_metroid/scripts/record/continuous.py --to supers --no-video --room-timing

# Human path record from Cathedral with on-screen route guide (F5 saves task JSON)
uv run python super_metroid/scripts/record/guided_human.py
uv run python super_metroid/scripts/record/guided_human.py --list
uv run python super_metroid/scripts/record/guided_human.py --from bubble --route bubble-to-bat

uv run python super_metroid/scripts/probe/post_spore_pb.py --to main
uv run python super_metroid/scripts/export/path_room_board.py

# Room timing (emulator frames; stock ROM — see docs/ROOM_TIMER.md)
uv run python super_metroid/scripts/probe/room_timer.py self-check
uv run python super_metroid/scripts/probe/room_timer.py offline -i samples.json

# Pure probe: nav-mode RAM + pin on RED; source catalog suggest
uv run python super_metroid/scripts/probe/kpdr.py suggest-source --room 0xA6E2 --segment varia-to-kraid
uv run python super_metroid/scripts/probe/kpdr.py pure varia-to-kraid \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_collected.state \
  --pin-json super_metroid/debug/varia_to_kraid_pin.json

# Scaffold next pure hop (dry-run checklist; --write to emit files)
uv run python super_metroid/scripts/scaffold_tip.py \
  --segment business_to_frog_save --from-room 0xA7DE --to-room 0xB167 \
  --module k4_norfair --card-id SM-K4-BUBBLE-01 --dry-run
```
