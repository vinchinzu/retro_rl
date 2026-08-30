# Super Metroid scripts

| Dir | Purpose |
|-----|---------|
| `record/` | Continuous power-on recordings (baselines) + guided human demos |
| `verify/` | Offline report / graph checks on recordings |
| `probe/` | A/B loop (`kpdr.py` load pin / play / compare) + Gravity `ws_main_climb` / Alcatraz WIP |
| `export/` | Regenerate maps / path board / plans |
| `room/` | Room-problem practice runner |
| `scaffold_tip.py` | Pure-first tip extension scaffold (controller stub + residual + checklist) |
| `setup_rom.py` | Install shared vanilla ROM into integration |
| `setup_practice_rom.py` | Patch practice-hack ROMs into `roms/` (presets + InfoHUD) |
| `export/practice_repertoire.py` | Rebuild practice-hack preset menu catalog JSON |
| `import_legacy_assets.py` | Pull legacy map assets |
| `tools/yt_ref.py` | YouTube KPDR ref VOD: fetch / chunk button+frame extract (gitignored `refs/yt_reference/`) |

Invoke from repo root, e.g.:

```bash
uv run python snes/super_metroid/scripts/record/continuous.py --no-video
uv run python snes/super_metroid/scripts/record/continuous.py --to supers --no-video --room-timing

# Human path record from Cathedral with on-screen route guide (F5 saves task JSON)
# Long takes: room/item anchors + F6 pins under tasks/<name>_anchors/ (default ON)
uv run python snes/super_metroid/scripts/record/guided_human.py
uv run python snes/super_metroid/scripts/record/guided_human.py --list
uv run python snes/super_metroid/scripts/record/guided_human.py --from bubble --route bubble-to-bat
# Post-supers Charge (Big Pink main shaft → collect + return); F5 → tasks/*.json
uv run python snes/super_metroid/scripts/record/guided_human.py \
  --from big-pink --route charge-collect-return --name charge_human
# Offline hop inventory / end fingerprint (open-loop = hop-replay / compose)
uv run python snes/super_metroid/scripts/tools/extract_human_tape.py \
  snes/super_metroid/tasks/maridia_grapple_human.json --summary

# Multi-take practice (Spazer Double Chamber missile ledge → Wave) — F5 save, reload, repeat
uv run python snes/super_metroid/scripts/record/practice_takes.py
uv run python snes/super_metroid/scripts/record/practice_takes.py --list-segments
uv run python snes/super_metroid/scripts/record/practice_takes.py \
  --segment dc-missile-wave --series dc_missile_v1 --list
uv run python snes/super_metroid/scripts/record/practice_takes.py \
  --segment dc-missile-wave --bot-check
# One-shot (same pin as practice default)
uv run python snes/super_metroid/scripts/record/guided_human.py \
  --from double-chamber --route double-chamber-to-wave --name dc_missile_wave_take01 --no-guide

# K5 Red Tower → Hellway multi-take (rr-av5s). Docs: docs/tasks/RED_CLIMB_HUMAN.md
./snes/super_metroid/scripts/record/red_climb_session.sh pure red_climb_v1
./snes/super_metroid/scripts/record/red_climb_session.sh human
./snes/super_metroid/scripts/record/red_climb_session.sh rank red_climb_v1
uv run python snes/super_metroid/scripts/tools/rank_red_climb_takes.py --series red_climb_v1

uv run python snes/super_metroid/scripts/probe/kpdr.py pure big-pink-to-ghz \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/dev_b1_bigpink_main_controller.state
uv run python snes/super_metroid/scripts/export/path_room_board.py

# Room timing (emulator frames; stock ROM — see docs/ROOM_TIMER.md)
uv run python snes/super_metroid/scripts/probe/room_timer.py self-check
uv run python snes/super_metroid/scripts/probe/room_timer.py offline -i samples.json

# Pure probe: nav-mode RAM + pin on RED; source catalog suggest
uv run python snes/super_metroid/scripts/probe/kpdr.py suggest-source --room 0xA6E2 --segment varia-to-kraid
uv run python snes/super_metroid/scripts/probe/kpdr.py pure varia-to-kraid \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_collected.state \
  --pin-json super_metroid/debug/varia_to_kraid_pin.json

# Scaffold next pure hop (dry-run checklist; --write to emit files)
uv run python snes/super_metroid/scripts/scaffold_tip.py \
  --segment business_to_frog_save --from-room 0xA7DE --to-room 0xB167 \
  --module norfair/cathedral --card-id SM-K4-BUBBLE-01 --dry-run

# YouTube reference VOD (default Kentroid KPDR; data under refs/yt_reference/, gitignored)
uv run python snes/super_metroid/scripts/tools/yt_ref.py list
uv run python snes/super_metroid/scripts/tools/yt_ref.py status
uv run python snes/super_metroid/scripts/tools/yt_ref.py chunk \
  --start 1338 --end 1351 --name moat_shinespark --spark
uv run python snes/super_metroid/scripts/tools/yt_ref.py chunk --segment-id k2_spazer --stride 2

# Shinespark / K6 (docs/tasks/SHINE_PRACTICE.md): Skill in routes/skills/shinespark.py
uv run python snes/super_metroid/scripts/probe/kpdr.py compose moat-to-ws \
  --source snes/super_metroid/scratch/post_moat_poweron.state
# Product WO → green Super WS 0xCA08
uv run python snes/super_metroid/scripts/probe/kpdr.py pure west-ocean-to-ws \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_moat_west_ocean_spark.state
# Edge bowling practice only (0xC98E — not product WS)
uv run python snes/super_metroid/scripts/probe/kpdr.py pure west-ocean-to-bowling \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_moat_west_ocean_spark.state \
  --place-x 350 --place-y 550
# Human: optional WO practice, or ship free-record from product WS pin
uv run python snes/super_metroid/scripts/record/guided_human.py --from west-ocean --list
uv run python snes/super_metroid/scripts/record/guided_human.py --from ws-entrance --name ws_ship_human
uv run python snes/super_metroid/scripts/record/practice_takes.py \
  --segment west-ocean-to-ws --series west_ocean_ws_v1
uv run python snes/super_metroid/scripts/record/practice_takes.py \
  --segment ws-entrance --series ws_ship_v1
```
