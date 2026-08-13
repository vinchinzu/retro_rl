# Super Metroid scripts

| Dir | Purpose |
|-----|---------|
| `record/` | Continuous power-on recordings (baselines) + guided human demos |
| `verify/` | Offline report / graph checks on recordings |
| `probe/` | Daily/dev CLIs (KPDR pure, bosses, route warps, room timer, post-Spore) |
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

uv run python snes/super_metroid/scripts/probe/post_spore_pb.py --to main
uv run python snes/super_metroid/scripts/export/path_room_board.py

# Room timing (emulator frames; stock ROM — see docs/ROOM_TIMER.md)
uv run python snes/super_metroid/scripts/probe/room_timer.py self-check
uv run python snes/super_metroid/scripts/probe/room_timer.py offline -i samples.json

# Pure probe: nav-mode RAM + pin on RED; source catalog suggest
uv run python snes/super_metroid/scripts/probe/kpdr.py suggest-source --room 0xA6E2 --segment varia-to-kraid
uv run python snes/super_metroid/scripts/probe/kpdr.py pure varia-to-kraid \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_collected.state \
  --pin-json super_metroid/debug/varia_to_kraid_pin.json

# Pure-chain video + reasonSpans (skill extraction source; not continuous evidence)
uv run python snes/super_metroid/scripts/probe/record_pure_chain.py --list
uv run python snes/super_metroid/scripts/probe/record_pure_chain.py --preset charge-collect-return
uv run python snes/super_metroid/scripts/probe/record_pure_chain.py --preset big-pink-to-ghz

# Scaffold next pure hop (dry-run checklist; --write to emit files)
uv run python snes/super_metroid/scripts/scaffold_tip.py \
  --segment business_to_frog_save --from-room 0xA7DE --to-room 0xB167 \
  --module k4_norfair --card-id SM-K4-BUBBLE-01 --dry-run

# YouTube reference VOD (default Kentroid KPDR; data under refs/yt_reference/, gitignored)
uv run python snes/super_metroid/scripts/tools/yt_ref.py list
uv run python snes/super_metroid/scripts/tools/yt_ref.py status
uv run python snes/super_metroid/scripts/tools/yt_ref.py chunk \
  --start 1338 --end 1351 --name moat_shinespark --spark
uv run python snes/super_metroid/scripts/tools/yt_ref.py chunk --segment-id k2_spazer --stride 2

# Shinespark gym + K6 pure (docs/tasks/SHINE_PRACTICE.md)
# Landing Site store drill (bot holds RIGHT+B; you press DOWN when e=4)
uv run python snes/super_metroid/scripts/probe/shine_practice.py drill
uv run python snes/super_metroid/scripts/probe/shine_practice.py human --series ls_edge_v1
uv run python snes/super_metroid/scripts/probe/shine_practice.py diagnose \
  snes/super_metroid/tasks/shine_practice/ls_edge_v1/take03.json
# Moat pure → West Ocean handoff
uv run python snes/super_metroid/scripts/probe/moat_spark_watch.py pure
# Product WO → green Super WS 0xCA08 (pin: post_west_ocean_ws_spark.state)
uv run python snes/super_metroid/scripts/probe/west_ocean_spark.py pure-ws
# Compose Kihunter/Moat → Moat spark → over-ocean → WS pin (Phantoon record handoff)
uv run python snes/super_metroid/scripts/probe/west_ocean_spark.py chain-ws
uv run python snes/super_metroid/scripts/probe/record_pure_chain.py --preset moat-to-ws
# Edge bowling practice only (0xC98E — not product WS)
uv run python snes/super_metroid/scripts/probe/west_ocean_spark.py pure
# Human: optional WO practice, or ship free-record from product WS pin
uv run python snes/super_metroid/scripts/record/guided_human.py --from west-ocean --list
uv run python snes/super_metroid/scripts/record/guided_human.py --from ws-entrance --name ws_ship_human
uv run python snes/super_metroid/scripts/record/practice_takes.py \
  --segment west-ocean-to-ws --series west_ocean_ws_v1
uv run python snes/super_metroid/scripts/record/practice_takes.py \
  --segment ws-entrance --series ws_ship_v1
```
