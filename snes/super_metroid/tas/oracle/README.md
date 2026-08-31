# Super Metroid TAS oracle tooling

Native-core truth path for full-movie reference. **Not** the snes9x Gym harness.

**Preferred 100% path:** lsnes rr2-β23 + `sniq_100_4010M.lsmv` (TASVideos [#4010M](https://tasvideos.org/4010M) authoring core: bsnes v085 Compatibility). The YouTube encode is not evidence. BizHawk `sniq_100p.bk2` is a converter copy; Linux Mono libsnes SEGV still blocks that track. Session plan: [`docs/tasks/LSNES_100_PLAN.md`](../../docs/tasks/LSNES_100_PLAN.md).

| Doc | Role |
|-----|------|
| [`docs/TAS_BSNES_ORACLE.md`](../../docs/TAS_BSNES_ORACLE.md) | Long-path plan + phases |
| [`tas/ref/ORACLE_ENV.md`](../ref/ORACLE_ENV.md) | Verified host/movie env + blockers |
| Beads epic | `rr-0lz6` |

## Layout

```text
tas/oracle/
  README.md                 # this file
  lsnes_dump_sm.lua         # lsnes WRAM dump + GREEN (Landing/morph)
  run_lsnes_100.sh          # Wine + lsnes-bsnes.exe + #4010M LSMV
  verify_movie_sync.lua     # BizHawk Phase 1: room/item progress + GREEN
  run_verify_100.sh         # launch BizHawk with absolute paths + oracle config
  oracle_out_dir.txt        # generated sidecar (out_dir for Lua; gitignored locally)
  oracle_flags.txt          # generated early_exit / max_frames
```

## Native 100% (lsnes #4010M)

```bash
# monorepo root — intro-aware verify (Ceres elev → Landing/morph)
MAX_FRAMES=60000 EARLY_EXIT=1 \
  ./snes/super_metroid/tas/oracle/run_lsnes_100.sh \
  snes/super_metroid/recordings/tas_oracle/sniq_100_lsnes

# full movie dump (no early exit; ~222788f)
EARLY_EXIT=0 MAX_FRAMES=230000 \
  ./snes/super_metroid/tas/oracle/run_lsnes_100.sh \
  snes/super_metroid/recordings/tas_oracle/sniq_100_lsnes
```

Requirements:

- `lsnes-bsnes.exe` from TASVideos `lsnes-rr2-beta23.7z` at `~/.local/opt/lsnes-rr2-beta23/`
- Wine (system, or Kron4ek `wine-*-amd64-wow64` unpacked to `~/.local/opt/wine`) — exe is 32-bit PE
- ROM `roms/SuperMetroid.sfc` SHA256 `12b77c4bc9c1832cee8881244659065ee1d84c70c3d29e6eaf92e6798cc2ca72`
- Movie `tas/ref/sniq_100_4010M.lsmv` (222 788 frames)

Outputs under the out dir (gitignored `recordings/`):

| File | Meaning |
|------|---------|
| `dump_log.txt` | Heartbeats + trusted rooms |
| `events.jsonl` | room_enter / item_gain / control |
| `room_timeline.csv` | Same events, tabular |
| `proof.json` | `GREEN` / `PARTIAL` |
| `meta_launch.json` | lsnes/wine paths + hashes |

**GREEN:** Landing `0x91F8` or morph bit after Ceres elev `0xDF45`.

Button parse (no emulator): `uv run python -m super_metroid.tas.lsmv` is not a CLI; use tests / `parse_lsmv(tas/ref/sniq_100_4010M.lsmv)`.

## BizHawk Phase 1 — verify 100% BK2 (converter; blocked on this host)

**Intro is long.** Title + Ceres arrival is **~3–5 minutes** of content (~8–12k frames before first elev `0xDF45`). Scripts must wait past that before claiming Ceres/Zebes progress.

```bash
# monorepo root — intro-aware long count (preferred)
LUA_SCRIPT=long_count.lua MAX_FRAMES=60000 EARLY_EXIT=1 \
  ./snes/super_metroid/tas/oracle/run_verify_100.sh \
  snes/super_metroid/recordings/tas_oracle/sniq_100_bsnes_verify

# full verify Lua (room/item events)
./snes/super_metroid/tas/oracle/run_verify_100.sh
```

Requirements:

- `bizhawk` on `PATH` (or `BIZHAWK=…`)
- ROM `roms/SuperMetroid.sfc` SHA1 `DA957F0D63D14CB441D215462904C4FA8519C613`
- Movie `tas/ref/sniq_100p.bk2` (authoring core **BSNES** / libsnes)

Outputs under the out dir (gitignored `recordings/`):

| File | Meaning |
|------|---------|
| `long_count.txt` / `verify_log.txt` | Heartbeats + trusted rooms |
| `long_proof.json` / `verify_proof.json` | `GREEN` / `PARTIAL*` |
| `meta_launch.json` | BizHawk version, paths, hashes |
| `bizhawk_config.ini` | Temp config (FrameSkip=0, BSNES preferred) |

**GREEN:** first elev seen **and** (Landing `0x91F8` or morph bit).

**Blocker:** authoring **libsnes** `PPU::render_line` SEGV on Mono Linux during intro. BSNESv115+ can soak past intro but **desyncs** — not oracle. See `ORACLE_ENV.md`.

## Notes for agents

1. Absolute CLI paths — wrapper cds to `~/.bizhawk`.  
2. Movie forces **BSNES** (`libsnes.wbx`); default user config may say Snes9x.  
3. No L+R sanitize on dumps.  
4. No STATUS from movie frames alone.  
5. Product pure-first remains the continuous tip.  
