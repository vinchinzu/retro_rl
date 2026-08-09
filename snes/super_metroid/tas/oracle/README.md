# Super Metroid TAS oracle tooling (BizHawk BSNES)

Native-core truth path for full-movie reference. **Not** the snes9x Gym harness.

| Doc | Role |
|-----|------|
| [`docs/TAS_BSNES_ORACLE.md`](../../docs/TAS_BSNES_ORACLE.md) | Long-path plan + phases |
| [`tas/ref/ORACLE_ENV.md`](../ref/ORACLE_ENV.md) | Verified host/movie env + blockers |
| Beads epic | `rr-0lz6` |

## Layout

```text
tas/oracle/
  README.md                 # this file
  verify_movie_sync.lua     # Phase 1: room/item progress + GREEN milestone
  run_verify_100.sh         # launch BizHawk with absolute paths + oracle config
  oracle_out_dir.txt        # generated sidecar (out_dir for Lua; gitignored locally)
  oracle_flags.txt          # generated early_exit / max_frames
```

Phase 2 (blocked by Phase 1 green) will add `bizhawk_dump_sm.lua`, `run_bizhawk_100.sh`, `import_oracle.py`, `schema.md`.

## Phase 1 — verify 100% BK2

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
