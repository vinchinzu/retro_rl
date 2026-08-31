# TAS Oracle environment — Super Metroid

**Status:** Phase 1 **partial** (BizHawk libsnes SEGV). Native **lsnes 100%** path added 2026-08-31.  
**Plan:** [`docs/TAS_BSNES_ORACLE.md`](../../docs/TAS_BSNES_ORACLE.md) · beads `rr-wbsr`, `rr-3vfm`  
**Tooling:** [`tas/oracle/`](../oracle/)

## Native 100% movie (preferred)

TASVideos [#4010M](https://tasvideos.org/4010M) is **lsnes rr2-β23**, core `bsnes v085 (Compatibility core)`, not BizHawk.

| Item | Value |
|------|--------|
| Movie | `tas/ref/sniq_100_4010M.lsmv` (byte-identical to TASVideos download) |
| Frames | 222 788 |
| `systemid` | `lsnes-rr1` (rr2 movies still write this) |
| `coreversion` | `bsnes v085 (Compatibility core)` |
| ROM SHA256 | `12b77c4bc9c1832cee8881244659065ee1d84c70c3d29e6eaf92e6798cc2ca72` |
| ROM SHA1 | `DA957F0D63D14CB441D215462904C4FA8519C613` |
| lsnes | `~/.local/opt/lsnes-rr2-beta23/lsnes-bsnes.exe` (32-bit PE from TASVideos 7z) |
| Wine | `~/.local/opt/wine` (Kron4ek 10.0 amd64-wow64) or system wine |
| Prefix | `~/.local/share/lsnes-wine` |
| Wrapper | `tas/oracle/run_lsnes_100.sh` |

`sniq_100p.bk2` is a **TASConverter** BizHawk copy of the same inputs. Use the LSMV + lsnes for bit-exact authoring-core replay.

### Verified lsnes replay (2026-08-31)

`run_lsnes_100.sh` + Wine 10.0 wow64 + `lsnes-bsnes.exe` (rr2-β23):

| Milestone | Frame | Notes |
|----------:|------:|-------|
| `post_load` | 0 | 222 788f, readonly |
| Intro `gs=30` | 0–10 961 | Title / Ceres arrival |
| Ceres Elev `0xDF45` | **10 962** | area=6, energy=99 |
| first_control `gs=8` | **11 182** | pose 0 @ (128,0) — matches harness any% first_control |
| Falling Tile `0xDF8D` | 17 881 | |
| Magnet / hall chain | 19 339–23 915 | `0xDFD7` → `0xE021` → `0xE06B` |
| Ceres Ridley `0xE0B5` | **37 269** | energy 99→25 (real fight) |
| f45 663+ | still Ceres | elev e=99, then Falling, **second Ridley f62639 e=99** — restart-shaped; **not** Landing |

Six unique Ceres rooms, items 0, **no Zebes by f70k**. Through first control + first Ridley this is native-core sync. After Ridley is **unverified**. Plan: [`docs/tasks/LSNES_100_PLAN.md`](../../docs/tasks/LSNES_100_PLAN.md).

## Intro length (do not misread early frames)

Super Metroid’s **title + Ceres arrival cutscene** is multi-minute. Under this movie:

| Milestone | Approx movie frames | Wall @ 60 fps |
|-----------|--------------------:|---------------|
| Boot / menus (`gs≈30`, room 0) | 0 → ~8 000 | ~0–2.2 min |
| First Ceres Elevator `0xDF45` | **~8 500–11 200** | ~2.4–3.1 min |
| (Harness snes9x first_control) | 11 183 | ~3.1 min |
| Deeper Ceres (flat / Ridley) | **tens of thousands** | many more minutes |

**Rules of thumb**

1. Expect **~3–5 minutes of content** before “real” Ceres play is obviously underway.  
2. **Do not** treat WRAM room IDs as truth until `area∈[0,6]`, room ∈ `0x9000–0xEFFF`, and preferably `gs∈{8,9,11,…}` after first elev.  
3. Early “Ridley @ 10k” under the old libsnes probe was a **false positive** (untrusted boot RAM), not a synced TAS.  
4. Verify scripts must use `max_frames ≥ 60000` (or full movie) — not a 12k abort.

## Verified host

| Item | Value |
|------|--------|
| BizHawk | **2.11** · `~/.local/bin/bizhawk` → Mono `EmuHawk.exe` |
| ROM | `roms/SuperMetroid.sfc` |
| ROM SHA1 | **`DA957F0D63D14CB441D215462904C4FA8519C613`** ✓ |
| Movie | `tas/ref/sniq_100p.bk2` (222 789f) |
| Header Core | **BSNES** (LibsnesCore, Profile=**Compatibility**) |
| Waterbox cores present | `libsnes.wbx` (authoring), `bsnes.wbx` (BSNESv115+) |

## Two cores, two outcomes

### A) Authoring core — BSNES / `libsnes.wbx` (needed for oracle truth)

| | |
|--|--|
| Load | Movie forces Core=BSNES → mounts **`libsnes.wbx`** |
| Result | **SIGSEGV** during intro (often before first elev) |
| Stack | `SNES::PPU::render_line()` inside waterbox `/memfd:MemoryBlockUnix` |
| coredump | `coredumpctl` · `si_code=SI_KERNEL` · mono-sgen |
| Lua | Not required to crash (no-Lua movie also dies) |
| Oracle | **Blocked** until this SEGV is cleared |

### B) BSNESv115+ retarget (tooling soak only — **not** oracle truth)

Retarget Header `Core BSNESv115+` + v115 SyncSettings (local copy under `recordings/…/sniq_100p_v115.bk2`). Mounts **`bsnes.wbx`**.

| Frame | Room | Notes |
|------:|------|-------|
| 0–8000 | — | Intro `gs=30` |
| **8500** | **`0xDF45`** | **FIRST elev** after long intro |
| 11000 | `0xDF8D` | Falling Tile |
| 19500+ | `0xDFD7` / `0xE021` / `0xE06B` | Ceres chain |
| **37500** | **`0xE0B5`** | Ridley; energy 99→22 |
| 46000 | `0xDF45` | Elev restart e=99 → **desync thrash** (wrong core) |

**Usable for:** proving BizHawk can run SM for tens of thousands of frames, WRAM domain = `WRAM`, intro timing.  
**Not usable for:** truth pins / STATUS / extract boards (inputs desync vs authoring BSNES).

Evidence (gitignored):  
`recordings/tas_oracle/sniq_100_bsnes_verify/{v115_sparse.txt,intro_core_proof.json}`

## Launch rules

```bash
# Preferred Phase 1 long wait (intro-aware)
LUA_SCRIPT=long_count.lua MAX_FRAMES=60000 \
  ./snes/super_metroid/tas/oracle/run_verify_100.sh \
  snes/super_metroid/recordings/tas_oracle/sniq_100_bsnes_verify
```

1. Absolute paths (wrapper `cd`s to `~/.bizhawk`).  
2. **FrameSkip=0** (frameskip can desync movies).  
3. Prefer GDI (`--gdi`) on Linux.  
4. Force `memory.usememorydomain("WRAM")` — default domain may be System Bus.  
5. Trust only elev/landing/morph-style pins after intro.  
6. Never sanitize L+R on dumps.  

## Phase 1 residual (clear `rr-3vfm`)

Goal: play **authoring** BSNES (`libsnes`) through intro → Ceres → **Landing `0x91F8` or morph**, without SEGV.

Candidates:

1. **Windows BizHawk** under Wine / real Windows (same 2.11 or movie-era build).  
2. Different BizHawk 2.x Linux build if libsnes PPU crash is fixed upstream.  
3. Upstream bug: waterbox `libsnes` `PPU::render_line` SEGV on Mono Linux during SM intro.  

**Do not** declare Phase 1 GREEN from v115 thrash or snes9x thrash.

## Relation to harness

| Path | Core | Full BK2 |
|------|------|----------|
| Gym / stable-retro | snes9x only | Ceres thrash, items `0x0000` |
| BizHawk libsnes (authoring) | BSNES Compatibility | SEGV mid-intro (this host) |
| BizHawk BSNESv115+ retarget | wrong core | Survives intro; desyncs mid-Ceres |

Product pure-first and STATUS rules unchanged.
