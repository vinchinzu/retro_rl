# Hal's Hole in One Golf

Harvest-style bot track for the SNES golf game: custom integration, save
states, menu bootstrap, timed swings, and human ↔ autoplay recovery on the
shared `PlaySession`.

## Quick start

```bash
cd hals_golf

# Human
./run_bot.sh play --state Title
./run_bot.sh play --state Hole1_Command

# Autoplay (menus + shots)
./run_bot.sh play --autoplay --state Title

# Already on the tee
./run_bot.sh play --autoplay --state Hole1_Command --skip-bootstrap

# Fast Title → 18-hole verification; refreshes Hole1_Command and latest
HEADLESS=1 ./run_bot.sh clear --state Title

# VS HAL match vs computer Hal (Amateur)
# A fresh VS boot enters METAL PLAY and uses metal-calibrated routes.
HEADLESS=1 ./run_bot.sh clear --mode vs-hal --state Title

# Existing in-round saves default to the original standard-club routes.
HEADLESS=1 ./run_bot.sh clear --mode vs-hal --club-set standard \
  --state VsHal_Hole1_Command --skip-bootstrap

# Identify an in-round metal-club save explicitly.
HEADLESS=1 ./run_bot.sh clear --mode vs-hal --club-set metal \
  --state latest --skip-bootstrap

# Record a full 60fps MP4 (playable length ≈ frames/60)
HEADLESS=1 PYTHONUNBUFFERED=1 ./run_bot.sh clear --state Title --video

# Reproduce the VS HAL win as a VLC-compatible Theora/Vorbis video with
# emulator sound and a 30-second post-win result sequence.
./record_vs_hal_win.sh
# Full metal stroke-play clear video (validates scorecard + codecs):
./record_metal_clear.sh
# or windowed autoplay + video:
./run_bot.sh play --autoplay --state Title --video
./run_bot.sh play --autoplay --mode vs-hal --state Title
```

Hotkeys: `~` or L+R+SELECT toggles human/autoplay; F5 QuickSave; F7/F8 load.
Videos land in `recordings/` (gitignored). Recording needs `ffmpeg` on PATH;
the VS HAL recording script also uses `ffprobe` to verify both output streams.

## ROM

Headerless USA ROM at `../roms/HalsHoleInOneGolf.smc` (gitignored).
`hals_golf.runtime.retro_setup` repairs `custom_integrations/.../rom.sfc`.

SHA1: `45baf328efa1e573aef81b2a936207f8979206a4`

## Status

Bronze bot clears Amateur stroke play from `Title.state` (`course_complete`).
`--mode vs-hal` selects the title-menu VS HAL match and a fresh bootstrap uses
the built-in `METAL PLAY` club unlock. Standard-club save states remain
supported through `--club-set standard`; `auto` assumes standard clubs when
`--skip-bootstrap` is present. A win writes `VsHalWin.state`. Confirmed RAM: hole, stroke,
REST, lie, aim offset (`0x10B1`), and opponent stroke (`0x10A3`). The round
total is derived from the per-hole stroke counter. Includes
deterministic hazard routes, green/bunker-aware shots, stall/futile-shot
nudges, per-tee `latest` checkpoints, final `CourseComplete.state`, and human ↔
autoplay recovery. Default swing timing is 42/26 frames. Current verified
Title clear: 61,485 frames, total 63 (-9), with every hole at par or better.
Current verified VS HAL Title win: 70,663 frames, record 3-2-7; Hole 1 is a
three-stroke birdie against Hal's four and establishes an immediate 1-up lead.

```bash
HEADLESS=1 ./run_bot.sh clear --state Title
# prints total=…, to_par=…, over_par=[], and scorecard=[…] on success

HEADLESS=1 ./run_bot.sh clear --mode vs-hal --state Title
# prints match_won and match=W-L-T on success
```
