# SMW Status

Verified 2026-08-10.

## Facts

- The installed USA ROM has SHA1
  `6b47bb75d16514b6a476aa0c73a683a2a4c18765` and SHA256
  `0838e531fe22c077528febe14cb3ff7c492f1f5fa8de354192bdff7137c27f5b`.
- Local BizHawk 2.11 loads native BK2 movies headlessly under Xvfb/Mono and
  exposes SMW WRAM to the oracle Lua script on Snes9x and BSNESv115+.
- TASVideos user file `637823197083827931` (2022, native BizHawk 2.3.2
  Snes9x) replays unchanged on BizHawk 2.11. Yoshi's Island 2 is GREEN from
  power-on: translevel `0x2A`, entry frame 1649, exit frame 6138. Its next
  stage contains repeated deaths and a game-over, so it is not promoted as a
  clean multi-level skill source.
- TASVideos user file `34596324054209273` (the authors' native BizHawk port of
  the published warps run) preserves optimized movement. With input rows
  unchanged and only old core metadata retargeted to BSNESv115+, two
  independent BizHawk 2.11 runs match exactly through two exits:
  - Yoshi's Island 2 / translevel `0x2A`: frames 1634-3943, maximum X 4858,
    normal end-timer completion.
  - Yoshi's Island 3 / translevel `0x27`: frames 4331-6314, maximum X 2721,
    one same-life wings sublevel and exits-completed transition.
  Both segments have entry/exit states and `clean_single_attempt` exact-input
  skills. The next translevel (`0x26`) repeatedly loses lives under v115, so
  the trusted movement prefix ends after the second exit.
- The original Snes9x 1.43 SMV and bsnes v085 LSMV both load after conversion,
  but current-core power-on playback dies and stalls in Yoshi's Island 2.
  Those lanes are RED and must not provide skills.
- Submission `10095S` (2025) matches the ROM and was independently sync-
  verified on BizHawk 2.11. It uses ACE to force exits immediately, so it is a
  compatibility/level-enumeration reference, not a movement-skill source.
- Raw movies, ROMs, states, and replay evidence remain ignored under
  `snes/SMW/tas/ref/` and `snes/SMW/recordings/tas_oracle/`.

## Maturity Gate

Produce two independent GREEN BizHawk 2.11 replays through the first three
normal route levels, with identical RAM boundaries, entry/exit states, and one
clean single-attempt skill artifact per level.
