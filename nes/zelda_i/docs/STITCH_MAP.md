# Zelda I honest stitch map — 2026-08-14

Development compose only. **Not Clean, not a STATUS promotion, and not a
continuous power-on run.** Every missing transition is disclosed on-screen and
in the JSON manifest.

## Delivered tape

| Field | Value |
|-------|-------|
| MP4 | `recordings/stitches/reels/power_on_to_level5_honest_seamed/power_on_to_level5_honest_seamed.mp4` |
| Manifest | `recordings/stitches/reels/power_on_to_level5_honest_seamed/power_on_to_level5_honest_seamed.json` |
| Media | H.264/AAC, 480×480, 60 fps, 55,784 frames, 929.75 s, 48,210,332 bytes |
| SHA-256 | `25d140b3d6ec50e3d1d2db8b40aeeb4ab2edcb57231a1dbaa8b19c6d916ccaa2` |
| Start | power-on |
| Last honest room | **Level 5 Triforce room (`0x14`)** |
| Final progression | Triforce `0x1c`; Level 5 bit `0x10` RAM-true |
| Tape kind | state-seamed viewing compose; `continuous_emulator_session=false` |

The manifest inventories 303 local states and 20 BK2s. Completion pins with
provenance exist for Levels 1–5. There is **no** route-eligible
`Level8Complete` state/provenance pair, so the Ganon fixture reel was not
attached.

## What the MP4 actually shows

| Order | Reel / seam | Evidence class |
|------:|-------------|----------------|
| 1 | power-on → Level 1 Triforce room (`0x36`) | continuous Clean reel |
| 2 | on-screen seam card: Level 1 complete → Levels 2–4 complete pins | missing continuous L2–L4 footage; not faked |
| 3 | Level 4 complete → Level 5 entrance (`0x76`) → first-key Gibdos (`0x66`) → East Key Pols Voice (`0x77`) | continuous Survival-assisted reel |
| 4 | on-screen seam card: East Key Pols Voice (`0x77`) → Whistle basement (`0x04`) | missing Recorder-acquisition footage; not faked |
| 5 | Whistle basement (`0x04`) → six-Darknut whistle room (`0x05`) → empty passage (`0x06`) → cellar (`0x07`) → Blue Darknut stairs (`0x64`) → west Gibdo pocket (`0x65`) | continuous Survival-assisted BK2/MP4 |
| 6 | west Gibdo pocket (`0x65`) bomb-east → first-key Gibdos (`0x66`) → north Dodongos (`0x56`) → east Zols (`0x57`) → north Gibdos (`0x47`) → Darknuts + compass (`0x37`) | same continuous reel |
| 7 | mixed Pols/Gibdo/Keese (`0x27`) → west Gibdos (`0x26`) → west Pols Voice (`0x25`) → Digdogger (`0x24`) → Level 5 Triforce room (`0x14`) | same continuous reel; Recorder shrink and sword kill |

Level 5 continuous evidence:

- BK2: `recordings/stitches/bk2_whistle04_to_tf/LegendOfZelda-Nes-Level5Whistle-000000.bk2`
- MP4: `recordings/stitches/reels/l5_whistle_to_tf/l5_whistle_to_tf_assisted.mp4`
- report: `recordings/stitches/l5_whistle04_to_tf_stitch.json`
- 10,776 route frames / 10,780 encoded video frames; no deaths; no resource,
  progression, or capacity pokes; Survival health refill only.
- Health drop to fix later: 43 damage units recorded; Survival restored 44
  health-counter units across 20 writes. Hot rooms were
  Digdogger (`0x24`: 27), first-key Gibdos (`0x66`: 10), west Gibdos
  (`0x26`: 4), north Dodongos (`0x56`: 1), and east Zols (`0x57`: 1).

## Fixture-only Ganon suffix — kept separate

The new backward boundary is **blade-trap/Like-Like room (`0x41`)**. Its north
shutter is sealed while enemies live; after a controller-only clear, north
lands in **east-bomb (`0x31`)** with no door poke. The one-session fixture
suffix then runs:

```text
blade-trap/Like-Like room (0x41)
  → east-bomb (0x31)
  → block-stairs (0x30)
  → cellar (0x67), right exit
  → west-bomb Wizzrobes (0x04)
  → Patra stairs (0x03)
  → final Patra (0x52)
  → Ganon (0x42)
  → Zelda (0x32)
  → credits/final page
```

Evidence: `recordings/l9_room41_dump.json`,
`recordings/l9_play41_north_patra_credits_recon.json`, and
`Level9Room41NorthReconFixture`. The controller portion has zero object, room,
door, inventory, progression, and capacity writes. The start still inherits a
composed full loadout and room loader, so `fixture_only=true` and
`route_eligible=false`.

Separate labeled endcard:
`recordings/stitches/reels/l9_ganon_fixture_endcard/REEL.json`.

## One-page missing-seam list

1. Continuous Level 1 exit through Levels 2, 3, and 4 to `Level4Complete`.
2. East Key Pols Voice (`0x77`) through the natural Recorder acquisition to
   Whistle basement (`0x04`).
3. Level 5 exit and all of Level 6, including its Triforce.
4. All of Level 7 and Level 8; specifically, no live `Level8Complete` pin.
5. Real post-Level-8 overworld entry and Level 9 interior to the
   blade-trap/Like-Like room (`0x41`).
6. Only after item 4 exists may the fixture-proven east-bomb → Patra stairs →
   Ganon/credits suffix be attached. Until then, the honest route stops in the
   Level 5 Triforce room (`0x14`).

Recordings remain gitignored. Do not commit controller pads or present this
compose as Clean.
