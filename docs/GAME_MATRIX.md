# Game Matrix

Generated from `docs/manifests/*.yaml`. Do not hand-edit the tables;
edit the manifests and run:

```bash
uv run python docs/generate_game_matrix.py
```

Games are **parallel genre capability tracks**, not a single ranked
ladder. Maturity uses M0–M8; runtime and intervention classes are
independent. See [ROADMAP.md](ROADMAP.md),
[DEVELOPMENT_LADDER.md](DEVELOPMENT_LADDER.md), and
[BENCHMARK_SPEC.md](BENCHMARK_SPEC.md).

Manifest count: **38**.

## Active and scaffolded workspaces

| Game | Genre track | Phase | State | Maturity | Runtime | Intervention | Full run | Blocker |
| ---- | ----------- | ----- | ----- | -------- | ------- | ------------ | -------- | ------- |
| `snes/great_waldo_search` / The Great Waldo Search | pipeline_and_menus, cursor_and_peripheral | P0 | verified | M8 | bronze | clean | yes | none — treat as completed pipeline fixture; keep capture current |
| `snes/hals_golf` / Hal's Hole in One Golf | pipeline_and_menus, continuous_vehicle_control | P0 | instrumented | M2 | bronze | clean | no | publish evaluation contract and first hole clear |
| `snes/mortal_kombat` / Mortal Kombat | fighting_game_policies | P0 | continuous-candidate | M3 | bronze | clean | no | elevate match wins into reset-to-ending arcade clear claims when desired |
| `snes/mortal_kombat_ii` / Mortal Kombat II | fighting_game_policies | P0 | scaffolded | M1 | bronze | clean | no | normalize match benchmark claims under maturity gates |
| `snes/street_fighter_ii` / Street Fighter II | fighting_game_policies | P0 | continuous-candidate | M3 | bronze | clean | no | arcade-mode continuous clear not yet the primary claim |
| `snes/super_street_fighter_ii` / Super Street Fighter II | fighting_game_policies | P0 | scaffolded | M1 | bronze | clean | no | normalize match benchmark claims under maturity gates |
| `nes/contra` / Contra | linear_combat | P1 | boot_verified | M1 | bronze | clean | no | first Stage 1 segment clear |
| `snes/final_fight` / Final Fight | linear_combat | P1 | segmenting | M3 | bronze | clean | no | Stage 3 West Side Area1 HP250 thug → Boss3; natural-entry hardening |
| `nes/punch_out` / Mike Tyson's Punch-Out!! | fighting_game_policies | P1 | instrumented | M3 | bronze | clean | no | Natural-entry Glass Joe win from power-on / Level1 (M4) |
| `snes/rival_turf` / Rival Turf! | linear_combat | P1 | instrumented | M2 | bronze | clean | no | clear opening Stage 1 combat lock |
| `snes/super_double_dragon` / Super Double Dragon | linear_combat | P1 | segmenting | M3 | bronze | clean | no | natural M3 gym stairs → Chin bosses |
| `nes/tmnt_i` / Teenage Mutant Ninja Turtles | linear_combat | P1 | boot_verified | M1 | bronze | clean | no | first Area 1 building/segment clear |
| `nes/tmnt_ii` / Teenage Mutant Ninja Turtles II: The Arcade Game | linear_combat | P1 | segment_clear | M3 | bronze | clean | no | extend past first wave (score≥5); natural-entry M4 |
| `nes/tmnt_iii` / Teenage Mutant Ninja Turtles III: The Manhattan Project | linear_combat | P1 | boot_verified | M1 | bronze | clean | no | first Stage 1 segment clear |
| `snes/tmnt_iv` / Teenage Mutant Ninja Turtles IV: Turtles in Time | linear_combat | P1 | verified | M8 | bronze | resource_assisted+protection_assisted | yes | Whole-run Bronze/Clean dry-run (ticket T4-CLEAN-FULL; infra + S2–S9 suites first) |
| `snes/battle_clash` / Battle Clash | cursor_and_peripheral | P2 | blocked | M1 | bronze | clean | no | infrastructure — Super Scope / light-gun injection unsupported |
| `snes/f_zero` / F-Zero | continuous_vehicle_control | P2 | instrumented | M2 | bronze | clean | no | one Mute City lap without crash |
| `snes/pilotwings` / Pilotwings | continuous_vehicle_control | P2 | instrumented | M2 | bronze | clean | no | complete light-plane Lesson 1 objective |
| `snes/star_fox` / Star Fox | continuous_vehicle_control | P2 | segmenting | M3 | bronze | clean | no | destroy Attack Carrier hatches and finish Corneria clear |
| `nes/castlevania` / Castlevania | platforming | P3 | boot_verified | M1 | bronze | clean | no | first Stage 1 segment clear |
| `snes/donkey_kong_country` / Donkey Kong Country | platforming | P3 | scaffolded | M1 | bronze | clean | no | first documented autonomous level/route clear |
| `nes/ducktales` / DuckTales | platforming | P3 | boot_verified | M1 | bronze | clean | no | enter a land stage and clear first segment |
| `snes/joe_and_mac` / Joe & Mac | platforming | P3 | instrumented | M2 | bronze | clean | no | first traversable Stage 1 segment |
| `nes/kirby_adventure` / Kirby's Adventure | platforming | P3 | boot_verified | M1 | bronze | clean | no | first stage/segment clear from Vegetable Valley hub |
| `snes/magical_quest` / The Magical Quest Starring Mickey Mouse | platforming | P3 | instrumented | M2 | bronze | clean | no | clear first room/checkpoint from Stage 1 |
| `nes/mega_man_2` / Mega Man 2 | platforming | P3 | segment_verified | M3 | bronze | clean | no | Air Man mid-stage / natural-entry M4 |
| `nes/smb` / Super Mario Bros. | platforming | P3 | verified_capture | M8 | bronze | clean | yes | — |
| `nes/smb3` / Super Mario Bros. 3 | platforming | P3 | segment_clear | M3 | bronze | clean | no | World 1-2 natural-entry clear |
| `snes/SMW` / Super Mario World | platforming | P3 | instrumented | M2 | bronze | clean | no | normalize route tooling into maturity gates and continuous clear path |
| `snes/alttp` / The Legend of Zelda: A Link to the Past | top_down_navigation, metroidvania_navigation | P4 | active | M1 | bronze | clean | no | opening route only to castle grounds; sword/uncle segments next |
| `nes/metroid` / Metroid | metroidvania_navigation, graph_navigation | P4 | segment_verified | M5 | bronze | clean | no | morph return + (5,14) door for first missiles; then bombs |
| `snes/smz3` / SMZ3 (Super Metroid + ALttP Combined Randomizer) | metroidvania_navigation, top_down_navigation | P4 | scaffolded | M2 | bronze | clean | no | map portal → settled Z3 Link; longer one-bot segment + video; dual-bot race later |
| `snes/super_metroid` / Super Metroid | metroidvania_navigation | P4 | route-building | M5 | bronze | resource_assisted | no | Super → farming → Big Pink main continuous; pure PB sill approach + maze bridges; then path board hops |
| `nes/zelda_i` / The Legend of Zelda | graph_navigation | P4 | segment_verified | M5 | bronze | clean | no | walk 0x4A → Level 2 door 0x3C (overworld health) |
| `nes/zelda_ii` / Zelda II: The Adventure of Link | graph_navigation | P4 | boot_verified | M1 | bronze | clean | no | leave North Palace / first side-scroll segment |
| `snes/harvest` / Harvest Moon | simulation_and_scheduling, tactical_planning | P6 | instrumented | M3 | bronze | clean | no | same-day water + harvest/ship income (money still ~$100 floor); planning stack skill composition in progress |
| `snes/alttp_rando` / A Link to the Past Randomizer | top_down_navigation, metroidvania_navigation | P7 | scaffolded | M1 | bronze | clean | no | M1 FirstPlay (Link's House) on JP 1.0; ./play records; ALTTPR patch + multi-seed next |
| `snes/sm_rando` / Super Metroid Randomizer | metroidvania_navigation | P7 | playable_boot | M1 | bronze | clean | no | M1 FirstPlay boot on vanilla SM (Ceres) done; real shuffled seed ROM / generator still open; prove multi-seed S/T next |

## Planned / external

| Game | Genre track | Phase | State | Maturity | Blocker |
| ---- | ----------- | ----- | ----- | -------- | ------- |

## Scoring fields (in manifests)

Each manifest may also carry `popularity`, `engineering_effort`,
`transfer_value`, `ending_definition`, evidence paths, and
`last_verified`. Those feed [PROGRAM_STATUS.md](PROGRAM_STATUS.md)
and local status docs.
