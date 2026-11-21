# Grapple Skip main route

`grapple_skip_fast` is the main Clean route and builds on `no_crocomire`.
Run `python -m super_metroid.splice
chain --replay --require-credits` to prove it from power-on and `python -m
super_metroid.splice benchmark` for its split report. The original finish
is available at the explicit `recordings/master_tape_clean` stem.
[Split history](SPEED_SPLITS.md)

The `no_crocomire` development policy goes directly from the west side of
Maridia Tube to Main Street, skipping the return to the tube from Crocomire.
It reaches Draygon through Full Halfie with Gravity, Hi-Jump, and Speed
Booster. The inventory is `0x3125` at Draygon's entrance. The Grapple
(`0x4000`) and Space Jump (`0x0200`) bits are both clear.

Samus enters Draygon at row 209 after 210,001 frames with 59 energy,
9 missiles, 1 Super, and 3 Power Bombs. That prefix saves 46,171 frames
(12:49.52 at 60 fps) against `master_tape_clean` through the same door.

The route defeats Draygon at row 210 and collects Space Jump at row 211.
Samus ends with 27 energy, 9 missiles, 1 Super, and 3 Power Bombs. The
full prefix takes 217,671 frames and saves 43,629 frames (12:07.15)
against `master_tape_clean` through row 211.

## Measured crossings

| Baseline rows | Replacement | Baseline frames | Candidate frames |
|---|---|---:|---:|
| 164–199 | Break the tube from the west, then climb to Main Street with one Power Bomb | 35,579 | 1,069 |
| 200 | Main Street tape, retimed at the exit | 2,409 | 2,408 |
| 201 | Mt. Everest, two vertical stutter-charge sparks | 3,859 | 1,264 |
| 202 | Crab Shaft, beam through the floor and one Super at the exit | 706 | 681 |
| 203 | Aqueduct, Power Bomb the floor then spark up to the left walkway | 2,706 | 906 |
| 204 | Botwoon Hallway, stutter charge through the speed blocks | 938 | 698 |
| 205 | Botwoon, searched fight and drop collection for this entry | 3,051 | 3,272 |
| 206–208 | Full Halfie, wall jumps and a ceiling spark across Colosseum | 7,229 | 1,552 |
| 209 | Precious Room, morph bomb shortcut and one Super for the eye door | 2,391 | 847 |
| 210 | Draygon, charged shots and exit to Space Jump Room | 4,334 | 6,876 |
| 211 | Space Jump pickup with the baseline inputs | 794 | 794 |
| **164–211** | | **63,996** | **20,367** |

The [entry skills](../routes/kpdr/maridia/entry.py) use the same vanilla
runway and room transitions as the candidate. The `no_crocomire` policy
uses the searched Botwoon inputs from its own hallway exit. The fight spends
missiles and Supers, then collects enough drops to enter Botwoon Energy Tank
Room at 186 energy with three Supers.

## Full Halfie

[Full Halfie](../routes/kpdr/maridia/halfie_climb.py) carries four Speed
Booster echoes across the Botwoon Energy Tank Room door and stores the
charge inside Halfie Climb near `x=150`. The wall kicks stay close to the
right wall above the spike pit. A Wave Beam shot during the final kick opens
the blue door before Samus lands. Samus enters Colosseum with 20 stored
charge frames, unspins at `y=104`, and sparks horizontally above the
pillars. The spark costs 127 energy and one Super opens Colosseum's green
exit.

The standalone `maridia.halfie_climb_wj` skill is a 416-frame backup from the
baseline r206 checkpoint. It reaches Colosseum with no damage and no charge
carry. The `speed_halfie` profile can run the Full Halfie and Precious Room
skills from the baseline r205 leave in 2,419 frames. That checkpoint carries
Grapple in inventory. The `no_crocomire` route reaches the same rooms without
collecting it.

## Draygon

The `draygon_grappleless_r210` policy rule plays searched inputs from the
Full Halfie entry. Twenty charged Ice/Wave/Spazer hits deal 300 damage
each. Samus takes one 32-energy hit and carries all ammo into row 211.
Draygon reaches zero HP at frame 215,870, and the Maridia boss bit
(`$7E:D82C` bit 0) sets at frame 216,599. The fight and left-door exit
cost 2,542 frames (42.37 seconds) more than the baseline Grapple kill.
The baseline row 211 inputs collect Space Jump, leaving inventory
`0x3325` with the Grapple bit clear.

## Replay

The candidate tape and proof are in `recordings/search_development_clean/`.
The vanilla ROM replay matched every planned crossing and settled leave
through row 211 with zero state loads, RAM writes, or deaths. The independent
audit reproduced the same emulator snapshot through row 211.
The report's `endpoint_sha256` comes from the loadless replay. The
checkpoint-seeded build's serialized endpoint is preserved separately as
`build_endpoint_sha256`.

Run from the repository root.

```bash
QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m super_metroid.splice chain --replay \
  --stem snes/super_metroid/recordings/search_development_clean/no_crocomire_draygon_clean \
  --through-row 211

QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python snes/super_metroid/scripts/tools/audit_master_tape.py \
  --stem search_development_clean/no_crocomire_draygon_clean
```

## Remaining work

1. **Completed locally (rows 212–339 + credits tail):** All post-Space-Jump rooms
   for `no_crocomire` (no Crocomire, no Grapple, no Spring Ball; inventory `0x332D`)
   have been patched and wired into `policies/route_profiles/no_crocomire.json`:
   - Maridia / Norfair / Lower Norfair (`229`, `237–239` skill, `244`, `251`, `254`,
     `265`, `266`, `268`, `271` Ridley with pause-RNG 4/5 PB drop sweep, `278`,
     `279`, `281` Maze +5M pack to 35M, `284`, `301` Red Tower Space Jump in 836f,
     `302` Hellway Space Jump in 478f).
   - Brinstar & Statues (`310` Reserve + Early Supers + 3x Missile packs to
     `50M/15S/100R` without Spring Ball in 10,464f; `312` Item Cancel with 1-frame
     entry idle prepended for 0-delay override execution).
   - Tourian & Finish (`313–315` `tourian.first_room_direct`, `316–319`
     `tourian.metroid_rooms_all` with Metroid 4 hatch x-alignment fix `104..118`,
     `322–323` `tourian.big_boy_room` with 14f spin-takeoff fix, `326` 1-Super
     Seaweed red door yielding `50M/12S = 86 eq` into `0xDD58`, `327` Item Cancel,
     `328–330` `tourian.rinka_shaft_direct`, `331` Clean Mother Brain 1+2 solve
     `no_crocomire/override331.npz` in 6,717f using morph bombs instead of Spring
     Ball, and live `finish_controller_for` escape controllers `332–339` + credits).
   - A full `policy build no_crocomire` ran cleanly through row 310 (`164–310`
     matched in-chain) before hitting the 1-frame delay on `override312.npz`, which
     has now been fixed in-place.
2. **Power-on proof, audit, and benchmark (done 2026-11-19: build proof, audit,
   and `v3_no_crocomire` all passed; `1:39:59.05` Ceres→Ship, 384,844f total).
   Commands kept for reruns:**
   ```bash
   QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
     uv run python -m super_metroid.splice policy build no_crocomire \
     --stem snes/super_metroid/recordings/speed_benchmarks/master_tape_no_crocomire_clean

   QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
     uv run python snes/super_metroid/scripts/tools/audit_master_tape.py \
     --stem speed_benchmarks/master_tape_no_crocomire_clean

   QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
     uv run python -m super_metroid.splice benchmark \
     --stem snes/super_metroid/recordings/speed_benchmarks/master_tape_no_crocomire_clean \
     --scan --record-history --label v3_no_crocomire \
     --markdown-out snes/super_metroid/recordings/speed_benchmarks/v3_no_crocomire.md
   ```
   [STATUS.md](STATUS.md#clean-track) carries the current proof.

## Tourian regression

The `speed_v2_tourian` profile was rebuilt into
`recordings/speed_benchmarks/grapple_skip_work/tourian_v2_check_clean`.
It reproduced the archived 443,259-frame input digest `46cfb95e072c…`
and passed the guarded power-on replay through credits. Metroid Room 4
uses the falling pose to choose its hatch alignment. Big Boy Room adds
the running spin-jump when Samus starts below the central pillar. The
Grapple Skip branch uses the same skills from its own entry.

## Direct Red Tower return

The v4 tape `recordings/speed_benchmarks/master_tape_grapple_skip_red_clean`
replaces rows 299–301. It keeps 2,101 frames of the lower ascent, then uses
the [upper shaft Skill](../routes/kpdr/red_tower/red_stack.py) to reach
Hellway in another 799 frames. The Skill taps Space Jump as Samus begins
falling, shoots up through the ceiling, and takes the upper door. The
change removes the pause and the Noob Bridge round trip, saving 3,090
frames at the crossing. Local A/B also passed through 17 downstream rows.

The Metroid Rooms Skill handled the shifted Tourian entry. At Mother Brain,
the existing Zebetite and glass inputs still fit. The corrected fight
spends the remaining 30 Missiles on MB1, then uses 20 charged shots on MB2.
It reaches the Rainbow kneel with 87 energy in the same 6,717 frames as
v3. The live escape saves another 185 frames on the new entry.

The full replay and independent audit passed through credits. The split
scan recorded `1:39:04.47` Ceres→Ship and cached 290 row checkpoints by
input digest. [V4 splits](SPEED_SPLITS.md#v4-direct-red-tower-return)

```bash
v4=snes/super_metroid/recordings/speed_benchmarks/master_tape_grapple_skip_red_clean
uv run python -m super_metroid.splice chain --replay --require-credits --stem "$v4"
uv run python -m super_metroid.splice benchmark --stem "$v4"
uv run python snes/super_metroid/scripts/tools/audit_master_tape.py \
  --stem speed_benchmarks/master_tape_grapple_skip_red_clean
```
