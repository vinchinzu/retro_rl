# ALTTP — Room / Save-State Work Queue

Sanctuary-path practice queue for `Zelda3-Snes` save states.
Ranked for progress after fighter sword: **exit room 0x55 / key / shutter** before random B1; escort and Sanctuary later.

Generated: `2026-07-30T22:40:10.433833+00:00`  
Catalog: `alttp_sanctuary_work_queue` schema 1  
States: **60**  
Sanctuary claimed: **False**

## Regenerate

```bash
uv run python alttp/scripts/export_work_queue.py
uv run python alttp/scripts/export_work_queue.py --json
```

Artifacts: `docs/routes/ROOM_WORK_QUEUE.md` · `recordings/room_work_queue.json`.

## Work focus (next toward Sanctuary)

| Rank | State | Group | Goal | Status | Tier | Notes |
|-----:|-------|-------|------|--------|------|-------|
| 1 | `Castle_55` | room_55 | exit_0x55 | blocker | blocker | Secret passage room 0x55. Uncle/south chamber partial: south chamber ~(2680,2... |
| 2 | `FighterSwordLamp` | post_sword | exit_0x55 | probe_state | standard | Sword + lamp checkpoint; still room-0x55 escape incomplete. |
| 4 | `CastleB1Key` | key_shutter | obtain_key | probe_state | blocker | Priority after sword: small key for 0x55 / shutter path. |
| 5 | `CastleB1SecondKey` | key_shutter | obtain_key | probe_state | blocker | Second key probe on B1 escape ladder. |
| 6 | `CastleB1Shutter` | key_shutter | open_shutter | probe_state | blocker | Shutter door path out of / through 0x55-adjacent B1. |
| 7 | `CastleB1ShutterGuard` | key_shutter | open_shutter | probe_state | blocker | Shutter room with guard; critical-path probe. |
| 8 | `CastleB1ShutterRoom` | key_shutter | open_shutter | probe_state | blocker | Shutter room probe state. |
| 9 | `CastleMainZeldaBoomerang` | zelda | reach_zelda_cell | probe_state | standard | Zelda path with boomerang loadout. |
| 10 | `CastleMainZeldaReady` | zelda | reach_zelda_cell | probe_state | standard | Main-hall Zelda approach probe. |
| 11 | `CastleZeldaFollower` | zelda | zelda_follower | probe_state | standard | Expected $F3CC==1 (Zelda tagalong). Not verified on natural path. |
| 12 | `CastleRoom51Zelda` | zelda | reach_zelda_cell | unstarted | standard |  |
| 13 | `CastleRoom52ZeldaBoomerang` | zelda | reach_zelda_cell | unstarted | standard |  |

## Status summary

```
byStatus: {'blocker': 1, 'natural_chain': 2, 'probe_state': 23, 'segment_scripted': 2, 'unstarted': 32}
byGroup:  {'b1': 20, 'b2': 2, 'b3': 5, 'escort': 1, 'key_shutter': 5, 'main': 2, 'opening': 4, 'post_sword': 2, 'room': 9, 'room_55': 1, 'zelda': 9}
byTier:   {'blocker': 6, 'easy': 4, 'later': 6, 'standard': 44}
```

Verified milestones (docs): title_to_castle_grounds, castle_to_fighter_sword, room_0x55_south_chamber_partial

## Full ranked table

| Rank | State | Group | Goal | Status | Tier | Predecessor |
|-----:|-------|-------|------|--------|------|-------------|
| 1 | `Castle_55` | room_55 | exit_0x55 | blocker | blocker | `HyruleCastleGrounds` |
| 2 | `FighterSwordLamp` | post_sword | exit_0x55 | probe_state | standard | `FighterSword` |
| 3 | `FighterSword` | post_sword | exit_0x55 | segment_scripted | standard | `HyruleCastleGrounds` |
| 4 | `CastleB1Key` | key_shutter | obtain_key | probe_state | blocker | `FighterSword` |
| 5 | `CastleB1SecondKey` | key_shutter | obtain_key | probe_state | blocker | `CastleB1Key` |
| 6 | `CastleB1Shutter` | key_shutter | open_shutter | probe_state | blocker | `CastleB1Key` |
| 7 | `CastleB1ShutterGuard` | key_shutter | open_shutter | probe_state | blocker | `CastleB1Shutter` |
| 8 | `CastleB1ShutterRoom` | key_shutter | open_shutter | probe_state | blocker | `CastleB1Shutter` |
| 9 | `CastleMainZeldaBoomerang` | zelda | reach_zelda_cell | probe_state | standard | `FighterSword` |
| 10 | `CastleMainZeldaReady` | zelda | reach_zelda_cell | probe_state | standard | `FighterSword` |
| 11 | `CastleZeldaFollower` | zelda | zelda_follower | probe_state | standard | `FighterSword` |
| 12 | `CastleRoom51Zelda` | zelda | reach_zelda_cell | unstarted | standard | `FighterSword` |
| 13 | `CastleRoom52ZeldaBoomerang` | zelda | reach_zelda_cell | unstarted | standard | `FighterSword` |
| 14 | `CastleZeldaB1East` | zelda | reach_zelda_cell | unstarted | standard | `FighterSword` |
| 15 | `CastleZeldaB1Island` | zelda | reach_zelda_cell | unstarted | standard | `FighterSword` |
| 16 | `CastleZeldaB1Pit` | zelda | reach_zelda_cell | unstarted | standard | `FighterSword` |
| 17 | `CastleZeldaB1West` | zelda | reach_zelda_cell | unstarted | standard | `FighterSword` |
| 18 | `CastleRoom51Cleared` | room | clear_room_0x51 | probe_state | standard |  |
| 19 | `CastleRoom62Cleared` | room | clear_room_0x62 | probe_state | standard |  |
| 20 | `CastleMain` | main | castle_main_nav | unstarted | standard |  |
| 21 | `CastleMainEast` | main | castle_main_nav | unstarted | standard |  |
| 22 | `CastleRoom01` | room | clear_room_0x01 | unstarted | standard |  |
| 23 | `CastleRoom50` | room | clear_room_0x50 | unstarted | standard |  |
| 24 | `CastleRoom51` | room | clear_room_0x51 | unstarted | standard |  |
| 25 | `CastleRoom52` | room | clear_room_0x52 | unstarted | standard |  |
| 26 | `CastleRoom60` | room | clear_room_0x60 | unstarted | standard |  |
| 27 | `CastleRoom62` | room | clear_room_0x62 | unstarted | standard |  |
| 28 | `CastleRoom62North` | room | clear_room_0x62 | unstarted | standard |  |
| 29 | `CastleB1BridgeCleared` | b1 | traverse_b1 | probe_state | standard | `FighterSword` |
| 30 | `CastleB1GreenRoomCleared` | b1 | traverse_b1 | probe_state | standard | `FighterSword` |
| 31 | `CastleB1GreenRoomDone` | b1 | traverse_b1 | probe_state | standard | `FighterSword` |
| 32 | `CastleB1IslandCleared` | b1 | traverse_b1 | probe_state | standard | `FighterSword` |
| 33 | `CastleB1PitCleared` | b1 | traverse_b1 | probe_state | standard | `FighterSword` |
| 34 | `CastleB1PitFull` | b1 | traverse_b1 | probe_state | standard | `FighterSword` |
| 35 | `CastleB1PitGuardCleared` | b1 | traverse_b1 | probe_state | standard | `FighterSword` |
| 36 | `CastleB1SingleGreenCleared` | b1 | traverse_b1 | probe_state | standard | `FighterSword` |
| 37 | `CastleB1UpperCleared` | b1 | traverse_b1 | probe_state | standard | `FighterSword` |
| 38 | `CastleB1Bridge` | b1 | traverse_b1 | unstarted | standard | `FighterSword` |
| 39 | `CastleB1FarDoor` | b1 | traverse_b1 | unstarted | standard | `FighterSword` |
| 40 | `CastleB1FarWest` | b1 | traverse_b1 | unstarted | standard | `FighterSword` |
| 41 | `CastleB1GreenRoom` | b1 | traverse_b1 | unstarted | standard | `FighterSword` |
| 42 | `CastleB1Guard` | b1 | traverse_b1 | unstarted | standard | `FighterSword` |
| 43 | `CastleB1GuardLamp` | b1 | traverse_b1 | unstarted | standard | `FighterSword` |
| 44 | `CastleB1Pit` | b1 | traverse_b1 | unstarted | standard | `FighterSword` |
| 45 | `CastleB1SingleGreen` | b1 | traverse_b1 | unstarted | standard | `FighterSword` |
| 46 | `CastleB1South` | b1 | traverse_b1 | unstarted | standard | `FighterSword` |
| 47 | `CastleB1West` | b1 | traverse_b1 | unstarted | standard | `FighterSword` |
| 48 | `CastleB1WestRoom` | b1 | traverse_b1 | unstarted | standard | `FighterSword` |
| 49 | `CastleB2Landing` | b2 | traverse_b2 | probe_state | standard | `CastleB1South` |
| 50 | `CastleB2` | b2 | traverse_b2 | unstarted | standard | `CastleB1South` |
| 51 | `CastleB3GuardCleared` | b3 | ball_and_chain | probe_state | later | `CastleB1South` |
| 52 | `CastleB3` | b3 | ball_and_chain | unstarted | later | `CastleB1South` |
| 53 | `CastleB3BallApproach` | b3 | ball_and_chain | unstarted | later | `CastleB1South` |
| 54 | `CastleB3Boomerang` | b3 | ball_and_chain | unstarted | later | `CastleB1South` |
| 55 | `CastleB3BossOneHitBoomerang` | b3 | ball_and_chain | unstarted | later | `CastleB1South` |
| 56 | `CastleMantleZelda` | escort | sanctuary | unstarted | later | `CastleZeldaFollower` |
| 57 | `FirstAction` | opening | opening_progress | probe_state | easy | `LinksHouseWake` |
| 58 | `LinksHouseWake` | opening | exit_links_house | segment_scripted | easy | `YazeSlot000` |
| 59 | `HyruleCastleGrounds` | opening | reach_secret_hole | natural_chain | easy | `LinksHouseWake` |
| 60 | `YazeSlot000` | opening | boot_title | natural_chain | easy |  |

## Notes

- `FighterSword` is a **dev checkpoint** after uncle sword; natural sword claim needs `--natural` on `castle_to_sword`.
- Room `0x55` south chamber is **partial**; no measured exit toward Zelda cell yet (`blocker`).
- Acceptance for rescue: `has_zelda_follower` (`$F3CC == 1`).
- Sanctuary: room `0x12` / OW screen `0x13` — not claimed.
- Random B1 cleared/island states are lower priority than key/shutter.

Units are Zelda3-Snes save states on the boot → fighter sword → room 0x55 → Zelda → Sanctuary path. Ranked for Sanctuary progress: after sword, prioritize exit_0x55 / key / shutter over random B1; escort/Sanctuary later. Sanctuary not claimed.
