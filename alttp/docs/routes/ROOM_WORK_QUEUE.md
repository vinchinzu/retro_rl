# ALTTP — Room / Save-State Work Queue

Sanctuary-path practice queue for `Zelda3-Snes` save states.
Continuous tip is **main hall room 0x61** (after `pocket_to_main_hall`).
Ranked for next work: **main hall / B1 → Zelda cell → follower → escort**.
Internal 0x55 key/shutter path is **alternate practice**, not primary.

Generated: `2026-07-31T20:40:21.295645+00:00`  
Catalog: `alttp_sanctuary_work_queue` schema 1  
States: **61**  
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
| 1 | `CastleMain` | main | main_hall_to_zelda | probe_state | blocker | Continuous tip: main hall room 0x61 after pocket_to_main_hall. Primary next w... |
| 2 | `CastleMainZeldaReady` | zelda | reach_zelda_cell | probe_state | blocker | Primary continuous-tip probe: main-hall Zelda approach from room 0x61. Rank a... |
| 3 | `CastleMainEast` | main | main_hall_to_zelda | probe_state | standard | Main-hall east nav from continuous tip room 0x61. |
| 4 | `CastleMainZeldaBoomerang` | zelda | reach_zelda_cell | probe_state | standard | Main-hall Zelda path with boomerang loadout; continuous-tip adjacent probe fr... |
| 5 | `CastleZeldaFollower` | zelda | zelda_follower | probe_state | standard | Expected $F3CC==1 (Zelda tagalong). Not verified on natural path. Predecessor... |
| 6 | `CastleRoom51Zelda` | zelda | reach_zelda_cell | unstarted | standard |  |
| 7 | `CastleRoom52ZeldaBoomerang` | zelda | reach_zelda_cell | unstarted | standard |  |
| 8 | `CastleZeldaB1East` | zelda | reach_zelda_cell | unstarted | standard |  |
| 9 | `CastleZeldaB1Island` | zelda | reach_zelda_cell | unstarted | standard |  |
| 10 | `CastleZeldaB1Pit` | zelda | reach_zelda_cell | unstarted | standard |  |
| 11 | `CastleZeldaB1West` | zelda | reach_zelda_cell | unstarted | standard |  |
| 12 | `CastleB1BridgeCleared` | b1 | traverse_b1 | probe_state | standard | Named progress save; treat as probe until scripted. |

## Status summary

```
byStatus: {'natural_chain': 2, 'probe_state': 25, 'segment_scripted': 3, 'unstarted': 31}
byGroup:  {'b1': 20, 'b2': 2, 'b3': 5, 'escort': 1, 'key_shutter': 5, 'main': 2, 'opening': 4, 'post_sword': 2, 'room': 9, 'room_55': 1, 'unknown': 1, 'zelda': 9}
byTier:   {'blocker': 2, 'easy': 4, 'later': 9, 'standard': 46}
```

Verified milestones (docs): title_to_castle_grounds, castle_to_fighter_sword, secret_entrance_clear, pocket_to_main_hall_0x61

## Full ranked table

| Rank | State | Group | Goal | Status | Tier | Predecessor |
|-----:|-------|-------|------|--------|------|-------------|
| 1 | `CastleMain` | main | main_hall_to_zelda | probe_state | blocker | `FighterSword` |
| 2 | `CastleMainZeldaReady` | zelda | reach_zelda_cell | probe_state | blocker | `CastleMain` |
| 3 | `CastleMainEast` | main | main_hall_to_zelda | probe_state | standard | `CastleMain` |
| 4 | `CastleMainZeldaBoomerang` | zelda | reach_zelda_cell | probe_state | standard | `CastleMain` |
| 5 | `CastleZeldaFollower` | zelda | zelda_follower | probe_state | standard | `CastleMain` |
| 6 | `CastleRoom51Zelda` | zelda | reach_zelda_cell | unstarted | standard | `CastleMain` |
| 7 | `CastleRoom52ZeldaBoomerang` | zelda | reach_zelda_cell | unstarted | standard | `CastleMain` |
| 8 | `CastleZeldaB1East` | zelda | reach_zelda_cell | unstarted | standard | `CastleMain` |
| 9 | `CastleZeldaB1Island` | zelda | reach_zelda_cell | unstarted | standard | `CastleMain` |
| 10 | `CastleZeldaB1Pit` | zelda | reach_zelda_cell | unstarted | standard | `CastleMain` |
| 11 | `CastleZeldaB1West` | zelda | reach_zelda_cell | unstarted | standard | `CastleMain` |
| 12 | `CastleB1BridgeCleared` | b1 | traverse_b1 | probe_state | standard | `CastleMain` |
| 13 | `CastleB1GreenRoomCleared` | b1 | traverse_b1 | probe_state | standard | `CastleMain` |
| 14 | `CastleB1GreenRoomDone` | b1 | traverse_b1 | probe_state | standard | `CastleMain` |
| 15 | `CastleB1IslandCleared` | b1 | traverse_b1 | probe_state | standard | `CastleMain` |
| 16 | `CastleB1PitCleared` | b1 | traverse_b1 | probe_state | standard | `CastleMain` |
| 17 | `CastleB1PitFull` | b1 | traverse_b1 | probe_state | standard | `CastleMain` |
| 18 | `CastleB1PitGuardCleared` | b1 | traverse_b1 | probe_state | standard | `CastleMain` |
| 19 | `CastleB1SingleGreenCleared` | b1 | traverse_b1 | probe_state | standard | `CastleMain` |
| 20 | `CastleB1UpperCleared` | b1 | traverse_b1 | probe_state | standard | `CastleMain` |
| 21 | `CastleB1Bridge` | b1 | traverse_b1 | unstarted | standard | `CastleMain` |
| 22 | `CastleB1FarDoor` | b1 | traverse_b1 | unstarted | standard | `CastleMain` |
| 23 | `CastleB1FarWest` | b1 | traverse_b1 | unstarted | standard | `CastleMain` |
| 24 | `CastleB1GreenRoom` | b1 | traverse_b1 | unstarted | standard | `CastleMain` |
| 25 | `CastleB1Guard` | b1 | traverse_b1 | unstarted | standard | `CastleMain` |
| 26 | `CastleB1GuardLamp` | b1 | traverse_b1 | unstarted | standard | `CastleMain` |
| 27 | `CastleB1Pit` | b1 | traverse_b1 | unstarted | standard | `CastleMain` |
| 28 | `CastleB1SingleGreen` | b1 | traverse_b1 | unstarted | standard | `CastleMain` |
| 29 | `CastleB1South` | b1 | traverse_b1 | unstarted | standard | `CastleMain` |
| 30 | `CastleB1West` | b1 | traverse_b1 | unstarted | standard | `CastleMain` |
| 31 | `CastleB1WestRoom` | b1 | traverse_b1 | unstarted | standard | `CastleMain` |
| 32 | `CastleB1Key` | key_shutter | obtain_key | probe_state | standard | `FighterSword` |
| 33 | `CastleB1Shutter` | key_shutter | open_shutter | probe_state | standard | `CastleB1Key` |
| 34 | `CastleB1SecondKey` | key_shutter | obtain_key | probe_state | later | `CastleB1Key` |
| 35 | `CastleB1ShutterGuard` | key_shutter | open_shutter | probe_state | later | `CastleB1Shutter` |
| 36 | `CastleB1ShutterRoom` | key_shutter | open_shutter | probe_state | later | `CastleB1Shutter` |
| 37 | `FighterSwordLamp` | post_sword | secret_entrance_clear | probe_state | standard | `FighterSword` |
| 38 | `FighterSword` | post_sword | secret_entrance_clear | segment_scripted | standard | `HyruleCastleGrounds` |
| 39 | `Castle_55` | room_55 | exit_0x55 | segment_scripted | standard | `HyruleCastleGrounds` |
| 40 | `CastleMantleZelda` | escort | sanctuary | unstarted | later | `CastleZeldaFollower` |
| 41 | `CastleRoom51Cleared` | room | clear_room_0x51 | probe_state | standard |  |
| 42 | `CastleRoom62Cleared` | room | clear_room_0x62 | probe_state | standard |  |
| 43 | `CastleRoom01` | room | clear_room_0x01 | unstarted | standard |  |
| 44 | `CastleRoom50` | room | clear_room_0x50 | unstarted | standard |  |
| 45 | `CastleRoom51` | room | clear_room_0x51 | unstarted | standard |  |
| 46 | `CastleRoom52` | room | clear_room_0x52 | unstarted | standard |  |
| 47 | `CastleRoom60` | room | clear_room_0x60 | unstarted | standard |  |
| 48 | `CastleRoom62` | room | clear_room_0x62 | unstarted | standard |  |
| 49 | `CastleRoom62North` | room | clear_room_0x62 | unstarted | standard |  |
| 50 | `CastleB2Landing` | b2 | traverse_b2 | probe_state | standard | `CastleB1South` |
| 51 | `CastleB2` | b2 | traverse_b2 | unstarted | standard | `CastleB1South` |
| 52 | `CastleB3GuardCleared` | b3 | ball_and_chain | probe_state | later | `CastleB1South` |
| 53 | `CastleB3` | b3 | ball_and_chain | unstarted | later | `CastleB1South` |
| 54 | `CastleB3BallApproach` | b3 | ball_and_chain | unstarted | later | `CastleB1South` |
| 55 | `CastleB3Boomerang` | b3 | ball_and_chain | unstarted | later | `CastleB1South` |
| 56 | `CastleB3BossOneHitBoomerang` | b3 | ball_and_chain | unstarted | later | `CastleB1South` |
| 57 | `FirstAction` | opening | opening_progress | probe_state | easy | `LinksHouseWake` |
| 58 | `CourtyardSecretPocket` | unknown | probe | unstarted | standard |  |
| 59 | `LinksHouseWake` | opening | exit_links_house | segment_scripted | easy | `YazeSlot000` |
| 60 | `HyruleCastleGrounds` | opening | reach_secret_hole | natural_chain | easy | `LinksHouseWake` |
| 61 | `YazeSlot000` | opening | boot_title | natural_chain | easy |  |

## Notes

- Continuous tip is **main hall room 0x61** after `pocket_to_main_hall` (courtyard pocket → main door).
- Secret-entrance clear (stairs → outdoor pocket) is already continuous; do **not** treat `Castle_55` internal exit as the top blocker.
- Primary next work: main hall / B1 → Zelda cell → follower → escort → Sanctuary.
- Internal 0x55 key/shutter path is **alternate practice** only.
- `FighterSword` is a **dev checkpoint** after uncle sword; natural sword claim needs `--natural` on `castle_to_sword`.
- Acceptance for rescue: `has_zelda_follower` (`$F3CC == 1`).
- Sanctuary: room `0x12` / OW screen `0x13` — not claimed.

Units are Zelda3-Snes save states on the boot → fighter sword → secret-entrance clear → courtyard pocket → main hall room 0x61 → Zelda → Sanctuary path. Continuous tip is **main hall 0x61**; next work is B1 → Zelda cell → follower → escort. Internal 0x55 key/shutter is alternate practice only. Sanctuary not claimed.
