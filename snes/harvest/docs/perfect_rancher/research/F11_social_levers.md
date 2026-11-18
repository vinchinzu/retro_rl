# F11 social levers: Star Night, the nightly wife drain, and what the walkthroughs say

Date: 2026-11-03. Static decode of the USA ROM (`event_dis group 15`, `bank_83.asm` `$83E144`/`$83E210`) plus two ROM
measurements on `s4` pins. Web walkthroughs (fogu.com, almarsguides.com, GameFAQs-style fan wikis) cover the ten power berry
sources only; they say nothing about heart mechanics, festival choices or glitches, so every lever below comes from the ROM.

## 1. Star Night (Winter D24) is a menu, and its default is the wrong answer

The forced festival (the farmer is warped to the festival map at 06:03) runs group `0x0F` event 0 (`$B48020`). Text `0x02B6`:
"It's the Star Night Festival today! Where should I go?" with five options laid out in two columns: Public square, Mountain top /
Church, Spa / Bar. Choice index *n* (`JCHOICE` 0-4) sets bit *n* of `$7F1F74` and transits to a map; the matching scene
(events 1-5) then plays and rewards one girl:

| choice index | transit map (`TRANSIT`) | scene | girl who gets +30 hearts | wife who takes -30 (skipped for the girl herself) |
| ---: | --- | --- | --- | --- |
| 0 | `0x0A` (the Star Night square) | event 1 (`$B48077`) | Ellen `$7F1F25` (`$B480B4`) | Maria, Ann, Nina, Eve |
| 1 | `0x1B` (the church, where Maria stands: F9) | event 2 (`$B48210`) | Maria `$7F1F1F` (`$B48259`) | Ann, Nina, Ellen, Eve |
| 2 | `0x1E` | event 3 (`$B48402`) | Eve `$7F1F27` (`$B48446`) | Maria, Ann, Nina, Ellen |
| 3 | `0x39` | event 4 (penalty chain at `$B48634`) | Nina (inferred: the chain skips wife bit 2) | Maria, Ann, Ellen, Eve |
| 4 | `0x14` | event 5 (penalty chain at `$B48718`) | Ann (inferred: the chain skips wife bit 1) | Maria, Nina, Ellen, Eve |

Which on-screen option is which index is **not** settled. The transit maps say index 1 is the church, so the cursor does not
walk the text row by row; column by column (Public square, Church, Bar down the left, Mountain top, Spa on the right) would put
index 2 on the Bar, which fits Eve working there, but only index 0 (square) and index 1 (church) are backed by a map id.

The +30 lands only when the scene's conversations are all done (`op0C 0..n` then `ADD16`); the wife's -30 is in a child script
(`cc[6]`) spawned at the scene's start, so it lands at once. The wife bit is `$7F1F66` bit 0 Maria, 1 Ann, 2 Nina, 3 Ellen, 4 Eve.
Events 4 and 5 stop the disassembler early (opcode `>= 0x51`), so their +30 target is inferred, not read.

The bot dismisses dialogs with default presses, so the cursor stays on Public square: every Star Night is **+30 Ellen** and
**-30 for the wife**. `s4` is exactly that: Eve 410 (Y2 Winter D5) to 376 (Y3 Spring D10) while Ellen went 38 to 68. Picking the
wife's scene (index 2 for Eve) turns -30 into +30, a 60-heart swing a year (~4 points of the 31-point window), and in an unmarried
year it puts the +30 on the girl you will marry.

**Measured 2026-11-08** on `PR_e2ec1_Y1_Winter_D24` (Eve the wife, 430): the menu is not live until one A turns the first
page (~300 frames after text `0x02B6` appears, then ~200 more); Down/Up then step `$018F` (`dialog_menu_cursor`) 0..4 and
Left/Right do nothing, so the index is driven by the RAM cursor, not the screen layout. The morning settle presses A+B every
frame, which is why every run so far picked index 0. Index 0: Ellen +30, Eve -30. Index 2: transit to the bar `0x1E` at 18:03,
Eve keeps 430. Eve's scene (event 3) waits for six villagers (signals 0,1,2,3,5,6), then Eve's line adds +30 and ends the scene;
the farmer wakes Winter D25 with Eve 458 (+30, then the hour-18 house entry -2), against 398 on the default path: **+60 for the
wife each Star Night**. Wired: `festival_escape.star_night_menu_press` (called first in `MultiDayPlannerTask._step_inner`) and
`StarNightSceneTask` (a fixed corridor path: seated actors seal the room BFS; Eve wanders, so her stop follows her sprite).
Only Eve's scene has stands; another wife keeps the default. Gated with the church calendar (`HARVEST_SOCIAL_CHURCH`).

## 2. Every house entry at hour 18 takes 2 hearts from a married wife (hearts >= 200)

`CODE_83DED3` loads the wife's group `0x43` script on any house entry before Y3 Summer D30 (`CODE_83E144`: the wife bit picks the
girl, hearts `>= 0xC8` goes to `CODE_83E210`, which loads the script unless `$7F1F6E & 0x0002`). Its first test is
`hour == 0x12` and the branch is a signed `-2` (`$B5805A` Maria, `$B58514` Ann, `$B589CA` Nina, `$B58E88` Ellen, `$B5934D` Eve).

Measured 2026-11-03 on `PR_e2es4_Y3_Spring_D10` (Eve married, 376 hearts): exit to the farm, set the clock to 17:05 or 18:05,
walk home, idle 600 frames. Entry at hour 17: Eve 376 to 376. Entry at hour 18: Eve 376 to 374. Ellen unchanged both times.

The bed refuses A before 18:00 (`home_sleep.earliest_sleep_hour` notes "ROM rejects bed A until evening") and the clock is
frozen indoors, so a farmer who sleeps must walk in at 18:xx: the drain is one fixed -2 per married night. Over `s4`'s Y2 Fall D8
to Y3 Summer D29 that is about 250 nights, ~500 hearts. `$7F1F6E & 0x0002` is a transient pregnancy-scene flag set at 06:00:00
(`$83E1F5` `ORA #$0003`), not a switch. A wife under 200 hearts is exempt, which is no use here.

What it means for the plan: the wife needs a net positive every day. `eve_gift.py` assumes it ("an egg is +4 and every hour-18
entry -2") but only runs in the remodeled house (0x16) with an egg laid; Eve slid ~96 hearts over `s4`'s last 110 days. Milk and
the other shipped goods (items `0x10`-`0x17`) are +4 to +5 for Eve per the F9 ladder and are not used. The final night still
costs -2: enter the last night with 498-511 (F0).

## 3. Things checked and found not to help

- Skipping Star Night: it is a forced festival (`calendar_schedule.FORCED_FESTIVAL_KEYS`).
- Avoiding the nightly -2 by entering at hour 17 and waiting inside: indoor time is frozen, the bed needs hour 18.
- Day 30 gives Maria, Ann, Nina and Ellen +5 twice (F9 site scan); the triggers were not decoded here, so it is not scheduled.
- Web walkthroughs: confirm the ten power berry sources only (see F6); no heart, festival-choice or glitch content.

## 4. The other four girls: where a talk pays (decoded 2026-11-08, measured where marked)

Town scripts are chosen by weekday at load (`CODE_83D46D`: group 1 weekday, 2 Saturday, 3 Sunday; none in rain `0x02`, snow
`0x08`, `0x200`, on a hurricane eve or with `$7F1F64 & 0x20`); the church by `CODE_83D689` (clear Sunday group 3 event 1,
Saturday 2/2, weekday 1/2, rain 4/1, snow 7/1, storm eve 6/1). A question pays once per map load: after it the actor repeats
a done line until the map reloads. A line that pays nothing returns to the main loop, so an RNG question re-rolls on the next
talk. Gifts always return to the main loop and repeat (one held item each). Choice 0 is always the larger answer.

| girl | venue | when | pays | notes |
| --- | --- | --- | --- | --- |
| Maria, Ann | Sunday church `0x1B` | clear Sunday, any season, door shut from 17:00 | +4 each per church entry (Maria `0x008E`, Ann `0x00A2`) | **measured**: 20 entries 8 -> 88 each, 42,499 frames, clock 07:11 -> 07:13. `ChurchCourtTask` |
| Ellen | Sunday church | same | talk 0; gifts +5 (crops, forage), +8 (eggs, milk `0x14`-`0x17`) | |
| Ellen | weekday town (648,904) | Spring `0x00CB`, Fall `0x00C7`, RNG 1/3 | +8 per town load | **measured** Fall D4: 6 loops 8 -> 56, 12,517 frames house to farm, the animal shop beside her reloads the town. `TownCourtTask` |
| Ellen | Saturday town (536,792) | every season | +8 per load (`0x00D2`, no RNG); 150-200: +10 (`0x00D9`); 120-149 nothing | not wired: no open door near her on Saturday |
| Maria | weekday town (312,280) | Spring `0x0086`, Fall `0x008A`, RNG | +8 per load | not wired: the church covers her |
| Nina | weekday town (520,264) | Summer `0x00AF`, Winter `0x00B3`, RNG | +8 per load | wired (flower shop reload), not yet run on the ROM |
| Nina | Sunday mountain (40,200) | Winter `0x00BC` +4; 150-200 `0x00C2` +10; 120-149 nothing | per load | not wired |
| Ann | weekday general store `0x22` | Summer `0x0097`, Winter `0x009B`, RNG | +8 per load | not wired: the church covers her |

**The 200-heart event** (`CODE_83CC72`, 06:00:00 on days 2-7, 13-17, 25-28, clear weather): only while the farmer is married
to nobody, for the first unblocked girl at 200+ in the order Maria, Ann, Nina, Ellen, Eve (each branch skips while an earlier
girl's flag is set; in `f4all` Maria's 0x1000 stepped to 0x2000 two days later and Ann's then fired too). It loads groups `0x19`-`0x1D` and sets `$7F1F6A` bit 12 (Maria), bit 14 (Ann), `$7F1F6C` bit 0 (Nina), bit
2 (Ellen), bit 4 (Eve). The church script skips Maria with bit 12 and gives Ann a non-paying line with bit 14, so courting a
girl past 199 before the wedding would stop her church income. `planner/social_calendar.PRE_WEDDING_CAP` is 199.
Measured in `f4all` (an unmarried Sunday took Maria and Ann to 200): `$7F1F6A` went 0x1000 (D14), 0x2000 (D16), 0x6000 (D17),
0xA000 (D24); Ann's church line paid nothing at 0x6000 and paid again at 0xA000, and Maria kept paying. So the flags are a
short event sequence, not a permanent block, but the cap stays: it cost a Sunday and the scenes' effects are not decoded.
Bits stayed 0 through Year 3 on the `t1` power-on pins.
