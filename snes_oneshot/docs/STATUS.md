# Status — top-10 SNES agent ladder

See program notes: [EASIEST_SNES_GAMES.md](EASIEST_SNES_GAMES.md)

**Working rule:** save-state development, segment clears, and retries are
valid early milestones. “Segmented ending” does not mean a continuous run.

| Rank | Game | Proven progress | Continuous full run | Video | Next milestone |
|------|------|-----------------|---------------------|-------|----------------|
| 1 | Great Waldo Search | Scenes 1–5 + five-scroll ending, segmented | no | — | optional title → ending run |
| 2 | Final Fight | Stages 1–2; S3 West Side w1–w5 + Area1 | no | — | Area1 HP250 thug → Boss3 |
| 3 | TMNT IV | Continuous low-assist Hard clear + full staff/cast credits | **yes — 01:05:41.709, 0 life losses** | [continuous capture](../../tmnt_iv/recordings/tmnt_iv_full_hard_credits.mp4) / [latest dry manifest](../../tmnt_iv/recordings/tmnt_iv_full_hard_dry_run.json) | reduce Technodrome/Starbase assists and form-2 iframe guard |
| 4 | Super Double Dragon | M1–M2 + M4; M3/M5 partial | no | — | natural M3 gym stairs → Chin bosses |
| 5 | Rival Turf! | Reset → fight-ready Stage 1 checkpoint | no | — | clear opening combat lock |
| 6 | F-Zero | Reset → Mute City race-start checkpoint; speed/lateral RAM | no | — | one lap without crash |
| 7 | Magical Quest | Reset → controllable Stage 1 checkpoint; X/progress RAM | no | — | clear first room/checkpoint |
| 8 | Pilotwings | Reset → airborne Lesson 1 checkpoint; altitude/pitch/heading RAM | no | — | complete light-plane objective |
| 9 | Battle Clash | Title boots; input blocked (joypad only, no Super Scope cursor API) | no | — | add light-gun peripheral injection |
| 10 | Joe & Mac | Reset → controllable Stage 1 checkpoint; progress RAM | no | — | first traversable segment |

Status meanings: **scaffold** = integration/layout only; **boot** = reset to a
controllable checkpoint verified; **input blocked** = boot tested but the
required controller is unavailable; **segments** = gameplay clears from
development states; **segmented ending** = ending reached with checkpoint
cuts; **continuous** = one reset-to-ending attempt without state loads or RAM
writes; **assisted continuous** = the same single-session standard with
disclosed, counted writes allowed by a game-local assist contract. Progression
writes do not qualify unless the benchmark explicitly says otherwise.

Shared package: `snes_oneshot/`

## Next assisted long-horizon target

Super Metroid is being prepared as an assisted navigation/full-run target
outside the original easiest-games top ten. Unlimited energy and naturally
unlocked ammo are allowed; progression writes are not. See the
[game plan](../../super_metroid_rl/docs/plan.md) and
[assist contract](../../super_metroid_rl/docs/ASSIST_CONTRACT.md).
