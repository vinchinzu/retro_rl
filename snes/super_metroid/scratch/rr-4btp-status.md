# rr-4btp manager — WS Main Shaft → basement (`ws_main_to_basement`)

Claimed **IN PROGRESS**. Manager coordinates / reviews / residuals / lands.
Worker implements the hop. Manager does **not** implement unless worker is
stuck after a bump.

## Ownership

| Owner | Files |
|---|---|
| WORKER | `routes/kpdr/wrecked_ship.py` (`play_ws_main_to_basement` only — do **not** rewrite `play_ws_entrance_to_main` / `play_ws_basement_to_phantoon`); new or extended probe (prefer `scripts/probe/ws_main.py` or a `basement` subcommand — do **not** smash Entrance hop); hop unit tests; scratch bench JSON + leave pin |
| MANAGER | residual `docs/tasks/rr-4btp-residual.md`; STATUS/plan/AGENTS/SOURCE_STATES next-line (**NOT** promote); beads close/sync; next bead basement→Phantoon; commit. Do **not** edit `wrecked_ship.py` while worker is mid-edit |

## Pin (source)

`scratch/post_ws_entrance_to_main.state` `0xCAF6` `(1063,907)` p9 gs=8
items `0x3105` beams `0x1007` max PB 5 selected=0.

Target: `0xCC6F` Wrecked Ship Basement, ordinary gs=8 (`door_transition=0`).
Leave pin (worker): `scratch/post_ws_main_to_basement.state` (name TBD by worker).

## Public policy (unpowered first visit toward Phantoon, NOT post-Phantoon climb)

Enter mid-height. Pin is on the RIGHT of a wide room (`x=1063`; save is
across). Descend the stairs (do **not** go up = attic `0xCA52`). Ignore
grey locked doors. Skip the optional left-wall missile. At the bottom:
shoot floor pipes (aim down), morph, Super the green floor hatch, drop
into basement. Coverns/Sbugs only; Atomics stay in glass (unpowered).
Energy assist on — tank. Wiki is thin on the descent (KPDR section is
the post-Phantoon climb):
https://wiki.supermetroid.run/Wrecked_Ship_Main_Shaft

Green door = Super (`select_weapon(2)`). Morph via `ensure_morph`
(double-tap DOWN). Floor-door patterns: `routes/skills/kraid_return.py`
and horizontal Super door in `west_ocean.open_green_super_ws`.

## Hard nos

- Do **not** rewrite `play_ws_entrance_to_main` / `play_ws_basement_to_phantoon` / attic / Phantoon fight
- Do **not** STATUS-promote. Do **not** change `DEFAULT_CONTINUOUS_TIP` (ice)
- Do **not** append this hop to `POST_ICE_SPINE` / `WS_ONLY_HOPS` / `--to ws`. Tip `ws` still ends at `0xCA08`
- Do **not** write `recordings/ws.json`. Scratch only
- Do **not** close `rr-g3nj`. Do **not** replace mid→thin WJ. Do **not** push

## Worker progress

- 2026-08-24 start: `play_ws_main_to_basement` is still `_scaffold_exit` 240f RIGHT+B.
- 13:18: worker wrote `scripts/probe/ws_main.py` (not yet run; no scratch JSON). `wrecked_ship.py` still scaffold. Watching.

## BEFORE (scaffold 240f RIGHT+B)

Timeout **240f** still `0xCAF6` `(1243,907)` p137 — save-door crash wall.
Boot after settle 5: `(1077,907)` p9 (source leave was `(1063,907)`).
Report: `scratch/ws_main_to_basement_before.json`. Success vs this timeout
matters more than a speed Δ.

## Dump (DOWN+RIGHT, halt no_stair_progress)

Path: `(1079,907)` p1 idle → crouch p17 along the floor → dip `(1157,916)` p111
→ fall `(1177,944)` p111 vy=2 → back up `(1193,930)` p17 → `(1206,917)` p17.
DOWN+RIGHT on the flat pin **crouches** (p17); it is not a stair walk.
Stair/drop start is ~x=1157. At x=1177 y=944 they were falling — continuing
RIGHT climbed back toward the save. Next hypothesis: walk RIGHT (no DOWN)
to ~1150–1180 then drop / keep falling; do not hold DOWN on the entry floor.

Coverns `0xD87F` near pin and down the shaft. Enemy `0xE77F` at `(56,984)`
is the left-hallway missile area (skip).

## Dump RIGHT-only (halt save_door_x_band)

Screenshot `scratch/ws_main_pin.png`: entry left, save right, valley/hole
in the middle with stairs visible below.

RIGHT path: `(1102,907)` p9 → fall `(1143,911)` p41 → `(1163,923)` → land
`(1180,943)` p9 → climb out `(1204,919)` → save `(1240,907)` p9.

The valley is a 2-tile V (not the shaft). Continuing RIGHT from the V
bottom climbs back to save. From `(1180,943)` try LEFT / drop (no RIGHT)
to follow the stairs down the shaft. Do not walk to x≥1240 at y≈907.

## Later dumps (save-ledge)

`save_approach.png`: Samus on the save island, stairs clearly down-left.
`dump_downleft`: walked INTO the door `(1243,907)` p137 then DOWN+LEFT stuck
p39. `dump_hop`: LEFT+A from `(1232,907)` jumped UP `(1236,867)` p26.

**Split is the valley bottom `(~1180,943)`.** RIGHT from there → save.
DOWN+RIGHT / door-crash does not descend. Next: walk RIGHT only until
y≥930 (into the V), then **release RIGHT** and hold DOWN or LEFT to follow
the lower stairs. Do not jump (goes up = attic direction). Back off the
save door — x=1243 is the wall.

## Morph hole (progress)

`ensure_morph` at pin drops y 907→921. Roll RIGHT falls through the hole:
`(1143,926)` → `(1176,961)` p29 morph **on the real stairs**
(`shots/02_on_stairs.png`). Halt `fell_through` is GREEN geometry.

Unmorph at that seat pops back to the bridge `(1176,909)` then save.
Continuous RIGHT from the pin overshoots the hole to save morph-floor
`(1243,921)` p30.

**Stay morph at `(1176,961)`. Do not unmorph. From that seat roll to
follow the stairs (try LEFT toward the shaft; RIGHT from pin overshoots).**

## Zigzag descent (working)

Morph-roll path:
1. Morph at pin → `(1077,921)`
2. RIGHT through hole → land `(1176,961)`
3. LEFT down stairs → left-wall `(1045,1097)` (do not sit there)
4. RIGHT drop → `(1123,1132)` p30 (`p3_final.png`) — still morph, grey
   sponge-bath door on the right (ignore).

Keep the zigzag (LEFT/RIGHT on y-progress) until the floor-pipe / green
hatch band (~y 1800). Then unmorph, aim-down shoot pipes, Super, drop.
Do not STATUS-promote. Phantoon stays scaffold.

## Bottom platform (y~1689)

Pong morph-roll reaches `(1061,1689)` left-wall grey door (ignore). Floor
hole to the pipes is **right of that**, visible in `bwr_1146_1680.png` /
`bwalk_final.png`. Shooting DOWN+X from x=1061 misses the pipes. Walking
to `(1250,1680)` is the right wall past the hole.

## Product (not dual-green — hatch drop still RED)

`play_ws_main_to_basement` is no longer `_scaffold_exit`. Phantoon still is.
Probe: `scripts/probe/ws_main.py` `bench` / `dump` / `pure --dual`.
Unit tests: `test_wrecked_ship_scaffold.py` 12 passed (seat / don't-go-up /
Super floor door / morph / settle gs=8 / product not scaffold).

Controller:
1. `ensure_morph` + RIGHT through grated hole → y≥950
2. Morph ping-pong LEFT/RIGHT on stuck to y≥1650 (node-3 platform)
3. Unmorph, beam, jump+aim-down+X floor pipes
4. Morph into well y≈1721, unmorph, aim-down Super, morph-roll/bomb drop

Hatch well is real: pipes open a hole at ~`(1143,1707)` standing /
`(1142,1721)` morph. Aim-down Super (jump, DOWN, DOWN+X) **does** hit the
green floor door (white open ring in `v8_after.png` / `after_drop.png`).
Drop still misses: remaining lip at y=1721 is not bomb-breakable; jump
climbs out; grounded morph sits on the lip; Coverns/Atomic at feet eat
Supers (pose 164 lag / p137 knockback). Timeout pin
`scratch/ws_main_to_basement_timeout.state` + `timeout_*.png`.

| | frames | seconds | clock | result |
|---|---:|---:|---:|---|
| BEFORE scaffold | 240 | 3.993 | 00:04.00 | TIMEOUT `0xCAF6` `(1243,907)` p137 save wall |
| AFTER product | 2261 | 37.621 | 00:37.68 | TIMEOUT `0xCAF6` `(1158,1721)` p30 morph in well |
| Δ | +2021 | +33.628 | +00:33.68 | fail → fail (reached hatch well, no basement) |

Times via `format_segment_time` (NTSC 60.0988). No dual-green. No leave pin.
No STATUS-promote. `WS_ONLY_HOPS` / `--to ws` / `DEFAULT_CONTINUOUS_TIP` untouched.

### Next (hatch drop only)

Stand **in** the well at x∈[1138,1155] y≈1721 (not the right lip x=1163).
Aim-down Super with fuse (do not mash X → pose 164). Then **fall** — no A.
If the remaining 1721 floor is the closed door shell, the Super must open
it **from this y** (closer than the 1688 platform). Morph bombs do not
break it. Do not go attic. Do not enter save. Do not rewrite Phantoon.

Over the hole (~x=1146–1220): aim-down shoot pipes, morph, drop, Super
green hatch (`select_weapon(2)`), `wait_ordinary_room` `0xCC6F` gs=8.

## Hatch (so close)

`dump_full` halt still `0xCAF6`. Path: over_hatch `(1135,1675)` → layer
`(1143,1707)` → morph_on_pipes `(1143,1721)` → unmorph Super `(1144,1707)`
p2 selected=2 → remorph/bounce `(1144,1663)`.

`pre_super.png`: standing on the pipe layer, Super selected, hatch in the
**hole below**. Horizontal Super misses. `supered.png` morph is on the
hatch but you cannot Super while morphed.

**From `(1144,1707)` selected=2: DOWN+X (not remorph) until the hatch
flashes/opens, then drop. Same pattern as `kraid_return` floor door.**

## Hatch v2–v7 still `0xCAF6`

v7 hops stayed p1 standing `(1143,1707)` — never aimed down, never dropped.
v3 morph sits ON the hatch `(1143,1721)` p29; unmorph pops to 1707.
v7_end.png: Super selected, standing over the hole, Covern tank-ok.

Recipe to try (one change):
1. Reach pre_super `(1143,1707)` selected=2
2. Hold **DOWN+X** (pose should leave p1; p23/p83 are aim-down)
3. Do **not** remorph until the hatch opens (green shell gone)
4. Then DOWN or morph-drop through. `wait_ordinary_room` 0xCC6F gs=8
   (state 11 can last 50–100+f)

Do not bomb. Do not walk to x=1227. Phantoon stays scaffold.

## v8 Super hits the WRONG floor

v8: six DOWN+X Super cycles at `(1143,1704)` p23 — hatch never opened.
v9_supered.png: Super explodes on the **pipe-layer floor** she's standing
on, not the hatch in the hole. `(1143,1707)` is the floor *around* the
hole. Morph-unmorph from `(1143,1721)` pops back to 1707.

**Correct Super seat:** fall *into* the hole standing, `(1143,1721)` p55
(`dump_hatch` already landed there). Select Super, DOWN+X **on the hatch**
(y=1721), then drop. Do not Super from y=1707.

## LAND NOW

Worker+manager dual-green is on disk. Do NOT re-implement the hop.
Leave pin: scratch/post_ws_main_to_basement.state
Reports: scratch/ws_main_to_basement.json + _dual.json (1208f ×2, 0xCC6F (657,92) p24 gs=8)
Probe: scripts/probe/ws_main.py

Land: residual rr-4btp-residual.md; STATUS/plan/AGENTS/SOURCE_STATES next-line only
(NOT promote); new bead Basement → Phantoon discovered-from:rr-4btp;
bd close rr-4btp; bd sync; tests; commit code+beads; do NOT push.
Do NOT append to --to ws / WS_ONLY_HOPS. Default CLI stays ice. rr-g3nj stays open.

## Manager close
- Dual GREEN accepted **1208f** ×2 `0xCC6F` `(657,92)` p24 gs=8.
- Residual `docs/tasks/rr-4btp-residual.md`. Next bead `rr-cjpp`.
- `bd close rr-4btp`. No STATUS-promote. No push. Commit `39462909`.

## Manager hatch take-over (2026-08-24 13:41)

Worker wrote `play_ws_main_to_basement` at 13:43 (morph hole + pong +
pipe hop + Super drop). Manager is **not** editing wrecked_ship.py.
Watching their bench. Hatch Super still the risk (DOWN+X on ground morphs;
L+X from y=1707 hits the pipe-layer floor). Phantoon stays scaffold.

## Bench RED (product)

13:45 **1830f** timeout `(1144,1707)` p164 sel=2. 13:47 **1988f** timeout
`(1163,1707)` p164 sel=2. enemy0 `(1163,1772)` hp 300→200 — Super is
hitting the **Covern on the hatch**, not the door. Wait for Covern phase
out (or beam it), then Super the empty hatch. Do not STATUS-promote.

13:50 bench still RED but **on the hatch** `(1158,1721)` sel=2. Super from
this y with **L+X** (DOWN+X morphs). Then DOWN to drop. Do not hop away.

## Parent bump 20m (both children still running — cannot resume)

Do **not** re-dump the stair path. Geometry is enough. Finish the hatch.

Known seats:
- BEFORE scaffold timeout **240f** `0xCAF6` `(1243,907)` p137 save-door crash
- Morph at pin → hole land `(1176,961)` p29 — stay morph, zigzag LEFT/RIGHT on y-progress
- Bottom left wall `(1061,1688)` p50 — ignore grey door
- Over hatch `(1135,1675)` p9 — aim-down shot then fell `(1143,1721)` p55 **still `0xCAF6` selected=0**
- Pipes/hatch band x≈1146–1220, y≈1675–1721. Green floor door still closed (no Super yet)

Worker: stop new dump scripts. Write `play_ws_main_to_basement` from these seats:
1. morph-roll zigzag to bottom (already works)
2. stand over hole ~x=1140 y=1675
3. DOWN+X pipes (beam still selected=0 is fine for pipes)
4. drop onto hatch y~1721 — unmorph/stand if p55, `select_weapon(2)`, shoot DOWN Super, drop
5. `wait_ordinary_room` `0xCC6F` gs=8
6. dual-green from `post_ws_entrance_to_main.state`; leave pin `post_ws_main_to_basement.state`

Do **not** jump from the hatch (p55 was a spin/fall, not a Super). Do **not** walk to save x≥1240. Do **not** go up. Phantoon stays scaffold.

Manager: do **not** implement. Watch for dual-green JSON then land residual/tests/commit. `play_ws_main_to_basement` is still `_scaffold_exit`. No leave pin yet. Schedule stays.

## Parent bump — worker DONE RED, manager still running

Worker 01a034f9-0cff-75e2-b845-510b4bafe6f5 completed. Hop is **RED**.
Timeout pin: `scratch/ws_main_to_basement_timeout.state` `(1158,1721)` p30 morph sel=2 still `0xCAF6`.
Parent is resuming the worker on **hatch drop only**. Manager: do **not** land as green. Do **not** re-implement the stair descent. Do not STATUS-promote. Stay on residual watch.

One-change hatch recipe (do not dump stairs again):
- Seat **in** the well x∈[1138,1155] y≈1721 standing (not the right lip x=1158/1163).
- DOWN+X morphs — use **L+X** (angle-down Super) from standing, fuse, don't mash.
- Then hold **DOWN** to fall. No A (climbs out).
- `wait_ordinary_room` `0xCC6F` gs=8. Dual-green + leave pin `post_ws_main_to_basement.state`.

## Parent bump 20m #2

Manager still running (cannot resume). Worker resume `01a0351d-d6a2-7150-8c86-b112cecb2f36` is live on hatch drop only.

Hop still **RED**. Latest `ws_main_to_basement.json`: fail 2321f `0xCAF6` `(1158,1721)` p30 morph sel=2. No dual JSON. No leave pin.

Manager: do **not** land. Do **not** edit `wrecked_ship.py` while the resumed worker is mid-hatch. Residual watch only.
Worker: one change remains — stand in well x∈[1138,1155] y≈1721, **L+X** Super, DOWN fall, `wait_ordinary_room` 0xCC6F. Do not dump stairs. Do not go attic/save/Phantoon.
