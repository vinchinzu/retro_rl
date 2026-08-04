# RAM map — TMNT IV: Turtles in Time (USA)

WRAM offsets as seen by stable-retro `get_ram()` (bank `$7E` stripped).
Seeded from [GameHacking PAR codes](https://gamehacking.org/game/45110);
player/enemy coordinates confirmed by walk/UP/attack differentials from
`Stage1.state`.

## Player (base `0x0400`, P2 at `+0x70`)

| Offset | Addr | Type | Notes |
|--------|------|------|-------|
| `+0x08` | `0x0408` | `<u2` | Player X (screen-space on Stage 1 locks) |
| `+0x0C` | `0x040C` | `<u2` | Player Y (UP decreases; normal screen Y) |
| `+0x14` | `0x0414` | `u8` | Turtle / weapon id (Leo≈2) |
| `+0x28` | `0x0428` | `u8` | Animation |
| `+0x4A` | `0x044A` | `u8` | Health (Leo 80, Mike 96, Don 64, Raph 48) |
| `+0x6E` | `0x046E` | `u8` | Invuln flash timer |

`0x0412` mirrors Y during probes (treat `+0x0C` as canonical).

## Enemies (base `0x08D0 + i*0x70`, i=0..6)

Same relative layout as the player. HP PAR one-hit-kill codes land at
`base+0x4A`:

| Slot | Base | HP addr |
|------|------|---------|
| 0 | `0x08D0` | `0x091A` |
| 1 | `0x0940` | `0x098A` |
| 2 | `0x09B0` | `0x09FA` |
| 3 | `0x0A20` | `0x0A6A` |
| 4 | `0x0A90` | `0x0ADA` |
| 5 | `0x0B00` | `0x0B4A` |
| 6 | `0x0B70` | `0x0BBA` |

Foot Clan street thugs start around HP **16** (chars
`0x5E`/`0x60`/`0x62`/`0x68`). Baxter Stockman char `0x44`, HP ≈96.
Stage 2 alley boss **Metalhead** char `0x46`, HP 128 (HUD `M. HEAD`).
Stage 3 sewer **spike props** char `0x1C` / `0x2C` (HP 0, −16) are in
`HAZARD_CHAR_IDS` — not living enemies. Stage 3 sewer boss **Rat King**
char `0x4A`, spawn HP 96 (HUD
`R. KING`) — stays boss via char id after HP drops below 80. Stage 4
Technodrome duo **Tokka** `0x48` + **Rahzar** `0xA0`, spawn HP 96 each
(HUD `TOKKA`/`RAHZAR`). Stage 5 Prehistoric boss **Slash** `0x50`,
spawn HP **160**. Stage 6 Skull and Crossbones duo **Bebop** `0xA8` +
**Rocksteady** `0xAC`, spawn HP 128 each. Stage 7 Wounded Knee boss
**Leatherhead** `0xA2`, spawn HP **172** (HUD `L. HEAD`). Stage 8
Neon Night Riders boss **Krang** `0x4E`, spawn HP **160**. Despawned
slots often show `x=65504` with HP 0 — filter `x >= 512` or HP 0.
Sewer Surfin' can leave ghost slots with `char=0` / `x=0` and residual
HP — also excluded.

**NPCs:** April O'Neil uses the same table with char `0xC4`, HP 48 —
excluded via `NPC_CHAR_IDS` (do not fight). Prehistoric pterodactyl
carrier `0xEE` also filtered. Dinos `0x6C` are combat targets but need
jump-slash (B+Y); grounded Y does not chip them. Stage 7 stacked
bazooka Foot top `0xB0` likewise needs jump-slash. Stage 8 Mode-7
props (`NEON_PROP_CHAR_IDS`: boards/debris `0x36`/`0x3C`/`0xAC`@HP2
etc.) filtered only when stage byte **7**. Starbase hover/teleporter
Foot `0x6A` (plus `0x6C`/`0xB0`/`0xB2`/`0xB4`/`0xF2`) need jump-slash
or grounded Y soft-locks mid-stage.

**Pickups:** Ground pizza box char **`0x30`** (blue “PIZZA” crate). HP
byte stays **0**, so it never appears in `living_enemies`. Adapter
exposes on-screen boxes as `extras["pickups"]` = `(x, y, char)` tuples.
Full restore to Leo max (**80**). Policy `PizzaSeek` walks to the box
and taps Y when HP is not full (screen-wide seek when HP ≤ 32).

**Hazards:** `extras["hazards"]` lists char **`0x32`** / **`0x36`**
wrecking-ball props (HP 0). Ceiling **`0x36`** can deal a −24 chip;
do not seek either as pizza.

**Boss chars:** `BOSS_CHAR_IDS = {0x44, 0x46, 0x48, 0x4A, 0x4E, 0x50,
0x52, 0xAE, 0xA0, 0xA2, 0xA8, 0xAC}` (Baxter, Metalhead, Tokka, Rat
King, Krang, Slash, Super Shredder form1/form2, Rahzar, Leatherhead,
Bebop, Rocksteady).

## Globals

| Addr | Notes |
|------|-------|
| `0x0032` | Menu / mode (`0x00` title, `0x02` char select, `0x06` playing) |
| `0x0070` | In-game event / scene (`0x0A` playing; `0x19` stage-clear fade; `0x04`–`0x09` intermission; `0x0D`/`0x0E`/`0x0F` ending sequence) |
| `0x0082` | **Stage id** (0 = Big Apple / S1, 1 = Alleycat Blues / S2, **2 = Sewer Surfin' / S3**, **3 = Technodrome / S4**, **4 = Prehistoric / S5**, **5 = Skull and Crossbones / S6**, **6 = Wounded Knee / S7**, **7 = Neon Night Riders / S8**, **8 = Starbase / S9**, **9 = Super Shredder form 2**, **≥10 = ending sequence**) |
| `0x0096` | Timer |
| `0x003A` | Progress heuristic (increases while advancing; not scroll origin) |
| `0x046E` | Player invulnerability/flash timer; capture uses value 1 as a disclosed form-2 assist |
| `0x1AA0` | Lives (P1). `0` = last life still playable if HP > 0 |
| `0x1A9A` | Lifebar modifier (PAR); not required for adapter |
| `0x1FEE` | Difficulty (`2` = hard; verified continuously by recorder) |
| `0x1FF2` | Continue setting selected in options |

## Combat notes

- Attack button: **Y**. Jump: **B**. Special (**A**) drains HP — avoid.
- Vertical align uses normal screen coords (`invert_vertical=False`).
- Player/enemy X are **screen-space** during locks. `0x003A` is a
  progress counter (wave unlock / walk stall), not camera-left; policy
  zeros `camera_x` for combat edge clamps. Progress can tick while Leo
  is stuck on Stage 2 dumpster collision — policy stalls on frozen
  `player_x` and cycles DOWN → JUMP+RIGHT → UP (dumpster breakers).
  Foot parked past ~screen-right (`x > 244`) widen the combat right
  margin so we walk in instead of `edge_wait` forever.
- First Stage 1 lock: walk right from spawn (~50f) until living enemies
  appear; clear → scroll unlock.
- Stage 1 boss is **Baxter Stockman** (fly), HP ≈96 in the enemy slot —
  not Rocksteady (Stage 6). Stage clear: `ADDR_STAGE` 0→1, event `0x19`.
- Stage 2 bridge: idle ~240f after stage advance, light START through
  intermission (`0x04`…`0x0C`), walk right until Foot spawn + HUD.
- Stage 2 alley boss is **Metalhead** (`char 0x46`, HP 128). Clear:
  `ADDR_STAGE` 1→2, event `0x19`. Next stage is **Sewer Surfin'**
  (hoverboard), not Prehistoric — Tokka & Rahzar are later.
- Stage 3 bridge: idle ~240f after stage advance, light START through
  intermission, walk/surf right until Foot spawn + HUD (`Stage3.state`,
  stage byte **2**).
- Stage 3 (Sewer Surfin'): hanging spikes punish `align_up` — policy
  clamps Foot fight Y to the lower water lane (`y≥160`) and holds
  RIGHT to match auto-scroll. Boss is **Rat King** (`char 0x4A`,
  HUD `R. KING`, spawn HP 96). Long poke (`attack_range≈140`) from
  mid/left water lane reduces HP to 0; top-lane jump-slashes whiff.
  Left auto-scroll chip → JUMP+RIGHT. After HP 0: event `0x0A→0x0B`,
  char despawns. Older isolated low-HP probes died ~444–480f into `0x0B`;
  the HP-safe continuous hard run now proves a natural transition. Remnants:
  boat-ish `char 0x6A`, props `0x66`. Historical `Stage3_Clear` was a
  cloned development state and is not used by the full-run recorder.
  Deep-Y ghosts (`y≥256`) filtered.
- Stage 4 (Technodrome, stage byte **3**): corridor beat-'em-up (not
  sewer auto-scroll). Bosses **Tokka & Rahzar** (`0x48`/`0xA0`, HP 96
  each). Policy uses `PreferredFlank.LEFT` so Leo does not overshoot
  past Rahzar into the right door. The continuous policy adds tank-screen
  throws and close blocker handling; the full run transitions naturally to
  `Stage5` Prehistoric (stage byte **4**).
- Stage 5 (Prehistoric, stage byte **4**): pterodactyl drops (`0xEE`
  filtered), dinos `0x6C` (jump-slash B+Y), cave bruisers `0xB0`/
  `0x76`. Boss **Slash** (`0x50`, spawn HP 160). The continuous policy
  jumps through Slash, attacks from behind, and transitions naturally to
  `Stage6` (stage byte **5**).
- Stage 6 (Skull and Crossbones, stage byte **5**): pirate ship.
  Foot/pirates `0x60`/`0x62`/`0x68`/`0x70`/`0x66`, bruisers
  `0xB0`/`0xB2`. Bosses **Bebop** (`0xA8`) + **Rocksteady** (`0xAC`),
  spawn HP 128 each. Left-flank poke; Bebop HP→0 often despawns the
  duo while Rocksteady still has HP → `0x0A→0x0B` → **natural**
  `event=0x19` / `stage=6` (~580f) → `Stage7` (stage byte **6**).
- Stage 7 (Bury My Shell at Wounded Knee, stage byte **6**): train.
  Foot `0x60`/`0x68`/`0x66`/`0x6A`, bruisers `0xB8`, stacked bazooka
  Foot `0xB0`/`0xB6` (top needs jump-slash B+Y). Boss **Leatherhead**
  (`0xA2`, spawn HP 172, HUD `L. HEAD`). Grounded Y → HP 0 →
  `0x0A→0x0B` → **natural** `event=0x19` / `stage=7` → `Stage8`
  Neon Night Riders (stage byte **7**, Mode-7 highway).
- Stage 8 (Neon Night Riders, stage byte **7**): Mode-7 highway
  (event often `0x16`). Foot boards `0x86`/`0x88`/`0x8A` (HP 2);
  fight only the near band (`y≥140`) — far slots approach in depth.
  Props `0x36`/`0x3C` and Rocksteady-id boards `0xAC`@HP2 filtered via
  `NEON_PROP_CHAR_IDS`. Boss **Krang** (`0x4E`, spawn HP **160**).
  Left-flank Y poke → HP 0 → `0x16→0x0B` → **natural** `event=0x19`
  / `stage=8` (~1180f) → `Stage9` Starbase (stage byte **8**).
- Stage 9 (Starbase, stage byte **8**): Foot `0x5C`/`0x60`/`0x62`/
  `0x66`/`0x68`/`0x6A`, bruisers `0xB0`/`0xB2`/`0xB4`. Hover/
  teleporter slots (`0x6A`, also `0x6C`/`0xF2`) need jump-slash (B+Y).
  Boss **Super Shredder** form 1 `0x52` (spawn HP **128**) → natural
  fade → form 2 arena (stage byte **9**). Form 2 `0xAE` (spawn HP
  **~190**): grounded Y (default policy; forced left-flank / B+Y
  whiff). HP→0 → `0x0A→0x0B` → **natural** `event=0x19` / `stage=10`
  normal-difficulty ending dialogue → title. With difficulty value 2, the
  continuous recorder observes hard-credits event `0x1A` and follows the full
  staff/cast roll and final scene. Stage ≥10 is `CUTSCENE` even with HP 0.
- Boss candidate heuristic: living enemy HP ≥ **96** **or** (char in
  `BOSS_CHAR_IDS` and HP ≥ 4) → `boss_active`. HP floor raised so Neon
  jetpack Foot at HP 80 does not false-trigger.
