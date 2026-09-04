# rr-5eb2 — L8 four-head Gleeok model + live kill

**Sitting: live kill 2/2.** Body type **`0x45` HP160** at `(124,111)` in
play `0x3C`. South-stand F6/F7 byte-identical, 5124 controller frames,
body-gone f5029, hc 3→4, leftover `(32,181)`.
`GLEEOK_FOUR_HEAD_OBJECT_TYPE = 0x45` is live RAM, not a ROM assumption.
`assumed_0x45` stays False. `route_eligible=false`. Not on `L8_THROUGH`.
Do not close this bead (parent will close). Do not STATUS.

Predict-path split is Verified / ROM claims / Assumed / Plan / Dead beliefs.
Walkthroughs are not proof. L8 Gohma already broke a source assumption
(walkthrough blue `0x34`; live RAM type `0x33` HP 96 in `0x1E`). The same
trap applies here: a ROM-claimed type is not a live observation.

## Power-on note

True `--through level8` is blocked on **rr-8t4.3** (measured post-L7 leave)
and **rr-6o7.1** (natural bush entry) and **rr-6o7.2** (Magical Key, plus the
return passage to the west Gleeok suffix). The first live fight cannot be a
power-on claim this sitting. The model exists so that first live attempt —
from a disclosed fixture, after Magical Key and the return passage — is not a
blind grind. `Level8FourHeadGleeokController` is the south-stand fight in
`level8/gleeok.py`. Hypothesis Gleeok room is still `room_id=None`
(walkthrough grid col 1 row 3, feature `four_head_gleeok`).

---

## Verified (live RAM / L4 / L6 / L8 census + kill)

Live only.

| Fact | Evidence |
|------|----------|
| L8 four-head body type **`0x45`**, room **`0x3C`**, start HP **160**, pose **`(124, 111)`** | `l8_4c_north` N7/N8 idle census 90f; F6/F7 fight. `LEVEL8_INTERIOR_0X3C_NORTH_RECON`. Arrival census empty; body appears after idle. Fireball residual **`0x56`**. `GLEEOK_FOUR_HEAD_OBJECT_TYPE = 0x45` is this live type |
| L8 Gleeok room `room_item_id=0x1A` (heart container) | N7/N8 arrival; still `0x1A` after pickup (id leftover) |
| L8 Gleeok fight pin **`(120, 189)`** south mouth, doors `4` (DOWN bomb hole) | `Level8Interior3CNorthReconFixture` |
| L8 south-stand kills body: type `0x45` **absent** at f5029; `saw_0x46` mid-fight | probe `l8_3c_gleeok` F6/F7 2/2, 5124 ctl / 5184 census. Clone L6 `STAND_DY=22`, bare UP then UP+A, fb dodge ≤14. ghp samples at 250f still read 160 until type-gone |
| L8 HC is room treasure slot 19 **`(32, 192)`**, not mid-room | F5dump `$83/$97`; F1–F4 PNG misses. Pickup hc 3→4 (`0x22`→`0x33`) |
| L8 post-kill leftover **`(32, 181)`**, doors `12` (UP+DOWN), hc **4** | `Level8Interior3CKillReconFixture`; north shutter RAM-open; TF unclaimed |
| L4 2-head body type **`0x43`**, room **`0x13`**, start HP **≈160** | `level4/boss_combat.py`; `LEVEL4_ROUTE.md`; dual-green from `Level4GleeokEnter` |
| L4 detached head type **`0x46`** mid-fight; fireball residual **`0x56`** | same; south-stand on body, do not chase heads while body remains |
| L4 bombs do **not** damage Gleeok | same |
| L4 dead when body type `0x43` **absent** (heads/fireballs may linger) | `level4_gleeok_cleared`; HC RoomItemId `0x1A` mid-room (spine tape often missed it) |
| L4 south-stand `(body.x, body.y+STAND_DY=22)` face **UP+A**; fireball dodge manhattan **≤14** horizontal | `dungeon/gleeok.py`; rr-vdnc Clean |
| L4 assisted **~3649f** from GleeokEnter; spine TF suffix **3564f**; `FIGHT_MAX_FRAMES=20000` | `boss_combat.py` |
| L6 3-head body type **`0x44`**, **not** `0x43`; room **`0x18`** | `l6_settle18_continuous_v1`; `level6/gleeok18.py` |
| L6 head `0x46` **not** seen during idle settle; seen **mid-fight**; fireball `0x56` | settle + `l6_gleeok18_continuous_v1` |
| L6 reuse L4 south-stand; hop **2848f** body-gone; `GLEEOK_18_MAX_FRAMES=20000` | same |
| L6 post-body: residual `0x46`/`0x56`; stop when body gone (heads optional) then census | `Level6PostGleeok18Controller` |
| Shared sensors: body type is dungeon-specific; `0x46` and `0x56` are shared | `dungeon/gleeok.py` |
| `ids.py`: `GLEEOK_OBJECT_TYPE=0x43`, `GLEEOK_3HEAD_OBJECT_TYPE=0x44`, `GLEEOK_HEAD_OBJECT_TYPE=0x46`; **no 4-head constant** | `dungeon/ids.py` |
| L8 `GLEEOK_FOUR_HEAD_OBJECT_TYPE = 0x45` from live RAM; `assumed_0x45` False | `level8/dungeon.py`; `test_four_head_gleeok_factory_does_not_assume_0x45` |
| L8 fixture has **Magical Sword `sword=3`** (disclosed poke on interior recon) | `Level8InteriorReconFixture` provenance |
| L8 Gohma live type **`0x33` HP 96** in `0x1E` (not walkthrough `0x34`) | `LEVEL8_INTERIOR_0X1E_RECON` |

Live L4/L6 body pose used by south-stand: **x≈124, y≈111** (L6 leftover body
`(124, 111)`). Stand target ≈ `(124, 133)`.

---

## ROM claims (not live)

Dumped from local `nes/zelda_i/roms/Legend of Zelda, The.nes` (iNES header
16 bytes; PRG offsets below are **without** the header; add `0x10` for file
offsets). Producer: `scratch/dump_gleeok_rom_tables.py`. Disassembly:
[aldonunez/zelda1-disassembly](https://github.com/aldonunez/zelda1-disassembly)
(`src/Z_04.asm`, `src/Z_07.asm`, `src/Z_05.asm`, `src/ObjVars.inc`). Data
Crystal ROM map for the Gleeok pointer table. **These are ROM claims. Code
must still refuse to assume them.**

### 1. Body type — ROM says `0x45`; unobserved live

Head-count is **the object type itself**, consecutive IDs, not a flag:

| ObjType | Disassembly name | Last neck index (`type - $42`) | Heads | Live |
|---------|------------------|--------------------------------|-------|------|
| `$42` | Gleeok1 | 0 | 1 (unused) | never spawned in first-quest tables we care about |
| `$43` | Gleeok2 | 1 | 2 | **L4 live** |
| `$44` | Gleeok3 | 2 | 3 | **L6 live** |
| `$45` | Gleeok4 | 3 | 4 | **ROM only** |
| `$46` | GleeokHead | — | flying residual | **L4/L6 live** |

`UpdateGleeok` (`Z_04.asm`):

```text
; Calculate the index of the last neck of this kind of gleeok.
; Subtract $42 (Gleeok1) from ObjType[1].
LDA ObjType+1
SEC
SBC #$42
```

Bank 7 init/update jump tables (`Z_07.asm` around the `$3C` Manhandla /
`$3D` Aquamentus / `$41` Moldorm cluster) map **four** consecutive entries
to `InitGleeok` / `UpdateGleeok`, then `InitGleeokHead` / `UpdateGleeokHead`.
That is `$42..$45` plus `$46`.

**L8 hypothesized boss room `0x3C` encodes object type `0x45` the same way
L4 `0x13` encoded `0x43` and L6 `0x18` encoded `0x44`:**

UW spawn (`Z_05.asm`): monster list ID = `(LevelBlockAttrsC & $3F)` plus
`$40` if `LevelBlockAttrsD` bit 7 is set. If that ID `< $62`, it **is** the
object type (single template, not a 4-byte list).

| Room | Block | AttrC (PRG / iNES) | AttrD (PRG / iNES) | ID | Live type |
|------|-------|--------------------|--------------------|-----|-----------|
| L4 `0x13` | L1–6 `0x18700` | `0x03` @ `0x18813` / `0x18823` | `0x85` @ `0x18893` / `0x188A3` (bit7) | `0x43` | **`0x43`** |
| L6 `0x18` | L1–6 `0x18700` | `0x04` @ `0x18818` / `0x18828` | `0x85` @ `0x18898` / `0x188A8` (bit7) | `0x44` | **`0x44`** |
| L8 `0x3C` | L7–9 `0x18A00` | `0x05` @ `0x18B3C` / `0x18B4C` | `0x85` @ `0x18BBC` / `0x18BCC` (bit7) | **`0x45`** | **unobserved** |

Same AttrD `$85` (bit7 set) on all three Gleeok layouts. Calibration of the
same formula:

- L8 Gohma `0x1E` AttrC=`0x33` AttrD bit7 **clear** → ID `0x33`, which
  **did** match live RAM (walkthrough said `0x34`).
- L6 boss `0x1C` AttrC=`0x34` AttrD bit7 clear → ROM ID `0x34`. L6 fight
  code accepts both `0x33` and `0x34`; ids.py labels L6 as `0x33`. Do not
  treat ROM colour/id as live.

Encoding is calibrated on L4/L6 Gleeok **and** still not a live L8 Gleeok
census.

LevelInfo L8 (PRG `0x19AE0`, iNES `0x19AF0`): entrance `0x7E` (matches
fixture-live), TF room `0x2C`, boss room `0x3C`, level# `0x08`. Boss `0x3C`
is the same static decode as `l8-handoff.md`. **Hypothesis until live
`$EB`.** L6 LevelInfo boss is `0x1C` (Gohma), not Gleeok `0x18` — the
LevelInfo "boss" byte is the dungeon-boss room, which for L8 is the
four-head fight if the table is honest.

Room item AttrE low 5 bits, L8 `0x3C`: **`0x1A`** (heart container) at PRG
`0x18C3C`. L4 `0x13` is also `0x1A`. L8 hypothesized TF `0x2C` AttrE=`0x1B`.
L8 entry `0x7E` AttrE=`0x03` matches live `room_item_id=0x03`.

**ROM-claimed L8 body type: `0x45`. Label: ROM, not live. Do not write it
into `GLEEOK_FOUR_HEAD_OBJECT_TYPE` or fight code.**

### 2. Head-count encoding

Not a flag. Not a separate spawn. **Next object id after L6 `0x44`.**
`InitGleeok` always initializes **four** neck RAM arrays; `UpdateGleeok`
only iterates necks `0 .. (type-$42)`. Unused 1-head `$42` exists in the
jump table (Zelda Wiki: unused one-head variant in the code).

### 3. Per-head HP and body HP

Packed HP table `ObjectTypeToHpPairs` PRG **`0x1FB4E`** / iNES **`0x1FB5E`**
(`Z_07.asm`; `ExtractHitPointValue` in `Z_04.asm`: even type = high nibble
as `$x0`, odd type = low nibble `<< 4`):

| Type | Pair | Table HP |
|------|------|----------|
| `$42`–`$45` | `$AA` | **`$A0` = 160** |
| `$46` flying head | `$FB` | `$F0` = 240 (unkillable in practice) |

`InitGleeok` then **overwrites** neck/object-slot HP to **`$A0`** anyway
("Set HP of each neck to $A0"). Live L4 start HP≈160 matches.

HP is **not** saved per neck in the neck RAM arrays (those store X, Y, and
`Gleeok_ObjHeadInfo` only). Slots 1–6 are reused as each neck is loaded.
When a neck dies (`Gleeok_DrawHeadAndCheckCollisions`):

```text
LDA #$60
STA ObjHP, X    ; $60 = 96; remaining heads inherit this shared slot HP
```

Leftover damage does **not** carry (GameFAQs 2013-03-17 thread; matches the
`$60` store after `DealDamage` already zeroed the slot). First head 160,
each remaining head 96. Four-head total **160+96+96+96**.

Sword damage table `SwordDamagePoints` (`Z_01.asm`): **`$10 / $20 / $40`**
= 16 / 32 / 64 for wooden / white / **magical**. Magical is in the L8
fixture (`sword=3`). Hits (ceil, no leftover):

| Sword | 2-head | 3-head | 4-head |
|-------|--------|--------|--------|
| Wooden 16 | 10+6=16 | 22 | 28 |
| White 32 | 5+3=8 | 11 | 14 |
| **Magical 64** | **3+2=5** | **7** | **9** |

**Yes: sword tier changes damage.** Magical does not half White's hit count
because of the `$60` reset (3+2+2+2 = 9, not 14/2).

Invincibility mask on necks: **`$FE`** = sword only. Bombs/fire/boomerang
do not apply. Matches L4 live.

### 4. Detach, necks, what kills Link

When the **head** slot (object slot 5 during a neck's update) dies:

1. Spawn object type **`$46`** in slot `(neck_index + 7)` (necks 0–3 → slots
   7–10), copy head X/Y, flag uninitialized.
2. Hide that neck's sprites; OR its bit into `GleeokDeadNeckMask`.
3. If dead-neck count == `(ObjType+1 - $41)` (4 for type `$45`), the **whole
   boss dies**: metastate `$11` on slot 1, **clear ObjType slots 2–$A**.
   Comment: **"Beware a left over fireball in slot $B."** Same residual that
   killed L4 post-boss at health 106 (rr-gjey).

Flying `$46` (`UpdateGleeokHead`): Keese-like flyer, speed `$BF` max `$E0`;
states SpeedUp / Decide (Random `< $D0` → chase else wander) / Chase /
Wander; **cannot die** (`ResetObjMetastate` + clear invuln every frame;
Red Candle notes: "vulnerable unkillable"). Still runs
`CheckMonsterCollisions` (contact). Shoots `$56` if `Flyer_ObjDistTraveled`
even, Random `< $20`, and slot `$B` empty.

Neck **segments are not separate object types**. Four arrays of 6 segment
XY (`Gleeok_NeckXs0..3` / `NeckYs0..3` in `ObjVars.inc`). Drawn as tile
`$DA` (neck) / `$DC` (head). Collision **only** on head (slot 5) and base
(slot 1). Mid-neck: draw only. Base **cannot die** (`ResetObjMetastate`).

`InitGleeok` geometry (shared by all head counts):

- Neck X all **`$7C` = 124**
- Segment Y `$6F,$74,$79,$7E,$83,$88` → base slot 1 **Y=`$6F`=111**, head
  slot 5 **Y=`$83`=131**
- Body sprites (`Gleeok_DrawBody`): Y `$57`/`$67`, X `$74`+col×8 — visual
  body north of the object slot

That is the live L4/L6 `(124, 111)` pose. South-stand dy=22 → **(124, 133)**,
just under the heads at y≈131.

**What actually hurts Link:** attached heads, body/base contact, flying
`$46`, fireballs **`$56`** (unblockable; Magical Shield does not block —
Zelda Wiki TLoZ Gleeok; `ShootFireball` type `$56`). Necks between base and
head are not collision objects.

Fireball cap from this path: attached **and** flying heads all refuse to
shoot if slot `$B` is occupied, so typically **one** Gleeok `$56` at a time.
Four heads still raise **attempt rate** (one attached neck per frame via
`FrameCounter & 3`, plus each `$46`).

### 5. Hitbox vs south-stand

ROM spawn matches the L4/L6 stand. Predicted policy: **same south-stand
positional loop**, not a new geometry. Four-head is **attrition**: more
attached heads → more `$56` attempts → more `$46` kites after detaches.
Do **not** chase `$46` while the body type remains (L4 rr-vdnc).

### 6. Magical Sword

`SwordDamagePoints-1[Items]` with `Items=3` → **`$40` = 64** per connected
slash (sword state must be fully extended `$02`). That is why L8 fixture
sword=3 matters: 9 hits vs 28 wooden.

---

## Assumed (until a live census)

- Live L8 body type will be **`0x45`**. Calibrated ROM encoding; **still
  assumed**. First frame of the fight logs slot 1 type+HP as-is, including
  a disagreeing value.
- Live start HP **160** (`$A0`), then **96** (`$60`) after first detach.
- Live room id **`0x3C`**, item **`0x1A`**, TF room **`0x2C`**.
- Body pose still ≈ `(124, 111)`; south-stand dy=22 still reaches heads.
- Detached heads still type **`0x46`**, fireballs still **`0x56`**.
- One `$56` in slot `$B` at a time; residual still kills after body-gone.
- Magical Shield still does not block `$56`.

---

## Predicted policy

**South-stand, not a new loop.** Reuse `dungeon/gleeok.py`
(`STAND_DY=22`, `FIREBALL_DODGE_DIST=14`, `_south_stand_action`,
`_fireball_dodge_dir`). Parameterize **body type from live RAM**, never
from a literal `0x45`.

Sketch (do not implement this sitting):

1. Settle until a Gleeok-family body is present (`0x43` / `0x44` / **whatever
   live L8 shows**). If the live type is `0x45`, record it; do not pre-load
   it. If it is something else, **that** is the type.
2. Approach south (`y` to ~165) then align `body.x`, same as L4.
3. Stand `(body.x, body.y+22)` UP+A. Horizontal dodge if `$56` manhattan ≤14.
4. While body present: **do not chase `0x46`**.
5. Body gone: brief face-and-slash only if `0x46` remains in melee; else
   2D fireball flee (rr-gjey `allow_vertical=True`) and hunt HC `0x1A`.
6. Bombs stay in the pocket.

### What would make L4 south-stand fail on 4-head

| Risk | Why | Mitigation |
|------|-----|------------|
| Fireball rate | 4 necks share `FrameCounter&3`; plus up to 4× `$46` also rolling `$56` | Keep dodge; if deaths cluster on `$56`, widen **approach** dodge only (L4 `FIREBALL_DODGE_DIST_LOW_HP=22` is approach-only — widening mid-fight walked into the body) |
| Flying-head contact | Up to 4 unkillable `$46` kiting the south tile | Stay on the stand; do not chase; accept some Survival refill |
| 4th neck hangs south | All necks start at x=`$7C` then spread; one head may overlap y≈133 | If live head.y ≥ stand.y, step 8px south or 8px off-x and re-stand |
| Longer exposure | 9 Magical hits vs L4's 5 / L6's 7 | Budget more frames; same stand |
| Slot `$B` residual | Boss-died path does **not** clear slot `$B` | Never idle after body-gone (rr-gjey) |
| Wrong body type | Gohma-class trap: ROM `0x45` vs live something else | Fail closed until census; do not hardcode |
| Wrong room | LevelInfo `0x3C` unobserved | Do not register `DungeonRoomSpec` from this file |

Attrition, not a different positional policy — unless live RAM shows body.y
or a 4th-head hitbox that the dy=22 stand cannot slash.

---

## Frame budget and stop

| Phase | Proposal | Precedent |
|-------|----------|-----------|
| Spawn settle | 512f | L6 `settle18` |
| Fight | **20000f**, same as L4/L6 `FIGHT_MAX_FRAMES` | L4 ~3.6k incl. TF; L6 2848f body-gone. 4-head is ~2 extra Magical hits + more dodge. If 20000 expires with body still live, **budgeted failure + RAM dump**, then consider 25000 |
| Post-body residual | 4000f | L6 `POSTGLEEOK_18_MAX_FRAMES` |
| Whole Magic-Key→shard chapter | already `max_frames=30000` | `MAGIC_KEY_TO_SHARD_SPEC` |

**Stop (fight):** live body type **absent** (whatever type the census
recorded — not a hardcoded `0x45`) **and** `0x46` absent, or body absent
plus residual-census timeout like L6. **Do not** treat `room_all_dead` as
sufficient (L4 heads/fireballs linger). HC `0x1A` is ROM-claimed for
hypothesized `0x3C`; L4 spine often collected it on the UP path — do not
fail the fight solely on missing mid-room HC. TF `0x80` is **out of scope**
for the first live attempt.

**Budgeted failure must include:** screen, mode, slot table (type, hp, x, y
for slots 1–12), `GleeokDeadNeckMask` if readable, `room_item_id`,
`cur_opened_doors` / `open_doorway_mask`, Link xy/health. No HP/room/TF/door
pokes.

---

## Plan (next sitting)

1. From `Level8Interior3CKillReconFixture` play `0x3C` `(32,181)`, one
   dest hop UP through the RAM-open north shutter to the TF room.
   Do not poke TF. Heart already collected.
2. Keep `route_eligible=false` / not on `L8_THROUGH` until power-on
   predecessors exist. Parent closes rr-5eb2.

---

## Dead beliefs

- **`0x45` is only a ROM encoding.** Struck: live N7/N8 census + F6/F7
  kill observed type `0x45` HP160. `GLEEOK_FOUR_HEAD_OBJECT_TYPE = 0x45`.
- **Walkthrough / Zelda Dungeon "4-head" proves object type or HP.** Source
  only (`DUNGEON_WALKTHROUGHS.md`, `LEVEL8_ROUTE.md`).
- **Walkthrough blue Gohma `0x34` as a type-table lesson.** Live L8 `0x1E`
  was `0x33` HP 96. ROM AttrC for that room *was* `0x33` (matched live;
  walkthrough was wrong). L6 boss room `0x1C` ROM AttrC is `0x34` while
  ids.py / fight comments call L6 `0x33`. ROM, walkthrough, and RAM are
  three channels. For L8 Gleeok, ROM and walkthrough agree on "4-head"
  while RAM had not spoken (now spoken: live `0x45`).
- **Bombs damage Gleeok.** Mask `$FE`; L4 live no.
- **Flying `$46` is a kill target.** Unkillable; chasing it while the body
  remains is the L4 failed policy.
- **Neck segments are extra object types.** They are RAM arrays + draw
  tiles; collision is head+base only.
- **LevelInfo boss `0x3C` is a live `$EB`.** Static decode; `room_id=None`
  until RAM.
- **AttrC `0x05` means blue Goriya.** Low 6 bits of the list ID; AttrD bit7
  adds `$40` → `$45`.
- **L6 LevelInfo boss is Gleeok.** It is `0x1C` Gohma; L6 Gleeok is mid-boss
  `0x18`.
- **HP is one pool of 160 for the whole dragon.** First head 160, then 96
  per remaining head after the `$60` reset.
- **Magical sword halves White's hit count on Gleeok.** Reset wastes
  leftover; 4-head Magical is 9, not 7.

---

## Citations

- Local live: `nes/zelda_i/dungeon/gleeok.py`, `dungeon/ids.py`,
  `level4/boss_combat.py`, `level6/gleeok18.py`, `level8/dungeon.py`,
  `level8/path.py`, `docs/LEVEL4_ROUTE.md`, `docs/LEVEL6_ROUTE.md`,
  `docs/LEVEL8_ROUTE.md`, `docs/tasks/l8-handoff.md`
- Disassembly: [aldonunez/zelda1-disassembly](https://github.com/aldonunez/zelda1-disassembly)
  `InitGleeok` / `UpdateGleeok` / `Gleeok_DrawHeadAndCheckCollisions` /
  `UpdateGleeokHead` in `src/Z_04.asm`; jump tables + `ObjectTypeToHpPairs`
  in `src/Z_07.asm`; UW spawn + `ObjLists` in `src/Z_05.asm`; neck RAM in
  `src/ObjVars.inc`; `SwordDamagePoints` in `src/Z_01.asm`
- Data Crystal: [ROM map](https://datacrystal.tcrf.net/wiki/The_Legend_of_Zelda/ROM_map)
  Gleeok pointer table PRG `0x12809` (body tiles `C0 C4 C8…`, graphics not
  type); [dungeon data](https://datacrystal.tcrf.net/wiki/The_Legend_of_Zelda/Dungeon_Data)
- Red Candle: [Technical Information](https://redcandle.us/Legend_of_Zelda/Technical_Information)
  (attached heads as multi-part; flying heads unkillable; `$0485` HP × `$10`)
- ZeldaHacks: UW 6-byte room attrs, LevelInfo offsets
  (`Zelda 1 Hack Information.txt`)
- GameFAQs thread "Gleeok contact damage is weird" (HP 160 then 96; Magical
  hit table) — **secondary**; ROM `$A0`/`$60`/`SwordDamagePoints` are the
  primary
- Local ROM dump: `scratch/dump_gleeok_rom_tables.py` → HP pairs iNES
  `0x1FB5E`; L8 info iNES `0x19AF0`; L7–9 AttrC/D for `0x3C` iNES
  `0x18B4C` / `0x18BCC`

---

## This sitting did not

Poke boss HP, room state, Triforce, or doors. Chain TF. Close rr-5eb2.
STATUS. Push.
