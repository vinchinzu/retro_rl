# Zelda I enemy constants (ROM primary sources)

Question: `dungeon/threat.py` carries `BODY_HALF = 8` and `SHOT_HALF = 4` for
**every object in the game**, and no module in this tree knows any per-type
hitbox extent, speed or damage number — speed is *observed* (`dungeon.tracking`),
never known. **What does the ROM actually say?**

This is **not** a port of enemy AI. `rollout.py`'s module docstring measured a
savestate rollout against the real ROM as bit-exact at ~6 us, so a
re-implementation would be less accurate and slower to build. What the
disassembly is worth is the **constants**, because they prune a search.

**Sources.** [aldonunez/zelda1-disassembly](https://github.com/aldonunez/zelda1-disassembly),
pinned at commit **`50a1c86`** (`master` head, fetched 2026-09-17). Every URL
below is line-anchored at that commit, so the anchors do not drift when the
repo moves. Prose twin: [Red Candle / Technical Information](https://redcandle.us/Legend_of_Zelda/Technical_Information).
Prior sitting, same method: [`drop_mechanics_rom.md`](drop_mechanics_rom.md).

Already established there and **not** re-derived here: object type ids are the
`$03xx` type byte; types `>= $53` are shots; `NoDropMonsterTypes` is
`Z_04.asm` ~L11031; no-drop types are `$5D`, `$14/$15`, `$1B/$1C/$1D`, `$17`.

Consumer: `dungeon/species.py` (data only this sitting — nothing reads it yet).

---

## 0. The three tables that cover every type

### Verified

Every enemy constant below comes out of three ROM byte arrays, indexed by the
`$03xx` object type. Nothing is hand-entered per monster.

| Array | What it holds | Decode | URL |
|---|---|---|---|
| `ObjectTypeToAttributes` | 95 bytes, one per type `$00-$5E` | direct index | [Z_07.asm#L5242](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_07.asm#L5242) |
| `ObjectTypeToHpPairs` | 38 bytes, **two types to a byte** (`$00-$4B`) | `ExtractHitPointValue`: even type → `byte & $F0`; odd type → `(byte & $0F) << 4` | [Z_07.asm#L5256](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_07.asm#L5256) · [Z_04.asm#L11002](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L11002) |
| `ObjTypeToDamagePoints` | 93 bytes, one per type `$00-$5C` | `HarmLink`: low nibble → whole hearts, `byte & $F0` → `HeartPartial` (so `$80` = ½ heart) | [Z_01.asm#L5574](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L5574) · [Z_01.asm#L5726](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L5719) |

Attributes and HP are read once, at spawn, in `@FetchAttrs`
([Z_07.asm#L5592](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_07.asm#L5592)).

**Object attribute bits** (each verified at the site that reads it):

| bit | meaning | URL |
|---|---|---|
| `$01` | self checks collisions and draws | [Z_07.asm#L1969](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_07.asm#L1969) |
| `$02` | half-width **draw** | [Z_01.asm#L5120](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L5120) |
| `$04` | self-draw | [Z_07.asm#L1974](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_07.asm#L1974) |
| `$08` | ignore sprite attribute table | [Z_01.asm#L5126](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L5126) |
| `$10` | reverse when blocked — **dead code, set on no type** | [Z_07.asm#L2939](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_07.asm#L2939) |
| `$20` | **invincible to every weapon** (skips the whole weapon pass) | [Z_01.asm#L5480](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L5480) |
| `$40` | **half width for collision detection** — X midpoint is `x+4`, not `x+8` | [Z_01.asm#L5553](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L5553) |
| `$80` | reverse after hitting Link (suppresses Link's shove) | [Z_01.asm#L6616](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L6616) |

---

## 1. Hitbox extents

### Verified — and the headline is that there is **no per-type extent**

`GetObjectMiddle` gives every object the midpoint `(x + 8, y + 8)`, except that
attribute `$40` moves the **X** midpoint to `x + 4`
([Z_01.asm#L5550](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L5550)).
`DoObjectsCollide` then compares `|Δmid|` against **one threshold on both axes**
([Z_01.asm#L6426](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L6426)).
The threshold belongs to the **weapon**, never to the monster:

| contact | threshold (both axes unless stated) | damage type | damage points | URL |
|---|---|---|---|---|
| **monster ↔ Link body** | **`$09` (9)** | — | `ObjTypeToDamagePoints` | [Z_01.asm#L5647](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L5647) |
| sword (stab) | `$10` along the facing axis, `$0C` across | `$01` | `SwordDamagePoints` | [Z_01.asm#L6202](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L6202) |
| sword **beam** / magic shot | `$0C` horizontal | `$01` beam, `$10` magic | beam = `SwordDamagePoints`; magic `$20` | [Z_01.asm#L6039](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L6039) |
| arrow | `$0B` horizontal | `$04` | `$20` wooden, `$40` silver | [Z_01.asm#L6242](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L6242) |
| rod | `$10` / `$0C` (stab) | `$01` | `$20` | [Z_01.asm#L6242](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L6242) |
| boomerang | `$0A` | `$02` | `$00` (stun `$10` → ~`$A0` frames) | [Z_01.asm#L5835](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L5835) |
| fire (candle) | `$0E` | `$20` | `$10` | [Z_01.asm#L6108](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L6108) |
| bomb (detonating) | **`$18`** | `$08` | `$40` | [Z_01.asm#L6108](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L6108) |

**What this means for `threat.py`.** `BODY_HALF = 8` + `LINK_HALF = 8` gives
`MIN_DODGE_BODY = 16`. The ROM's body contact is a **single 9 px centre
distance**. The tree's pad is ~1.8x the ROM's, on both axes, for every type.
That is a safety margin, not an error — but it is not the ROM number, and the
one per-type fact the ROM does carry (attribute `$40`, X midpoint `+4`) is
absent from the tree entirely. `SHOT_HALF = 4` has **no ROM counterpart at
all**: a shot's contact with Link uses the same `$09` as a body
([Z_01.asm#L5647](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L5647)) —
the shot/body split lives only in `dungeon.tracking.HazardClass`.

`combat.SWORD_HALF_WIDTH = 12` **is** the ROM's `$0C`. `combat.SWORD_REACH = 20`
has no ROM twin: the ROM's `$10` is measured from the *sword object's*
midpoint, which sits ahead of Link, not from Link.

### Unverified / absent

- **Per-type body extents do not exist.** Verified by absence: `DoObjectsCollide`
  is the only overlap test and its threshold is a caller parameter; no table is
  indexed by `ObjType` to produce one. The only type-keyed hitbox datum in the
  whole ROM is attribute bit `$40`.
- Sprite (draw) size is not collision size — bit `$02` is a *draw* flag and is
  set independently of `$40`.

---

## 2. Movement speeds and cadences

### Verified — the unit

Speeds are held as `ObjQSpeedFrac` (`$3BC`), a **quarter speed**: `MoveObject`
applies it **four times per frame** to an 8-bit position fraction, so

> **px/frame = qspeed / 64**

[Z_07.asm#L2768](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_07.asm#L2768) ·
[Z_01.asm#L3518](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L3518).
The disassembly's own comments agree (`$20` = 0.5, `$40` = 1, `$C0` = 3).

**The default every monster starts with is `$20` = 0.5 px/frame**, written for
slots `$B..1` in `InitMode_EnterRoom`
([Z_05.asm#L1692](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_05.asm#L1692)).
A type that never writes `ObjQSpeedFrac` moves at 0.5.

### Verified — per type

| type(s) | qspeed | px/frame | when | URL |
|---|---|---|---|---|
| **Link** | `$60` | **1.5** | always, UW and flat OW | [Z_05.asm#L7123](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_05.asm#L7123) |
| **Link** | `$30` | 0.75 | standing on mountain stairs (tile `$74`/`$75`) | [Z_05.asm#L7123](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_05.asm#L7123) |
| Link's sword beam / arrow / boomerang | `$C0` | 3 | on wield | [Z_05.asm#L2983](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_05.asm#L2983) |
| `$07`/`$09` slow Octorok | `$20` | 0.5 | every frame via `_TryShooting` | [Z_04.asm#L2969](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L2969) |
| `$08`/`$0A` fast Octorok | `$40` | **1.0** | `$20` doubled (`ASL`) | [Z_04.asm#L2969](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L2969) |
| any Octorok | `$00` | 0 | on the frame it shoots | [Z_04.asm#L1969](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L1969) |
| `$02`/`$01` Lynel, `$04`/`$03` Moblin | `$20` | 0.5 | restored after shooting | [Z_04.asm#L1950](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L1950) |
| `$0B` red Darknut | `$20` | 0.5 | init | [Z_04.asm#L6449](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L6449) |
| `$0C` blue Darknut | `$28` | 0.625 | init | [Z_04.asm#L6449](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L6449) |
| `$0F` blue Leever | `$08 $0A $10 $20 $10 $0A` | 0.125 … 0.5 | **per `ObjState` 0-5** | [Z_04.asm#L2592](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L2592) |
| `$10` red Leever | `$00 $00 $00 $20 $00 $00` | **0 except state 3** | per `ObjState` 0-5 | [Z_04.asm#L2735](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L2735) |
| `$13` Zol / `$15` Gel | `$40` | 1.0 | state 2 (moving) | [Z_04.asm#L1470](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L1470) |
| `$15` Gel | `$20` | 0.5 | state 0→1, 5-frame timer | [Z_04.asm#L1461](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L1461) |
| `$27` Wallmaster | `$18` | 0.375 | on the crawl | [Z_04.asm#L4242](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L4242) |
| `$28` Rope | `$20` / `$60` | 0.5 / **1.5 charge** | slow on turn, `$60` when it rushes | [Z_04.asm#L4586](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L4586) · [Z_04.asm#L4618](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L4618) |
| `$2A` Stalfos | `$20` | 0.5 | — | [Z_04.asm#L4670](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L4670) |
| `$40` Bubble | `$40` | 1.0 | init | [Z_04.asm#L1095](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L1095) |
| `$5B` Moblin arrow | `$80` | 2.0 | — | [Z_04.asm#L2098](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L2098) |
| `$5C` Goriya boomerang | `$A0` | 2.5 | on throw | [Z_04.asm#L602](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L602) |
| `$55`/`$56` fireball | `$70` major axis | **1.75** | `FireballQSpeedsX/Y`, 9 quantized angles | [Z_04.asm#L981](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L981) |

**Flyers** (`$1B`/`$1C`/`$1D` Keese, `$1A` Peahat, `$21`/`$22` Ghini,
`$46` Gleeok head) do **not** use `ObjQSpeedFrac`. `MoveFlyer` adds
`Flyer_ObjSpeed & $E0` to an 8-bit fraction and steps 1 px per component axis
on carry, so **px/frame/axis = `(speed & $E0) / 256`**
([Z_04.asm#L11560](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L11560)).
Speed ramps ±1 per frame between `$1F` and a per-type maximum
([Z_04.asm#L11530](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L11530)):

| type(s) | start | max | max px/frame/axis | URL |
|---|---|---|---|---|
| `$1B` blue Keese | `$1F` | `$C0` | 0.75 | [Z_04.asm#L1100](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L1100) |
| `$1C`/`$1D` red/black Keese | `$7F` | `$C0` | 0.75 | [Z_04.asm#L1114](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L1114) |
| `$1A` Peahat, `$21`/`$22` Ghini | `$1F` | `$A0` | 0.625 | [Z_04.asm#L1893](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L1893) |
| `$46` Gleeok head | `$BF` | `$E0` | 0.875 | [Z_04.asm#L7659](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L7822) |

A flyer whose speed masks to `$00` sits still and goes to flying state 5 —
which is exactly the state in which a Peahat can be cut.

**Jumpers** (`$0D`/`$0E` Tektite, `$20` boulder) keep `ObjQSpeedFrac` for the
horizontal and add a vertical ballistic arc: `JumperStartSpeedsHi` =
`$FD, $FC, $FE` (−3, −4, −2 px/frame) by kind index `type − $0D` clamped to 2
([Z_04.asm#L2244](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L2244) ·
[Z_04.asm#L2521](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L2521)),
accelerations in `JumperYAccelerations`
([Z_04.asm#L2238](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L2238)).
In `ObjState 0` a tektite is **on the ground and not moving at all**, waiting on
a random timer — the ROM reason `rollout.py` measured a 0 px median error and a
34 px p90 for tektites.

**Turn cadence.** `Wanderer_TargetPlayer` re-aims only when the object is on a
16 px grid line, and turns toward Link only when `ObjTurnRate >= Random`
([Z_04.asm#L314](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L314)).
Rates: slow Octorok `$70`; fast/blue Octorok `$A0`
([Z_04.asm#L2969](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L2969));
Moblin `$A0` ([Z_04.asm#L1950](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L1950));
blue Leever `$A0` ([Z_04.asm#L2601](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L2601));
Stalfos / Gibdo / Like-Like `$80`
([Z_04.asm#L4670](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L4670) ·
[Z_04.asm#L6465](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L6465) ·
[Z_04.asm#L6816](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L6816));
Zol / Gel `$20` ([Z_04.asm#L1476](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L1476));
Bubble `$40` ([Z_04.asm#L1117](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L1122)).
The same routine also fixes the "aim at Link" band at **9 px** on each axis —
the same number as the contact threshold.

**Shoot cadence.** `_TryShooting` gates on `Random,X < $F8` (≈3 % a chance
frame) **except** for the blue overworld walkers `$01`, `$03`, `$09`, `$0A`,
which take every opportunity
([Z_04.asm#L1969](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L1969)).
Stalfos cannot shoot at all in quest 1
([Z_04.asm#L4670](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L4670)).

### Unverified

- Per-frame speeds for the bosses (`$32` Dodongo, `$3C` Manhandla, `$3D`
  Aquamentus, `$43`-`$45` Gleeok body, `$47` Patra, `$33`/`$34` Gohma) are
  written inside long per-boss state machines rather than at one init site.
  They are sourceable but each one is its own reading; none is in the table.
- `$1E` Armos, `$16` Pol's Voice, `$17` Like-Like, `$23`/`$24` Wizzrobe,
  `$30` Gibdo, `$49` blade trap: **no `STA ObjQSpeedFrac` of their own** was
  found, so they run at the `$20` default. Recorded as "default", not as a
  measured per-type constant.

---

## 3. Weapon damage table

### Verified

| weapon | damage points | note | URL |
|---|---|---|---|
| wooden sword (`Items == 1`) | **`$10` (16)** | also the sword **beam** | [Z_01.asm#L6162](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L6162) |
| white sword (`Items == 2`) | **`$20` (32)** | | [Z_01.asm#L6162](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L6162) |
| magical sword (`Items == 3`) | **`$40` (64)** | | [Z_01.asm#L6162](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L6162) |
| wooden arrow | `$20` (32) | | [Z_01.asm#L6242](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L6242) |
| silver arrow | `$40` (64) | | [Z_01.asm#L6242](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L6242) |
| bomb | `$40` (64) | only while detonating (`ObjState $13`) | [Z_01.asm#L6108](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L6108) |
| candle fire | `$10` (16) | | [Z_01.asm#L6108](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L6108) |
| magic rod | `$20` (32) | sword damage **type** | [Z_01.asm#L6242](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L6242) |
| magic (wand) shot | `$20` (32) | damage type `$10` | [Z_01.asm#L6039](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L6039) |
| boomerang | **`$00`** | never damages; stuns `$10` (~`$A0` frames) | [Z_01.asm#L5885](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L5915) |

A monster dies when `damage points >= ObjHP` (`DealDamage`,
[Z_01.asm#L5969](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L5969)),
so **hits to kill = ceil(HP / damage points)** and an HP-0 type dies to one
touch of any weapon it is not immune to.

**Damage-type bits** (`ObjInvincibilityMask & damage type != 0` ⇒ immune,
[Z_01.asm#L5921](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L5921)):
`$01` sword/rod, `$02` boomerang, `$04` arrow, `$08` bomb, `$10` magic shot,
`$20` fire.

| type | mask | immune to everything but | URL |
|---|---|---|---|
| `$0B`/`$0C` Darknut | `$F6` | sword, bomb — **and parries a sword from the front** | [Z_04.asm#L6449](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L6449) · [Z_01.asm#L6311](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L5954) |
| `$16` Pol's Voice | `$FE` | sword — **but any arrow sets HP to 0** | [Z_04.asm#L6656](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L6656) · [Z_01.asm#L6279](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L6295) |
| `$23`/`$24` Wizzrobe | `$F6` | sword, bomb | [Z_04.asm#L7590](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L7590) |
| `$32` Dodongo | `$FF` → `$FE` | nothing, until the bomb-swallow window | [Z_04.asm#L6055](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L6055) |
| `$33`/`$34` Gohma | `$FB` | **arrows only** | [Z_04.asm#L7804](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L7804) |
| `$3C` Manhandla | `$E2` | sword, arrow, bomb, magic (not fire/boomerang) | [Z_04.asm#L7738](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L7738) |
| `$3D` Aquamentus | `$E2` | sword, arrow, bomb, magic | [Z_04.asm#L4842](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L4842) |
| `$43`-`$45` Gleeok | `$FE` | sword | [Z_04.asm#L7641](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L7641) |
| `$47`/`$48` Patra (+ eyes) | `$FE` | sword | [Z_04.asm#L9526](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L9526) |
| Ganon | `$FA` | sword, arrow | [Z_04.asm#L9571](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L9571) |

**Gohma is not an HP fight in the usual sense.** `Gohma_HandleWeaponCollision`
takes damage only when the arrow hits sprite part 3 or 4, the eye state is 3,
**and the arrow's direction is UP** — Link must be below it
([Z_04.asm#L8466](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L8466)).
Everything else is a parry.

### Unverified

- Which weapon slot maps to which `$0D`-`$12` index is read off
  `CheckMonsterCollisions` ([Z_01.asm#L5475](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L5475))
  and is **not** in the table here; the tree's `$0E` sword-beam slot agrees
  with that listing but nothing in this sitting re-measured it live.

---

## 4. Damage to Link, per type

### Verified

`ObjTypeToDamagePoints[ObjType]`: low nibble → whole hearts, `byte & $F0` →
`HeartPartial` (`$80` = ½ heart). The ring halves it once per level
([Z_01.asm#L5712](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L5751)).

Live-confirmed by this tree: a rock (`$53`) is `$80` — one half heart, the
`$FF → $7F` partial that `docs/PRE_L1.md` traced. Types `$2B`-`$2E` (bubbles,
whirlwind) are `$00`: they still run `HarmLink` → `Link_BeHarmed`, so they
**zero the kill streak while costing no hearts** — the exact mechanism
`drop_mechanics_rom.md` ranked #2.

The full per-type table is section 6.

---

## 5. Spawn tables per room

### Verified — the decode

`InitMode_EnterRoom`
([Z_05.asm#L1696](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_05.asm#L1696)):

- object list / template id `= (LevelBlockAttrsC[RoomId] & $3F) | ($40 if bit 7 of LevelBlockAttrsD[RoomId])`
- count index `= LevelBlockAttrsC[RoomId] >> 6` → `LevelInfo_FoeCounts[index]`
- id in `[$32, $62)` → **count forced to 1** (bosses and one-offs)
- id `< $62` → the id **is** the object type, repeated *count* times
- id `>= $62` → index `id - $62` into `ObjListAddrs` → a mixed list in `ObjLists`

RAM addresses (`Variables.inc`
[#L325](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Variables.inc#L325)):
`LevelBlockAttrsA $687E`, `B $68FE`, `C $697E`, `D $69FE`, `E $6A7E`,
`F $6AFE`, `LevelInfo_FoeCounts $6BA2`. These are cart WRAM — **this tree can
already read them live**, the same way `dungeon/tilemap.py` reads `$6530`.

### Not sourceable — say so

The lists and the per-room attribute bytes are **not in the disassembly source**.
`ObjLists` is `.INCBIN "dat/ObjLists.dat"`
([Z_05.asm#L1444](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_05.asm#L1444)) and that `.dat`
**404s** on raw.githubusercontent (it is built from the ROM, not committed);
`LevelBlockAttrs*` is level data loaded into cart WRAM. Only
`dat/ObjListAddrs.inc` (30 offsets) is present, and offsets without the data
are worth nothing.

**So: no spawn rows in `species.py`, on purpose.** The actionable finding is
the RAM decode above, which belongs to a later card.

---

## 6. Per-type table

### Verified

HP, attribute and contact damage for every type the three ROM arrays cover.
"wood / white / magic sword hits" = `ceil(HP / $10 | $20 | $40)` and ignores
per-type immunity (read the mask table in §3 first). Names are this tree's
`dungeon.ids.OBJECT_NAMES`, which are **not** ROM labels — a blank name means
the tree has never seen that id, not that the ROM lacks it.

| type | tree name (`dungeon.ids`) | HP | wood / white / magic sword hits | attr | X-mid | invuln | contact damage |
|---|---|---|---|---|---|---|---|
| `$01` | lynel_blue | 96 | 6 / 3 / 2 | `$00` | +8 | · | `$02` 2 hearts |
| `$02` | lynel | 64 | 4 / 2 / 1 | `$00` | +8 | · | `$01` 1 heart |
| `$03` | moblin_blue | 48 | 3 / 2 / 1 | `$00` | +8 | · | `$80` 0.5 heart |
| `$04` | moblin | 32 | 2 / 1 / 1 | `$00` | +8 | · | `$80` 0.5 heart |
| `$05` | goriya_blue_or_residual | 80 | 5 / 3 / 2 | `$00` | +8 | · | `$01` 1 heart |
| `$06` | goriya | 48 | 3 / 2 / 1 | `$00` | +8 | · | `$80` 0.5 heart |
| `$07` | octorok | 16 | 1 / 1 / 1 | `$05` | +8 | · | `$80` 0.5 heart |
| `$08` | octorok_fast | 16 | 1 / 1 / 1 | `$05` | +8 | · | `$80` 0.5 heart |
| `$09` | octorok_blue | 32 | 2 / 1 / 1 | `$05` | +8 | · | `$80` 0.5 heart |
| `$0A` | octorok_blue_fast | 32 | 2 / 1 / 1 | `$05` | +8 | · | `$80` 0.5 heart |
| `$0B` | darknut | 64 | 4 / 2 / 1 | `$81` | +8 | · | `$01` 1 heart |
| `$0C` | — | 128 | 8 / 4 / 2 | `$81` | +8 | · | `$02` 2 hearts |
| `$0D` | tektite_blue | 16 | 1 / 1 / 1 | `$81` | +8 | · | `$80` 0.5 heart |
| `$0E` | tektite | 16 | 1 / 1 / 1 | `$81` | +8 | · | `$80` 0.5 heart |
| `$0F` | leever_blue | 64 | 4 / 2 / 1 | `$01` | +8 | · | `$01` 1 heart |
| `$10` | leever | 32 | 2 / 1 / 1 | `$01` | +8 | · | `$80` 0.5 heart |
| `$11` | zora | 32 | 2 / 1 / 1 | `$81` | +8 | · | `$80` 0.5 heart |
| `$12` | vire | 64 | 4 / 2 / 1 | `$01` | +8 | · | `$01` 1 heart |
| `$13` | zol | 32 | 2 / 1 / 1 | `$01` | +8 | · | `$01` 1 heart |
| `$14` | gel_or_zol_split_residual | 0 | 1 / 1 / 1 | `$43` | +4 | · | `$80` 0.5 heart |
| `$15` | gel | 0 | 1 / 1 / 1 | `$43` | +4 | · | `$80` 0.5 heart |
| `$16` | pols_voice | 160 | 10 / 5 / 3 | `$81` | +8 | · | `$02` 2 hearts |
| `$17` | like_like | 144 | 9 / 5 / 3 | `$81` | +8 | · | `$01` 1 heart |
| `$18` | — | 128 | 8 / 4 / 2 | `$81` | +8 | · | `$02` 2 hearts |
| `$19` | — | 240 | 15 / 8 / 4 | `$81` | +8 | · | `$00` **0** |
| `$1A` | peahat | 32 | 2 / 1 / 1 | `$01` | +8 | · | `$80` 0.5 heart |
| `$1B` | keese | 0 | 1 / 1 / 1 | `$81` | +8 | · | `$80` 0.5 heart |
| `$1C` | vire_split_keese | 0 | 1 / 1 / 1 | `$81` | +8 | · | `$80` 0.5 heart |
| `$1D` | — | 0 | 1 / 1 / 1 | `$81` | +8 | · | `$80` 0.5 heart |
| `$1E` | armos | 48 | 3 / 2 / 1 | `$01` | +8 | · | `$01` 1 heart |
| `$1F` | — | 240 | 15 / 8 / 4 | `$81` | +8 | · | `$80` 0.5 heart |
| `$20` | boulder | 240 | 15 / 8 / 4 | `$81` | +8 | · | `$80` 0.5 heart |
| `$21` | ghini | 144 | 9 / 5 / 3 | `$81` | +8 | · | `$01` 1 heart |
| `$22` | ghini_flying | 240 | 15 / 8 / 4 | `$81` | +8 | · | `$01` 1 heart |
| `$23` | wizzrobe_blue_walkthrough_correlated | 160 | 10 / 5 / 3 | `$81` | +8 | · | `$02` 2 hearts |
| `$24` | wizzrobe_orange | 64 | 4 / 2 / 1 | `$81` | +8 | · | `$01` 1 heart |
| `$25` | patra_eye | 96 | 6 / 3 / 2 | `$C3` | +4 | · | `$02` 2 hearts |
| `$26` | — | 96 | 6 / 3 / 2 | `$C3` | +4 | · | `$02` 2 hearts |
| `$27` | wallmaster | 32 | 2 / 1 / 1 | `$89` | +8 | · | `$80` 0.5 heart |
| `$28` | rope | 16 | 1 / 1 / 1 | `$89` | +8 | · | `$80` 0.5 heart |
| `$29` | — | 16 | 1 / 1 / 1 | `$81` | +8 | · | `$80` 0.5 heart |
| `$2A` | stalfos | 32 | 2 / 1 / 1 | `$81` | +8 | · | `$80` 0.5 heart |
| `$2B` | invuln_mover_residual | 240 | 15 / 8 / 4 | `$89` | +8 | · | `$00` **0** |
| `$2C` | — | 240 | 15 / 8 / 4 | `$89` | +8 | · | `$00` **0** |
| `$2D` | — | 240 | 15 / 8 / 4 | `$89` | +8 | · | `$00` **0** |
| `$2E` | — | 240 | 15 / 8 / 4 | `$89` | +8 | · | `$00` **0** |
| `$2F` | — | 240 | 15 / 8 / 4 | `$83` | +8 | · | `$00` **0** |
| `$30` | gibdo | 112 | 7 / 4 / 2 | `$81` | +8 | · | `$02` 2 hearts |
| `$31` | — | 240 | 15 / 8 / 4 | `$89` | +8 | · | `$01` 1 heart |
| `$32` | dodongo | 240 | 15 / 8 / 4 | `$89` | +8 | · | `$01` 1 heart |
| `$33` | gohma_red | 96 | 6 / 3 / 2 | `$C9` | +4 | · | `$02` 2 hearts |
| `$34` | gohma_blue | 32 | 2 / 1 / 1 | `$C9` | +4 | · | `$02` 2 hearts |
| `$35` | l4_mid_11_cluster | 240 | 15 / 8 / 4 | `$81` | +8 | · | `$00` **0** |
| `$36` | — | 240 | 15 / 8 / 4 | `$81` | +8 | · | `$00` **0** |
| `$37` | — | 240 | 15 / 8 / 4 | `$81` | +8 | · | `$00` **0** |
| `$38` | — | 240 | 15 / 8 / 4 | `$A9` | +8 | **yes** | `$02` 2 hearts |
| `$39` | — | 240 | 15 / 8 / 4 | `$A9` | +8 | **yes** | `$02` 2 hearts |
| `$3A` | — | 32 | 2 / 1 / 1 | `$41` | +4 | · | `$02` 2 hearts |
| `$3B` | — | 32 | 2 / 1 / 1 | `$41` | +4 | · | `$02` 2 hearts |
| `$3C` | manhandla | 64 | 4 / 2 / 1 | `$89` | +8 | · | `$01` 1 heart |
| `$3D` | aquamentus | 96 | 6 / 3 / 2 | `$89` | +8 | · | `$01` 1 heart |
| `$3E` | — | 240 | 15 / 8 / 4 | `$81` | +8 | · | `$04` 4 hearts |
| `$3F` | — | 16 | 1 / 1 / 1 | `$81` | +8 | · | `$80` 0.5 heart |
| `$40` | bubble | 240 | 15 / 8 / 4 | `$81` | +8 | · | `$80` 0.5 heart |
| `$41` | — | 32 | 2 / 1 / 1 | `$C1` | +4 | · | `$80` 0.5 heart |
| `$42` | — | 160 | 10 / 5 / 3 | `$C1` | +4 | · | `$01` 1 heart |
| `$43` | gleeok | 160 | 10 / 5 / 3 | `$C1` | +4 | · | `$01` 1 heart |
| `$44` | gleeok_3head | 160 | 10 / 5 / 3 | `$C1` | +4 | · | `$01` 1 heart |
| `$45` | — | 160 | 10 / 5 / 3 | `$C1` | +4 | · | `$01` 1 heart |
| `$46` | gleeok_head | 240 | 15 / 8 / 4 | `$81` | +8 | · | `$01` 1 heart |
| `$47` | patra | 176 | 11 / 6 / 3 | `$81` | +8 | · | `$02` 2 hearts |
| `$48` | — | 176 | 11 / 6 / 3 | `$81` | +8 | · | `$02` 2 hearts |
| `$49` | blade_trap | 240 | 15 / 8 / 4 | `$A1` | +8 | **yes** | `$01` 1 heart |
| `$4A` | — | 240 | 15 / 8 / 4 | `$A1` | +8 | **yes** | `$01` 1 heart |
| `$4B` | — | 0 | 1 / 1 / 1 | `$81` | +8 | · | `$00` **0** |
| `$4C` | — | — | — | `$81` | +8 | · | `$00` **0** |
| `$4D` | old_man_or_npc | — | — | `$81` | +8 | · | `$00` **0** |
| `$4E` | trap_or_fire_residual | — | — | `$81` | +8 | · | `$00` **0** |
| `$4F` | — | — | — | `$81` | +8 | · | `$00` **0** |
| `$50` | — | — | — | `$81` | +8 | · | `$00` **0** |
| `$51` | — | — | — | `$81` | +8 | · | `$00` **0** |
| `$52` | — | — | — | `$81` | +8 | · | `$00` **0** |
| `$53` | rock_projectile | — | — | `$E3` | +4 | **yes** | `$80` 0.5 heart |
| `$54` | — | — | — | `$E3` | +4 | **yes** | `$80` 0.5 heart |
| `$55` | fireball_or_statue_projectile | — | — | `$E3` | +4 | **yes** | `$80` 0.5 heart |
| `$56` | manhandla_projectile_residual | — | — | `$E3` | +4 | **yes** | `$01` 1 heart |
| `$57` | lynel_sword_shot | — | — | `$E3` | +4 | **yes** | `$02` 2 hearts |
| `$58` | — | — | — | `$E1` | +4 | **yes** | `$02` 2 hearts |
| `$59` | — | — | — | `$E1` | +4 | **yes** | `$04` 4 hearts |
| `$5A` | — | — | — | `$E1` | +4 | **yes** | `$04` 4 hearts |
| `$5B` | moblin_arrow | — | — | `$E1` | +4 | **yes** | `$80` 0.5 heart |
| `$5C` | boomerang_projectile | — | — | `$E1` | +4 | **yes** | `$01` 1 heart |

### Unverified

- Types `$4C`-`$52` and `$5D`-`$5E` have an attribute byte but **no HP entry**
  (`ObjectTypeToHpPairs` ends at type `$4B`) — blank, not zero.
- Types `$53`-`$5C` (shots) have an attribute byte and a damage byte but no HP:
  they are destroyed by `DestroyMonsterShot`, never by `DealDamage`.
- The **colour names** in `dungeon.ids.OBJECT_NAMES` are this tree's, not the
  ROM's. The disassembly only ever labels a colour in a comment, and for Gohma
  those comments disagree with the tree — see §7.

---

## 7. Where the ROM contradicts something this tree believes

Reported, **not** fixed. C5 is data; nothing was rewired.

1. **`threat.LINK_SPEED = 1.0` is wrong. Link moves `$60 / 64` = 1.5 px/frame**
   ([Z_05.asm#L7123](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_05.asm#L7123)),
   dropping to 0.75 only on overworld mountain-stair tiles `$74`/`$75`.
   `threat.py`'s comment — *"Link walks ~1 px/frame; a step button therefore
   buys ~1 px of separation per frame of horizon"* — under-states the walk by a
   third, which makes every `MIN_DODGE_BODY` / `TRIGGER_TTC` derivation
   pessimistic: a 16 px pad is ~11 frames of walking, not 16.

2. **`$1D` is a Keese and `behaviors._TYPE_TO_KIND` does not know it.**
   `NoDropMonsterTypes` lists `$1B, $1C, $1D` together as Keese, and the ROM
   gives `$1D` HP **0** and `InitRedOrBlackKeese`'s `$7F` flyer speed
   ([Z_04.asm#L1114](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L1114)).
   The tree maps `$1B` and `$1C` to `EnemyKind.KEESE` (`type_only=True`) but
   leaves `$1D` to fall through to `EnemyKind.UNKNOWN`, whose `alive_rule` is
   `TYPE_AND_HP`. A black Keese would therefore read as **dead on arrival** —
   the exact failure the KEESE policy note was written to prevent.

3. **The Gohma colour labels look swapped.** `dungeon/ids.py` has
   `0x33 = "gohma_red"` with *"L6 0x1C; one wooden arrow to open eye"* and
   `0x34 = "gohma_blue"` with *"(3 arrows)"*. The ROM's HP is
   `$33` = **96** (3 wooden arrows at `$20` each) and `$34` = **32** (1 arrow);
   `HandleMonsterWeaponCollision`'s comment reads *"Blue Gohma or Red Gohma"* in
   the order `CMP #$33` then `CMP #$34`
   ([Z_01.asm#L5928](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L5928)).
   Either the colour labels are reversed or the "one wooden arrow" note is.
   **The HP is the fact; the colour is not.** `species.py` carries HP by id and
   no colour claim. A live census on L6 `0x1C` settles it in one run.

4. **`SHOT_HALF = 4` has no ROM basis.** Shots hit Link through the same
   `$09` centre-distance test as bodies
   ([Z_01.asm#L5647](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L5647));
   what is genuinely per-type is attribute `$40`, which every shot `$53`-`$5C`
   carries and which shifts the X midpoint to `x+4` rather than shrinking the
   box. `species.py` records the attribute and leaves `SHOT_HALF` alone.

5. **A dormant burrower is `ObjState 0` for the Zora too.**
   `combat.dormant_body` covers leever types `{$0F, $10}` at state 0. The ROM
   rule is in `Burrower_AnimateDrawAndCheckCollisions` — *"If state = 0,
   return"* — before any collision check
   ([Z_04.asm#L2664](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L2664)),
   and `UpdateZora` runs the **same** updater
   ([Z_04.asm#L1915](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L1915)).
   A submerged Zora (`$11`, state 0) is as uncuttable and as harmless as a
   buried leever. `species.py` records it as `no_contact_states`; nothing
   consumes it yet.

6. **Red leever state 0 is stationary in the ROM; blue leever state 0 is not.**
   `RedLeeverStateQSpeeds` is `$00` in every state but 3
   ([Z_04.asm#L2735](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L2735)),
   which is exactly the tree's measured "3844 still frames". But
   `BlueLeeverStateQSpeeds` state 0 is `$08` = 0.125 px/frame
   ([Z_04.asm#L2592](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L2592)) —
   a buried blue leever creeps ~1 px every 8 frames. The *uncuttable /
   harmless* half of `dormant_body` holds for both; the *immobile* half is a
   red-leever fact that happens to be nearly true of blue ones.

---

## 8. Cross-checks: ROM numbers this tree had already measured live

These are the reason to trust the rest of the table.

| tree constant | measured how | ROM | verdict |
|---|---|---|---|
| `behaviors.ZORA_SHOT_SPEED = 1.75` | `scratch/zora1.json`, 4 surfacings | `FireballQSpeedsX[0] = $70` = 1.75 px/f ([Z_04.asm#L981](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L981)) | **exact** |
| `ZORA_SHOT_DELAY = 2` | live | fires at `ObjState 3`, `ObjTimer == $FD` — 2 ticks after `$FF` ([Z_04.asm#L1928](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L1928)) | **exact** |
| `ZORA_CYCLE` states 1/2/4/5 = 32/15/16/96 | live | `BlueLeeverStateTimes = $80 $20 $0F $FF $10 $60` ([Z_04.asm#L2595](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L2595)) | **exact** on 4 of 6 |
| `ZORA_CYCLE[3] = 34` | live | `$FF` timer, cut to `$20` on the shot ⇒ 2 + 32 = 34 ([Z_04.asm#L1928](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L1928)) | **exact** |
| "aim is quantized at launch, not a bearing" | 4 shots off-bearing | 9-entry `FireballQSpeedsX/Y` via `_CalcDiagonalSpeedIndex` | **explained** |
| `combat.SWORD_HALF_WIDTH = 12` | tuned | `$0C` across the blade ([Z_01.asm#L6202](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_01.asm#L6202)) | **exact** |
| `ids` Vire `$12` HP 64, splits to `$1C` | L4 live (rr-5lu) | HP 64; `UpdateVire` makes **two** type `$1C` ([Z_04.asm#L6915](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L6915)) | **exact** |
| `ids` `$2B` HP 240, sword/bomb no damage | L3 live | HP 240, contact damage `$00` | **exact** |
| Keese HP stays 0 while alive | L1+ live | `ObjectTypeToHpPairs` gives `$1B/$1C/$1D` **0** | **exact** |
| Digdogger `$38` HP 240, whistle-first | L5 live | HP 240 **and attribute `$20` = invincible to every weapon**; `$18` (shrunk) HP 128, attribute `$81` | **exact** |
| Peahat killable only in flying state 5 | `behaviors` note | `UpdatePeahat`: `CheckMonsterCollisions` only at state 5 ([Z_04.asm#L4022](https://github.com/aldonunez/zelda1-disassembly/blob/50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/Z_04.asm#L4022)) | **exact** |
| Rock (`$53`) costs `$FF → $7F` | `docs/PRE_L1.md` | `ObjTypeToDamagePoints[$53] = $80` | **exact** |
| Zora banks no streak | `drop_mechanics_rom.md` | `UpdateZora` → `DestroyMonster`, no `HandleMonsterDied` | **exact** |

Twelve independent hits, no misses. The three arrays in §0 are the same kind of
data as `FireballQSpeedsX`, read the same way.

---

## What is in `species.py` and what is not

**In:** per-type HP, attribute byte (with `half_width` / `invincible` decoded),
contact damage, contact class, the sourced q-speeds, the weapon damage points,
`no_contact_states`. Every one of those has a URL above.

**Out, deliberately:**

- Spawn tables — not in the source (§5).
- Boss per-frame speeds — sourceable, not sourced this sitting (§2).
- Anything about *how to fight* a type. `dungeon/behaviors.py::KIND_POLICY`
  owns policy (`preferred_distance`, `alive_rule`, `projectile_aware`,
  `off_wall_only`, `whistle_then_sword`) and stays the seam 17 modules import.
  `species.py` holds only ROM fact. Where the two touch —
  `KIND_POLICY.alive_rule == TYPE` ⇔ ROM `hp == 0`, and
  `whistle_then_sword` ⇔ attribute `$20` — `tests/test_species.py` asserts the
  agreement instead of storing the value twice.
