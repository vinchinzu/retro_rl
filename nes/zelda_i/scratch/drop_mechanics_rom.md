# Zelda I drop / kill-streak mechanics (ROM primary sources)

Question: on a 0x77→0x4A OW walk, 14 slot-kills, 0 rupees, `$50`/`$627` peaked at 4 and went to 0 six times, 0 damage on `$066F`+`$0670`. `ram.py` “a hit clears all three” is incomplete. **What actually writes 0 to `$0050` and `$0627`?**

**Sources.** `zeldaret/zelda1` does not exist. Matching decomp is [aldonunez/zelda1-disassembly](https://github.com/aldonunez/zelda1-disassembly) (`Variables.inc`, `Z_01.asm`, `Z_04.asm`, `Z_07.asm`, `Z_05.asm`). Prose twin: [Red Candle / Technical Information § Forced drops](https://redcandle.us/Legend_of_Zelda/Technical_Information). [Qwertymodo/Zelda1-Disassembly](https://github.com/Qwertymodo/Zelda1-Disassembly) not found (Red Candle’s other listing is [camthesaxman/zeldasource](https://github.com/camthesaxman/zeldasource)). Data Crystal RAM map 403’d this sitting; older revisions label `$0627` “killed enemy count / current screen” (wrong) and leave `$0050` unmarked.

Names (`Variables.inc`): `HelpDropCount=$50`, `HelpDropValue=$51`, `WorldKillCount=$627`, `WorldKillCycle=$52A` (0–9 table column, **not** the streak).

---

## 1. Every store to `$50` / `$51` / `$627`

### Verified

| Where | What | URL |
|---|---|---|
| **`Link_BeHarmed`** (`Z_01.asm` ~L5759–5764) | `LDA #0` → **STA `$627`, `$50`, `$51`** | [Z_01.asm#L5759](https://github.com/aldonunez/zelda1-disassembly/blob/master/src/Z_01.asm#L5759) |
| **`HandleMonsterDied`** (`Z_01.asm` ~L5997–6015) | `INC $627`; `INC $50` if `<$0A`; if that INC hits `$0A` **and** damage type = bomb (`$09==$08`) then `INC $51` | [Z_01.asm#L5997](https://github.com/aldonunez/zelda1-disassembly/blob/master/src/Z_01.asm#L5997) |
| **`SetUpDroppedItem` `@SetHelpItem`** (`Z_04.asm` ~L11166–11173) | On **any forced drop** (fairy **or** 10-kill rupee/bomb): `STA $50`, `STA $51` **only**. **`$627` is not written.** | [Z_04.asm#L11166](https://github.com/aldonunez/zelda1-disassembly/blob/master/src/Z_04.asm#L11166) |
| **`Dodongo_CheckCollisions` `@Die`** (`Z_04.asm` ~L6046–6050) | `LDA #$0A` → STA `$50` and `$51` (force bomb). No `$627`. | [Z_04.asm#L6046](https://github.com/aldonunez/zelda1-disassembly/blob/master/src/Z_04.asm#L6046) |
| **`ClearRam`** (`Z_05.asm` ~L7440–7454) | ZP `$00–$EF` (includes `$50`/`$51`) **and** `$0300–$07FE` (includes `$627`). Called from `RunGame` boot only (`Z_07.asm` ~L354). | [Z_05.asm#L7440](https://github.com/aldonunez/zelda1-disassembly/blob/master/src/Z_05.asm#L7440) |

`Link_BeHarmed` callers: **`HarmLink`** (`CheckLinkCollision`, type `<$53` or unparried shot) and candle **fire** (`Z_07.asm` ~L4756–4760, always `$80` partial). Red Candle: also **bubbles** and **recorder whirlwind** (0-damage types `$2B–$2E` still take the `HarmLink` path).

**Does not write `$50`/`$627` (verified by absence of STA):** room/screen load (`InitMode_EnterRoom` clears `$0300–$051F` only — that **does** zero `WorldKillCycle=$52A` and `RoomKillCount`, not the streak); cave enter/exit; death `InitMode11` (zeros `HeartPartial`, not streak); stun / Link i-frames / clock (`CheckLinkCollision` **returns before** `HarmLink`); `DestroyMonster` (no INC); ringleader mass-die (`ObjMetastate=$10`, skips `HandleMonsterDied`).

GitHub code search: those are the **only** `STA HelpDrop*` / `STA WorldKillCount` sites in the repo.

---

## 2. Increment rules

### Verified

Increment is **`HandleMonsterDied`**, reached when weapon damage `>=` HP (`DealDamage`). Then `UpdateDeadDummy` (death cry + metastate `$10` spark). Drop + `WorldKillCycle` happen **later** in **`UpdateMetaObjectEnd`** (`Z_07.asm` ~L5441–5488) when spark finishes (`metastate=$14`).

**`WorldKillCycle` skip** (still may INC `$50`/`$627` at kill time): types `$5D`, `$14` child Gel, `$1C` red Keese. **`RoomKillCount` skip:** Zora `$11` (still increments streak + cycle).

**No-drop types** (`NoDropMonsterTypes`, `Z_04.asm` ~L11031): `$5D`, `$14/$15` Gel, `$1B/$1C/$1D` Keese, `$17` Like-Like. They never run drop code, so they **cannot consume** a forced drop (counters stay). Stalfos `$2A` / Gibdo `$30` in **slot 1** also skip drop.

**Projectiles** (`>=$53`): `DestroyMonsterShot` → `DestroyMonster`, **no** `HandleMonsterDied`. **Zora submerge** (`UpdateZora` state 0 → `DestroyMonster`): same. **Peahat:** only `CheckMonsterCollisions` (can die) in flyer state 5; other states `CheckLinkCollision` only (can **HarmLink**, cannot die). **Armos `$1E`:** not an object until touched; then a normal kill. **Scroll despawn:** objects are wiped without `HandleMonsterDied` — **does not** count. Hunt slot-census **will** count Zora-submerge / scroll / shot death; `$50`/`$627` will not.

---

## 3. Forced-drop thresholds / wrap

### Verified (`SetUpDroppedItem` ~L11146–11175)

- `$627==$10` (16) → force fairy `$23`, then **zero `$50`/`$51` only**. `$627` **keeps 16** (can force more fairies on later drops this streak; wraps at 256).
- Else if `$50>=$0A` (10) → force 5-rupee `$0F` if `$51==0`, else bomb `$00`; then zero `$50`/`$51`.
- `$50` **holds at 10** (`HandleMonsterDied` `CMP #$0A / BCS`). `$627` has **no** cap besides 8-bit wrap.
- A hunt that treats `$627: n→0` as a “reset” is **not** seeing this wrap (wrap does not zero `$627`). Peak 4 never hits 10 or 16.

---

## 4. Random drop tables (groups A–D)

### Verified (`Z_04.asm` ~L11034–11182)

Rows chosen by ObjType lists; **else row 3**. Item IDs: `$00` bomb, `$0F` 5-rupee, `$18` rupee, `$21` clock, `$22` heart, `$23` fairy.

`WorldKillCycle` is **INC’d (0→9 wrap) before lookup**, so kill 1 uses column 1.

| ROM row | Baxter name | rate | 10-slot (cols 0–9) | ObjTypes (this walk) |
|---|---|---|---|---|
| 0 `Types0` | **A** | `$50`=80/256 **31%** | `22 18 22 18 23 18 22 22 18 18` | `$07/$08` red Oct, `$0E` blue Tektite, `$04` red Moblin, `$0F` blue Leever |
| 1 `Types1` | **B** | `$98`=152/256 **59%** | `0F 18 22 18 0F 22 21 18 18 18` | `$0D` red Tektite, `$10` red Leever, `$21/$22` Ghini, `$13` Zol, `$28` Rope, `$2A` Stalfos |
| 2 `Types2` | **C** | `$68`=104/256 **41%** | `22 00 18 21 18 22 00 18 00 22` | `$09/$0A` blue Oct, `$03` blue Moblin, `$01` blue Lynel |
| 3 rest | **D** | `$68`=104/256 **41%** | `22 22 23 18 22 23 22 22 22 18` | **`$1A` Peahat**, **`$1E` Armos**, `$02` red Lynel, Zora `$11` if it actually dies |

**CLEANUP_PLAN 4.5.1 “group B”** = Baxter **B** = ROM row 1 (two 5-rupees / 10-cycle, highest rate). Red Candle lists B=41% / C=59% — **rates swapped vs ROM**; table *contents* match Baxter. Chance: `Random,X >= DropItemRates[row]` cancels the drop (`@RandomlyCancel`).

---

## 5. `WorldFlags[$067F+screen]` kill bits

### Verified

**OW** uses bits **`$07`** (0–7), not `$C0` (`SaveKillCountOW` / `ModifyObjCountByHistoryOW`, `Z_05.asm` ~L3537–3605). **UW** uses **`$C0`** (`SaveKillCountUW` ~L4063–4118). Local `WORLD_FLAG_KILLS=0xC0` is UW-only.

2+ kills: subtract from next spawn count (history + flags). At max, room can suppress or fully respawn (OW: flags==7 and **not** in 6-room history → **clear** the kill bits and respawn). **Does not** read or write `$50`/`$627`. **Does not** change drop rates.

---

## 6. Do `$50` and `$627` always move together?

### Verified: **no**

Together: every `HandleMonsterDied` until `$50` hits 10; `Link_BeHarmed` / `ClearRam` zero **both**.

Diverge: `$50` caps at 10 while `$627` climbs; any forced drop zeros **`$50`/`$51` only**; Dodongo writes `$50`/`$51`:=10; `$627` 8-bit wrap. A live walk that never hits 10 and never `HarmLink`s will look lockstep.

---

## What would reset a streak 6 times on a 0-damage 6-screen walk

Ranked, each falsifiable. ROM fact: **the only gameplay `STA 0` of `$627` is `Link_BeHarmed`.** Forced wrap and screen load do **not** zero `$627`. Hunt `streak_resets` is `$627: n→0`.

1. **`Link_BeHarmed` ran 6 times; the heart census missed it.** Falsify: log `$066F`/`$0670` **on the same frame** `$627` falls. A `$80` partial (`$FF→$7F`) is a rock/octorok/peahat/fire hit and **must** show if sampled that frame (including mode 6/7 scroll — hunt currently observes play-mode only).
2. **0-damage `HarmLink`** (bubble `$2B–$2D`, whirlwind `$2E`: `ObjTypeToDamagePoints=$00`). Hearts unchanged, both counters zeroed. Falsify: no such `ObjType` on 0x77–0x4A; dump collider type at reset.
3. **Watching `$52A` by mistake.** `InitMode_EnterRoom` zeros `$0300–$051F` **every screen** (6 screens → 6 wraps of the 0–9 column, peak ≤ kills-on-that-screen). Falsify: `$50` would **not** fall with it; confirm the probe address is `$0627` not `$052A`.
4. **Forced-drop wrap** — **reject** unless peak was 10/16 and only `$50` fell.
5. **Slot despawn (Zora submerge, scroll, Peahat not in state 5)** — explains 14 hunt kills vs ~13 counter INCs; **does not** write 0 to `$627`.
6. **Stun / i-frames / clock / cave / death / `WorldFlags`** — **reject** as writers of 0.

**Bottom line:** `ram.py` “hit clears all three” is the **only** in-game zero of `$627`, and it runs even when damage bytes are 0. It is **not** the only zero of `$50` (forced drop also). Six `$627` resets on that walk are six `Link_BeHarmed` calls unless the probe is on the wrong byte.
