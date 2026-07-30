# SMZ3 combo RAM map

Verified against test seed 1337 combo ROM under stable-retro `SMZ3-Snes`
(2026-07-29). Vanilla SM and Z3 offsets apply **while that world owns WRAM**.

## World ownership

| Source | Address | Encoding |
|--------|---------|----------|
| Combo SRAM flag (upstream) | bus `$A1:73FE` (`!SRAM_CURRENT_GAME`) | 8-bit NMI: `0` = ALTTP, negative (`$80–$FF`) = SM, positive nonzero = credits |
| Reset default | stores `#$00FF` | Always boots into Super Metroid |
| stable-retro today | **not mapped** in `get_ram()` / `memory.blocks` | Use WRAM heuristic in `smz3.world.detect_world` |

Upstream reference: tewtal `alttp_sm_combo_randomizer_rom` `src/sram.asm` +
`src/common.asm`.

## Super Metroid (active world)

Same as `super_metroid/docs/ram_map.md` for low WRAM. Confirmed on combo:

| Field | Offset | Notes |
|-------|--------|-------|
| `game_state` | `$0998` | `8` = ordinary controllable |
| `room_id` | `$079B` | Landing Site `0x91F8` after fresh file |
| `area_index` | `$079F` | `0` = Crateria at start |
| `door_transition` | `$0797` | `0` when settled |
| `health` | `$09C2` | `99` at start on test seed |
| `samus_x` / `samus_y` | `$0AF6` / `$0AFA` | ~`(1152, 1088)` ship start |

`super_metroid.ram.parse_state` works on combo `get_ram()` while SM is active.

## ALttP (active world)

Vanilla module/submodule/Link offsets (`alttp.ram`) apply after a portal.

| Field | Offset | Role |
|-------|--------|------|
| `module` | `$10` | `0x07` dungeon / `0x09` overworld when controllable; `$0F` seen mid-portal |
| `submodule` | `$11` | `0` when Link has control |
| `link_y` / `link_x` | `$20` / `$22` | |
| `room_id` | `$A0` | Indoors; Crateria map portal leaves cave `$0122` |

While SM owns WRAM these bytes are **not** Z3 state (often garbage).

Partial portal residue observed (dev door assist, not settled): module `$0F`,
room `$0122` (Fortune Teller), world detect → `ALTTP`. Controllable Link not
yet verified — see `docs/EARLY_ROOMS.md`.

## Detection policy (`smz3.world`)

1. `sm_controllable` (`game_state==8`, door `0`, room≠0, hp>0) → SM  
2. Known SM menu `game_state` → MENU  
3. Known SM engine state + plausible room pointer → SM  
4. Else if Z3 active module and SM state not in SM engine set → ALTTP  
5. Else UNKNOWN  

## Memory blocks (stable-retro snes9x)

| Block base | Size | Role |
|------------|------|------|
| `0x0` | 8 KiB | Low WRAM mirror (`get_ram()[:0x2000]`) |
| `0x7E0000` | 128 KiB | Bank `$7E` WRAM |
| `0x206000` | 8 KiB | Cart save-ish (not full combo SRAM) |
