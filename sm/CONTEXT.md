# Super Metroid Lua / lsnes side thread

In-emulator Clean KPDR through **Gravity Suit**. Lives under `sm/`.
Not the Python gym in `snes/super_metroid/`. Not the `../sm_ceres`
authoring tree.

Tip of this thread: power-on → Gravity collect (`0xCE40`, items bit
`GRAVITY_MASK=0x0020`). Ceres is the prefix, not the product.

## Language

**Side thread**:
This folder. Lua 5.1 that lsnes loads. Buttons + RAM reads. Clean:
no energy/ammo/item/boss/door writes.

**Product gym**:
`snes/super_metroid/` on stable-retro snes9x. Untouched by this thread.

**Authoring reference**:
`../sm_ceres` (sibling of this repo). Read-only. lsnes rr2-β25 + bsnes
v085 Compatibility, `input.joyset`, mailbox Lua. Do not add files there.

**Tape**:
SNES-12 button names in order `B Y select start up down left right A X L R`.
Same as lsnes P1 and the gym.

**Session**:
A Lua coroutine. `session:step(names, reason)` yields one frame of
buttons; lsnes `on_input` applies them via `input.joyset(1, ...)`.
After the frame, `session.state` is a fresh WRAM snapshot.

## Layout

```
sm/
  CONTEXT.md
  run.sh                 # native lsnes ELF + --lua=sm/lua/run_gravity.lua
  lua/
    path.lua             # package.path for this tree
    ram.lua              # WRAM snapshot (same offsets as ram.py)
    input.lua            # names <-> joyset table
    runtime.lua          # coroutine session, hold, wait_until
    rooms.lua            # room ids (full KPDR through Gravity)
    enemies.lua          # slot scan + Ceres steam/door + WS species
    takeoff.lua          # TakeoffWindow / PlatformHop / arm-pump
    skills/              # knockback, moonfall, walljump, shinespark,
                         # charge_shot, morph_bomb, door, runway
    ceres/               # boot → Landing
    morph/               # Landing → Morph (seeds + parlor/climb moonfall)
    bombs/               # Morph → Bomb Torizo exit
    brinstar/            # Terminator → Spore → Supers → Pink → GHZ → Red
    red_tower/           # Red → Bat → Spazer/Warehouse
    kraid/               # Hi-Jump, Kraid, Varia, return
    norfair/             # Cathedral, Speed, Wave, Ice
    wrecked_ship/        # Moat, WS, Phantoon, Attic→Bowling→Gravity
    combat/              # ridley, bomb_torizo, spore, kraid, phantoon
    spine.lua            # compose power-on → gravity
    run_ceres.lua        # prefix entry (Landing)
    run_gravity.lua      # product entry (Gravity collect)
```

## Rules

1. Lua 5.1 only. No Python files under `sm/`.
2. Do not write energy, ammo, items, bosses, doors, pose, or coordinates.
   Moonwalk `$09E4` is a file option; Ceres 1 may poke it on, then off
   before the first door, same as the Python Clean path.
3. Do not STATUS-promote. Do not touch `snes/super_metroid/docs/STATUS.md`.
4. Do not edit `../sm_ceres`.
5. Leave proof is RAM: room, game state, xy, pose, health, items,
   timer_type. Prefix Landing is `0x91F8` gs=8. Product Gravity is
   `0xCE40` gs=8 with `collected_items & 0x0020`. Write
   `sm/recordings/<tip>_leave.json`. Never overwrite assisted
   `snes/super_metroid/recordings/*.json`.
6. Hop side is D-pad `LEFT`/`RIGHT`. Shoulders are `L`/`R`.
7. Soft max ~1000 LOC per file. Merge into the spine composer, no sibling
   extract.

## Boot

Native ELF: `~/.local/opt/lsnes-rr2-beta25/lsnes`. ROM:
`roms/SuperMetroid.sfc` (SHA-256 `12b77c4b…`). Launch with `--rom-a=`
and `--lua=` pointing at `sm/lua/run_gravity.lua` (or `run_ceres.lua`
for the prefix). No startup movie: this thread *scripts* the run.

## Source of truth for the port

Python controllers (read, do not copy into this tree as `.py`):

- Ceres prefix: `../sm_ceres/super_metroid/` (lsnes Lua reference too)
- Morph→Gravity: `snes/super_metroid/routes/kpdr/` and `combat/`
  (this repo). Spine order: `routes/kpdr/spine_hops.py` +
  `post_ice_spine.py`. Gravity leave: `wrecked_ship/gravity_collect.py`.
- Hash-pinned button seeds (`policies/morph/*.json`, hop JSON bodies)
  become Lua tables under the matching `sm/lua/` package. Do not keep
  a Python loader.
- lsnes joyset names: `../sm_ceres/super_metroid/tas/oracle/lsnes_worker.lua`
