---
name: sm-no-assist-boss
description: >
  Build Super Metroid no-assist (Clean) boss policies from human tape.
  Measure hit geometry before writing a loop; do not treat assisted spray as
  proof of damage. Use when the user says "no assist", "clean boss", "spore
  spawn policy", "farm droplets", "look at human tape" for a boss fight, or
  runs /sm-no-assist-boss.
---

# SM no-assist boss policy

Work from a **human enter pin**, **assist off**, and **one measured hit**
before a 20k-frame fight. Product residual for the current Spore Spawn loop:
`snes/super_metroid/docs/tasks/SPORE_NOASSIST.md`.

## Probe first

```bash
# Idle dump: room, seat xy, enemy0 x/y/hp/spritemap, pickups
uv run python snes/super_metroid/scripts/probe/<boss>_combat.py dump --frames 300

# Fight (default no assist)
uv run python snes/super_metroid/scripts/probe/<boss>_combat.py strategy \
  --report snes/super_metroid/scratch/<boss>_noassist.json
```

Mirror `scripts/probe/phantoon_combat.py` / `spore_spawn_combat.py` for a new
boss. `UnlimitedResourcesAssist(unlimited_energy=False, unlimited_ammo=False)`.

## Human tape is a seat, not a hitbox

1. Extract the hop (`full_start_v1` tape `trace` rows for the boss room).
2. Use morph/stand **mode** of `x/y/pose` as the **safe seat**.
3. Ignore spray `X` frames under assist — missiles never drop, so you cannot
   see hits. Confirm damage only via `enemy0_hp` delta.

## One-window loop (do this before a full fight)

1. `_go_to_seat` from the enter pin. Print seated xy + health.
2. Idle until the vuln condition (spritemap / phase). Log enemy `x/y/spritemap`
   for every distinct map — **open windows may only exist in one corner of
   the room**.
3. Fire **one** window with a per-frame log: samus xy/pose, enemy xy/sm,
   `missiles`, `enemy0_hp`.
4. Count shots by **`missiles` decreasing**, not by pressing `X` (spin/crouch
   eats the input).
5. Do not start a 20k-frame run until that window chips HP.

## Geometry rules that keep showing up

- **Shell / invuln facing:** a "safe corner" can be the wrong angle. Close
  to the vulnerable point (measured hit xy + pose) and aim the way that hit.
- **Jump into a wall (`x<50` + `A`)** in a tall pit **climbs out** (Spore:
  `y≈150`, death). Walk off the wall before any jump; hop only from a
  mid-ledge x band.
- **`select_weapon` at 0 ammo raises** — game forces beam. Farm first, then
  select.
- **Pickup RAM is not "18 bytes at `$19BB` = 0x16/0x18"** until you watch a
  frame where `missiles` actually increases. Spore bouncing spores showed
  only projectile **ID 2** at that table.

## Policy shape

```
seat (morph / kite band) → wait for measured vuln →
close to measured hit pose → spend N shots (ammo delta) →
retreat to seat → if ammo < N, farm without standing under the body
```

Keep the assisted continuous controller untouched until this policy is green
on the natural enter pin with **zero** resource writes. Then swap the fight
body and re-verify the assisted tip.

## Tests

Unit-test seat / mouth-open / "don't fire at 0 ammo" without the emulator.
Emulator proof is the probe report (`success`, `final.enemy0_hp`,
`assist.energy_writes == 0`, `assist.missile_writes == 0`).
