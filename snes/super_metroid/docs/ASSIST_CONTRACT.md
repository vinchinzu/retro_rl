# Assist contract — unlimited health/energy and ammo

This project intentionally uses disclosed survival/resource assists. The goal
is reliable navigation, item routing, room transitions, bosses, and the
endgame—not resource conservation.

Three explicit intervention profiles. Mixed evidence is rejected.

| Profile | Runtime use | Allowed writes | Claim |
|---------|-------------|----------------|-------|
| `clean` | Later privilege removal | none | Existing Clean rules |
| `survival` | Primary product | current energy + naturally unlocked ammo only | Eligible for living Tip after normal gates |
| `scaffold` | Fast first bot credits chain | Survival plus an allowlisted live-enemy HP clamp | Development-only, never STATUS/Finish |

## Allowed writes

### Unlimited energy

- During ordinary controllable gameplay, restore current energy to the
  naturally available maximum (see **refill policy** below).
- Do not increase maximum energy or grant Energy Tanks.
- Observe and count natural damage before restoring it.
- Do not revive a completed death transition.
- Suspend or specialize the refill during cutscenes and scripted sequences
  whose progression depends on energy values.
- Suspend the refill throughout Ceres ordinary gameplay: Ridley's natural
  damage must reach the evacuation-countdown threshold. Energy refill begins
  only after the run reaches Zebes.
- Suspend the refill during energy-drain scripts:
  - Metroid latch (Big Boy): pose `$E8`, movement `$15`/`$1B`, or baby-Metroid
    (HP `$7FFF`) stuck within ~48px of Samus.
  - Mother Brain rainbow / stun: pose `$54` (rainbow drain), `$E9`/`$EB`,
    movement `$0A` (plus latch poses above).
  Refill mid-drain softlocks gray-door release and the MB baby-Metroid cutscene.

### Unlimited ammo

- Refill only ammo types that have been naturally unlocked.
- Refill current ammo only up to its naturally collected capacity
  (see **refill policy** below).
- A zero capacity means the ammo type is still locked and must remain zero.
- Do not grant Missiles, Super Missiles, Power Bombs, or capacity upgrades.

### Refill policy (`refill_when`)

Two policies share the same write guards; only the **when** differs:

| Policy | Behavior | Default for |
|--------|----------|-------------|
| `always` | Restore whenever current &lt; capacity (energy: while still alive) | Continuous / product bots |
| `at_zero` | Ammo: only at **0**. Energy: at **0** or ≤ floor (40) — death-save before death phase latches | Human `./play` practice |

Energy under `at_zero` uses a low floor (`AT_ZERO_ENERGY_FLOOR = 40`), not
only exact zero. Boss one-shots (e.g. Phantoon flame, GT 40-chip) often jump
ordinary → death game-state in one tick without a usable `health==0` ordinary
frame, so a pure zero energy policy never tops up (`energy_restored=0` while
ammo still works). Death / game-over phase still never revives a completed
death transition.

Metroid-latch / MB-rainbow energy suspend is **Tourian-only** (plus baby-
Metroid proximity). Movement `$0A` is ordinary knockback in LN/GT — matching
it globally holds the refill for 180f after every hit and looks like the
health boost is off.

Telemetry per resource: `restored`, `writes`, and **`top_ups`** (discrete
empty/floor→full events). Practice goal: drive `top_ups` down by taking less
damage and firing fewer shots / farming less. Human record: `./play` uses
`at_zero`; `--assist-full` restores the continuous-style always top-up;
`--no-assist` is fully clean.

The implementation should be a separate assist controller, not scattered
`set_value` calls in route policy.

### File options (moonwalk)

Moonwalk is a Special Setting Mode flag (`$09E4`, 1 = on), off on every new
file. It is **not** energy, ammo, items, or pose. `ram.set_moonwalk` may
write that word so moonfall (Climb first descent, later shafts) can run
without menuing Option Mode. Telemetry must not count it as
`progression_writes` or `capacity_writes`.

On a seed-backed prefix, poke **on** only for the moonfall hop and poke
**off** before the next hash-pinned seed (those tapes were recorded with
moonwalk off). A full-run moonwalk-on rewrite is a later sitting.

Menuing Special Setting Mode at boot is the buttons-only equivalent; it
costs real time and is not required while the poke is contracted.

## Forbidden writes

- item/equipment ownership
- ammo or energy capacity
- collected-item bits
- boss/event flags
- door, room, area, map, or elevator state
- player position, velocity, pose, or movement ability
- timers
- save-file completion state

If a new write is needed, update this contract before using it.

## Phase guard

The assist controller must distinguish at least:

- ordinary controllable gameplay
- room/door/elevator transition
- pause/inventory/menu
- cutscene/scripted sequence
- death/game over
- ending/credits

Default behavior outside ordinary gameplay is no write. Add a phase exception
only after a focused probe proves it preserves natural progression.

## Required telemetry

The full-run report records:

- total energy restored
- energy write count and affected frames
- ammo restored and write count per ammo type
- first frame each ammo type became naturally available
- any phase in which assists were suspended
- maximum single-frame damage
- deaths and game-over entries
- forbidden/progression writes (must be zero)
- scaffold HP-clamp writes and counts by room/entity (empty on Survival)

## Integrity assertions

A successful assisted clear must show:

- continuous emulator session from the declared start
- no state loads after power-on
- natural inventory/capacity acquisition
- no progression writes
- natural room and boss/event advancement
- natural endgame escape and ending/credits evidence

## Clean mode (parallel track)

**Clean** means both unlimited energy and unlimited ammo are **off**: zero
resource restores and zero resource writes. Observation may still be Bronze
(read-only RAM). Clean is a **parallel** privilege-reduction workstream; it
does not replace this assisted contract or the primary KPDR continuous tip.

Rules, artifact isolation (`*_clean` stems), tickets, and the Bomb Torizo
clean tip ladder: [`CLEAN_TRACK.md`](CLEAN_TRACK.md).

Hard constraints:

- Default continuous CLI remains resource-assisted.
- Clean runs must not overwrite assisted `recordings/<tip>.json` / videos.
- STATUS primary program gate stays assisted until an explicit program decision
  changes it; Clean results are documented as a secondary track.

## Scaffold mode (development only)

**Scaffold** is Survival plus an allowlisted live-enemy HP clamp. Conservatively
label it **Bronze / Progression-assisted development**. It is a splice and
traversal unblocking tool. It is **never** STATUS, Finish, or the living Tip.

The clamp is not a generic `enemy0_hp = 0` switch:

- Allowlist by room, species/boss, optional spawn state, and optional phase.
  Unknown `(room, species)` pairs are never written. An empty allowlist writes
  no HP.
- Change a live target from positive HP to **1**, once per eligible phase, so a
  real controller hit triggers the game's death / phase / event logic. Never
  write HP to 0.
- Scan every enemy slot. Multi-phase bosses need an explicit phase on the
  allowlist row; a new phase may be clamped once. Fail closed for unknown
  layouts. Suspend during non-ordinary gameplay (door, death, cutscene).
- Log every write as `{frame, room_id, slot, enemy_id, old, new, reason}` and
  expose counts by room/entity. These writes are **not**
  `progression_writes` or `capacity_writes`.
- Removable per task (`enabled=False` default). Product
  `UnlimitedResourcesAssist` does not enable the clamp unless constructed with
  `profile="scaffold"` or an explicit allowlist.

The first ordinary-enemy pilot is Attic `0xCA52` (gray-door kill-all). The
factory species is a documented placeholder until a ROM-backed header id is
recorded; unknown species in Attic still fail closed.

Use the clamp to unblock traversal and validate splicing, not to skip doors,
items, boss flags, escape, or credits. A room whose phase logic breaks under
the clamp gets a normal reactive boss Skill immediately.

Item, door, boss, event, room, timer, position, and capacity writes remain
**forbidden**. Survival refill rules still apply. Scaffold evidence must not
be claimed as Survival or Finish.
