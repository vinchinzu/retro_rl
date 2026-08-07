# Assist contract — unlimited health (first-pass routing)

Zelda I uses a **disclosed Survival assist** for first-pass dungeon/overworld
mapping. The goal is reliable navigation, room transitions, keys, bosses, and
route graph evidence—not heart conservation.

**Default runners stay Clean** (no RAM writes). Enable only via
`--infinite-life` / `UnlimitedHealthAssist(enabled=True)`.

Any published assisted result must record this contract path and the assist
telemetry block in the run report.

## Allowed writes

### Unlimited health (hearts)

- During ordinary controllable gameplay (mode 5 overworld/dungeon play, or
  cave mode 11), restore **filled hearts** to the natural full sentinel
  while **preserving the high nibble** (heart containers − 1).
- Implementation: `health = (health & 0xF0) | 0x0F` via `data.set_value("health", …)`.
- Do **not** increase heart containers.
- Observe natural damage before restoring; count writes / restored units.
- Do not revive a completed death transition (mode 17): suspend writes.
- Suspend during boot/menu modes, scroll transitions (6/7/16), and Triforce
  fanfare (mode 18).

The implementation is `zelda_i.assist.UnlimitedHealthAssist`, applied from
`chain.run_controller_stage` / probe loops—not scattered policy writes.

## Forbidden writes

- sword, bombs, keys, rupees, arrows, inventory items
- triforce / dungeon completion bits
- room, screen, door, object, or map state
- Link position, facing, or mode
- heart **containers** (high nibble of `ADDR_HEALTH`)
- timers / dialog counters
- save-file completion state

If a new write is needed, update this contract before using it.

## Phase guard

| Phase | Mode cues | Assist |
|-------|-----------|--------|
| ordinary_gameplay | mode 5 or cave 11, not dead | refill |
| transition | 6 / 7 / 16 | suspend |
| triforce_fanfare | 18 | suspend |
| death | 17 | suspend; count death entry |
| menu_or_boot | 0–4 | suspend |

## Required telemetry

Assisted reports include:

- `assist.enabled`, write count, hearts restored
- first frame assist became active
- maximum single-frame damage observed before refill
- death entries
- suspended phase frame counts
- `progression_writes` / `capacity_writes` (must stay 0)

## Integrity assertions (assisted clear)

- continuous emulator session from declared start
- no state loads after power-on (for natural-entry claims)
- natural inventory / triforce acquisition
- no progression or capacity writes
- natural room and boss advancement

## Dual track (Clean vs assisted)

| Track | Stem / flag | Claims |
|-------|-------------|--------|
| Clean | default CLI | may promote STATUS Clean / M5+ |
| Survival-assisted | `--infinite-life`, `*_assisted` artifacts | first-pass geometry + route graph only |

Do not mix assisted greens into Clean STATUS rows. Prefer SM-style dual-track
stems when both exist.

## CLI

```bash
# First-pass: door path + Moon entry without heart farm
uv run python zelda_i/scripts/probe_level2_suffix.py \
  --infinite-life --enter-dungeon --tag l2_assist

uv run python zelda_i/scripts/run_to_level2_prefix.py --infinite-life --trials 1
```
