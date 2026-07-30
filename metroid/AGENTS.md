# Agent Instructions — metroid

Scripted NES completion agent for **Metroid** (graph_navigation /
metroidvania track; maturity **M5** — boot → Maru Mari; first missiles WIP).

## Identity

| Field | Value |
|-------|-------|
| Status | natural first-missiles prefix; upper west shaft frontier |
| Integration | `Metroid-Nes` |
| Shared ROM zip | `roms/Nintendo/NES/Metroid.zip` |
| Local ROM | `metroid/roms/` (via `scripts/setup_rom.py`) |

## Commands

```bash
uv run python metroid/scripts/setup_rom.py
uv run python metroid/scripts/boot_probe.py
uv run python metroid/scripts/run_morph_ball.py              # isolated Level1
uv run python metroid/scripts/run_morph_ball.py --natural-entry
uv run python metroid/scripts/run_first_missiles.py         # AfterMorph diagnostic
uv run python metroid/scripts/run_first_missiles.py --from-level1
uv run python metroid/scripts/run_first_missiles.py --natural-entry
uv run python metroid/scripts/run_first_missiles.py --natural-entry --screen-timing
# Screen hop timing (emulator frames; self-check/offline need no ROM)
uv run python metroid/scripts/probe_screen_timer.py self-check
uv run python metroid/scripts/probe_screen_timer.py offline \
  -i metroid/tests/fixtures/screen_timer_sample.json
uv run pytest metroid/tests adventure_common/tests -q
```

## Layout

| Path | Role |
|------|------|
| `ram.py` | System RAM + WRAM equipment snapshots |
| `screen_timer.py` | Map-cell hop timing (emulator frames; optional observer) |
| `screen_timing_session.py` | Opt-in runner observer + bottleneck summary |
| `brinstar.py` | Early map graph (start → morph; east probe; planned missiles) |
| `routes.py` | Named routes / milestones |
| `morph_ball.py` | Morph segment controller (verified) |
| `first_missiles.py` | Natural missiles prefix through west-shaft frontier |
| `adventure_common/` | Shared capability-aware `RouteGraph` (repo root) |
| `docs/SCREEN_TIMER.md` | Screen-timer semantics / limitations |

## Traps

- Morph is **left** of start (map 3,14 → 1,14), not right.
- Equipment lives in cart WRAM `$6878` — use `env.data.memory.extract`, not
  `get_ram()` (only 2 KiB system RAM). Missiles capacity is `$687A`.
- Do not mash START after play begins (pauses the game).
- West corridor needs LEFT+A climb spans; floor-only LEFT sticks at x≈4.
- After morph, return through the low ball tunnel; map (3,14), x≈76/y=200 is
  the unmorph point before jumping into the real skree room.
- AfterMorph often starts in item fanfare (`game_mode == 9`); idle until mode 3.
- The three east doors are elevated. Stop each transition on the first
  controllable target-map frame or the downstream enemy timing changes.

## Next milestone

Morph → first missiles (capacity `$687A` > 0), natural-entry. Continue from
the verified map (11,13), x≈106/y=225 west-shaft platform through the bridge
and east shaft.

## Norms

- Prefer nearest local docs (`docs/STATUS.md`, `docs/plan.md`) over root notes.
- Keep RAM maps, save states, and policies inside this game directory.
- NES actions use `retro_harness.nes` (9-button fceumm layout).
- Line length 88; type hints; `uv run pytest` for tests.
