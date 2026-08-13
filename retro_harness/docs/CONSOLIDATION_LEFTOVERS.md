# retro_harness consolidation leftovers

Live architecture: [TOOLSET.md](TOOLSET.md). Hygiene board: [docs/REPO_HYGIENE.md](../../docs/REPO_HYGIENE.md).

## Fix wave 1 (2026-08-04) — complete

| Item | Status |
|------|--------|
| Extract FF edge/patient policy out of shared `combat.fight_nearest_action` | **done** — `snes/final_fight/edge_combat.py` |
| Split `platformer/runner.py` + `route.py` | **done** — `platformer/cli/*`, `platformer/route/*` |
| Rename `video.RecordingSession` → `CaptureSession` | **done** |
| Move SM platformer levels under `snes/super_metroid/` | **done** |
| Document Task/`WorldState` vs BT/`GameState` dual stack | **done** |

## Fix wave 2 (2026-08-04) — complete

| Item | Status |
|------|--------|
| Move remaining platformer levels out of harness (DKC/SMB/SMB3/SMW) | **done** — game `platformer_levels.py` + thin shims only |
| Ladder from `docs/manifests` `setup:` (no hardcoded game list) | **done** — `ladder.load_ladder()` |
| Purge SMB RAM maps from neuro; split neuro package | **done** — `platformer/neuro/*`; obs via `smb.obs` |
| Move `SMB_BOTTLENECK_WINDOWS` to `nes/smb/rle_windows.py` | **done** |
| Typed `GameState.go_flashing` / `area_clear` | **done** |
| `player_is_down(..., max_live_health=)` | **done** — default 128 beat-em-up; `None` for platformers |
| Hash helper dedup (SM runtime, zelda dungeon_trace) | **done** — `adventure.hashutil` only |
| Editor registry discovery (no hardcoded earthbound) | **done** — `*/editor_registration.py` |
| Remove dead `super_metroid_rl` / `super_mario_bros` imports from platformer CLI | **done** — game-owned renderer hook |

**Base-folder hygiene (this wave):**

- Repo root has no loose game packages (only `snes/`, `nes/`, `retro_harness/`, `docs/`, `roms/`).
- `retro_harness/platformer/levels/` is shims only (no RAM maps).
- `retro_harness/ladder.py` has no game slug/title/zip literals.
- No SMB/SM hex RAM addresses under `retro_harness/`.
- Single `sha256_file` implementation in harness (`adventure.hashutil`).
- Fighters/genre multi-game registries remain under `retro_harness/fighters/` (intentional second-consumer package).

## Fix wave 3 (2026-08-04) — complete

| Item | Status |
|------|--------|
| Nested package roots without game-slug literals | **done** — `repo.discover_nested_package_roots()` layout scan; conftest uses `ensure_import_paths`; pyproject pythonpath is only `.`/`snes`/`nes` |
| Slim harvest + hals_golf `AGENTS.md` | **done** — commands/traps/pointers; facts → STATUS |
| hals_golf `docs/STATUS.md` | **done** — gate, menu flow, RAM table, acceptance |
| `get_route` name collision | **done** — `get_platformer_route` / `get_named_route` primary; `get_route` alias |
| Probe PNG gitignore | **done** — `**/probe*.png`, debug_frames, SM/tmnt/FF probe globs |
| Stale `platformer/neuro.py` path refs | **done** — smb AGENTS/plan point at package |

## Fix wave 4 (2026-08-04) — complete

| Item | Status |
|------|--------|
| Slim remaining fat AGENTS | **done** — tmnt_iv, zelda_i, MK, MKII, smb, alttp, smz3 (~40–55 lines) |
| MK / MKII `docs/STATUS.md` | **done** — results + RAM moved out of AGENTS |
| Shared `clean_artifact_stem` / recording paths | **done** — `retro_harness.artifacts`; SM/TMNT re-export |
| Shared `game_paths` layout | **done** — `retro_harness.game_layout`; migrated FF/Waldo/MQ/Rival/Pilotwings/F-Zero/SDD/TMNT |
| harvest/hals `ensure_monorepo_on_path` | **done** — delegates to `repo.ensure_import_paths` after local path insert |

---

## Still open

### P1 — shared size / honesty

1. **`platformer/neuro/train.py` (~642 lines)**  
   Split further only if GA grows (net/train already separated).

2. **`editor/gui_emulator_panel.py`**  
   **partial** — split recording mixin + pure `emulator_loop` / `transcribe` helpers (panel now ~770 LOC). Remaining: Qt session/bridge lifecycle + HUD still panel-local; PlaySession still owns pygame I/O (shares turbo preview helper only).

3. **`adventure` capability aliases are Metroid-shaped**  
   Fine for Metroid-family graphs; do not grow Zelda item aliases into the same map without namespacing.

### P2 — dual systems

4. **Task vs BehaviorNode bridge**  
   Documented in TOOLSET; adapter still optional.

5. **Package root barrel**  
   Root stays old SNES-controls surface; scripted APIs via submodules (TOOLSET). Optional `retro_harness.scripted` facade later.

### P3 — remaining game coupling (acceptable or later)

6. **Nested package *layout* (harvest, hals_golf)**  
   Still discovered automatically. Flattening packages later would empty the discovery set — blocked by workspace name collisions (`tasks/`, `maps/` assets vs package modules). Not policy debt.

7. **Remaining `paths.py` still hand-rolled**  
   NES packs + SM/alttp/smz3/smb still custom (extra dirs/constants). Migrate when next touched via `game_paths`.

8. **Docstring examples** (`PlaySession`, `recorder`, CLI help) still mention DKC as examples — cosmetic.

9. **`fighters/game_configs.py`**  
   Multi-game registry is intentional for the fighters domain. Do not move unless a single-game path is preferred.

10. **TMNT IV custom combat paths**  
    Re-check composition against slim shared melee when polishing TMNT; do not re-introduce FF edge policy into shared.

---

## Out of scope

- Replacing platformer GA/hillclimb algorithms.
- Unifying adventure graphs with SM room topology rewrites.
- Full NES genre packages before a second NES consumer needs them.
- Moving `fighters/` game tables into each fighter game tree (optional later).
- Flattening harvest/hals nested packages (layout collisions).

## How to close an item

1. Prefer deleting a concept over wrapping it.
2. Shared harness must not gain game RAM maps, ROM hashes, or deleted package names (`super_metroid_rl`, `super_mario_bros`).
3. Run the narrowest tests + `uv run pytest tests/test_docs.py -q` when docs/manifests change.
4. Strike the item here when merged.
