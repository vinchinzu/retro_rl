# TASK SM-K4.2-PURE: Frog Speedway → Upper Norfair Farm

## Recipe step

1. Pure controller plus pure-probe registration. Geometry green before any
graph, catalog, continuous tip, or STATUS.

## Model

Luna

## Wave type

implement

## Own files only

- `routes/kpdr/k4_norfair.py` — replace only `play_speedway_to_farm`; leave
  Frog Save→Speedway, Business→Frog, and `play_farm_to_bubble` untouched.
- `routes/kpdr/__init__.py` and `routes/kpdr/registry.py` — export/register
  this one segment if missing.
- `scripts/probe/kpdr.py` — add pure choice `speedway-to-farm` only.
- `tests/test_k4_norfair_scaffold.py` — registration contract for this segment.
- `docs/tasks/SM-K4.2-PURE-residual.md` — required PROCESS residual.

Do not edit `continuous.py`, `routes/catalog.py`, `progression.py` verification,
`STATUS.md`, or `SOURCE_STATES.md`. Do not grant Speed Booster or forge items.

## Source and contract

- Source: `custom_integrations/SuperMetroid-Snes/scratch/post_frog_save_to_speedway_pure.state`
- Expected source room: `0xB106` Frog Speedway; continuous-like pure successor
  (catalog: x=39 y=139 pose=11 door_transition=0 after reload settle).
- Target: ordinary Upper Norfair Farming Room `0xAF72` through the real
  **right-hand** door (graph: right exit → left entry).
- Required capabilities: Morph, Bombs, Missiles, Supers, Hi-Jump, Varia —
  **no Speed** (continuous loadout at this tip; do not grant Speed).
- One named controller change only: `speedway_to_farm`.

## Context (minimal)

- Predecessor pure GREEN: Frog Save → Speedway (`SM-K4-SPEEDWAY-PURE`).
- SRC catalog GREEN: `SM-K4-SPEEDWAY-SRC` / `post_frog_save_to_speedway_pure`.
- Practice reverse (AF72→B167) used Speed for Boost Blocks — that is dual-track
  only. Continuous eastbound must work **without** Speed grant.
- Process: pure-first; residual → next card + one change.

## Read first

- `routes/kpdr/k4_norfair.py` (`play_frog_save_to_speedway` pattern)
- `docs/SOURCE_STATES.md` row `post_frog_save_to_speedway_pure`
- `docs/tasks/SM-K4-FROG-SPEEDWAY-PURE-residual.md`
- `docs/tasks/PROCESS.md` residual schema
- `scripts/probe/kpdr.py` pure segment map

## Do

1. Replace `_scaffold_exit` in `play_speedway_to_farm` with a bounded
   source-backed controller (run/shoot right door → settle in `0xAF72`).
   Timeout must expose room/pose/x/y/door_transition.
2. Register pure `speedway-to-farm` in `kpdr.py` (controller-only; no
   `--place-x` on acceptance).
3. Run the pure command below. On GREEN, write successor to
   `scratch/post_speedway_to_farm_pure.state`. On RED, keep one-knob residual
   with pin + redDiag paths if present.
4. Focused unit/registration test green (not a pure-green claim alone).

## Acceptance

- [ ] Source fingerprint loads at `0xB106`.
- [ ] Pure controller reaches ordinary `0xAF72` without placement, warp, or
  item grants.
- [ ] Successor source captured only if pure GREEN.
- [ ] Focused unit test green.
- [ ] Residual PROCESS fields; next card ID + one change; no continuous/STATUS claim.

## Verify

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure speedway-to-farm \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_frog_save_to_speedway_pure.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_speedway_to_farm_pure.state \
  --pin-json super_metroid/debug/speedway_to_farm_pure_pin.json

uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
```

## Residual routing

- GREEN → `SM-K4.2-SRC` (catalog farm successor) **or** planner opens
  `SM-K4.2-GRAPH` / `SM-K4.3-PURE` (farm→Bubble). Prefer SRC if state captured.
- RED → `SM-K4.2-PURE-R1`: one named phase change from the same Speedway source
  (cite pin / redDiag). Do not touch frog_save_to_speedway geometry.

Backlog: `SM-K4.2-PURE`. Live tip: [`QUEUE.md`](QUEUE.md).
