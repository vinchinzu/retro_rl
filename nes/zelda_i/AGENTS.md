# Agent Instructions — zelda_i

NES Legend of Zelda (graph nav; **M5** Clean power-on → Level 1 Triforce).
Shared: `retro_harness.adventure`, `retro_harness.nes`.
Docs: `docs/STATUS.md`, `docs/plan.md`, `docs/HYGIENE.md`,
`docs/ASSIST_CONTRACT.md`. Session: `.grok/skills/zelda-session/SKILL.md`.
Tracker: `bd ready -l zelda_i -l spine`.

## Dual track

**Survival** (`--infinite-life` / health refill) vs **Clean**. Assisted greens
are not Clean STATUS. Planner owns STATUS. Clean M5 =
`run_level1_complete` without `--infinite-life`. Do not overwrite.

## Immediate goal

Bead `rr-tne2` (L6 Survival compose → stairs/Gohma/TF `0x20`). Living residual:
[`docs/tasks/rr-tne2-residual.md`](docs/tasks/rr-tne2-residual.md).

## Commands

```bash
bd ready -l zelda_i -l spine

uv run python nes/zelda_i/scripts/run_survival_spine.py --no-video
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level6-clear3a --no-video --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level6-stairs3a-warp --no-video --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level6-cellar08 --no-video --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level6-south1d --no-video --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level6-west2d --no-video --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level6-north2c --no-video --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level6-gohma --no-video --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level1-bow --no-video --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level1-bow-cellar --no-video --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level1-bow-pickup --no-video --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level2-entry --no-video --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level6-east3a --no-video --trials 1

# Clean M5 (do not overwrite)
uv run python zelda_i/scripts/run_level1_complete.py --natural-entry --trials 2

uv run pytest zelda_i/tests -q
```

`--no-video` on spine CLIs. Leave proof is RAM + `zelda_i.screen_glance`,
not an MP4. Segment CLIs (L2–L9, TAS, lab): `docs/plan.md`.

## Layout

| Path | Role |
|------|------|
| `ram.py`, `overworld/graph.py`, `overworld/nav.py` | Snapshots + OW graph / L1 path |
| `overworld/path.py` | Shared hop engine (L2–L8) |
| `walk/physics.py`, `walk/predict.py` | OccupancyWalker + RAM claims |
| `dungeon/engine.py` + `level*/dungeon.py` | Combat + **specs/stop predicates only** |
| `spine/hops.py` | `SpineHop` rows + `attach_hops` / `ready` |
| `dungeon/hop_controller.py` | Dest-hop timeout/death/scroll guard |
| `dungeon/token_path.py` | L4 maze hold-token walker |
| `level*/path.py`, `level*/spine.py` | Path controllers + dest spine tables |
| `level*/overworld.py` | Hop tables + thin `overworld.path` subclasses |
| `runner.py` | Script env/assist/report helpers |

Split a file **before 500 lines**; refuse a new knob on a file **≥800**.
Do not boil already-split `level4/`. Named pins stay named. Probe PNG /
window JSON go gitignored scratch — not an AGENTS novel.

## Traps (burned once)

- Sword cave is **NW** of spawn on 0x77. Cave = mode **11**. Pickup x≈120
  then UP; after cave exit ~(64,77): **DOWN first**.
- `$066F` low nibble is whole hearts, not `0xF` full. Full is `lo==hi`
  (`0x22`=3/3) plus `$0670=$FF`.
- L2 prefix: `37→38→48→58→59→49→4A`; never 0x79.
- Stuck nav: stand still (`*_wait`). Do not loop LEFT/RIGHT/DOWN wiggle.
- `$0656` B-item: **1=bombs, 2=arrows, 4=candle**.
- Do not poke doors/keys/undiscovered items. Do not grant Map/Whistle.
- L2 entry bombs=0; Survival count top-up until farm `rr-doua`.

## Pointers

[docs/STATUS.md](docs/STATUS.md) · [docs/plan.md](docs/plan.md) ·
[docs/ASSIST_CONTRACT.md](docs/ASSIST_CONTRACT.md) ·
[docs/HYGIENE.md](docs/HYGIENE.md) · session skill `zelda-session`.
