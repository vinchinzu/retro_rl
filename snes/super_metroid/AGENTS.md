# Agent Instructions — Super Metroid

Scripted full-clear package `super_metroid` (disk: `snes/super_metroid/`).
Docs: `CONTEXT.md`, `docs/STATUS.md`, `docs/plan.md`,
`docs/ASSIST_CONTRACT.md`, `docs/ram_map.md`. Session loop:
`.grok/skills/sm-session/SKILL.md`.
Tracker: `bd ready -l super_metroid -l spine`. Empty ready while a
spine bead is in_progress means continue the residual.

## Evaluation contract

- Primary: unlimited energy + ammo only
  ([`docs/ASSIST_CONTRACT.md`](docs/ASSIST_CONTRACT.md)); no free
  items/doors/map/bosses/capacity.
- Natural ending/credits required; final boss alone is not a clear.
- Clean track: [`docs/CLEAN_TRACK.md`](docs/CLEAN_TRACK.md);
  `*_clean` stems only — never overwrite assisted baselines.
- Two tracks: spine continuous vs room practice. Practice greens are
  not continuous evidence. Planner owns STATUS.

## Immediate goal

Living tip: `--to phantoon` ([STATUS.md](docs/STATUS.md),
[CONTEXT.md](CONTEXT.md)). Next spine bead: `rr-kw8t` Gravity.
Pin, checkbox, and probe CLI:
[`docs/tasks/rr-kw8t-residual.md`](docs/tasks/rr-kw8t-residual.md).

## Commands

```bash
# Watch (headed first when the user says watch). --headed is retro_harness.headed.
uv run python snes/super_metroid/scripts/probe/kpdr.py pure <hop> --source <pin> --headed
./snes/super_metroid/play <pin> --headed --assist-full

uv run python snes/super_metroid/scripts/record/continuous.py --to phantoon --no-video

# Scratch Gravity collect from the s23 enter pin (not STATUS / not --to gravity green)
uv run python snes/super_metroid/scripts/probe/kpdr.py pure gravity-collect \
  --source snes/super_metroid/tasks/full_start_v1_anchors/f022887_enter_0xCE40_0xCE40.state

bd ready -l super_metroid -l spine
```

`--no-video` on every run. Leave proof is RAM + the run JSON
(`super_metroid.hop_glance`), not an MP4. The residual owns one probe CLI and
one run — run it twice only to compare against TAS (`settled_gs8` hop clock)
or another skill (the A/B loop), never to prove determinism.

## Layout

| Path | Role |
|------|------|
| `routes/continuous.py`, `early_continuous.py`, `catalog.py` | Power-on chain + tip registry |
| `routes/kpdr/` | Pure **movement** hops (no env ownership, no fight loops) |
| `combat/` | Boss fight policies after natural entry; hops are thin adapters |
| `routes/skills/` | Reusable movement Skills (walljump, moonfall, shinespark, poses) |
| `tas/` | Sniq movies + harness replay (`docs/TAS_ADAPT.md`) |
| `recordings/` | Watch MP4s in `recordings/<area>/` (`RECORDINGS_DIR`) |
| `scratch/` | Hop JSON + phase pins in `scratch/<hop>/` |
| `custom_integrations/SuperMetroid-Snes/` | Named anchors |

Hop ≠ fight. Soft max ~1000 LOC: merge into the **Composer**
(`tips.play_hops`) or delete ([CODING_STANDARDS.md](../../CODING_STANDARDS.md)).
Import `play_*` from the owner
(`registry.get_segment`). Continuous hops only via `tips.play_hops`.
New-file table:
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) “Where a new file goes”.

## Traps

- Door-warp settle: wait for **game state 8**; state 11 can last 50–100+f.
- High WRAM (`$7E:D820+`): `read_bank7e_wram` / `write_wram_u8` — raw
  `get_ram()[0xD820]` is open-bus garbage.
- Named anchors in `SuperMetroid-Snes/`. Hop JSON + phase pins in
  `scratch/<hop>/` (one run JSON per hop; do not stack `_vN`). Watch MP4s in
  `recordings/<area>/` (`RECORDINGS_DIR`).
- Practice / door-warp / boss probes are practice, not continuous evidence.
- Morph bombs are **X** while morph (not A).
- Hop `side` is D-pad `LEFT`/`RIGHT`. Shoulders are `L`/`R`.
- Phase dumps are named scratch pins; leftover still is the next boot when
  the residual says so. A RED run keeps the controller.
- RNG (`$05E5`): boot writes `$0061` after clearing WRAM — do not re-power-on
  for a new seed. Pin first Ceres/Zebes `gs=8` and `ram.set_rng`. Empty rooms
  only unless the door is re-entered. [docs/RNG.md](docs/RNG.md).
