# Agent Instructions: Super Metroid

Scripted full-clear package `super_metroid` (disk: `snes/super_metroid/`).
Docs: `CONTEXT.md`, `docs/STATUS.md`, `docs/plan.md`,
`docs/ASSIST_CONTRACT.md`, `docs/ram_map.md`. Session loop:
`.grok/skills/sm-session/SKILL.md`.
Tracker: `bd ready -l super_metroid -l spine`. Empty ready while a
spine bead is in_progress means continue the residual.

## Commands

```bash
# Watch (headed first when the user says watch). --headed is retro_harness.headed.
uv run python snes/super_metroid/scripts/probe/kpdr.py pure <hop> --source <pin> --headed
./snes/super_metroid/play <pin> --headed --assist-full

uv run python snes/super_metroid/scripts/record/continuous.py --to phantoon --no-video

bd ready -l super_metroid -l spine
```

`--no-video` on every run. Leave proof is RAM plus the run JSON
(`super_metroid.hop_glance`), not an MP4. One spine bead. The residual
owns the pin and the one probe CLI:
[`docs/tasks/rr-kw8t-residual.md`](docs/tasks/rr-kw8t-residual.md).
Run that probe twice only to compare against TAS (`settled_gs8` hop clock)
or another skill, never to prove determinism. Do not write STATUS from a pin.

## Traps

- Survival assist is current energy plus unlocked ammo only. No free
  items, doors, map, bosses, or capacity. `*_clean` stems never overwrite
  assisted baselines. Practice greens are not continuous evidence.
  Planner owns STATUS.
- Door-warp settle: wait for game state 8. State 11 can last 50-100+ frames.
- High WRAM (`$7E:D820+`): `read_bank7e_wram` / `write_wram_u8`. Raw
  `get_ram()[0xD820]` is open-bus garbage.
- Named anchors in `SuperMetroid-Snes/`. Hop JSON and phase pins in
  `scratch/<hop>/` (one run JSON per hop; do not stack `_vN`). Watch MP4s
  in `recordings/<area>/`.
- Morph bombs are X while morph, not A. Hop `side` is D-pad LEFT/RIGHT.
  Shoulders are L/R.
- A RED run keeps the controller. RNG (`$05E5`): boot writes `$0061` after
  clearing WRAM. Do not re-power-on for a new seed. Pin first Ceres/Zebes
  `gs=8` and `ram.set_rng`. See [docs/RNG.md](docs/RNG.md).
- Where a new file goes: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
  Hop is not a fight. Soft max ~1000 LOC: merge into the Composer
  (`tips.play_hops`) or delete.
