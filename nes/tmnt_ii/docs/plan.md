# Plan — TMNT II (NES)

## Goal

Advance from M3 (first wave) toward a verified continuous clear of
Teenage Mutant Ninja Turtles II: The Arcade Game.

## Next milestones

1. **Past first lock** — score≥8–10 and camera advance; map lock flag.
2. **M2 complete** — enemy slots, stage/area, death/continue.
3. **M4 natural-entry** — same first wave from power-on / boot script
   (not only `Level1.state`).
4. Chain Stage 1 packs under `Stage1Policy` / shared combat helpers.

## Bottleneck

Right-edge screen lock after early kills: pure RIGHT+B stalls at score 4;
need face-LEFT B (done for score 5). Next packs need unlock + targeting.

## Notes

- Platform: NES (fceumm via stable-retro custom integration).
- Shared ROM root: `roms/Nintendo/NES/`.
- Transfer target: SNES TMNT IV combat stack (`retro_harness.combat`).
