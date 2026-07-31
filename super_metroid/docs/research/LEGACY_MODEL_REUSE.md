# Legacy climbing and Bomb Torizo model handoff

The previous project at `../snes_editor/super_metroid_rl` contained four
candidate **vision** policies. `scripts/import_legacy_assets.py` imports them
into `models/imported/` after checking the SHA-256 values in
`models/manifest.json`.

| Model | Intended reuse | Current status |
|-------|----------------|----------------|
| `bc_nav_model.pth` | General navigation and Climb ascent candidate | Imported, hash-verified, inference adapter tested, **parked until gold** |
| `nav_ppo.zip` | PPO navigation/climbing candidate | Imported, hash-verified, parked |
| `boss_bc.pth` | Bomb Torizo combat (pixels) | Imported; wins on `BossTorizo` train state; **fails natural Flyway entry**; parked until gold |
| `boss_ppo.zip` | PPO Bomb Torizo combat (pixels) | Imported, parked |

## Policy for hard spots (2026-07-30)

**Vision-only networks are not the path for bosses until gold.** Prefer:

1. Full-knowledge strategy (RAM hitboxes / HP / spritemap) — see
   [STRUCTURED_BOSS_RL.md](STRUCTURED_BOSS_RL.md).
2. Structured-state RL to clean up speed and damage.
3. Keep accepted hash-pinned replays and controllers on continuous routes.

The accepted start-to-Torizo run uses hash-pinned controller replays because
they passed the continuous natural-entry gate. Vision checkpoints remain
offline artifacts; they do not own accepted graph edges.

## Compatibility contracts (parked)

The navigation BC checkpoint expects one 112×128 grayscale frame and emits
12 SNES button logits. The Boss BC checkpoint expects four RGB frames
stacked channel-first (12×112×128). `visual_models.py` still implements
those contracts for later gold experiments only.

## Structured promotion path (active)

1. Catalog boss hitbox / HP / weapons (`combat/features.py`).
2. Hand strategy controller that wins from an **active** fight state
   (`combat/bomb_torizo.py`).
3. Capture natural-entry mid-fight states from continuous prefix.
4. Optional: RL on `feature_vector`, not pixels — Gym + short PPO loop live
   in `combat/env.py` / `scripts/probe/bomb_torizo_combat.py train`.
5. Promote only with natural continuous evidence; no boss/event/item writes.
