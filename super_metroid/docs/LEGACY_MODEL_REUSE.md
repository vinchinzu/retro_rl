# Legacy climbing and Bomb Torizo model handoff

The previous project at `../snes_editor/super_metroid_rl` contained four
candidate policies. `scripts/import_legacy_assets.py` imports them into
`models/imported/` after checking the SHA-256 values in `models/manifest.json`.

| Model | Intended reuse | Current status |
|-------|----------------|----------------|
| `bc_nav_model.pth` | General navigation and Climb ascent candidate | Imported, hash-verified, inference adapter tested, not natural-entry accepted |
| `nav_ppo.zip` | PPO navigation/climbing candidate | Imported, hash-verified, wrapper compatibility pending |
| `boss_bc.pth` | Bomb Torizo combat candidate | Imported, hash-verified, inference adapter tested, not natural-entry accepted |
| `boss_ppo.zip` | PPO Bomb Torizo combat candidate | Imported, hash-verified, wrapper compatibility pending |

“Chorizo” in the request is interpreted as Bomb Torizo.

The accepted start-to-Torizo run does not claim a neural-model promotion. It
uses hash-pinned controller replays from the same legacy corpus because they
passed the stronger continuous natural-entry gate. The neural checkpoints
remain useful candidates for boundary recovery and later state coverage, but
they do not currently own accepted graph edges.

## Compatibility contracts

The navigation BC checkpoint expects one 112×128 grayscale frame, normalized
inside the network, and emits 12 independent SNES button logits. The original
runtime used a sigmoid threshold of 0.5 plus room-direction bias.

The Boss BC checkpoint expects four resized RGB frames stacked channel-first
(12×112×128) and emits the same 12-button logits.

`visual_models.py` now implements those exact two BC contracts, verifies each
checkpoint hash before loading, derives channel/stack dimensions from the
state dict, applies the original 2× downsample and `/255` normalization, emits
12 environment-order button probabilities, and sanitizes opposite directions.
This makes the checkpoints callable candidates for future boundary/recovery
evaluations; it does not promote their gameplay success.

The PPO archives depend on the original discrete action table, sanitization,
action-hold behavior, 112×128 resize, and four-frame channel stack. Loading an
archive alone is insufficient; those wrappers must be reproduced exactly.

## Promotion path

1. Add adapters that preserve each model's original observation/action
   contract.
2. Evaluate candidates from their original development states.
3. Re-evaluate from states captured from the real predecessor segment.
4. Assign a graph edge only after deterministic or bounded-success evidence.
5. Chain Morph → first Missiles → Climb ascent → Flyway → Bomb Torizo.
6. Require natural Bomb Torizo defeat and Bombs pickup; model inference may
   choose controller inputs but may not write boss/event/item state.

The imported `maps/legacy/full_game_route.json` is a useful objective baseline,
but its research-only anchors are not accepted room edges. Full-run work should
promote them incrementally into the typed graph with live transition and
inventory evidence.
