# Status — Star Fox completion run

## Target

Bronze Route 1 completion: RAM may select/recover policies, while all game
progress is made through ordinary controller actions.

## Current milestone

Destroy Attack Carrier's three hatches and finish the Route 1 Corneria clear.

## Run status

- Super FX boot and controller injection verified in Stable-Retro.
- True reset → controls → game → Route 1 map verified.
- Route 1 `CorneriaStart.state` captured under the custom integration.
- Rev 2 RAM verified for health, lives, bombs, kills, player X, and vertical
  phase.
- `scripts/run_completion.py` runs the Bronze rapid-fire/flight policy with
  healthy save-state checkpoints and varied recovery retries.
- The recovery runner reached Attack Carrier with all three lives preserved.
- `AttackCarrierStart.state` captures the opening overhead pass. The current
  controller policy survives that pass, damages a tracked carrier component,
  and alternates long hatch-alignment and evasion variants on retries.

## Compatibility note

Stable-Retro includes an experimental `StarFox-Snes` integration for USA 1.0.
This project uses the available USA Rev 2 ROM, so bundled save states are not
assumed compatible. The RAM addresses are provisional until verified.

The published 2007 TASVideos maximum-score movie uses this exact Rev 2 ROM,
but its Snes9x 1.43 WIP1 timing omits Super FX lag. Its input log is retained
as a reference; direct playback on the modern core desynchronizes.
