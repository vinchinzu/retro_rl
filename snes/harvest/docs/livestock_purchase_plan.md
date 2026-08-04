# Livestock Purchase Plan

## Goal

Use verified save-state editing as the first path for Harvest Moon scenario and farm-data hacking, then anchor later gameplay automation against those same bytes.

## Phase 1: Compact State Builder

Implemented in `harvest/tools/livestock_builder.py`.

- Start from `Y1_After_Buy_Potato`.
- Inject working purchase resources:
  - money
  - stored grass / hay
  - chicken feed
  - cow feed
- Write three compact validation states:
  - `<prefix>_resources.state`
  - `<prefix>_chicken.state`
  - `<prefix>_chicken_cow.state`
- Verify each output two ways:
  - parse the written `.state` directly
  - load it through `stable-retro` and parse `env.initial_state`

This keeps verification tied to the actual Snes9x snapshot bytes instead of trusting `get_ram()`, which does not currently reflect the same address space.

## Current Scope

- Confirmed edit path:
  - money
  - hay / stored grass
  - chicken feed
  - cow feed
  - chicken count + slot 0 data
  - cow count + slot 0 data
- Deferred until live capture proves the layout:
  - scenario/event flags
  - romance/event progression flags
  - any animal structures beyond the current chicken/cow slots

## Sheep Status

Do not assume sheep exist in this SNES build.

- The local integration exposes cow/chicken counts and feed only.
- The validated livestock slot work so far only yields chicken and cow structures.
- Sheep should stay blocked until a real purchase capture or ROM evidence in this repo shows a separate sheep structure.

## Phase 2: Real Purchase Automation

Use the existing task stack instead of inventing a parallel system.

- `NavTask` handles day-time movement from farm to map exit.
- `CrossMapRecordedTask` handles off-farm store/menu playback.
- Record dedicated tasks once the compact states are stable:
  - `buy_chicken.json`
  - `buy_cow.json`
- Chain them into a livestock day plan:
  - leave house
  - navigate to farm exit
  - replay chicken purchase
  - save post-purchase state
  - on a later day, replay cow purchase
  - save post-purchase state

## Validation Loop

For each new live purchase state:

- diff the raw snapshot RAM against the compact builder output
- confirm money/feed/hay deltas
- confirm animal counts
- confirm slot bytes for the purchased animal
- load the state in the editor and confirm the same values render there

## Commands

Build and verify compact livestock states:

```bash
uv run python -m harvest.tools.livestock_builder --base Y1_After_Buy_Potato --verify
```

Run the narrow editor/state tests:

```bash
uv run python -m unittest tests.test_extract_tiles tests.test_rom_tools tests.test_harvest_state tests.test_livestock_builder tests.test_editor_app
```
