# TASK SM-K4-SPEEDWAY-SRC: Fingerprint-register Frog Speedway pure successor

## Recipe step
source capture / catalog

## Model
Flash

## Wave type
implement

## Own files only
- `docs/SOURCE_STATES.md`
- optional residual: `docs/tasks/SM-K4-SPEEDWAY-SRC-residual.md`

No controller geometry, no continuous, no STATUS, no progression.

## Context (minimal)
- Predecessor pure card: `SM-K4-SPEEDWAY-PURE` / `SM-K4.1-PURE`
- Expected successor path:
  `custom_integrations/SuperMetroid-Snes/scratch/post_frog_save_to_speedway_pure.state`
- Expected room: ordinary Frog Speedway `0xB106`
- Residual (if pure GREEN) listed frames ~295, pose/x/y pin on exit
- Next geometry after catalog: backlog `SM-K4.2-PURE` (Speedway → farm)
  and planner `SM-K4.1-GRAPH`

## Read first
- `docs/SOURCE_STATES.md` (table format + gaps section)
- `docs/tasks/SM-K4-FROG-SPEEDWAY-PURE-residual.md` (if present)
- `docs/tasks/SM-SRC-INVENTORY.md` (catalog style)

## Do
1. Confirm the successor `.state` exists and loads at room `0xB106` with
   ordinary gameplay (door_transition=0). Record pose/x/y from a one-shot
   load probe or residual pin — **no** placement / warp.
2. Add a SOURCE_STATES row for pure Speedway entry (label continuous-like
   pure successor of Frog Save; not continuous tip evidence).
3. Update or remove the gaps row that said Frog→Speedway still needs first
   geometry controller (only if pure residual is GREEN and state verifies).
4. Residual: next card `SM-K4.2-PURE` **or** `SM-K4.1-GRAPH` / PLANNER-GATE
   for graph — one change only (catalog complete).

## Do not
- Edit `k4_norfair.py` / continuous / STATUS / progression
- Claim continuous integrity or tip promotion
- Use development full-loadout anchors as “natural” Speedway sources

## Acceptance
- [ ] SOURCE_STATES has a Speedway pure-successor row with repo-relative path
- [ ] Room/fingerprint notes honest (GREEN pure only if state verified)
- [ ] Residual next card ID + one change
- [ ] Non-claims: not continuous evidence

## Verify commands
```bash
test -f super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_frog_save_to_speedway_pure.state
# optional load-only probe if a dump/inspect helper exists; else residual pin paste
rg -n "post_frog_save_to_speedway|0xB106|Speedway" super_metroid/docs/SOURCE_STATES.md
```

## Done when
Flash residual filed. Planner opens graph edge / Speedway→farm pure.
