# TASK SM-SRC-INVENTORY: Expand SOURCE_STATES catalog from disk (Flash)

## Recipe step
docs / source catalog

## Model
Flash

## Wave type
implement

## Own files only
- `docs/SOURCE_STATES.md`
- optional residual note

No code, no STATUS continuous frame tables, no progression promote.

## Context
- Many `scratch/` and `dev_*` states exist on disk but are not indexed.
- Executors block on “no source”; catalog expansion unblocks parallel pure cards.

## Read first
- `docs/SOURCE_STATES.md`
- List (shell): `custom_integrations/SuperMetroid-Snes/scratch/` and top-level
  `dev_*.state` names (do not commit binaries).

## Do
1. Add rows for high-value anchors: reverse pure outs, business climb,
   phantoon/botwoon/ridley/mb **dev** anchors (label developmentOnly).
2. Gaps table: zeela source, warehouse reverse, bubble entry, moat+speed.
3. Keep rules section; no absolute home paths.

## Acceptance
- [ ] Catalog denser; gaps honest
- [ ] Non-claims: catalog ≠ continuous evidence

## Verify
```bash
# docs only — ensure markdown tables parse-ish
test -f super_metroid/docs/SOURCE_STATES.md && wc -l super_metroid/docs/SOURCE_STATES.md
```
