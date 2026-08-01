# TASK SM-PURE-ISO-REV: Pure isolation CLI for reverse hops (harness epic)

## Recipe step
harness (diagnostics — not pure green claims)

## Model
Luna

## Wave type
implement

## Own files only
- `scripts/probe/kpdr.py` (add isolation helpers / segment aliases only as needed)
- optional: `scripts/probe/pure_iso_reverse.py` (**create** if cleaner)
- `docs/tasks/SM-PURE-ISO-REV-note.md`

Do not edit controllers geometry. Do not continuous / STATUS.

## Context
- SM-PURE-ISO added business isolation. Reverse chain needs one-command
  probes for eye/baby/kihunter/zeela hops from cataloged sources.
- This is harness throughput for executors — green geometry is separate cards.

## Read first
- `scripts/probe/kpdr.py` pure subcommand
- `docs/SOURCE_STATES.md`
- `routes/kpdr/registry.py` segment ids

## Do
1. Ensure pure segment names exist for reverse hops already registered.
2. Add a small “iso-reverse” help path or docs note listing exact commands
   with source paths for K3.3–K3.6.
3. Optional: smoke each command once; report RED/GREEN honestly in note.
4. Never force-pass.

## Acceptance
- [ ] Documented command matrix in note
- [ ] No controller geometry edits
- [ ] Residual next: R-02B if zeela still red

## Verify
```bash
uv run python super_metroid/scripts/probe/kpdr.py pure --help
# optional smokes with catalog sources (report only)
```
