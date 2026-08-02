# TASK SM-K4-SPEEDWAY-PURE: Frog Savestation → Frog Speedway

## Recipe step

1. Pure controller plus its bounded pure-probe registration. Geometry green is
required before any graph, catalog, or continuous wiring.

## Model

Luna

## Wave type

implement

## Own files only

- `routes/kpdr/k4_norfair.py` — replace only
  `play_frog_save_to_speedway`; leave the Business→Frog controller and all
  other K4 scaffold functions untouched.
- `routes/kpdr/__init__.py` and `routes/kpdr/registry.py` — export/register
  this one segment.
- `scripts/probe/kpdr.py` — add this one controller-only `pure` choice.
- `tests/test_k4_norfair_scaffold.py` — import/registry contract only.
- `docs/tasks/SM-K4-FROG-SPEEDWAY-PURE-residual.md` — required PROCESS
  residual.

Do not edit `continuous.py`, `routes/catalog.py`, `progression.py`,
`STATUS.md`, `SOURCE_STATES.md`, or Business→Frog geometry. Do not promote
the graph edge from `unverified`.

## Source and contract

- Source: `custom_integrations/SuperMetroid-Snes/scratch/post_frog_continuous.state`
- Expected source room: `0xB167` Frog Savestation; after the normal five-frame
  reload settle: x=60, y=139, pose=1, ordinary gameplay.
- Target: ordinary Frog Speedway `0xB106` through the real right-hand door.
- Required capabilities: Morph, Bombs, Missiles, Supers, Hi-Jump, Varia.
- The only geometry change is the named `frog_save_to_speedway` controller;
  no shared primitive extraction in this card.

## Do

1. Replace the `_scaffold_exit` call in `play_frog_save_to_speedway` with a
   bounded source-backed controller. Keep its room precondition and include a
   timeout that exposes room/pose/x/y/door-transition.
2. Register exactly this controller for `kpdr.py pure frog-save-to-speedway`.
   The probe must use controller input and normal resource assists only;
   `--place-x` is forbidden.
3. Run the pure command below. On GREEN, write the successor to
   `scratch/post_frog_save_to_speedway_pure.state`; on RED, retain no unrelated
   timing experiment and record the final probe pin.
4. Run the focused unit test. A unit test or registration alone is never a
   pure-green claim.

## Acceptance

- [ ] Source fingerprint loads at `0xB167`.
- [ ] Pure controller reaches ordinary `0xB106` without placement or warp.
- [ ] Successor source is captured only if the pure probe is GREEN.
- [ ] Focused unit test is green.
- [ ] Residual has all PROCESS fields, one exact next card, and no continuous
  or STATUS claim.

## Verify

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure frog-save-to-speedway \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_frog_continuous.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_frog_save_to_speedway_pure.state \
  --pin-json super_metroid/debug/frog_save_to_speedway_pure_pin.json

uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
```

## Residual routing

- GREEN → [`SM-K4-SPEEDWAY-SRC`](SM-K4-SPEEDWAY-SRC.md): fingerprint-register
  the captured Speedway successor before opening Speedway→Farm.
- RED → `SM-K4-FROG-SPEEDWAY-R1`: change one named controller phase from the
  same Frog source; do not change Business→Frog or graph verification.

Backlog alias: [`SM-K4.1-PURE`](SM-K4.1-PURE.md). Wave: [`WAVE-11.md`](WAVE-11.md).
