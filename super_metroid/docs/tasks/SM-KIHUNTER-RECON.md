# TASK SM-KIHUNTER-RECON: Door-band recon for Kihunter reverse exit (diagnostic)

## Recipe step
diagnostic recon (not pure green claim)

## Model
Luna

## Wave type
implement

## Own files only
- `scripts/probe/kihunter_zeela_recon.py` (**create**)
- `docs/tasks/SM-KIHUNTER-RECON-report.md` (**create**)

Do **not** edit `kraid_return.py` (geometry owned by SM-K4-R-02C). No STATUS.

## Context
- Pure geometry stuck: climb OK, door setup hits Baby `0xA521`.
- Need measured post-climb x/y bands and which door PLM/hatch is live when
  holding DOWN at various x.
- Source: `scratch/post_baby_to_kihunter_return.state`.

## Read first
- Prior recon scripts: `scripts/probe/kraid_door_*_recon.py` (style)
- `docs/tasks/SM-K4-R-02B-residual.md`
- `routes/kpdr/kraid_return.py` climb section (read only)

## Do
1. Probe script: load source, optional short climb-or-warp-to-upper helper
   (controller only; no progression forge), sweep x positions, sample room /
   door_transition / pose while aiming DOWN.
2. Write report: candidate Zeela x-band vs Baby x-band with table.
3. Recommend one numeric window for R-02C residual consumers.
4. Never claim pure green.

## Acceptance
- [ ] Report with numeric bands
- [ ] Non-claims

## Verify
```bash
uv run python super_metroid/scripts/probe/kihunter_zeela_recon.py \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_baby_to_kihunter_return.state
test -f super_metroid/docs/tasks/SM-KIHUNTER-RECON-report.md
```
