# TASK SM-K4-R-02D: Kihunter→Zeela — real climb + recon Zeela door band

## Recipe step
1 pure controller (one-knob geometry residual)

## Model
Luna

## Wave type
implement

## Own files only
- `routes/kpdr/kraid_return.py` (`play_kihunter_to_zeela_return` only)
- optional residual: `docs/tasks/SM-K4-R-02D-residual.md`

## Context (minimal)
- Prior: SM-K4-R-02 / 02B / 02C. Climb was **mis-reported green**: RIGHT
  bias walks into the **east Baby hatch** before any true upper land.
- Fresh pure pin (post-approach fix):
  `error: upper traverse crossed wrong door` → `roomIdHex: 0xA521`
  `samusX: 65521` `samusY: 116` `pose: 105` `frame: 154`.
- Source start (probed): `0xA4DA` **x≈465 y≈378** — already on the east edge.
  RIGHT climb hits door transition at **x≈492** (~80f), room → `0xA521`.
- `y < 280` alone is a **false climb success** once Baby loads (spawn y is
  low). Accept climb **only** while still `room_id == 0xA4DA`.
- Recon (`SM-KIHUNTER-RECON-report.md`): natural climb never reaches upper band
  (`climbed=False` every trial). **Dev-warp** upper y≈240 shows Zeela down-door
  only at **x∈[96,160]** → `0xA471`. Do not use natural edge x≈492.
- Continuous still Varia-only; pure K3.6 only. No graph / STATUS.

## Read first
- `routes/kpdr/kraid_return.py` (`play_kihunter_to_zeela_return`)
- `docs/tasks/SM-KIHUNTER-RECON-report.md`
- `docs/tasks/SM-K4-R-02C-residual.md`
- `docs/SOURCE_STATES.md` (`post_baby_to_kihunter`)

## Do
1. **One knob — climb exit + east-door avoidance** (then door band is fixed
   constants from recon, not a second free experiment):
   - During climb: **never** set `climbed=True` unless
     `room_id == ROOM_WAREHOUSE_KIHUNTER` **and** `samus_y < 280` **and**
     `door_transition == 0`.
   - Fail loud immediately if `room_id == ROOM_BABY_KRAID` during climb
     (same as drop path).
   - Stop walking **RIGHT into x≥~480** / the east hatch. Source is already
     x≈465; prefer vertical / left-of-door launch into the shot-block tunnel,
     then land upper while still in Kihunter.
   - After a **true** upper land: position into recon Zeela window
     **`96 <= samus_x <= 160`** (center ~128), then DOWN+shot/drop only from
     that band. Drop RIGHT backoff that pushes past 160.
2. ≤3 bounded climb-or-window strategies; residual with pin if still red.
3. On pure green: `--output` → `scratch/post_kihunter_to_zeela_return.state`.
4. No graph promote, continuous, STATUS, zeela→warehouse.

## Do not
- Implement SM-K4-R-03
- Progression/door RAM forges or `place_samus` warps in the route controller
- Edit `kraid_approach.py` / multi-room compose

## Acceptance
- [ ] Pure green → ordinary `0xA471` **or** residual with **in-Kihunter**
      post-climb x/y pin (room still `0xA4DA`, y band upper, x noted)
- [ ] Fail loud on `0xA521` (no silent green; no climb success on Baby y)
- [ ] `uv run pytest super_metroid/tests/test_controller_common.py -q` green

## Verify
```bash
uv run python super_metroid/scripts/probe/kpdr.py pure kihunter-to-zeela-return \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_baby_to_kihunter_return.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_to_zeela_return.state
uv run pytest super_metroid/tests/test_controller_common.py -q
```

## Done when
Pure exit 0 into Zeela, or residual after ≤3 knobs with PROCESS schema +
**in-source-room** pose/x/y and one next change.
