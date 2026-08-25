## Residual — rr-7lc5 wiki KPDR Phantoon fight benches

Four public [Phantoon](https://wiki.supermetroid.run/Phantoon) KPDR recipes,
same natural pin, assist ON. Product left-corner charge **not** rewritten.
Do **not** STATUS-promote. Default CLI stays `ice`. Do **not** wire into
`--to phantoon` / `phantoon_fight`. Power-on compose is still `rr-8g2u`.

**Pin:** `scratch/post_ws_basement_to_phantoon.state` — `0xCD13` ~(39,124)
p81 gs=8 HP 2500 health 299 missiles 20 supers 5 max PB 5 items `0x3105`
beams `0x1007` (Charge `0x1000` + Wave `0x0001` + Ice `0x0002` + Spazer
`0x0004`).

**Product baseline:** assist charge-only **20537f** ×2 (rr-tlaq), 9 chips
(8×300 + 100), HP 0 + `$D82B` bit 0, never Super.

### Bench (assist, pin reload, `format_segment_time`)

| Wiki recipe | Result | frames | clock | vs 20537f | Wire? |
|---|---|---:|---|---:|---|
| Ice/Wave/Spazer charge only | 9 one-chip rounds. `shots_per_window=3` == `=1` (300 closes the eye before a second charge) | **20537** ×2 | 05:42.28 | 0 | **keep product** |
| Charge + missiles (2+2+charge ×4) | Kill, but 12 windows; barrage B never landed | **27645** ×2 | 07:40.75 | **+7108** | no |
| Missile doppler (2-2-N, 10f) | Kill. Recipe landed **2-2-1**, close-eye extra **0**. Super finisher (4 spent, 2 miss then 600) | **12118** ×2 | 03:21.97 | **−8419** | **no** (not true doppler; Super sloppy) |
| popTOON / X-Factor | Ice-on Wave Shield: charge 120, PB 5→5, HP 2500→2500, no SBA particles | window 1818f | 00:30.30 | n/a | no |

Reports under `scratch/phantoon_wiki_{charge_only,charge_missiles,doppler,xfactor}*.json`.
Dual rows reloaded the pin. Boss bit always `$7E:D82B` via `read_bank7e_wram`.

### Why product stays charge-only

- Wiki “three charges per round” is a *spaced* barrage. Ice/Wave/Spazer
  charge is 300; 300+ in one barrage closes the eye; recharge (~60f) loses
  the window. Spine `shots_per_window=3` is a no-op vs probe `=1`.
- MassHesteria 2+2+charge is slower here: he leaves after the 2-missile
  opener (`missiles_b=0` every round).
- Doppler **is faster** on this pin, but extras during the ~10f close never
  counted (`close_eye_extra=0`). Super was gated on HP ≤ 600 and did kill;
  two spends missed first. Do not replace the never-Super product body from
  a pin bench. Do not clobber `post_phantoon_poweron.state`.
- True X-Factor is Charge+Wave+PB with Ice **off**. This pin has Ice on and
  there is no pause-menu unequip helper. Measured miss, not a 2-round claim.

### Files (experimental; unused by spine)

- `combat/phantoon_{charge_missiles,doppler,xfactor}.py`
- `scripts/probe/phantoon_{wiki_charge,charge_missiles,doppler,xfactor}.py`
- `tests/test_phantoon_{wiki_charge,charge_missiles,doppler,xfactor}.py`

### Verify paste

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  snes/super_metroid/tests/test_phantoon_combat.py \
  snes/super_metroid/tests/test_phantoon_wiki_charge.py \
  snes/super_metroid/tests/test_phantoon_charge_missiles.py \
  snes/super_metroid/tests/test_phantoon_doppler.py \
  snes/super_metroid/tests/test_phantoon_xfactor.py \
  snes/super_metroid/tests/test_continuous_tips.py -q
# → unit GREEN; DEFAULT_CONTINUOUS_TIP still ice; fight wrapper still
#    PhantoonStrategy(weapon=beam, shots_per_window=3)
```

### Acceptance

- [x] Four wiki recipes documented + pin-benched from the Basement→room leave
- [x] Charge-only dual 20537f reproduced; `shots_per_window=3` measured no-op
- [x] Charge+missiles dual-green but slower (not 2+2+charge)
- [x] Doppler dual-green faster but not wiki extras; not wired
- [x] Ice-on X-Factor window miss (HP drop 0, PB unspent)
- [x] Product `combat/phantoon.py` / `play_phantoon_room_fight` unchanged
- [ ] Power-on / pin compose dual (rr-8g2u)

### Next action (required)

- **Follow-on done:** `rr-asyg` wired doppler + loot/exit. Charge-only
  stays research. Next is `rr-8g2u` pin compose with the doppler+leave body.

### Non-claims

- Did not STATUS-promote past Ice
- Did not change `DEFAULT_CONTINUOUS_TIP` or `WS_ONLY_HOPS`
- Did not rewrite `combat/phantoon.py` or the spine fight wrapper
- Did not write `recordings/phantoon.json`
- Did not run power-on `--to phantoon`
- Did not clobber `post_phantoon_poweron.state` / `post_phantoon_defeated.state`
- Did not Super-spray / enrage
- Did not claim a no-assist kill
- Did not claim wiki 4-round charge, 4-round 2+2+charge, true doppler extras,
  or a 2-round X-Factor
