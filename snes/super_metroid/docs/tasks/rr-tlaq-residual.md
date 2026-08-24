## Residual — rr-tlaq Phantoon fight (0xCD13)

**Status:** Assist dual-green kill **20537f** ×2. HP 0 + wrecked-ship `$D82B`
bit 0, gs=8, not dead. No-assist energy floor still real (do not re-prove).
**Pin in:** `scratch/post_ws_basement_to_phantoon.state`
**Pin out:** `scratch/post_phantoon_poweron.state` (did **not** clobber
`post_phantoon_defeated.state`)
**Probe:** `strategy --assist --weapon beam --max-frames 40000 --save-state`
**Reports:** `scratch/phantoon_assist_kill.json` + `_dual.json`

Do **not** STATUS-promote. Default CLI stays `ice`. Super-spray is not a hit.
Do **not** append to `POST_ICE_SPINE` / `WS_ONLY_HOPS` / `--to ws`.

### Assist kill (product spine)

Same seat-window policy as the no-assist windows. `--assist` energy+ammo ON
so `$D82A` / skip parks do not eat the tank. 9 charge chips (300 each; Super
unused). Boss-bit wait peeks `$7E:D82B` (low `env.get_ram()` never contains
that byte).

  | run | frames | body 0 | boss bit | HP | gs | health | shots |
  |-----|-------:|-------:|---------:|---:|---:|-------:|------:|
  | 1 | **20537** | 19507 | 20537 | 0 | 8 | 299 | 9 |
  | 2 | **20537** | 19507 | 20537 | 0 | 8 | 299 | 9 |

Assist telemetry ×2: energy_restored 500 / 25 writes, max hit 20, deaths 0,
missile_writes 0, supers still 5.

```bash
uv run python snes/super_metroid/scripts/probe/phantoon_combat.py strategy \
  --assist --weapon beam --max-frames 40000 --save-state \
  --report snes/super_metroid/scratch/phantoon_assist_kill.json
```

### No-assist ceiling (do not re-prove)

Jump only rain (48, 96) y 88–104. Skip x=219, (128, 96), (88, 64), (53, 82),
(56, 113), (83, 64). Snipe-wait pose 3. No sit-charge. No 2k farm.

  | W | Park | Spend | HP | Health in→out |
  |---|------|-------|----|-------:|
  | 1 | (120, 108) | (104, 149) p43 | 2500→2200 | 279→239 |
  | 2 | (48, 96) | (37, 148) p21 | 2200→1900 | 219→199 |
  | 3 | (48, 96) | (37, 148) p21 | 1900→1600 | 139→119 |
  | 4 | (48, 96) | (37, 148) p21 | 1600→1300 | 79→59 |
  | 5 wait | `$D5E7` then `$D82A` pose 3 | — | 1300 | 59→39→**19** halt |

Best streak W1–W6 2500→700 then `$D82A` flames eat the rest. 54–59 HP cannot
tank `$D82A` to the next (48, 96).

### Next actions

1. Leave Phantoon's Room / WS power-on from `scratch/post_phantoon_poweron.state`.
   Do **not** append to `--to ws` until a planner STATUS pass.
2. Planner STATUS for `moat` / `ws` is a follow-on (`rr-g3nj`). Default CLI
   stays `ice`.

### Non-claims

- Did not STATUS-promote past Ice
- Did not change `DEFAULT_CONTINUOUS_TIP`
- Did not write `recordings/ws.json`
- Did not append to `POST_ICE_SPINE` / `WS_ONLY_HOPS` / `--to ws`
- Did not close `rr-g3nj`
- Did not rewrite `play_ws_entrance_to_main` / `play_ws_main_to_basement` /
  `play_ws_basement_to_phantoon`
- Did not clobber `post_phantoon_defeated.state`
- Did not Super-spray (enrage)
- Did not start a 16k
