# Agent Instructions — final_fight

SNES Final Fight oneshot ladder entry (rank 2). Shared helpers:
`snes_oneshot/`. Program notes:
`snes_oneshot/docs/EASIEST_SNES_GAMES.md`.

## Norms

- Prefer development save states and segment scripts over uninterrupted runs.
- Store `.state` files under `custom_integrations/FinalFight-Snes/`.
- Keep RAM maps and game policy here; elevate reusable beat-em-up primitives
  to `snes_oneshot/`.
- Headless probes: `SDL_VIDEODRIVER=dummy` (and audio dummy as needed).
- Docs: `docs/STATUS.md`, `docs/plan.md`, `docs/ram_map.md`.

## Immediate goal

**Stage 3 West Side Area1** — wave5 dual cleared (verified). Prefer
`Stage3_Clear_w5_real_p48_cam640` / `…_hp48` or
`Stage3_Mid_w5_true1v1_p60_e142_cam640` → LEFT+Y wait-KD. Post-clear:
scroll → cam931 softlock → CLEAR_AREA → **`Stage3_Area1_hp50_L1_cam2560`**.
Area1 HP≈250 thug (`ENTITY_HP_MAX=255`) chips hard — Boss3 open.
Heal poke `player_hp` 60–70 used on crumb→1v1 (document). Continuous
wave2 still prefers Mid_w2_p66. Damnd `0x0CD2` still open.

## Scripts

- `scripts/setup_rom.py` — extract/link shared zip
- `scripts/boot_probe.py` — headless menus → fight-ready `Stage1.state`
- `scripts/ram_probe.py` — walk/attack differentials from Stage1
- `scripts/run_stage1_segment.py` — multi-wave chain; JSON + PNGs in
  `recordings/`; mid clears as `Stage1_Clear_w*_cam*.state`
- `scripts/damnd_probe.py` — Boss/door fight via `Stage1Policy`; HP deltas
- `scripts/door_jump_clear.py` — park-bait / JD door clear + Damnd spam-Y
- `scripts/stage2_advance.py` — Stage1_Clear → subway `Stage2` + early
  waves; also resumes `Stage2*` mid-states (`--state Stage2_Clear_w2_cam537`)
- `scripts/stage3_advance.py` — Stage2_Clear → West Side `Stage3` (+
  Break Car) + early waves; resumes `Stage3*`
- `scripts/stage3_bridge_probe.py` — minimal CLEAR_AREA → Stage3 probe
- `scripts/sodom_probe.py` — Sodom UP+Y throw kill (`--mode kill`) /
  cold Drawn chip+flee Mid (`--mode chip`)
- `scripts/leftover_kill_probe.py` — Area2 leftover attack sweep (+ jd90)
- `scripts/wave4_instrument.py` — per-chip HP/life/food log for waves 3–4
- `scripts/alley_probe.py` — hit/miss + geometry for alley combat
