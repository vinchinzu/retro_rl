# Agent Instructions — tmnt_iv

SNES TMNT IV: Turtles in Time linear-combat reference clear (M8). Shared helpers:
`snes_oneshot/`. Program notes:
`snes_oneshot/docs/GAME_SELECTION_NOTES.md`.

## Norms

- Prefer development save states and segment scripts over uninterrupted runs.
- Store `.state` files under `custom_integrations/TMNTIV-Snes/`.
- Keep RAM maps and game policy here; reuse `snes_oneshot.combat` /
  `segment_runner` — elevate only when clearly shared.
- Headless probes: `SDL_VIDEODRIVER=dummy` (and audio dummy as needed).
- Docs: `docs/STATUS.md`, `docs/plan.md`, `docs/ram_map.md`.
- Do **not** mash START once Stage 1 HUD is live (pauses the game).
- Avoid special (**A**) — it drains HP.

## Immediate goal

**Continuous low-assist hard clear done** (M8, Bronze / Resource+Protection).
Next publication target: **Bronze / Clean** (unassisted — maturity stays M8,
not a new gate). Stage 1 segment **heal=none clear** (2026-07-25):
**14,921f / 130 dmg / 0 heals** from `Stage1.state` (3/3); Baxter `Boss`
**4,293f / 64 dmg** heal=none. Exact power-on dry-run:
**00:57:19.635 / 4,667 dmg / 65 heals / 0 lives lost** (−6:47.496).
The production menu route selects Raphael; keep the Starbase launch guard
and pulsed stack jumps intact. Next: reconcile Stage 1/2 entry context and
reduce later assists without regressing the sub-hour clear.

## Scripts

- `scripts/setup_rom.py` — extract/link shared zip
- `scripts/boot_probe.py` — headless menus → fight-ready `Stage1.state`
- `scripts/ram_probe.py` — walk/attack differentials from Stage1
- `scripts/run_stage1_segment.py` — multi-wave Stage 1 / Baxter
- `scripts/run_stage2_bridge.py` — Stage1_Clear / BeforeBoss → `Stage2`
- `scripts/run_stage2_segment.py` — Alleycat Blues wave chain
- `scripts/run_stage3_bridge.py` — Stage2_Clear → fight-ready `Stage3`
- `scripts/run_stage3_segment.py` — Sewer Surfin' wave chain / Rat King
- `scripts/run_stage4_segment.py` — Technodrome wave chain / Tokka+Rahzar
- `scripts/run_stage5_segment.py` — Prehistoric wave chain / Slash
- `scripts/run_stage6_segment.py` — Skull and Crossbones / Bebop+Rocksteady
- `scripts/run_stage7_segment.py` — Wounded Knee / Leatherhead
- `scripts/run_stage8_segment.py` — Neon Night Riders Mode-7 / Krang
- `scripts/run_stage9_segment.py` — Starbase waves / Super Shredder
- `scripts/record_full_hard_run.py` — deterministic power-on → hard credits
  capture with native audio, live footer, final metrics, and JSON manifest
- `scripts/run_local_grind_agent.py` — **preferred**: multi-turn Ollama
  tool agent (`list_knobs` / `run_baseline` / `run_trial` / `inspect_trial`
  / `finish`). Whitelist knobs in `grind_knobs.py`; prompts in
  `local_grind/prompts/agent_system.md`. Artifacts:
  `recordings/local_grind_agent/{agent_trace.jsonl,summary.json,trials/}`.
  Does **not** auto-edit `policy.py`. Example:
  `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy uv run python -m tmnt_iv.scripts.run_local_grind_agent --focus slash --max-trials 2`
- `scripts/run_local_grind.py` — older single-shot propose/eval loop
  (non-agent). Prefer the tool agent above.

## RAM quick ref

Player base `0x0400` (`X+0x08`, `Y+0x0C`, `HP+0x4A`). Enemies
`0x08D0 + i*0x70`. Lives `0x1AA0` (0 = last life). Menu `0x0032`
(`0x06` = playing). Stage id `0x0082` (0=S1, 1=S2, **2=S3 Sewer
Surfin'**, **3=S4 Technodrome**, **4=S5 Prehistoric**, **5=S6 Skull
and Crossbones**, **6=S7 Wounded Knee**, **7=S8 Neon Night Riders**,
**8=S9 Starbase**, **9=Super Shredder form 2**, **≥10=ending sequence**).
Progress heuristic `0x003A`. Difficulty `0x1FEE` (`2` = hard); continue
setting `0x1FF2`; invulnerability timer `0x046E`. Stage 1 boss = **Baxter**
(~96 HP). Stage 3 boss = **Rat King** (`char 0x4A`). Stage 4 bosses = **Tokka**
(`0x48`) + **Rahzar** (`0xA0`). Stage 5 boss = **Slash** (`0x50`,
spawn HP 160). Stage 6 bosses = **Bebop** (`0xA8`) + **Rocksteady**
(`0xAC`, spawn HP 128). Stage 7 boss = **Leatherhead** (`0xA2`, spawn
HP 172). Stage 8 boss = **Krang** (`0x4E`, spawn HP **160**). Stage 9
boss = **Super Shredder** form 1 `0x52` (HP 128) / form 2 `0xAE`
(HP ~190). See `docs/ram_map.md`.
