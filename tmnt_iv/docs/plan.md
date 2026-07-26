# Plan — TMNT IV: Turtles in Time

Ladder #3 (tier 1). See
`snes_oneshot/docs/GAME_SELECTION_NOTES.md` for program context.

## Control style

Side-scrolling beat-'em-up (move, jump, attack). Attack = **Y**; avoid
special **A** (HP cost).

## Useful RAM

Documented in `docs/ram_map.md`. Highlights:

- Player `0x0400 + {X:0x08, Y:0x0C, HP:0x4A}`
- Enemies `0x08D0 + i*0x70` (same relative layout; skip April `char 0xC4`)
- Menu `0x0032`, event `0x0070`, stage `0x0082`, lives `0x1AA0`

## Development approach

1. `uv run python scripts/setup_rom.py`
2. `SDL_VIDEODRIVER=dummy uv run python scripts/boot_probe.py`
3. Clear one segment at a time from save states
   (`scripts/run_stage1_segment.py`, `run_stage2_bridge.py`,
   `run_stage2_segment.py`, `run_stage3_bridge.py`,
  `run_stage3_segment.py`, `run_stage4_segment.py`,
  `run_stage5_segment.py`, `run_stage6_segment.py`,
  `run_stage7_segment.py`, `run_stage8_segment.py`,
  `run_stage9_segment.py`).
4. Continuous validation/capture:
   `uv run python -m tmnt_iv.scripts.record_full_hard_run`.

## Milestones

- Stage 1 Foot locks → Baxter → `Stage1_Clear` (done)
- Bridge → fight-ready `Stage2` (done)
- Stage 2 alley waves → Metalhead → `Boss2` / `Stage2_Clear` (done)
- Bridge → fight-ready `Stage3` Sewer Surfin' (done; stage byte **2**)
- Stage 3 waves → Rat King → natural next stage in continuous run (done)
- Stage 2–3 checkpoint damage pass (done): `Stage2` 293→124 damage /
  4→1 emergency heals; standalone `Stage3` 572→112 / 9→2; back-to-back
  natural-entry Stage 3 is 170 / 4. Exact power-on dry-run completed:
  Alleycat **346 damage / 5:16.163** (checkpoint gain did not transfer);
  Sewer **240 / 3:08.889** (**−226 / −1:22.747** vs prior full run).
- `Stage3_Clear` + bridge → fight-ready `Stage4` Technodrome (done;
  historical Clear state is a clone; continuous run transitions naturally)
- Stage 4 Technodrome waves → Tokka & Rahzar → natural next stage in the
  continuous run (done)
- `Stage4_Clear` + bridge → fight-ready `Stage5` Prehistoric (done;
  Clear is `Stage3_Clear_post` clone with `stage=4`)
- Stage 5 Prehistoric waves → Slash → natural next stage in the continuous
  run (done)
- `Stage5_Clear` + bridge → fight-ready `Stage6` (done; Clear is
  `Stage4_Clear` clone with `stage=5`)
- Stage 6 Skull and Crossbones waves → Bebop+Rocksteady → natural
  `Stage6_Clear` → fight-ready `Stage7` (done; stage byte **6**)
- Stage 7 Wounded Knee waves → Leatherhead → natural `Stage7_Clear`
  → fight-ready `Stage8` Neon Night Riders (done; stage byte **7**)
- Stage 8 Neon Night Riders waves → Krang → natural `Stage8_Clear`
  → fight-ready `Stage9` Starbase (done; stage byte **8**)
- Stage 9 Starbase waves → Super Shredder form 1 (`0x52`) → form 2
  (`0xAE`, stage byte **9**) → hard credits/event `0x1A` (done in one
  low-assist power-on run; stage byte **≥10**).
- Low-assist hard capture + manifest (done, post-Slash): 01:15:34.050, 8,085 damage,
  166 emergency heals (HP≤16→80), form-2 iframe only, zero life losses.
  Old every-hit restore-to-96 removed.
- Stage 1 Clean track (2026-07-25): `Stage1` + Baxter **heal=none** segment
  clears (pizza seek + Baxter left-lane + disengage 6f). The power-on
  low-assist bucket remained **324 damage**; reconcile entry context before
  the Bronze / Clean attempt (maturity stays M8).
- Stage 2–3 full dry-run (2026-07-25): **01:04:07.131 / 6,869 damage /
  91 heals / 3,824 iframe frames / 0 life losses**, improving the prior
  runtime by **1:34.578**. Big Apple pizza seek is stage-0-only; global
  pickup seeking soft-locked Skull & Crossbones on an unreachable slot.

## Local grind (optional)

Farm short damage/speed probes to Ollama (`gemma4:12b`) via
`scripts/run_local_grind.py`. Whitelist knobs only (`grind_knobs.py`);
KEEP does not auto-edit `policy.py`. Prefer `--focus slash` or
`technodrome_tank` before burning a full dry-run.

## Notes

Faster than Final Fight; same align-then-poke loop. Y axis is normal
screen coords (do not invert). Reuse `snes_oneshot.combat` rather than
forking Final Fight policy. Stage 2 dumpster freezes `player_x` while
`0x003A` still ticks — policy uses DOWN + JUMP+RIGHT escapes. Far-park
Foot need a widened right combat margin (not idle `edge_wait`).
Sewer Surfin' auto-scrolls with hanging spikes — clamp Foot fight Y
down, hold RIGHT for pace (stage byte **==2** only). Rat King:
extended-range poke (not constant jump-slash); JUMP+RIGHT only to
escape left chip. Older isolated low-HP probes died during the `0x0B`
transition; the HP-safe continuous run transitions naturally. Technodrome
duo: tank-screen throws plus close-range blocker handling.
Prehistoric: pterodactyl `0xEE` filtered; dino `0x6C` needs jump-slash
(B+Y); Slash (`0x50`, HP 160) is crossed and struck from behind before the
natural transition. Skull and Crossbones: left-flank duo poke;
Bebop HP0 often clears while Rocksteady still has HP — **natural**
fade to `Stage7`. Neon Night Riders (Mode-7): wait for near-band Foot
(`y≥140`); Krang (`0x4E`, HP 160) left-flank Y → natural fade to
`Stage9`. Starbase: hover Foot `0x6A` needs jump-slash; Super Shredder
form 1 `0x52` (HP 128) → form 2 `0xAE` (HP ~190). Capture assists: emergency
HP only when ≤16, plus form-2 iframe=1 against demutation; hard credits
via `event=0x1A`. Duo bosses use an explicit left-flank poke so Leo does not
freeze on the wrong side mashing Y.
