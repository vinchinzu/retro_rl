# Status — TMNT IV: Turtles in Time

| Field | Value |
|-------|-------|
| Ladder rank | 3 |
| Tier | 1 |
| Status | **Continuous low-assist hard clear + staff/cast credits** |
| Integration | `TMNTIV-Snes` |
| ROM zip | `Teenage Mutant Ninja Turtles IV - Turtles in Time.zip` |
| Final replay state | None — capture boots from power-on (`NONE`) |
| Video | [Continuous hard clear with sound](../recordings/tmnt_iv_full_hard_credits.mp4) ([manifest](../recordings/tmnt_iv_full_hard_credits.json)) |
| Latest dry-run | [manifest](../recordings/tmnt_iv_full_hard_dry_run.json) |
| Frozen baseline | [BASELINE_METRICS.md](BASELINE_METRICS.md) |

One unbroken emulator session from power-on through hard-mode staff/cast
credits. Zero life losses, zero state loads, zero stage/lives writes, no
A-special uses. **Low-assist** (not the old full-bar spam): emergency HP
top-up only when HP ≤ 16 (restore to 80), plus form-2 Super Shredder iframe
hold at 1. Manifest records every intervention.

## Continuous hard-run proof (2026-07-24 / tank + wall recovery)

| Metric | Post-Slash whiplash | Previous re-probe | **Tank + wall fixes** |
|--------|----------------------|-------------------|-----------------------|
| Power-on → credits | 01:15:34.050 | 01:09:46.389 | **01:05:41.709** |
| Damage taken | 8,085 | 7,959 | **6,851** |
| HP interventions | 110 (HP ≤ 16 → 80) | 108 | **93** (HP ≤ 16 → 80) |
| Form-2 iframe frames | 7,467 | 4,482 | **3,887** |
| Life losses | 0 | 0 | **0** |
| Lives start / peak / end | 2 / 6 / 6 | 2 / 6 / 6 | 2 / 6 / 6 |
| Min HP seen | 2 | 2 | **2** |
| Frames to credits | 272,491 | 251,597 | **236,892** |

Δ vs previous re-probe: **−4:04.680**, **−1,108 damage**, **−15 heals**,
and **−595 iframe-guard frames**. Zero life losses held.

### Damage by stage (new)

| Stage | Damage | Δ |
|-------|--------|---|
| Big Apple | 322 | 0 |
| Alleycat Blues | 288 | 0 |
| Sewer Surfin' | 466 | 0 |
| Technodrome (duo + tank) | **1,412** | **−988** |
| Prehistoric (Slash) | 982 | −56 |
| Skull & Crossbones | 970 | −8 |
| Wounded Knee | 916 | −41 |
| Neon Night Riders | 407 | +261 |
| Starbase | 1,088 | **−276** |
| Final Shell Shock | 0 (iframe guard) | 0 |

Policy: duo left-flank poke with right-door jump recovery; Super Shredder
form-2 wall-aware dodge cycle;
emergency-only HP assist; **Slash whiplash** (lab-ported) — FullHardBoss5
probe **13.6k f / 616 dmg / 10 heals**; whole-run Prehistoric segment
**~7.2 min**.

- Hard flag stayed at WRAM value `2`; hard-credits event `0x1A` observed
- Re-encode video with `uv run python -m tmnt_iv.scripts.record_full_hard_run`
  when a fresh capture is needed

## Done

- ROM setup + integration stubs (`data.json` with known vars)
- Headless boot past intro / char select → fight-ready `Stage1.state`
- RAM map: player X/Y/HP, enemy slots HP/X/Y, menu/event/lives/stage
- `Stage1Policy` via `snes_oneshot.combat.fight_nearest_action`
  (`invert_vertical=False`; combat zeros progress-as-camera)
- Multi-wave Stage 1 chain (`WaveChainTracker`) + mid clears
- Stage 1 boss **Baxter Stockman** defeated → stage byte `0→1`
- Stage 1→2 bridge (`scripts/run_stage2_bridge.py`) → fight-ready
  `Stage2.state` (Alleycat Blues, lives 1, HUD + Foot)
- Early Stage 2 wave clears + mid states; April NPC filtered
  (`char 0xC4`); dumpster X-stall → DOWN + JUMP+RIGHT escape
- Far-park Foot (`x > ~244`) no longer `edge_wait` soft-lock
- Stage 2 alley boss **Metalhead** (`char 0x46`, HP 128, HUD
  `M. HEAD`) → `Boss2.state` → `Stage2_Clear` (stage byte `1→2`)
- Stage 2→3 bridge (`scripts/run_stage3_bridge.py`) → fight-ready
  `Stage3.state` (**Sewer Surfin'**, hoverboard, stage byte **2**)
- Stage 3 early waves (spike lane) + mid states; surf ghost slots
  (`char 0` / `x 0` / `y≥256`) filtered; sewer Y-clamp avoids spikes
- Stage 3 boss **Rat King** (`char 0x4A`, HUD `R. KING`, spawn HP
  96): long poke (`attack_range=140`) from water lane reduces HP to
  **0**; char despawns; event `0x0A→0x0B` (same pattern as Metalhead).
  States: `Boss3`, `Boss3_mid`, `Boss3_low` (HP18 / boss~24)
- **Historical `Stage3_Clear` / `Stage3_Clear_post` segment states**:
  post-clear cutscene walk
  (`event` 4→…→12, `stage=3`) → fight-ready **`Stage4`**
  (Technodrome corridor, Foot `0x5E`/`0x60`). Built by cloning
  `Stage2_Clear_post` with `ADDR_STAGE` set to 3 (natural Rat King
  fade was missing in the old isolated probe; the continuous hard run now
  traverses this transition naturally.
- Stage 4 Technodrome wave chain (`scripts/run_stage4_segment.py`);
  stage byte **3** confirmed. Mids through pre-boss
  (`Stage4_Clear_w*`). Policy: sewer rules only for stage `==2`.
- Stage 4 boss **Tokka & Rahzar** (`chars 0x48` / `0xA0`, spawn HP
  96 each, HUD `TOKKA`/`RAHZAR`). Both reducible to HP 0 with left-flank
  poke from `Boss4_hp80`; event `0x0A→0x0B` then stalls (same open
  fade as Rat King). States: `Boss4`, `Boss4_hp80`, `Boss4_low`.
- **Historical `Stage4_Clear` / `_post` segment states**: template clone of
  `Stage3_Clear_post` with `stage=4`. The continuous hard run reaches
  **`Stage5`** naturally (Prehistoric Turtlesaurus, stage byte **4**).
- Stage 5 Prehistoric wave chain (`scripts/run_stage5_segment.py`);
  stage byte **4** confirmed. Early mids `Stage5_Clear_w1`–`w7`.
  NPC filter: pterodactyl `0xEE`. Dinos `0x6C` need **jump-slash**
  (B+Y); cave bruisers `0xB0`/`0x76` (HP 40/32).
- Stage 5 boss **Slash** (`char 0x50`, spawn HP **160**). Grounded Y
  poke from `Boss5` / `Boss5_mid` reduces HP to **0**; event
  `0x0A→0x0B` then stalls (same open fade). States: `Boss5`,
  `Boss5_mid`, `Boss5_low`, `Boss5_hp0`.
- **Historical `Stage5_Clear` / `Stage6` segment states**: template clone of
  `Stage4_Clear` with `stage=5`. The continuous hard run reaches fight-ready
  **`Stage6`** naturally (stage byte **5**, Foot `0x62`).
- Stage 6 **Skull and Crossbones** wave chain
  (`scripts/run_stage6_segment.py`); stage byte **5** confirmed.
  Mids `Stage6_Clear_w1`–`w14` (deeper mids often need HP heal).
  Foot `0x60`/`0x62`/`0x68`, pirates `0x70`/`0x66`, bruisers
  `0xB0`/`0xB2` (HP 40).
- Stage 6 bosses **Bebop** (`0xA8`) + **Rocksteady** (`0xAC`), spawn
  HP **128** each. Left-flank poke (same as Tokka/Rahzar). Bebop HP→0
  (Rocksteady may still have HP) → despawn → event `0x0A→0x0B` →
  **natural** `event=0x19` / `stage=6` (~580f). States: `Boss6`,
  `Boss6_hp80`, `Boss6_mid`, `Boss6_low`.
- **`Stage6_Clear` / `_post`** (natural, not a template clone) →
  fight-ready **`Stage7`** (stage byte **6**, Foot `0x60`).
- Stage 7 **Bury My Shell at Wounded Knee** wave chain
  (`scripts/run_stage7_segment.py`); stage byte **6** confirmed.
  Early mids `Stage7_Clear_w1`–`w8` + deeper `*_cam59*`. Foot
  `0x60`/`0x68`/`0x66`/`0x6A`, train bruisers `0xB8`, stacked bazooka
  Foot `0xB0`/`0xB6` need **jump-slash** (B+Y) for the top soldier.
- Stage 7 boss **Leatherhead** (`char 0xA2`, HUD `L. HEAD`, spawn HP
  **172**). Grounded Y poke → HP 0 → event `0x0A→0x0B` → **natural**
  `event=0x19` / `stage=7` (~720f after 0x0B). States: `Boss7`,
  `Boss7_mid`, `Boss7_low`, `Boss7_hp0`.
- **`Stage7_Clear` / `_post`** (natural) → fight-ready **`Stage8`**
  (Neon Night Riders, stage byte **7**, Mode-7 highway).
- Stage 8 **Neon Night Riders** wave chain
  (`scripts/run_stage8_segment.py`); stage byte **7** confirmed.
  Mode-7: wait for near-band Foot (`y≥140`, chars `0x86`/`0x88`/
  `0x8A` HP2); do not chase vanishing-point slots. Props
  `0x36`/`0x3C`/`0xAC`@HP2 filtered. Mids `Stage8_Clear_w1`–`w11+`.
  Dev heals on low HP for long highway runs.
- Stage 8 boss **Krang** (`char 0x4E`, spawn HP **160**). Left-flank
  Y poke → HP 0 → event `0x16→0x0B` → **natural** `event=0x19` /
  `stage=8` (~1180f after HP0). States: `Boss8`, `Boss8_hp100`,
  `Boss8_hp5`, `Boss8_hp0`.
- **`Stage8_Clear` / `_post`** (natural) → fight-ready **`Stage9`**
  (Starbase, stage byte **8**, Foot `0x5C`).
- Stage 9 **Starbase** wave chain (`scripts/run_stage9_segment.py`);
  stage byte **8** confirmed. Hover/teleporter Foot `0x6A` (and
  `0x6C`/`0xB0`/`0xB2`/`0xB4`/`0xF2`) need **jump-slash** (B+Y) or
  grounded Y soft-locks. Dev heals + `player_lives` for long runs.
  Mids `Stage9_Clear_w*` / `Stage9_Clear_js_w*` through pre-boss.
- Stage 9 boss **Super Shredder** form 1 (`char 0x52`, spawn HP
  **128**) → natural fade → form 2 arena (stage byte **9**). Form 2
  (`char 0xAE`, spawn HP **~190**): grounded Y via default policy
  (forced left-flank / jump-slash whiff). HP→0 → `0x0A→0x0B` →
  **natural** `event=0x19` / `stage=10` normal-mode ending dialogue → title in
  the older segmented normal run. The continuous hard run instead observes
  `event=0x1A` and follows the staff/cast credits through the final scene. States:
  `Boss9`, `Boss9_mid`, `Boss9_phase2`, `Boss9_phase2_mid`/`_low`/
  `_hp0`, `Stage9_Clear`, `Ending`, `Ending_title`.

## Not done

- Unassisted zero-loss proof: remove emergency HP heals and the form-2
  iframe guard without losing continuity.
- Cut Slash time/damage (still ~40% of total damage and ~22 min of the run).
- True camera scroll word (still using `0x003A` progress heuristic)

## Preferred resumes

| State | Notes |
|-------|-------|
| `Stage1` | Fight-ready highway spawn |
| `Stage1_BeforeBoss` / `Boss` | Baxter spawn |
| `Stage1_Clear` | Post-Baxter fade (`stage=1`, event `0x19`) |
| `Stage2` | Alleycat Blues fight-ready (lives 1, Foot present) |
| `Stage2_Clear_w2_cam19460` | Past first dumpster (HP 36, lives 1) |
| `Stage2_Clear_w17_cam27882` | Pre-Metalhead alley (HP 64, last life) |
| `Boss2` | Metalhead spawn (HP 64, boss HP 128) |
| `Stage2_Clear` / `Stage2_Clear_post` | Post-Metalhead (`stage=2`) |
| `Stage3` | Sewer Surfin' fight-ready (HP 80, last life, Foot present) |
| `Stage3_Clear_w2_cam32273` | Early surf mid (full HP) |
| `Boss3` | Rat King spawn (HP 80, boss HP 96) |
| `Boss3_mid` | Rat King mid (HP 20, boss HP ~40) |
| `Boss3_low` | Rat King low (HP 18, boss HP ~24) |
| `Boss3_hp0_*` / `Boss3_ev11*` | Post-kill debug (dies ~444f into `0x0B`) |
| `Stage3_Clear` / `_post` | Post-sewer cutscene (`stage=3`, event `4`) |
| `Stage4` | Technodrome fight-ready (stage byte **3**, Foot present) |
| `Stage4_Clear_w2_cam34763` | Mid corridor (HP 68) |
| `Stage4_Clear_w6_cam40097` | Pre-boss approach (HP 24) |
| `Boss4` / `Boss4_hp80` | Tokka+Rahzar spawn (natural HP24 / healed 80) |
| `Boss4_low` | Duo nearly dead |
| `Stage4_Clear` / `_post` | Post-Technodrome cutscene (`stage=4`, event `4`) |
| `Stage5` | Prehistoric fight-ready (stage byte **4**, Foot present) |
| `Stage5_Clear_w2_cam33919` | Early Prehistoric mid (full HP) |
| `Stage5_Clear_w5_cam35034` | Cave approach mid |
| `Stage5_Clear_w7_cam36913` | Deeper mid (low HP — heal before resume) |
| `Boss5` / `Boss5_mid` | Slash spawn HP160 / mid HP64 |
| `Boss5_low` | Slash low (~24) |
| `Stage5_Clear` | Post-Slash cutscene template (`stage=5`, event `4`) |
| `Stage6` | Skull and Crossbones fight-ready (stage byte **5**) |
| `Stage6_Clear_w2_cam33939` | Early pirate mid (heal if low) |
| `Stage6_Clear_w7_cam37571` | Mid ship (heal recommended) |
| `Stage6_Clear_w14_cam42957` | Pre-boss approach |
| `Boss6` / `Boss6_hp80` | Bebop+Rocksteady spawn (natural low / healed 80) |
| `Boss6_mid` / `Boss6_low` | Duo mid / Bebop near death |
| `Stage6_Clear` / `_post` | Natural post-pirate clear (`stage=6`, event `0x19`/`4`) |
| `Stage7` | Wounded Knee fight-ready (stage byte **6**, Foot present) |
| `Stage7_Clear_w2_cam50673` | Early train mid |
| `Stage7_Clear_w8_cam57962` | Mid train (heal recommended) |
| `Stage7_Clear_w3_cam59213` | Pre-boss train car |
| `Boss7` / `Boss7_mid` / `Boss7_low` | Leatherhead spawn HP172 / mid / low |
| `Stage7_Clear` / `_post` | Natural post-train clear (`stage=7`, event `0x19`/`4`) |
| `Stage8` | Neon Night Riders fight-ready (stage byte **7**) |
| `Stage8_Clear_w4_cam3074` | Early highway mid |
| `Stage8_Clear_w11_cam4599` | Mid highway (heal recommended) |
| `Boss8` / `Boss8_hp5` / `Boss8_hp0` | Krang spawn HP160 / low / dead |
| `Stage8_Clear` / `_post` | Natural post-Krang clear (`stage=8`, event `0x19`) |
| `Stage9` | Starbase fight-ready (stage byte **8**, Foot present) |
| `Stage9_Clear_w17_cam24255` | Mid Starbase (pre-hover soft-lock; use jump-slash) |
| `Stage9_Clear_js_w29_cam36866` | Deep Starbase mid → Boss9 |
| `Boss9` / `Boss9_mid` | Super Shredder form 1 (`0x52`, HP128 / mid) |
| `Stage9_Clear` | Post-form1 → form2 arena (stage byte **9**) |
| `Boss9_phase2` / `_mid` / `_low` | Form 2 (`0xAE`, HP~190 / mid / low) |
| `Boss9_phase2_low` | Replays the old normal-mode final kill → dialogue → title |
| `Ending` / `Ending_title` | Old normal-mode checkpoints; not hard staff credits |

## Next

Whole-run baseline (2026-07-24): **01:05:41.709** / **6,851 dmg** /
**93 heals** / **0 lives lost** — see
[BASELINE_METRICS.md](BASELINE_METRICS.md).

Biggest remaining damage buckets: **Technodrome 1,412**, Starbase 1,088,
Prehistoric 982. Neon rose to 407 in this route and remains variance work.

### Tokka/Rahzar (2026-07-23)

`CombatPositionStall` was overriding the duo left-flank poke after ~4s of
stationary chip and jump-escaping into the pack. Stall is now suppressed while
Tokka/Rahzar (or Bebop/Rocksteady) are alive.

| Probe (emergency HP≤16→80) | Before | After |
|----------------------------|--------|-------|
| Boss4 | 7,432f / 364 dmg / 5 heals | **3,218f / 116 dmg / 1 heal** |
| Boss4_hp80 | 3,970f / 132 dmg / 2 heals | 3,970f / 132 dmg / 2 heals (stall was idle) |
| Boss6_hp80 | 4,551f / 256 / 3 | **3,236f / 176 dmg / 2 heals** |

Re-probe 2026-07-24 (with `--heal emergency` now default):

| Probe | Result |
|-------|--------|
| Boss4 (full stage to advance) | 15,345f / 470 dmg / 7 heals |
| Boss4_hp80 (full stage to advance) | 15,231f / 440 dmg / 6 heals |
| Boss6_hp80 | **3,888f / 176 dmg / 2 heals** (confirms stall-suppress held) |

The first post-fix continuous attempt exposed a checkpoint-only blind spot:
Leo entered Tokka/Rahzar pinned at the right door (`x=224`) and plain LEFT
made no progress. `duo_wall_escape` (37 jump-left frames in the successful
run) cleared it. The next full run reached Prehistoric at **26:36.304** with
Technodrome damage **1,412** (was 31:08.490 / 2,400).

Form-2 wall-aware flanking reduced the isolated `Boss9_phase2` checkpoint
from **3,825f** to **2,631f** and whole-run iframe guard use from 4,482f to
3,887f. Next: tank throw efficiency, Starbase below 1,000, and form-2 iframe
removal.
