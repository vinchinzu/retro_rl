# Status — TMNT IV: Turtles in Time


## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M8 |
| Best verified result | Continuous hard clear + credits |
| Last verification | 2026-07-25 |
| Runtime class | Bronze |
| Intervention class | Resource-assisted + Protection-assisted |
| Next publication target | Bronze / Clean (unassisted; maturity stays M8) |

| Field | Value |
|-------|-------|
| Status | **Continuous low-assist hard clear + staff/cast credits** |
| Integration | `TMNTIV-Snes` |
| ROM zip | `Teenage Mutant Ninja Turtles IV - Turtles in Time.zip` |
| Final replay state | None — capture boots from power-on (`NONE`) |
| Video | [Prior continuous hard clear with sound](../recordings/tmnt_iv_full_hard_credits.mp4) ([manifest](../recordings/tmnt_iv_full_hard_credits.json)); sub-hour route is dry-run verified |
| Latest dry-run | [manifest](../recordings/tmnt_iv_full_hard_dry_run.json) |
| Current baseline | [BASELINE_METRICS.md](BASELINE_METRICS.md) |

One unbroken emulator session from power-on through hard-mode staff/cast
credits. Zero life losses, zero state loads, zero stage/lives writes, no
A-special uses. **Low-assist** (not the old full-bar spam): emergency HP
top-up only when HP ≤ 16 (restore to 80), plus form-2 Super Shredder iframe
hold at 1. Manifest records every intervention.

Maturity gates stop at **M8**. “Unassisted bronze” means the same M8
continuous clear with intervention class **Clean** (zero HP/iframe RAM
writes), not a new M9 gate.

## Stage 1 Clean track (pizza-only, path-RNG suite 2026-07-27)

Full Big Apple **heal=none** = **no emergency HP writes**; only natural
pizza (`0x30`) restores. Path-RNG suite (2/2 identical full passes):

`SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy uv run python -m tmnt_iv.scripts.probe_stage1_clean --suite`

Evidence: [clean_suite.json](../recordings/stage1_clean_track/clean_suite.json).

| Entry | Outcome | Frames | Damage | Min HP | Notes |
|-------|---------|--------|--------|--------|-------|
| **`Stage1`** | stage_advance | **15,237** | **108** | **30** | fight-ready checkpoint |
| **`power_on`** (NONE→menus→S1) | stage_advance | **15,046** | **138** | **10** | full Big Apple path |
| **`Stage1_BeforeBoss`** | stage_advance | **5,323** | **40** | **44** | Baxter entry |
| **`Boss`** | stage_advance | **5,323** | **40** | **44** | Baxter only |

Prior rows: 2026-07-25 Stage1 heal=none **14,921 / 130**; Baxter **4,293 / 64**.
Current Baxter **40** (standoff poke); full stage **108–138** pizza-only.

Clean policy (Stage 0, path-flexible):

- **Pizza-first**: HP ≤ 68 seeks out to 260px; ≤ 48 screen-wide; Baxter
  survival grab when HP ≤ 32. No empty-screen `RIGHT+Y` spam.
- **Baxter left standoff** (HP-adaptive width/cadence): never walk into his
  body; jump-slash when elev ≥ 10 and HP > 20.
- **Hazard dodge offline in production** — A/B: jump-through dodge caused
  Clean mid-wave deaths; pizza + spacing survives path RNG better.
  `HazardAvoid` kept for tests / future phase-safe work.
- Elevated jump-slash for true-air targets (elev ≥ 44).

Suite stresses checkpoint vs power-on entry (different wave timing). Historical
`Stage1_Clear_w*` mid-locks remain optional probes (different spawn tables).

Whole-run continuous still uses emergency HP outside Stage 1 Clean probes.

**Do not relearn Stage 1:** [`CLEAN_PLAYBOOK.md`](CLEAN_PLAYBOOK.md)
(traps, rollout order for stages 2–9 Clean).

## Stage 3 Clean track (Sewer Surfin' — in progress 2026-07-27)

Probe: `scripts/probe_stage3_clean.py` (`--suite` / `--state` /
`--from-stage2-clear`). Evidence dir: `recordings/stage3_clean_track/`.

**Prefer `LiveHardStage3` (lives=2).** `Stage3` / `Boss3` are last-life
(lives=0) and die on the post-kill `event=0x0B` fade even after Rat King
HP hits 0 (known checkpoint artifact). Emergency on LiveHardStage3 still
stage-advances (~9–13k f / 2–4 heals).

| Entry | heal=none | Notes |
|-------|-----------|-------|
| **`LiveHardStage3`** | life_loss ~6.3k f / ~72 dmg / entry ~18 | Residual 0x1C spikes (−16) |
| **`Boss3`** | kills boss (68 dmg / min 12) then 0x0B death | Fade artifact |
| **`Stage3`** | life_loss mid-boss entry ~38 | Last-life checkpoint |
| Stage2_Clear bridge | same early/mid bottleneck | lives=0 after Metalhead |

Policy landings this pass (keep):

- **Sewer dumpster / WalkProgress stall thrash offline** — auto-scroll
  freezes X; UP/DOWN thrash walked into spikes (4×16 empty-band hits).
- **`SewerSpikeAvoid`**: jump-right when 0x1C/0x2C adx ≤ 56 (A/B cut
  empty-band spikes 3→1). 0x1C/0x2C in `HAZARD_CHAR_IDS`.
- **Rat King**: boss_active down to HP 1 (old floor 4 abandoned finishers);
  grounded RIGHT at left wall (B+RIGHT soft-lock x≈24); combat-stall
  suppress; between-wave pizza seek like Alleycat.
- Spike LEFT thrash **rejected** (4 spikes).

Next: zero residual 0x1C columns on LiveHard entry so boss entry HP ≥ 70
(Boss3 takes ~68 dmg to kill); then suite green.

## Stage 2 Clean track (Alleycat — in progress 2026-07-27)

Probe: `scripts/probe_stage2_clean.py` (`--suite` / `--state` /
`--from-stage1-clear`). Evidence dir: `recordings/stage2_clean_track/`.

| Entry | heal=none | Notes |
|-------|-----------|-------|
| **`Boss2`** (Metalhead) | **CLEAR** ~3,881f / 38 dmg / min 26 | Boss Clean OK |
| **`Stage2_Clear_w17_*`** pre-boss | **CLEAR** ~4,493f / 49 dmg / min 19 | Late alley OK |
| **`Stage2`** full checkpoint | **life_loss** ~8,391f / 104 dmg / min 8 / 1 pizza | Early/mid waves |
| Stage1_Clear bridge | life_loss | Same early-wave bottleneck |
| power-on through Alleycat | life_loss | Stage0 Clean holds; S2 dies |

Policy knobs kept for Alleycat Clean work:

- **Left flank + standoff 36** (emergency Stage2 **14,231f / 159 dmg / 2
  heals** without between-wave pizza; with between-wave pizza **~14.5k /
  183 / 1 heal**).
- **Pizza:** underfoot always; far seek **only between waves** (mid-wave
  chase → emergency 190→479).
- **Generic elev≥44 jump-slash stage-0-only** (Alleycat false air → ~443
  dmg emergency). Stage-specific B+Y (dino/stack/hover) unchanged.
- **Rejected:** mid-wave pizza seek, pack jump-hop thrash.

Bottleneck: early/mid Foot packs (incl. 0x5E 24-dmg pile-ons) after the
single pizza window. Metalhead is already Clean. Next: cut post-pizza
chip or a second safe pizza without mid-wave desync.

## Stage 2–3 damage pass (2026-07-25)

Deterministic fight-ready checkpoint probes under the production emergency
contract (HP ≤ 16 → 80). Each tuned result was reproduced twice and advanced
to the next stage naturally.

| Checkpoint | Policy | Outcome | Frames | Damage | E-heals | Lives |
|------------|--------|---------|--------|--------|---------|-------|
| `Stage2` | previous | stage_advance | 15,550 | 293 | 4 | 1→1 |
| **`Stage2`** | **tuned** | **stage_advance** | **15,453** | **124** | **1** | **1→1** |
| `Stage3` | previous | stage_advance | 20,172 | 572 | 9 | 0→1 |
| **`Stage3`** | **tuned** | **stage_advance** | **7,768** | **112** | **2** | **0→0** |

Checkpoint delta: Alleycat Blues **−169 damage (−58%) / −3 heals**;
Sewer Surfin' **−460 damage (−80%) / −7 heals / −12,404 frames**.

The production policy also cleared both levels back-to-back from `Stage2`
with one policy instance and no state load between them: **24,963f / 294
total damage / 5 heals / lives 1→1**. Subtracting the identical Stage 2
prefix gives the natural-entry Sewer segment **9,510f / 170 damage / 4
heals**. That is the closer continuous-context estimate; the 112-damage
`Stage3` row above is the standalone fight-ready checkpoint.

Policy changes:

- Alleycat Blues: attack range 65, vertical tolerance 6, cadence 1f Y /
  2f release.
- Sewer waves: attack range 64, vertical tolerance 36, cadence 2f Y /
  2f release. The wider lane prevents long spike-lane alignment chases.
- Rat King: attack range 120, vertical tolerance 32, and JUMP+RIGHT only
  at the true left edge (`player_x ≤ 80`).

The completed power-on dry-run below now provides the exact continuous
context. The checkpoint gain still did not transfer to Alleycat Blues
(**4:54.166 / 376 damage**), while Sewer Surfin' improved again to
**2:55.843 / 202 damage** in the sub-hour route.

## Continuous hard-run proof (2026-07-25 / sub-hour Raphael route)

| Metric | Previous re-probe | Tank + wall fixes | Stage 2–3 pass | **Sub-hour Raphael** |
|--------|-------------------|-------------------|----------------|------------------------|
| Power-on → credits | 01:09:46.389 | 01:05:41.709 | 01:04:07.131 | **00:57:19.635** |
| Damage taken | 7,959 | 6,851 | 6,869 | **4,667** |
| HP interventions | 108 | 93 | 91 | **65** (HP ≤ 16 → 80) |
| Form-2 iframe frames | 4,482 | 3,887 | 3,824 | **4,635** |
| Life losses | 0 | 0 | 0 | **0** |
| Lives start / peak / end | 2 / 6 / 6 | 2 / 6 / 6 | 2 / 6 / 6 | **2 / 6 / 6** |
| Min HP seen | 2 | 2 | 2 | **2** |
| Frames to credits | 251,597 | 236,892 | 231,208 | **206,718** |

Δ vs Stage 2–3 pass: **−6:47.496 / −24,490 frames**,
**−2,202 damage**, and **−26 heals**. Form-2 protection increased by
811 frames; zero life losses held.

### Damage by stage (new)

| Stage | Damage | Δ vs Stage 2–3 pass |
|-------|--------|---------------------|
| Big Apple | 334 | +10 |
| Alleycat Blues | 376 | +30 |
| Sewer Surfin' | **202** | **−38** |
| Technodrome (duo + tank) | **1,022** | **−240** |
| Prehistoric (Slash) | 861 | −283 |
| Skull & Crossbones | **306** | **−454** |
| Wounded Knee | **579** | **−580** |
| Neon Night Riders | 238 | −180 |
| Starbase | **749** | **−467** |
| Final Shell Shock | 0 (iframe guard) | 0 |

The production boot now selects Raphael through the real character menu.
Character-specific Wounded Knee cadence cut that stage from **6:46.447 /
1,159 damage** to **4:46.695 / 579**. Starbase releases B/Y between
closing jumps and bypasses the generic stall detector during its frozen
launch frames; this removed both observed Starbase soft-locks.

Policy also retains the duo left-flank poke with right-door jump recovery,
Super Shredder form-2 wall-aware dodge cycle, emergency-only HP assist, and
**Slash whiplash** (lab-ported). The current whole-run Prehistoric segment
is **6:47.579 / 861 damage**.

### Raph state capture + continuous lessons (2026-07-27)

**Infrastructure:** continuous-faithful Raphael (char 8) states + capture script.

| State | Meaning | Notes |
|-------|---------|-------|
| `RaphFullHardStage4` | Technodrome entry | Clears **30,379f / 886 dmg / 13 heals** |
| `RaphFullHardDuo` | Tokka/Rahzar first live | |
| `RaphFullHardTank` | Tank event 0x18 | Mid-empty-foot frame soft-locks; prefer Stage4 |
| `RaphFullHardBoss5` | Slash first live | Baseline **11,386f / 478 / 6** |
| `RaphFullHardBoss9` | Super Shredder form 1 | Baseline **9,120f / 136 / 2** (through stage 9) |

Capture: `uv run python -m tmnt_iv.scripts.capture_raph_states`  
Slash from prehistoric entry: start `RaphDiagStage5` / `RaphFastStage5`.

**Leo / early Raph probe KEEPs vs continuous dry-run:**

| Knob | Raph probe | Continuous dry-run |
|------|------------|--------------------|
| Slash approach 36 + cross 18/12 (Leo) | — | Soft-lock WK; Prehistoric worse |
| `blocker_hit_frames` 8 | Stage4 **worse** (1,157 dmg) | **01:01:28 / 5,431** |
| `slash_spin_dodge_adx` 40 | Boss5 **6,765f / 226 / 3** (5/5) | **00:57:52 / 5,474 / 78 heals** |
| `slash_spin_dodge_adx` 44 | Boss5 8,957 / 298 / 4 | **00:57:31 / 5,152 / 74 heals** |
| production spin 52 | Boss5 11,386 / 478 / 6 | **00:57:19 / 4,667 / 65** (best) |

Faster Slash changes later-stage RNG; probe wins that shrink fight length
need a **full-route re-tune**, not a blind port. Production keeps spin 52.

**Kept safety:** Wounded Knee stall Y-quantize + B+Y escape on elevated `0xb0`
(happy-path-neutral; continuous still **00:57:19.635 / 4,667 / 65**).

Next continuous ROI: re-tune the spin-40 trajectory (Skull/WK blew up) **or**
grind late-route Raph states (Starbase/Boss9) where path desync is smaller.

- Hard flag stayed at WRAM value `2`; hard-credits event `0x1A` observed
- Dry-run: `uv run python -m tmnt_iv.scripts.record_full_hard_run --dry-run`

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
  96): long poke (`attack_range=120`) from water lane reduces HP to
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

- **Bronze / Clean** continuous proof: zero emergency HP heals and zero
  form-2 iframe guard without losing continuity (Stage 1 segment
  heal=none is done; roll into whole-run dry-run next).
- Cut Slash time/damage (still a large whole-run bucket).
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

Ticket boards (imported from Super Metroid process learnings):

| Board | Path |
|-------|------|
| Live queue | [`tasks/QUEUE.md`](tasks/QUEUE.md) |
| Critical path | [`tasks/TRIAGE.md`](tasks/TRIAGE.md) |
| Backlog index | [`tasks/BACKLOG.md`](tasks/BACKLOG.md) |
| Clean track contract | [`CLEAN_TRACK.md`](CLEAN_TRACK.md) |
| ★ Clean full continuous | [`tasks/T4-CLEAN-FULL.md`](tasks/T4-CLEAN-FULL.md) |

1. **Wave-1 Clean infra:** `T4-CLEAN-CONTRACT` / `ARTIFACTS` / `CLI` /
   `INTEGRITY` (artifact isolation + `--clean` + zero-assist asserts).
2. **Clean stage rollout:** `T4-CLEAN-S2` (Alleycat) → `S3` (Sewer) → …
   → `S9` (form-2 without iframe) → ★ `T4-CLEAN-FULL`. Follow
   [`CLEAN_PLAYBOOK.md`](CLEAN_PLAYBOOK.md); do not re-open Stage 1
   hazard jump-dodge or global pizza seek.
3. **Parallel assisted improve:** `T4-ASSIST-TECHNO` (1,022 dmg) and other
   `T4-ASSIST-*` cards → `T4-ASSIST-DRYRUN` before BASELINE promote.
4. Publish Bronze / Clean when continuous hard clear has zero HP/iframe
   assists and 0 life losses (STATUS **secondary** until program decision).

Whole-run baseline (2026-07-25): **00:57:19.635** / **4,667 dmg** /
**65 heals** / **0 lives lost** — see
[BASELINE_METRICS.md](BASELINE_METRICS.md).

Biggest remaining damage buckets: **Technodrome 1,022**, Prehistoric 861,
Starbase 749, and Wounded Knee 579.

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
run) cleared it. The tank + wall run reached Prehistoric at **26:36.304**
with Technodrome damage **1,412** (was 31:08.490 / 2,400); the Stage 2–3
pass reached it at **23:05.901 / 1,262**, and the current route reaches it
at **22:23.654 / 1,022**.

Form-2 wall-aware flanking reduced the isolated `Boss9_phase2` checkpoint
from **3,825f** to **2,631f**. Route timing makes the current whole-run
iframe hold **4,635f** (up from 3,824f in the prior baseline). Next: tank
throw efficiency, fewer emergency restores, and form-2 iframe removal.
