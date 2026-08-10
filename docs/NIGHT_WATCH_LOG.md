# Night Watch Log

Overnight watcher: hourly lane managers for beads (separate game lanes).
Session start: 2026-08-09 ~21:00 CDT.

Protocol:
- Cap 6 concurrent game managers
- Each manager owns one game label only
- Claim one bead → spawn workers → tests → bd sync + commit (no push)
- Prefer worktree isolation

---

## Ticks

### 2026-08-10 night — harvest rr-5go9 CLOSED (CROP_WATER refill densify)
- Lane: harvest only (`snes/harvest/`)
- Claimed **rr-5go9** (blocks parent **rr-5in** Gate B full)
- **Root:** after fence open, east→south stuck (29,30); soft-band UP re-entered
  gap; east densify thrash (41,32); past-fence densify self-hop (31,29)
- **Fixes** (`crop_planter.py`): east-only/south-only corridor legs; gap-south
  fallback after 3× stuck (29,30); gap-safe soft/south lip (cap UP low x);
  east_pond only y≥32; past-fence pure-south bail; densify west-step east of pond
- **Evidence:**
  - Dry fixture GREEN: can_peak=20 refill=1 watered=3
    `recordings/empty_can_refill_probe.json`
  - Power-on continuous: CROP_WATER success D9–D11 watered=6 each, D9
    refills=1 can=20; crop wet=2 dry=4 alive; overnights=10 money=$160
    mid_run=0 Clean `recordings/power_on_spring_to_summer.json`
  - Terminal return_home multi_nav timeout D12 (not water thrash)
- **CLOSED** rr-5go9. **rr-5in** remains open (Summer D1 + return_home residual)
- Units: test_water_refill + test_crop_planter_logic 70; day_plan 201/202
  (pre-existing directional_transition fail). No push.
- Next tip: rr-5in return_home after water days / re-refill can=2 thrash


### 2026-08-10 night — mega_man_2 rr-f3nr PARTIAL (Heat→Item-1 dual-green)
- Lane: mega_man_2 only (`nes/mega_man_2/`)
- Claimed **rr-f3nr** (residual of rr-54ui PARTIAL)
- **Path:** Heat→Air Item-1 scaffold (no cloud solid re-grid; no FCEUX human)
- **Inventory:** no prior Heat states/policies; AirFan `weapons=$00`
- **Dual-green:**
  1. `Heat1` entry — stage select LEFT from Wily(0) → Heat(8)
  2. Heat screen ≥1 — `HeatManPolicy` 3/3 ~244f HP24 from `Heat1`
- Scripts: `boot_heat_probe.py`, `run_heat_segment.py`; doc `HEAT_ITEM1_PATH.md`
- **CLOSED PARTIAL** rr-f3nr; residual **rr-808** (Heat mid) → **rr-809** (boss+Item-1)
  → **rr-810** (Air+Item-1 cam≥5). FCEUX pin protocol documented.
- Units: `nes/mega_man_2/tests` 13/13. Commit `1f1cfb3`. No push.
- Next tip: **rr-808** Heat mid/late from `HeatScreen1`

### 2026-08-10 night — SM rr-av5s PARTIAL mid dual temp floor y1606
- Lane: super_metroid only (`snes/super_metroid/`)
- Claim kept **rr-av5s** Pure Red→Hellway (K5 hop 12)
- **Progress (dual-stable):**
  1. Bottom IBJ 18/30 c150 → tunnel peak ~y1820 (skip climb_lower-before-IBJ)
  2. Tunnel seat → midplat hop → x171
  3. Midplat IBJ → **temporary floor y1606 p49** ×2 exact
- **Discovery:** human mid ledges = frozen rippers (not solid tiles)
- **Still RED:** upper from y1600 lip → Hellway `0xA2F7`; no export
- Controllers: `routes/kpdr/k5/red_to_hellway.py` phased mid + `_tunnel_to_midplat`
- Residual: `snes/super_metroid/docs/tasks/rr-av5s-residual.md`
- Parent **rr-dbu.8** open; no STATUS
- Units: `tests/test_kpdr_dev.py` 8 passed
- Next: open temp floor + upper WJ to Hellway dual green
- No push

### 2026-08-10T~03:44 CDT — zelda_i rr-rvae CLOSED (Gleeok TF 0x08 dual-green)
- Lane: zelda_i only (`nes/zelda_i/`)
- Claim kept **rr-rvae** (map+Gleeok path already partial dual)
- **Result: CLOSED** — Gleeok melee + HC + TF bit `0x08` dual-green 2/2
- Live facts:
  - From `Level4GleeokEnter` room **0x13**: body type **`0x43`** HP≈160,
    detached head **`0x46`**, fireball **`0x56`** (ignore)
  - Melee A-spam prefer heads → body absent → HC `0x1a` (containers 7→8) →
    free UP → TF room **0x03** → mid walk → **`tf&0x08`** (mode 18)
  - Dual exact **~4286f** ×2 assisted (`l4_rvae_gleeok_tf_dual.json`)
- Landed: `level4_boss_combat.py`, `scripts/run_level4_gleeok.py`,
  checkpoint **`Level4Complete`** (+ provenance)
- Units: `test_level4_dungeon` + hygiene **15 passed**
- Parent **rr-05fz** residual: natural key KEY-UP + continuous PostLadder→TF
- Epic **rr-q3n** stays OPEN (continuous/Clean residual)
- Not Clean STATUS; no push
- Next: `bd ready -l zelda_i` tip **rr-05fz** natural continuous or parallel **rr-38p**

### 2026-08-10 night — SM rr-av5s PARTIAL mid pocket-spin ~y1932
- Lane: super_metroid only (`snes/super_metroid/`)
- Claim kept **rr-av5s** Pure Red→Hellway (K5 hop 12)
- **Progress:**
  - Lower past pocket ~(219,1942) holds (right-wall hard ceiling; pure-A gain 0)
  - New mid primitive: **pocket spin** `B+LEFT+A` from ~(235,2091) → mid crouch
    seat **~(174–183,1932–1942)** dual-stable (human-matched launch)
  - Controllers: `_pocket_spin_mid` + mid climb prefers spin→IBJ over recatch thrash
- **Still RED:** tunnel y≤1880 (~50px gap); morph from crouch lip falls through;
  bottom IBJ 18/30 peaked y1799 once not dual-stable; Hellway not reached
- Residual: `snes/super_metroid/docs/tasks/rr-av5s-residual.md`
- Parent **rr-dbu.8** stays open; no STATUS / no export `post_ice_red_to_hellway_pure`
- Units: `tests/test_kpdr_dev.py` 8 passed
- Next: ledge-hop chain x≈185–203 (human y2255→1878) or stabilize IBJ through tunnel
- No push


### 2026-08-10T~03:10 CDT — mega_man_2 rr-fpd6 CLOSED (LL spawn decode)
- Lane: mega_man_2 only (`nes/mega_man_2/`)
- Claimed **rr-fpd6** (no other MM2 in_progress)
- **Result: CLOSED** — Lightning Lord spawn decoded + live-confirmed
- Findings:
  - Type IDs: **0x3E** `objects_kaminari_goro` (+ **0x3D** move, **0x3F** bolt)
  - Goblin/Air Tikki is **0x40** (night3–5 “type36” misread of `$0400`)
  - ROM placement (lsmmega/mm2): mapset **4**, x=`0xC0`, y=`0x20` first LL
  - Live: LL enters slots at prog **~961** (scr3 cam_x~193) under Clean play
  - Cloud y≈32–36; pure-jump min_sy~34 reaches altitude but **~28px short in X**
  - No camera≥5 / cloud land yet — residual stays on parent **rr-54ui**
- Evidence: `nes/mega_man_2/docs/LL_SPAWN_DECODE.md`, probe_*.json
- Script: `nes/mega_man_2/scripts/ll_spawn_probe.py`
- Units: mega_man_2 tests 10/10; test_docs 8/8
- No push
- Next (rr-54ui): Clean land on first LL cloud (kill rider + stand), chain mapset 5–6




### 2026-08-10 night — harvest rr-5in PARTIAL (sparse water + refill residual)
- Lane: harvest only (`snes/harvest/`)
- Claimed **rr-5in** Gate B full power-on→Summer income
- **GREEN progress:**
  - D1 handoff + multi-day sleep Clean (rr-m0wq holds)
  - ENSURE_CROP_SEEDS plant path works (rr-6byj holds) — `planted=1` west pocket
  - **Sparse water detect** fix: `detect_crop_resume_plots(min_count=1)` + singleton dry centers so dry_crops=2 enters WATER (was instant `water fail` with no plots)
  - Densify thrash bail at ~(25,30) queues east→south corridor charge (incl. north-lip stand 33,30)
- **RED residual:** empty-can never fills on live continuous — post-charge re-multihop thrash ~(29–31,29); `watered=0`; money stays $100
- Child bead **rr-o00y** (P1 bug, discovered-from rr-5in)
- Units: `tests.test_crop_planter_logic` + `tests.test_water_refill` green (`test_water_only_detects_sparse_partial_plant`)
- Evidence: `snes/harvest/recordings/rr_5in_residual.json`, `power_on_spring_to_summer.log`
- Parent **rr-20w** / **rr-5in** stay open; tip next: **rr-o00y** fill path then re-run `--power-on --end-of-spring`
- No push
### 2026-08-10T01:52 CDT — SM rr-0ue1 Bat→Red pure dual GREEN
- Lane: super_metroid only (`snes/super_metroid/`)
- Bead **rr-0ue1** CLOSED: Pure Bat Room → Red Tower return (K5 hop 11)
- **Evidence:** `bat-to-red` dual pure **718f** ×2 exact, room `0xA253`
  xy=(216,2443) pose=10 from `post_ice_below_to_bat_pure` ~(472,139) p12/42
- Controller: LEFT platform chain reverse of bat_to_below + left door into Red bottom
  (`play_bat_to_red`); reverse of red_tower_to_bat bottom exit
- Export: `scratch/post_ice_bat_to_red_pure.state` (+ dual)
- Parent **rr-dbu.8** stays IN_PROGRESS PARTIAL (hop0–11 GREEN; Alpha PB open)
- Next bead **rr-av5s**: Red → Hellway climb reverse
- Units: kpdr_dev + source_states green; no continuous / STATUS
- No push


### 2026-08-10T01:37 CDT — SM rr-85c4 Glass→West pure dual GREEN
- Lane: super_metroid only (`snes/super_metroid/`)
- Bead **rr-85c4** CLOSED: Pure Glass Tunnel → West Tunnel return (K5 hop 7)
- **Evidence:** `glass-to-west` dual pure **211f** ×2 exact, room `0xCF54`
  xy=(216,139) pose=10 from `post_ice_east_to_glass_pure` ~(216,395) p12
- Controller: LEFT `play_run_shoot_exit` (reverse of west_to_glass RIGHT)
- Export: `scratch/post_ice_glass_to_west_pure.state` (+ dual)
- Parent **rr-dbu.8** stays IN_PROGRESS PARTIAL (hop0–7 GREEN; Alpha PB open)
- Next bead **rr-abx5**: West → Below Spazer reverse
- Units: kpdr_dev + source_states green; no continuous / STATUS
- No push

### 2026-08-09T21:00 CDT — bootstrap
- Scheduler: hourly task `019fe966cf19` created (session + durable)
- Lanes to launch: super_metroid, smb, harvest, zelda_i, alttp_rando, mega_man_2
- Initial managers dispatched by watcher (see below)

### 2026-08-09T21:01 CDT / 2026-08-10T02:01Z — hourly tick
- `bd ready`: idle board (no in_progress); git clean
- Skipped: sm_rando (rr-3f3e P2 only), alttp/smz3 (no clear solver tips beyond shared epic)
- Dispatched 6 managers (worktree isolation):
  - super_metroid → target **rr-5if** Pure Ice Snake→PLM (P0; not oracle)
  - smb → target **rr-k0x** pure HL 8-3 leave (no hybrid)
  - harvest → target **rr-bhr** Gate B (fallback rr-53g)
  - zelda_i → target **rr-5lu** Z4.2 pure residual (under rr-q3n)
  - alttp_rando → target **rr-gbd.23** house→uncle edge
  - mega_man_2 → target **rr-iyya** Air Man past screen 2
- Manager ids: 019fe967-c28d… (SM), …6d68… (smb), …6d73… (harvest), …6d83… (zelda), …6daa… (alttp_rando), …6dbc… (mm2)

### 2026-08-09T21:05 CDT — managers launched (wave 1)

| Lane | Manager ID | Target bead |
|------|------------|-------------|
| super_metroid | 019fe967-7ee9-7f10-b91b-276d39fb623f | rr-5if Ice Snake→PLM |
| smb | 019fe967-7ee9-7f10-b91b-277cc8d00311 | rr-k0x pure HL 8-3 leave |
| harvest | 019fe967-7ee9-7f10-b91b-2789db274a91 | rr-bhr Gate B (prefer) |
| zelda_i | 019fe967-7ee9-7f10-b91b-2795ff8302b6 | rr-5lu / Z4 tip |
| alttp_rando | 019fe967-7eea-7da3-8fd7-a6b5303e656c | rr-gbd.23 house→uncle |
| mega_man_2 | 019fe967-7eea-7da3-8fd7-a6c302a36827 | rr-iyya Air Man |

Hourly re-dispatch: scheduler task 019fe966cf19 (1h, durable). Cap 6 lanes. No push.

### 2026-08-09T21:04 CDT — anti-double-dispatch note
- Immediate fire of hourly tick raced wave-1 (claimed board looked empty briefly) and may have spawned a second manager set for the same 6 beads.
- Scheduler prompt updated: **busy lane = any in_progress bead with that game label → skip spawn**.
- Wave-1 managers still primary (ids `…276d39fb623f` SM, `…277cc8d00311` smb, `…2789db274a91` harvest, `…2795ff8302b6` zelda, `…a6b5303e656c` alttp_rando, `…a6c302a36827` mm2).
- Next hourly ticks must only refill IDLE lanes.

### 2026-08-09T21:10 CDT — alttp_rando lane complete
- **CLOSED** `rr-gbd.23` house→uncle natural_entry (15 tests pass).
- Free capacity → spawn **sm_rando** manager for `rr-3f3e` (only ready) / prep toward `rr-gbd.25` when harness unblocks.
- Still busy: SM `rr-5if`, smb `rr-k0x`, harvest `rr-bhr`, zelda `rr-5lu`, mm2 `rr-iyya`.

### 2026-08-09T21:15 CDT — mega_man_2 lane complete
- **CLOSED** `rr-iyya` AirScreen2→s3 (~241f 3/3) and →s4 (~502f 3/3) via `AirManPolicy(start=screen2)`.
- Opened follow-up `rr-54ui` past screen 4 toward boss door.
- Code merged from worktree to main; 10 tests pass. Free capacity → continue MM2 `rr-54ui` or next idle tip.

### 2026-08-09T21:20 CDT — SMB lane DROPPED (human)
- Killed SMB pure HL manager; discarded overnight pure_hl WIP.
- Closed pure HL + open SMB TAS/frame-cut beads as parked (product **M8 Clean** stands).
- Night watch: **never spawn smb** again this session.

### 2026-08-09T21:22 CDT — sm_rando lane complete
- **CLOSED** `rr-3f3e` Landing BC train/evaluate/report ownership split (7 tests).
- No more ready sm_rando work; `rr-gbd.25` still blocked on `rr-gbd.33` harness.

### 2026-08-09T21:25 CDT — super_metroid Ice PLM GREEN
- **CLOSED** `rr-5if` dual pure 1756f×2 room 0xA890 beams 0x1007 (not continuous).
- Next: `rr-dbu.11` Ice pure stack acceptance / compose.

### 2026-08-09T21:28 CDT — zelda_i L4 live recon
- `rr-5lu` still IN_PROGRESS (Stepladder residual).
- Live graph: 0x71 empty → UP 0x61 Vires → BOMB_UP 0x51 Keese+key → LEFT 0x50.
- Room beads: rr-zchy / rr-yr77 / rr-h278 / rr-wqdu / rr-2ysf. Tip next: pure Vires clear.


### 2026-08-09T21:19 CDT — solver/meta lane (rr-gbd.33)
- **CLOSED** `rr-gbd.33` Benchmark: fail-closed resumable seed campaign runner.
- Added `retro_harness/seed_campaign.py`: `SeedCampaignRunner`, `SeedExecutionStatus`, atomic ledger (`os.replace`), `policy_factory(seed)`.
- Acceptance: INFRA_ERROR ordered rows → non-claimable; resume byte-identical; contract mismatch refuses resume; missing audit ≠ Clean.
- Unblocks **rr-gbd.25** sm_rando multi-seed early tip S/T dry-run (now ready P1).
- Tests: 8 new campaign + full benchmark suite green (64 with docs).
- Next: `rr-gbd.25` sm_rando multi-seed dry-run consumer (thin harness use).

### 2026-08-09T21:30 CDT — solver harness complete
- **CLOSED** `rr-gbd.33` SeedCampaignRunner (fail-closed, atomic ledger, resume byte-identical) — commit 4cfbebb.
- Unblocks `rr-gbd.25` sm_rando multi-seed dry-run → next free lane.

### 2026-08-09T21:32 CDT — SM Ice pure stack closed
- **CLOSED** `rr-dbu.11` outbound dual GREEN (Business→Gate→Acid→Snake→Ice PLM). Residual doc on main.
- Next product tip: `rr-dbu.7` continuous `--to ice` compose only (no STATUS without dual continuous green).

### 2026-08-09T21:35 CDT — zelda L4 first rooms pure dual-green
- CLOSED rr-zchy/yr77/h278/wqdu pure 2/2 (~1278f chain to first key).
- rr-5lu residual; tip **rr-2ysf** 0x50+ toward Stepladder.

### 2026-08-09T21:38 CDT — sm_rando multi-seed dry-run closed
- **CLOSED** `rr-gbd.25` ship→morph SeedCampaignRunner dry S/T 3/3 claimable (vanilla fixtures).
- Unblocks `rr-gbd.13` SMZ3 multi-seed pattern. Live shuffled ROMs still open.

### 2026-08-09T21:42 CDT — SMZ3 multi-seed dry-run closed
- **CLOSED** `rr-gbd.13` portal→house SeedCampaignRunner dry S/T 3/3 claimable (fixture substrate).
- Next free capacity: alttp_rando multi-seed `rr-gbd.26` if ready, else other tips.


### 2026-08-09T21:40 CDT — SM continuous `--to ice` compose (rr-dbu.7)
- **CLOSED** `rr-dbu.7` TipSpec/spine/catalog/graph compose for continuous `--to ice`.
- Parent tip `wave`; hops Business→Gate→Acid→Snake→Ice PLM (pure stack greens).
- Graph outbound Ice edges spine-emitted `continuous`; default CLI tip remains **`wave`**.
- **Not** STATUS-promoted; **not** dual continuous green (room gap Wave `0xADDE` → Business `0xA7DE`).
- Residual: `snes/super_metroid/docs/tasks/rr-dbu.7-residual.md`
- Next: **`rr-vqv3`** Wave→Business pure return stack (unblock dual continuous Ice).
- Units: 33 passed (`test_continuous_tips` + `test_k4_speed_branches` + `test_k4_ice_scaffold`).
- No push.

### 2026-08-09T21:45 CDT — MM2 post-s4 probe (open)
- `rr-54ui` still open: no camera≥5 / boss door.
- Checkpoint AirFanPlatform (prog~949 scr3); pure late RIGHT pits ~1047.
- Next: fan updraft/ladder from AirFanPlatform, not mid-air AirScreen4.

### 2026-08-09T21:48 CDT — SM continuous --to ice compose closed
- **CLOSED** `rr-dbu.7` compose wired (TipSpec/spine/catalog). No dual continuous STATUS.
- Blocker: Wave tip ends 0xADDE; ice hops start Business 0xA7DE → next **rr-vqv3** Wave→Business pure return.


### 2026-08-09T21:52 CDT — alttp_rando multi-seed opening S/T dry-run closed
- **CLOSED** `rr-gbd.26` house→uncle SeedCampaignRunner dry S/T 3/3 claimable (substrate=vanilla, seed_source=fixture).
- Consumer: `snes/alttp_rando/opening_tip_campaign.py` + `scripts/run_opening_tip_campaign.py`.
- Published: `snes/alttp_rando/docs/opening_tip_seed_campaign_dry.json`.
- Tests: 7 passed (`test_alttp_rando_opening_tip_campaign.py`). Fail-closed INFRA + non-vanilla uncle failure modes covered.
- Not shuffled-seed robustness; live path fail-closed without ROM/FirstPlay.
- Next free alttp_rando tip: ALTTPR patch fixture / next graph edges (no more ready gbd beads in this lane).

### 2026-08-09T21:50 CDT — alttp multi-seed dry-run closed
- **CLOSED** `rr-gbd.26` house→uncle SeedCampaignRunner dry S/T 3/3 claimable (vanilla fixtures).
- Multi-seed dry stack complete: sm_rando 25 + smz3 13 + alttp 26 (all fixture/vanilla labeled).

### 2026-08-09T21:55 CDT — SM Wave→Double pure GREEN
- **rr-vqv3** in_progress: Wave→Business return stack.
- First hop dual GREEN 560f×2 Wave→Double (0xADDE→0xADAD). Next: Double→Single.
- **SMB re-parked** (rr-k0x reopened by stray work; closed again per human).


### 2026-08-09T21:55 CDT — solver L0 certified snapshots (rr-gbd.32)
- **CLOSED** `rr-gbd.32` SnapshotAdapter + SnapshotEnvelope (EMULATOR_ONLY vs CERTIFIED_FULL_ENV).
- Worktree commits: `44c6edb` + beads `b50e4aa` under subagent-019fe98a… (not on main clone tip).
- Pool: `save_snapshot` / `load_snapshot` / `fork_snapshot`; raw path uncertified.
- Maturity: **real-ROM tested** (SM single-lane smoke). Next: **`rr-gbd.34`** branch-rollout batches.
- No push.

### 2026-08-09T21:58 CDT — L0 certified snapshots closed
- **CLOSED** `rr-gbd.32` SnapshotAdapter/Envelope CERTIFIED_FULL_ENV; 100-step fake replay; identity fail-before-mutate; SM rom_smoke single-lane.
- Next free capacity: `rr-gbd.34` branch-rollout batches over certified snapshots.

### 2026-08-09T22:15 CDT — L0 branch-rollout batches closed (rr-gbd.34)
- **CLOSED** `rr-gbd.34` deterministic branch-rollout batches over certified snapshots.
- Module: `retro_harness/branch_rollout.py` — `RolloutSpec`/`RolloutResult`/`BranchSpec`/`BranchResult`, order-independent `replay_digest`, exact `RolloutAccounting`.
- Acceptance: widths 1 and 4 identical (and input-order independent); controller exception isolates invalid branch; accounting exact.
- Tests: 8 unit (`test_branch_rollout.py`) + SM rom_smoke width-1 batch with isolation (`test_snapshot_rom_smoke.py`).
- Maturity: **real-ROM tested** (SM single-lane batch; multi-width fake-tested — stable-retro one emu/process). Second game consumer still required for publication-ready L0.
- Next free capacity: L0 second consumer / planner search consumer of rollouts; or next ready solver tip.

### 2026-08-09T22:05 CDT — zelda KEY-RIGHT 0x62 pure dual-green
- **CLOSED** `rr-2ysf`: 0x50 dead-end; Stepladder path is KEY-RIGHT 0x61→0x62.
- Pure 2/2 clear_50 / key_right_62 / clear_62. Next tip **rr-7r24** maze+compass+ladder.

### 2026-08-09T22:10 CDT — harvest Gate B diagnosis (open)
- `rr-bhr` still open: ExitToFarm drops free-move after pure truck→sleep; softlock → 0x5F.
- Fail-fast `farm_control_lost` landed. Next: pure truck+sleep re-record keeping gs free-move.


### 2026-08-09T22:05 CDT — SM Double→Single pure GREEN (rr-qpkd)
- **CLOSED** `rr-qpkd` dual pure 1101f×2 Double `0xADAD`→Single `0xAD5E` via `play_double_to_single_chamber`.
- Source: `post_wave_to_double_chamber_pure` ~(984,139); export `post_double_to_single_chamber_pure` ~(216,630).
- Parent **`rr-vqv3` remains open** (Wave→Business stack; 5 hops left). No continuous Ice STATUS.
- Next tip: `rr-u0y8` Single→Bubble return. Residual: `docs/tasks/rr-qpkd-residual.md`.

### 2026-08-09T22:15 CDT — L0 branch rollouts closed
- **CLOSED** `rr-gbd.34` RolloutSpec/Result + replay digest; widths 1≡4; exception isolates branch; SM width-1 rom smoke.
- L0 stack: .32 certified snapshots + .34 branch batches. Free capacity → next solver/product tip.


### 2026-08-09T21:57 CDT — solver SkillPolicy multi-step lifecycle (rr-d03j)
- **CLOSED** `rr-d03j` Exercise multi-step SkillPolicy lifecycle (not API narrow).
- Shared helpers: `retro_harness/skill_policies.py` — `ScriptedSkillPolicy` (RUNNING ticks) + `OneShotSkillPolicy` (macro terminal).
- `SolverSession`: RUNNING with `action=None` now raises (prevents hang / false timeout).
- SM production adapter: `RouteCommandPolicy` + vertical `InjectedFailurePolicy` subclass OneShot; multi-frame path owned by ScriptedSkillPolicy.
- Tests: multi-step RUNNING→SUCCESS, mid-skill replan, hang rejection, one-shot double-step; facade exports. 16 solver + full sm_rando suite green.
- Next free capacity: ready solver tips `rr-gbd.14` / `.15` (P3 L2/L3 sketches) or higher-priority non-smb product tips; L0 stack complete (.32+.34).

### 2026-08-09T22:20 CDT — SkillPolicy lifecycle closed
- **CLOSED** `rr-d03j`: ScriptedSkillPolicy multi-frame + OneShotSkillPolicy; RUNNING hang guard; SM adapters use OneShot.
- Solver free capacity idle (L0 + SkillPolicy done; product lanes busy).


### 2026-08-09T22:05 CDT — SM Single→Bubble pure GREEN (rr-u0y8)
- **CLOSED** `rr-u0y8` dual pure 817f×2 Single `0xAD5E`→Bubble `0xACB3` via `play_single_to_bubble`.
- Source: `post_double_to_single_chamber_pure` ~(216,630) deep climb (no fake top pin); export `post_single_to_bubble_pure` ~(472,395).
- Parent **`rr-vqv3` remains open** (Wave→Business stack; 4 hops left). No continuous Ice STATUS.
- Next tip: `rr-czg9` Bubble→Farm return. Residual: `docs/tasks/rr-u0y8-residual.md`.

### 2026-08-09T22:07 CDT — harvest Gate B causal diagnosis (rr-bhr open)
- **Still open** `rr-bhr`: pure truck→D2 bed works; shed grass+can blocked.
- **Causal:** `event_flags_1f68` truck D2 = `0x0011` vs Y1 `0x00B1`. Min free-move-safe mask **`0x00A1`** (truck `0x01` + morning intro `0x20` + dog owned `0x80`). Offline A/B on `town_day1_rest_end`: only pre-set `0xA1`/`0xB1` keeps free-move; `house_size` not causal.
- ROM: `CODE_83CEAE` fires outdoor intro when `0x20` clear → free-move never recovers (mash/name/mid-warp/clock). D1 farm without truck cannot leave town.
- Landed: `outdoor_intro_flags_ready`, richer `farm_control_lost`, STATUS/recon docs, unit test.
- **Next:** pure human/auto complete D2 dog intro then shed; do not claim acceptance without free outdoor + grass+can. Alternate progress: `rr-53g` harvest+ship 5pm.
- No push.



### 2026-08-09T22:08 CDT — zelda_i L4 0x62 compass maze pure dual-green
- **CLOSED** `rr-9so0` (tip residual was noted as rr-7r24 pre-create): dark maze scripted path → `ADDR_COMPASS|0x08` @~(136,132) → return LEFT 0x61 (~471f pure 2/2).
- Segment `compass_62` + `Level4Compass62Controller` + checkpoint `Level4Compass`.
- Evidence: `nes/zelda_i/recordings/l4_compass62_pure_compass_62.json`.
- Parent **`rr-5lu` still open**. Tip residual **`rr-o0nn`**: post-Compass → Stepladder (`ADDR_LADDER`); 0x51 key spent, UP blocked.
- No push.

### 2026-08-09T22:35 CDT — harvest Gate B outdoor-intro causal (open)
- Free-move after ExitToFarm requires event_flags_1f68 ≥ 0x00A1 (dog owned 0x80 + morning 0x20 + truck 0x01).
- Truck D2 is 0x0011; CODE_83CEAE dog-intro softlocks free-move. house_size not causal.
- Landed outdoor_intro_flags_ready (e966d7c). Next: pure complete D2 outdoor dog intro then shed.

### 2026-08-09T22:40 CDT — zelda L4 Compass pure dual-green
- **CLOSED** `rr-9so0` 0x62 maze → Compass bit 0x08 ~471f 2/2; return LEFT→0x61.
- Next tip: `rr-o0nn` post-Compass → Stepladder. Parent rr-5lu open.


### 2026-08-09T22:11 CDT — MM2 post-s4 overnight probe (rr-54ui open)
- **Claimed** `rr-54ui` in_progress; stayed open (no camera≥5 / boss door).
- Started from **AirFanPlatform** (grounded scr3 prog949), not mid-air AirScreen4.
- Corrected geometry: pink head = **Goblin/Air Tikki** (not updraft fan); ladder bar never feet=2.
- New checkpoint **AirLeftPlatform** (prog~902 left of Goblin); temp probe states pruned.
- Bird (Pipi) bounce: min_sy~23–26 with damage; best press **prog~1085–1086** scr4 still pit.
- Dense Goblin-head / cloud hop grids: 0 elevated lands (sy&lt;82).
- Policy docstring + STATUS/plan/AGENTS updated; units 10/10 pass.
- Next: 5px Goblin head land or Lightning Lord cloud ride / controlled Pipi→cloud.
- No push.

### 2026-08-09T22:15 CDT — harvest Gate B CLOSED (rr-bhr)
- **CLOSED** `rr-bhr` Power-on full D1→D2 shed on house_size=0 (Gate B).
- Breakthrough: softlock tilemap `0x5F` is intentional **dog name entry** (`$099F=3`), not door glitch. Completing name `AAAA` sets dog-owned → `event_flags_1f68=0x00B1` and restores free-move `0x4000`.
- Landed `CompleteOutdoorMorningIntroTask` (inventory.py); wired first in `_shed_starter_tools`.
- ROM evidence:
  - `snes/harvest/recordings/gate_b_dog_intro_shed.json` — rest_end → intro → grass+can (~4.8k f)
  - `snes/harvest/recordings/gate_b_anneve_full_shed.json` — peak mask `0x3F`, D2, grass+can, free+intro, `mid_run_state_loads=0`, `ram_writes=0` (~17.6k f)
- Unit tests: outdoor intro already-ready + name-entry presses + free-move helpers.
- Docs: STATUS closed; town_day1_recon pure fix section.
- Next tip: `rr-53g` Harvest + ship + post-5pm money assert (Gate B full spring income path).
- No push.

### 2026-08-09T22:45 CDT — super_metroid Bubble→Farm pure GREEN
- **CLOSED** `rr-czg9` dual pure **1566f×2** Bubble `0xACB3`→Farm `0xAF72` ~(472,139).
- Export: `scratch/post_bubble_to_farm_pure.state` (Farm→Speedway predecessor).
- Parent `rr-vqv3` remains open (4/7 hops dual green).
- Next: `rr-z13h` Pure Farm→Speedway (needs Speed).
- Tests: `test_k4_wave_return_scaffold.py` 6 passed. No push. No Ice STATUS.

### 2026-08-09T22:50 CDT — super_metroid Farm→Speedway pure GREEN
- **CLOSED** `rr-z13h` dual pure **329f×2** Farm `0xAF72`→Speedway `0xB106` ~(2008,139) right entry.
- Export: `scratch/post_farm_to_speedway_pure.state` (Speedway→Frog predecessor; needs Speed for Boost Blocks).
- Parent `rr-vqv3` remains open (5/7 hops dual green).
- Next: `rr-05dp` Pure Speedway→Frog Save (LEFT across 8-screen tunnel).
- Tests: `test_k4_wave_return_scaffold.py` 8 passed. No push. No Ice STATUS.

### 2026-08-09T23:00 CDT — SM Speedway→Frog pure GREEN
- **CLOSED** `rr-05dp` dual pure 621f×2 (0xB106→0xB167). Stack 6/7.
- Next: `rr-vsjy` Frog Save→Business to finish rr-vqv3.

### 2026-08-09T23:20 CDT — super_metroid Frog→Business pure GREEN (stack complete)
- **CLOSED** `rr-vsjy` dual pure **347f×2** Frog Save `0xB167`→Business `0xA7DE` floor ~(216,1419).
- **CLOSED** parent `rr-vqv3` Wave→Business pure return stack (7/7 hops dual green).
- Export: `scratch/post_frog_save_to_business_pure.state` (Business floor; Ice Super is mid-shaft).
- Scaffold replaced: `play_frog_save_to_business` → `wave/frog_to_business.py`.
- Tests: `test_k4_wave_return_scaffold.py` 10 passed. **No continuous Ice STATUS.** No push.
- Next: compose Wave→Business return into continuous Ice prefix; dual continuous green required before STATUS.

### 2026-08-09T23:05 CDT — harvest ship money CLOSED
- **CLOSED** `rr-53g`: shipped 24, wallet +1920 overnight after farm 5pm ShippingScene. Clean.
- Next: `rr-y8n` end-of-spring soak Gate A.


### 2026-08-09T22:32 CDT — zelda_i L4 post-Compass recon (rr-o0nn open)
- **IN PROGRESS** `rr-o0nn`: live recon from `Level4Compass` toward `ADDR_LADDER`.
- **Finding:** post-compass graph component **CLOSED** at `{0x71,0x61,0x51,0x50,0x62}`.
  - free/BOMB UP 0x61→0x51; RIGHT re-enter 0x62 (no key); LEFT 0x51→0x50.
  - 0x51 UP/RIGHT **sealed** (key poke does not consume — not key doors).
  - 0x50 denser bomb-N no open; 0x62 bomb exits none; no Vire key-farm drops (8 cycles).
  - `ADDR_LADDER` still 0.
- Docs/code: `LEVEL4_ROUTE.md` post-compass section; `planning_interior_report` tip `rr-o0nn`; AGENTS tip.
- Child **`rr-xc3x`**: expand past closed component (first live room outside set).
- Parent **`rr-5lu` remains open**. No pure Stepladder segment yet. No push.
- Evidence: `nes/zelda_i/recordings/l4_o0nn_{focus,prod,bombs,keypoke}.json`.

### 2026-08-09T23:15 CDT — zelda post-compass closed component
- `rr-o0nn` open: live component {0x71,0x61,0x51,0x50,0x62} sealed; ADDR_LADDER=0.
- Child `rr-xc3x`: expand past closed set toward Stepladder (TAS/FM2 room-id leads OK).

### 2026-08-09T23:20 CDT — SM Wave→Business pure stack COMPLETE
- **CLOSED** `rr-vsjy` 347f×2 Frog→Business floor + **`rr-vqv3` 7/7 dual pure GREEN**.
- Unblocks continuous Ice compose (still NO STATUS without dual continuous green).
- Note: Business settle is floor ~(216,1419); Ice Super needs climb/re-pin.


### 2026-08-09T22:50 CDT — harvest Gate A CLOSED (rr-y8n)
- **CLOSED** `rr-y8n` End-of-spring soak with money growth (Gate A).
- Multi-day successor from `Y1_Day09_Harvest_Mode_Start` `--days 1`:
  - `final_money=3180` (start $1260), `mid_run_state_load=false`
  - `HARVEST_ROUTE` shipped=24 / harvested=24
  - `CROP_ESTABLISH` planted=6; `gate_a_economy_ok=true`
  - Evidence: `snes/harvest/recordings/run_spring_gate_a_day09.json`
- Wired Day09 farm 5pm path into calendar: `FarmShippingWaitTask` + MultiDay
  `wait_shipping` when `shipping_money>0` and hour<17 on farm (NightReset still
  credits wallet even if work already past 5pm).
- Journal summary: harvest/establish deltas + `gate_a_economy_ok`.
- Units: `tests.test_shipping_credit` 11/11.
- Full `--end-of-spring` from `Y1_Inside_House` still flaky (empty-can water fail;
  return_home hang ~D5) — parent **rr-20w** / Gate B remains open.
- Next tip: empty-can natural refill or power-on continuous spring (rr-5in).
- No push.

### 2026-08-09T23:05 CDT — zelda_i L4 expand CLOSED rr-xc3x
- **CLOSED** `rr-xc3x`: first live room outside early post-compass component
  `{0x71,0x61,0x51,0x50,0x62}` → **0x40** via 0x50 north (live BFS + long UP).
- Live: 0x40 = 5× Zol `0x13` + RoomItemId `0x19` key; DOWN returns 0x50.
- Pure segment `north_40` **2/2** from `Level4Compass` (~8254f); checkpoint
  `Level4Room40` (+ provenance). Evidence: `recordings/l4_xc3x_*.json`.
- Trap: 0x50 is **not** a dead-end; center+UP fails on interior blocks —
  runner uses online BFS from clear_50 end pose (pose varies).
- Parents **rr-o0nn** / **rr-5lu** stay open (`ADDR_LADDER` residual).
- Follow-up open: **rr-q8eq** 0x40 clear+key + next room toward ladder.
- Tests: `test_level4_dungeon.py` 8 passed. Commit only (no push).

### 2026-08-09T23:01 CDT — hourly watcher tick
- **in_progress (busy lanes — no re-dispatch):**
  - super_metroid: `rr-kxge` Dual continuous --to ice stabilize
  - zelda_i: `rr-o0nn` L4 Compass→Stepladder + `rr-5lu` L4 pure residual
  - mega_man_2: `rr-54ui` Air Man past screen 4
- **SKIP permanently:** smb / pure_hl (never spawn)
- **Idle priority action:** harvest Gate B/rr-53g CLOSED → tip **rr-5in** Gate B full power-on Spring→Summer
- **Spawned manager (worktree):** harvest → claim `rr-5in`
  - subagent_id: `019fe9d5-97f7-7851-8bee-7bb396619e73` (worktree isolation)
- **Not spawned (busy):** super_metroid, zelda_i, mega_man_2
- **Not spawned (low tip / closed):** solver rr-gbd.32 CLOSED; remaining gbd open are P3/P4 sketches; alttp P3 only
- **Cap:** 3 busy + 1 new harvest = 4 / 6 lanes
- Never push. Next tick: if harvest still in_progress skip; else next idle tip.

### 2026-08-09T23:45 CDT — zelda 0x50→0x40 breakthrough
- **CLOSED** `rr-xc3x`: 0x50 N→0x40 (Zols 0x13 + key 0x19). Pure north_40 2/2 ~8254f.
- Component was not a hard dead-end. ADDR_LADDER still 0. Next: clear 0x40 + key (rr-q8eq / rr-o0nn).


### 2026-08-09T23:10 CDT — SM dual continuous --to ice compose (rr-kxge)
- **Claimed** `rr-kxge` (discovered-from rr-vqv3): dual continuous `--to ice` stabilize.
- **COMPOSE LANDED:** Wave→Business return ×7 + Ice pure ×4 on ice tip spine (11 hops). Graph frog_save_to_business promoted continuous. Default tip still `wave`.
- **Pure:** frog→business 355f×2 ~(209,1419); floor Business→Gate climb dual **3219f×2**; elev pin still **891f**.
- **Continuous RED:** power-on return chain reaches Business f141870; fail `business_to_ice_gate` floor climb (1339→1227 / HJ exit). Report `snes/super_metroid/recordings/ice_r1.json`. Residual `snes/super_metroid/docs/tasks/rr-kxge-residual.md`.
- **No continuous Ice STATUS.** Tests 55 passed (continuous tips + ice + wave return + path summary). No push.
- Next: Super-directed continuous Business floor→Ice Super climb; dual continuous green before STATUS.

### 2026-08-09T23:50 CDT — SM continuous Ice compose landed (open)
- **rr-kxge** open: 11-hop ice tip compose Wave return + Ice pure LANDED (942f13f).
- Continuous power-on RED on business_to_ice_gate floor climb after frog_save_to_business.
- Pure floor→Gate dual 3219f×2 OK. No STATUS. Next: Super-directed continuous climb.

### 2026-08-09T23:00 CDT — MM2 post-s4 overnight (rr-54ui open)
- **Claimed** `rr-54ui`; stayed **OPEN** (no camera≥5 / boss door / clear).
- Mapped `AirFanPlatform` solid **prog 937–984** (left fall walk~14, right walk~33).
- Goblin obj slot14 type36 @~(39,49). Dense 5px + spike-cycle grids both sides:
  **0** feet=1 lands in gap (906,936) or sy<82. Long hop → left ledge only.
- `AirLeftPlatform` is short (prog 902–905). Best press still Pipi/shoot **prog~1086** min_sy~23 scr4 pit.
- Pruned ~2400 false `AirGoblinHead*` / `AirPastFan*` probe states.
- STATUS/plan/AGENTS/policy docstring updated; units 10/10; AirScreen2→4 smoke GREEN.
- Next: Goblin solidity confirmation or Lightning Lord alternate entry.
- No push.

### 2026-08-09T23:55 CDT — harvest empty-can path harden (open)
- `rr-jwju` open under rr-20w: F9-before-fence, pond reachability, fence local_drop, return_home timeout 5500f.
- Units green; ROM natural fill still can_peak=0 (stall ~tile 25,30). Next: multi-hop F0 act densify.

### 2026-08-09T23:20 CDT — zelda L4 0x40 key + 0x30 enter
- **CLOSED** `rr-q8eq`: 0x40 Zol/gel clear + key pure 2/2 (~3345f); free UP→**0x30** pure 2/2 (~228f).
- Live: gel split 0x14 type-only; key via `MAZE_40_TO_KEY` hold6 east corridor ~(136,117); L/R sealed.
- Checkpoints: `Level4Room40Cleared`, `Level4Room30`. ADDR_LADDER still 0.
- Next tip: **`rr-n1wn`** clear 0x30 + expand N/E (parent `rr-o0nn` / `rr-5lu` open until ladder).
- Evidence: `l4_q8eq_key40_pure_key_40.json`, `l4_q8eq_north30_north_30.json`.

### 2026-08-10T00:05 CDT — zelda 0x40 key + UP 0x30 pure
- **CLOSED** `rr-q8eq`: key_40 2/2 ~3345f; north_30 2/2 ~228f. ADDR_LADDER still 0.
- Next tip `rr-n1wn`: 0x30 Vires clear + expand N/E toward ladder.


### 2026-08-09T23:30 CDT — harvest rr-5in PARTIAL (power-on continuous)
- **Claimed** `rr-5in` Gate B full power-on Spring→Summer with income.
- **GREEN:** power-on → D1 handoff (talks+truck+dog intro+shed grass+can) → D2
  Clean; `recordings/power_on_d1_handoff_d2.json` (25840f, $300, mid_run=0).
- **Wired:** `run_to_day2 --power-on` auto TownDay1Handoff before multi-day
  (`--no-d1-handoff` to skip). GoToSleep X-cycle + more attempts.
- **RED residual (rr-5in stays OPEN):**
  1. Full spring attempt 1: sleep miss bed (70,86) D7 after 5 overnights ($300)
  2. Attempt 2: `ENSURE_CROP_SEEDS` multi_nav 1-wp hang S0D4 ~11:02 after buy ($100)
- Children: **`rr-6byj`** ENSURE_CROP_SEEDS hang; **`rr-m0wq`** sleep D7.
- Empty-can natural refill still **`rr-3q27`** (parallel; do not thrash crop_planter).
- Evidence: `snes/harvest/recordings/rr_5in_residual.json`
- Next tip: fix `rr-6byj` shed-seed equip hang, then re-run `--power-on --end-of-spring`.
- No push.

### 2026-08-09T23:30 CDT — zelda_i L4 0x30 clear + KEY-RIGHT 0x31 CLOSED rr-n1wn
- **CLOSED** `rr-n1wn`: 0x30 Vire clear pure **2/2** (~2016f) + KEY-RIGHT → **0x31** pure **2/2** (~348f).
- Live: 3× Vire `0x12` + 2× invuln `0x2b` (ignore for clear). Walkable band **y∈[128,208]** only —
  north-band patrol face UP (generic mid-room chase starves damage).
- Exit probe: free N/E/W sealed; DOWN→0x40; **KEY-RIGHT @y141** keys1→0 → **0x31** (5× Vire).
- Segments: `clear_30`, `key_right_31` in `run_level4_rooms.py`. Checkpoints
  `Level4Room30Cleared`, `Level4Room31`. Evidence: `l4_n1wn_clear30_*.json`,
  `l4_n1wn_key31_*.json`, `l4_n1wn_30_exits.json`.
- **ADDR_LADDER still 0.** Parents **rr-o0nn** / **rr-5lu** stay open.
- Next tip: **`rr-resv`** clear 0x31 + expand toward ladder (blocks rr-o0nn).
- Units: `test_level4_dungeon.py` 9 passed. Commit only (no push).

### 2026-08-09T23:45 CDT — zelda_i L4 0x31 clear + RIGHT 0x32 CLOSED rr-resv
- **CLOSED** `rr-resv`: 0x31 Vire clear pure **2/2** (~6865f) + free RIGHT → **0x32** pure **2/2** (~406f).
- Live: 5× Vire `0x12` maze; clear opens doors **2→3** (RIGHT free). N free sealed.
- East path: hold4/q4 BFS to x≥200 y≈136 then long RIGHT (hold6/q8 starves from clear pose).
- 0x32 recon: 2× Zol `0x13`, 2× LikeLike `0x17`, 2× invuln `0x2b`, `0x68`; LEFT→0x31; **ADDR_LADDER=0**.
- Segments: `clear_31`, `east_32` in `run_level4_rooms.py`. Checkpoints
  `Level4Room31Cleared`, `Level4Room32`. Evidence: `l4_resv_clear31_*.json`,
  `l4_resv_east32_*.json`, `l4_resv_31_bfs.json`, `l4_resv_room32_recon.json`.
- **ADDR_LADDER still 0.** Parents **rr-o0nn** / **rr-5lu** stay open.
- Next tip: clear **0x32** + expand toward ladder (blocks rr-o0nn).
- Units: `test_level4_dungeon.py` 9 passed. Commit only (no push).

### 2026-08-09T23:47 CDT — harvest empty-can multi-hop F9 path (rr-3q27 open)
- **Claimed** `rr-3q27` (tip under rr-20w; empty-can tip id not rr-jwju).
- **Units:** multi-hop after gap, densify, carry-drop, F9 multihop when viewport-blocked; `CROP_WATER refill_bounds` **(3,10,62,60)** includes north F9 y=12. 262 tests OK (water_refill + crop_planter + day_phase + day_plan_sequences).
- **ROM dry fixture** `Y1_Test_Crops_Planted_Dry`: still **`can_peak=0`** (`recordings/empty_can_refill_probe.json`).
  - Root maps: (1) old bounds y_min=14 **excluded F9**; (2) single y=31 fence clear is **not** empty-handed south corridor; (3) plant soft-block thrash ~(13,27) blocks F9 climb; (4) mid-carry soft-timeout left hands full.
- **Not closed** rr-3q27 (no natural fill GREEN). **rr-20w** remains open.
- Next: recorded west-climb to F9 act, or finish FenceClearLoop carry→pond toss without soft-timeout.
- No push.

### 2026-08-10T00:45 CDT — harvest multi-hop densify (open)
- `rr-3q27` still open: F9 densify + refill_bounds y_min=10 + carry-drop (95de9f6).
- Units green; ROM can_peak still 0 (plant soft-block / F9 west-climb residual).

### 2026-08-10T00:30 CDT — SM rr-kxge continuous Ice single GREEN (dual flaky)
- **Claimed** `rr-kxge` continuous Business floor→Ice Super climb stabilize.
- Climb harden: `pos_1339` (84 pure / 90 cont), RIGHT-biased floor recover, HJ door recover, bound LEFT setup on cont retries, wave floor dumps.
- **Pure** floor Business→Gate dual GREEN **3255f×2**; elev pin **891f**.
- **Single continuous GREEN** `ice_r3.json` **148192f** room `0xA890` beams `0x1007` (Ice). Splits include business_to_ice_gate @145255 through ice_snake_to_ice.
- **Dual continuous not stable** (r4/r5a/dual_a RED on floor climb). **No STATUS**. Default tip still `wave`.
- Residual `snes/super_metroid/docs/tasks/rr-kxge-residual.md`. Tests 55 passed. Bead left open.
- Next: dual continuous 2/2 integrity green before STATUS.

### 2026-08-10T01:00 CDT — SM single continuous Ice GREEN (dual open)
- `rr-kxge` open: climb harden landed (cbb46f8). Pure floor→Gate dual 3255f×2.
- Continuous 1× GREEN ice_r3 148192f room 0xA890 beams 0x1007. Dual flaky.
- No STATUS. Next: dual continuous integrity 2/2.

### 2026-08-10T00:01 CDT — hourly watcher tick
- **in_progress (busy lanes — no re-dispatch):**
  - harvest: `rr-3q27` Natural empty-can refill + return_home hang
  - super_metroid: `rr-kxge` Dual continuous --to ice stabilize (single GREEN; dual flaky)
  - zelda_i: `rr-tib8` L4 0x32 + `rr-o0nn` / `rr-5lu` stepladder residual
- **SKIP permanently:** smb / pure_hl (never spawn)
- **Idle priority action:** mega_man_2 `rr-54ui` OPEN (not in_progress) — fan platform → boss
- **Spawned manager (worktree):** mega_man_2 → claim `rr-54ui`
  - subagent_id: `019fea0c-198d-7f41-ba10-f7763de25a80` (worktree isolation)
- **Not spawned (busy):** harvest, super_metroid, zelda_i
- **Not spawned:** solver rr-gbd.32 CLOSED; remaining gbd open P3/P4 sketches only
- **Cap:** 3 busy + 1 MM2 = 4 / 6 lanes
- Never push. Next tick: skip busy lanes; harvest tip stays empty-can until free.

### 2026-08-10T00:20 CDT — harvest rr-3q27 empty-can (open; F9 sealed + gap residual)
- **Claimed** `rr-3q27` (tip under rr-20w). Prefer Clean.
- **ROM recon:** north F9 is **sealed** from west plant pocket (y=13–14 fence bar);
  full BFS never reaches F9 stands; prior multihop was false manhattan progress.
- **Code:** preferred multihop requires hop nearly-arrives (≤3); `FenceClearLoopTask.corridor_only`
  local-drop (no pond-toss thrash); gap densify never charges south from y=31;
  fence soft-timeout no longer aborts mid-carry; local_drop avoids north-first.
- **Units:** 63 passed (`test_water_refill` + `test_crop_planter_logic`).
- **ROM dry** `Y1_Test_Crops_Planted_Dry`: still **`can_peak=0`**
  (`snes/harvest/recordings/empty_can_refill_probe.json`) — fence_att=1 gap=true
  but soft-block on (13,31) gap transit / local-drop-while-carrying.
- **Not closed** rr-3q27 (no natural fill GREEN). **rr-20w** remains open.
- Next tip: carry-south cross of y=31 gap OR recorded `toss_fence_pond` from gap
  OR reliable empty charge from (12,29) after south-side drop.
- No push.

### 2026-08-10T01:15 CDT — harvest F9 sealed; gap soft-block residual
- `rr-3q27` open (c00bbfa): F9 false multihop sealed; corridor_only fence path.
- ROM: fence opens then soft-block (13,31) gap transit. can_peak still 0.
- Next: south gap cross / south local-drop / toss_fence_pond segment.

### 2026-08-10T02:30 CDT — harvest rr-3q27 natural empty-can GREEN (tip)
- **Claimed** `rr-3q27` (keep under rr-20w). Prefer Clean.
- **ROM dry** `Y1_Test_Crops_Planted_Dry`: **`can_peak=20`**, `refill_count=1`,
  `watered=2`, TaskStatus.SUCCESS (`recordings/empty_can_refill_probe.json`,
  ~7992f, wall ~16s). No RAM poke.
- **Route (ROM-mapped):** empty gap charge soft-blocks (13,31); east→south on
  y=30→x≈28 then south to (28,32); (28,32) RIGHT/DOWN dead — west→south-lip
  waypoints to F0 `(32,34)` face-up fill; post-fill water-return north charge.
- **Code:** `FenceClearLoop` carry-south from y≤31; CropWater east→south charge,
  west→south-lip charge, east-crawl densify (no empty gap south), water north
  return. Probe script `harvest.scripts.gap_transit_probe` kept for recon.
- **Units:** 64 passed (`test_water_refill` + `test_crop_planter_logic`).
- **Not closed** rr-3q27 (return_home hang residual; 2/3 water). **rr-20w** open.
- Next tip: return_home stabilize + 3/3 dry water / short Inside_House multi-day.
- No push.


### 2026-08-10T00:25 CDT — super_metroid rr-kxge dual continuous Ice GREEN
- Lane: super_metroid only (`snes/super_metroid/`)
- Bead **rr-kxge** CLOSED: Dual continuous `--to ice` stabilize
- **Evidence:** `ice_dual_d` + `ice_dual_e` → `ice.json` / `ice_dual.json`
  **148,167f** ×2 exact match, room `0xA890`, beams `0x1007`, outcome
  `ice_collected`, integrity 0 loads/prog/capacity/deaths
- Climb harden: cont-tuned 907 runup ladder (18/20/22 @ pos90) for Charge
  loadout; classic warehouse setup preserved (bound=False); safe re-center
  only on Ice retries; pure floor→Gate still 3255f×2; elev 891f
- STATUS program gate + `DEFAULT_CONTINUOUS_TIP = ice`; AGENTS immediate goal
- Tests: continuous tips + ice/wave scaffolds + segment contracts green
- Optional ice demo video still open; next bead: K5 Alpha PB (`rr-dbu.8`)
- No push

### 2026-08-10T01:30 CDT — SM dual continuous Ice GREEN ★
- **CLOSED** `rr-kxge`: dual continuous --to ice **148167f×2** room 0xA890 beams 0x1007.
- DEFAULT tip **ice**; STATUS promoted. Climb cont-tuned 907 ladder. No deaths/loads.
- Next product: `rr-dbu.8` K5 Alpha PB pure stack.

### 2026-08-10T00:30 CDT — zelda_i L4 0x32 clear + Stepladder ADDR_LADDER CLOSED rr-tib8
- **CLOSED** `rr-tib8`: 0x32 Zol+LikeLike pure **2/2** (~3141f) + stairs **0x60** → `ADDR_LADDER` pure **2/2** (~5328f).
- Also closed parents **`rr-o0nn`** / **`rr-5lu`** (ladder advanced).
- Live: ignore invuln `0x2b` + block `0x68`; push left @~(120,141) detour statues; NE ~(208,96) UP → mode-9 0x60; multi-grid BFS goal-state restore.
- Trap: stepladder needs **5 idle** preamble before clear (1 idle → 48-cell BFS miss).
- Segments: `clear_32`, `stepladder` in `run_level4_rooms.py`. Checkpoints `Level4Room32Cleared`, `Level4Stepladder`.
- Evidence: `l4_tib8_clear32_clear_32.json`, `l4_tib8_stepladder_stepladder.json`.
- Next tip: **`rr-05fz`** post-ladder residual (water/map/Gleeok/TF 0x08) under `rr-q3n`.
- Units: `test_level4_dungeon.py` 9 passed. Commit only (no push).

### 2026-08-10T01:45 CDT — zelda L4 Stepladder pure dual-green ★
- **CLOSED** `rr-tib8`/`rr-o0nn`/`rr-5lu`: 0x32 clear 2/2 + 0x60 ADDR_LADDER 2/2.
- Stairs push from clear pose; need 5 idle after reset (RNG trap). Next: `rr-05fz` post-ladder residual.

### 2026-08-10T00:35 CDT — MM2 post-s4 overnight (rr-54ui open)
- **Claimed** `rr-54ui`; stayed **OPEN** (no camera≥5 / boss door / clear).
- **KEY CORRECTION:** type36 Goblin is **not solid** — platforms are tiles
  (`tile_feet`/`tile_center`). Type36 = damage enemy (~111f teleport-hit cycle).
  AirScreen2 “on goblin” is y=52 **tile** platform (walk past type36 x, feet=1).
- Gap math: last solid prog **984** → scr5@**1280** ≈ **296px** open.
  Pure jump max **~1065–1071**; Pipi boost still ~1086; freefall tiles past 984 = 0.
- Swept: 1000+ phase/top-down goblin hops (0 elevated land); ladder UP never feet=2;
  no wind; camera_y=0; edge camp 400–600f types only 1/35/36 (no Lightning Lord).
- Evidence: `nes/mega_man_2/recordings/air_post4_night3/`. Units 10/10.
- Docs: STATUS/plan/AGENTS/policy residual updated. False probe states pruned.
- Next: Lightning Lord / intermediate cloud spawn via **earlier route fork**
  (before full y=84 descent); map-match segment at prog~950.


### 2026-08-10T01:45 CDT — SM rr-dbu.8 K5 pure stack PARTIAL (Ice→Snake dual)
- Lane: super_metroid only (`snes/super_metroid/`)
- Bead **rr-dbu.8** IN_PROGRESS (PARTIAL, not closed): K5 Alpha PB pure stack
- **First hop GREEN dual:** `ice-to-snake` **538f** ×2 exact, room `0xA8B9`
  xy=(472,395) pose=10 from `post_ice_snake_to_ice_pure`
- Package: `routes/kpdr/ice/ice_to_snake.py` + leave geometry; `routes/kpdr/k5/`
  hop map (return + reverse tunnels + Red→Alpha tape table)
- Child bead **rr-bf29** OPEN: Snake → Tutorial pure (next one-hop)
- Residual: `docs/tasks/rr-dbu.8-residual.md`
- Units: ice scaffold + docs green; no continuous / STATUS change (Ice tip remains)
- No push

### 2026-08-10T02:00 CDT — SM K5 first hop Ice→Snake GREEN
- `rr-dbu.8` PARTIAL: ice-to-snake dual pure **538f×2** room 0xA8B9. Next `rr-bf29` Snake→Tutorial.
- Also: harvest empty-can fill GREEN landed (ac78ec1) — check rr-3q27.

### 2026-08-10T02:05 CDT — harvest natural empty-can fill GREEN
- `rr-3q27` open residual: can_peak=20 refill=1 watered=2 Clean (ac78ec1).
- Path east→south + F0 south-lip. Next: return_home hang + 3/3 water + multi-day.


### 2026-08-10T00:57 CDT — SM rr-bf29 Snake→Tutorial pure dual GREEN
- Lane: super_metroid only (`snes/super_metroid/`)
- Bead **rr-bf29** CLOSED: Pure Ice Snake → Tutorial return (K5 hop 1)
- **Evidence:** `ice-snake-to-tutorial` dual pure **2386f** ×2 exact, room `0xA865`
  xy=(39,127) pose=81 from `post_ice_to_snake_pure`
- Controller: drop Ice alcove → morph tunnel → floor → multi-attempt 2WJ climb
  (reuse `_snake_platform_climb`; no freeze thrash RLE) → top-right Tutorial door
- Export: `scratch/post_ice_snake_to_tutorial_pure.state` (+ dual)
- Parent **rr-dbu.8** stays IN_PROGRESS PARTIAL (hop0+hop1 GREEN; Alpha PB open)
- Next bead **rr-81ek**: Tutorial → Gate pure from new pin
- Units: ice scaffold green; no continuous / STATUS change
- No push

### 2026-08-10T02:20 CDT — SM K5 Snake→Tutorial pure GREEN
- **CLOSED** `rr-bf29` dual pure 2386f×2 room 0xA865. Stack hop0+hop1 GREEN.
- Next: `rr-81ek` Tutorial→Gate. Parent rr-dbu.8 open.

### 2026-08-10T01:01 CDT — hourly watcher tick
- **in_progress (busy lanes — no re-dispatch):**
  - harvest: `rr-3q27` empty-can residual (fill GREEN landed; return_home/multi-day open)
  - super_metroid: `rr-81ek` Tutorial→Gate + parent `rr-dbu.8` K5 Alpha PB stack
  - zelda_i: `rr-05fz` L4 post-Stepladder (water/map/Gleeok/TF) — stepladder path advanced
- **SKIP permanently:** smb / pure_hl (never spawn)
- **Idle priority action:** mega_man_2 `rr-54ui` OPEN again (prior manager finished; no camera≥5)
- **Spawned manager (worktree):** mega_man_2 → claim `rr-54ui` (Lightning Lord / cloud fork tip)
  - subagent_id: `019fea42-fe32-7473-8999-a0f6583be0b9` (worktree isolation)
- **Not spawned (busy):** harvest, super_metroid, zelda_i
- **Not spawned:** solver rr-gbd.32 CLOSED; P3/P4 sketches only
- **Cap:** 3 busy + 1 MM2 = 4 / 6 lanes
- Notes: SM K5 hop0+hop1 GREEN; dual Ice continuous still residual on earlier kxge line. Harvest can fill GREEN. Never push.

### 2026-08-10T01:10 CDT — harvest rr-3q27 CLOSED (3/3 empty-can + return_home)
- Lane: harvest only (`snes/harvest/`)
- Bead **rr-3q27** CLOSED: Natural empty-can refill + return_home hang
- **Evidence:**
  - Dry fixture `Y1_Test_Crops_Planted_Dry`: `can_peak=20`, `refill=1`,
    **`watered=3`**, `dry_end=[]` Clean (~9097f)
    `recordings/empty_can_refill_probe.json`
  - Short Clean multi-day `Y1_Inside_House --days 3`: overnights=3,
    `mid_run_state_loads=0`, `ram_writes=0`, success goal reached
    `recordings/inside_house_3day_clean.json`
- **Code:** residual crop-walk recovery (wet-neighbor stand + full 3x3
  extra_walkable); water reorder cap 3/step; no temp_blocked near stand
  y≤30; return_home off-stand re-nav cap (force enter / fail clean)
- Units: 64 passed (`test_water_refill` + `test_crop_planter_logic`)
- Parent **rr-20w** stays OPEN (full Spring income; Inside_House D3–D4
  ENSURE_CAN / CROP_WATER soft fails residual)
- Next tip under rr-20w: live Inside_House multi-day water path after plant
  (day-plan CROP_WATER reliability), not fixture-only
- No push

### 2026-08-10T01:08 CDT — SM rr-81ek Tutorial→Gate pure dual GREEN
- Lane: super_metroid only (`snes/super_metroid/`)
- Bead **rr-81ek** CLOSED: Pure Ice Tutorial → Gate return (K5 hop 2)
- **Evidence:** `ice-tutorial-to-gate` dual pure **969f** ×2 exact, room `0xA815`
  xy=(807,131) pose=81 from `post_ice_snake_to_tutorial_pure`
- Controller: partial cleaned human RLE left→mid + double-DOWN morph tunnel
  + Ice freeze + long spin gap + door pressure (no boyon thrash RLE)
- Export: `scratch/post_ice_tutorial_to_gate_pure.state` (+ dual)
- Parent **rr-dbu.8** stays IN_PROGRESS PARTIAL (hop0–2 GREEN; Alpha PB open)
- Next bead **rr-e5i6**: Gate → Business pure from new pin
- Units: ice scaffold green; no continuous / STATUS change
- No push

### 2026-08-10T01:08 CDT — SM K5 Tutorial→Gate pure GREEN
- **CLOSED** `rr-81ek` dual pure 969f×2 room 0xA815. Stack hop0+1+2 GREEN.
- Next: `rr-e5i6` Gate→Business. Parent rr-dbu.8 open.

### 2026-08-10T02:30 CDT — MM2 rr-54ui Lightning Lord fork (OPEN residual)
- Lane: mega_man_2 only (`nes/mega_man_2/`)
- Bead **rr-54ui** IN_PROGRESS → remains **OPEN** (no camera≥5 / boss door)
- Pursued tip: LL/cloud via earlier route fork + map-match prog~950
- **Map-match:** prog~950 = goblin + stripe tile + Pipi (**not** Matasaburo E;
  no wind / fan robots / camera_y / drop D). Gap >984 looks like open LL sky
  but **no LL object ever**.
- Object types Level1→death hybrid + all forks: only **{1, 2, 35, 36}**
- Night4 sweeps (`recordings/air_post4_night4/` + RED_PIN.txt):
  - High forks before y84 descent: max prog~947
  - Descent interrupt UP/LEFT/hold: max~936
  - 186 micro-hops from AirFan: **0** feet=1 past prog 984; best ~1073
  - Edge void shoot 500f: no new types
  - Slow late waits / shoot goblin / WRAM novelty after prog800: no new `$0400` types
  - Smoke AirScreen2→4 still GREEN (502f)
- Units: `nes/mega_man_2/tests` 10/10
- Docs: STATUS/plan/AGENTS/policy residual updated
- **Next tip:** ROM stage-enemy placement / TAS compare at prog≥1000 (why LL
  never enters object table); or nametable platforms past 984. Do not re-run
  goblin-solid or pure-RIGHT grids.
- No push

### 2026-08-10T01:16 CDT — SM rr-e5i6 Gate→Business pure dual GREEN
- Lane: super_metroid only (`snes/super_metroid/`)
- Bead **rr-e5i6** CLOSED: Pure Ice Gate → Business return (K5 hop 3)
- **Evidence:** `ice-gate-to-business` dual pure **879f** ×2 exact, room `0xA7DE`
  xy=(41,907) pose=25 from `post_ice_tutorial_to_gate_pure`
- Controller: cleaned human RLE morph drop + tunnel roll RIGHT + Super door
  pressure (tunnel-mouth thrash trimmed; accept mid-top Gate settle)
- Export: `scratch/post_ice_gate_to_business_pure.state` (+ dual)
- Parent **rr-dbu.8** stays IN_PROGRESS PARTIAL (hop0–3 GREEN; Alpha PB open)
- Next bead **rr-3gh9**: Business Super → Warehouse pure from new pin
- Units: ice scaffold + docs green; no continuous / STATUS change
- No push

### 2026-08-10T01:16 CDT — SM K5 Gate→Business pure GREEN
- **CLOSED** `rr-e5i6` dual pure 879f×2 room 0xA7DE. Stack hop0+1+2+3 GREEN.
- Next: `rr-3gh9` Business→Warehouse. Parent rr-dbu.8 open.

### 2026-08-10T02:50 CDT — harvest ENSURE_CROP_SEEDS hang CLOSED
- **CLOSED** `rr-6byj`: shelf A replace thrash fixed; Clean hoe+seeds SUCCESS ~2248f.
- Next tip: rr-m0wq sleep bed miss or rr-20w live plant→water residual.

### 2026-08-10T01:25 CDT — SM rr-3gh9 Business→Warehouse pure dual GREEN
- Lane: super_metroid only (`snes/super_metroid/`)
- Bead **rr-3gh9** CLOSED: Pure Business Super → Warehouse return (K5 hop 4)
- **Evidence:** `business-to-warehouse` dual pure **10255f** ×2 exact, room `0xA6A1`
  xy=(37,139) pose=138 from `post_ice_gate_to_business_pure` Super lip ~(41,907)
- Controller adapt: Super/midshaft floor-fall first + Charge multi-attempt ladder
  (classic 14→8 lead; cont-tuned 18/20/22 retries; mid-right re-anchor on recover)
- Export: `scratch/post_ice_business_to_warehouse_pure.state` (+ dual)
- Parent **rr-dbu.8** stays IN_PROGRESS PARTIAL (hop0–4 GREEN; Alpha PB open)
- Next bead **rr-bw2w**: Warehouse → East Tunnel reverse
- Units: kpdr_dev + source_states + ice scaffold + controller_common green; no continuous / STATUS
- No push

### 2026-08-10T01:29 CDT — SM rr-bw2w Warehouse→East pure dual GREEN
- Lane: super_metroid only (`snes/super_metroid/`)
- Bead **rr-bw2w** CLOSED: Pure Warehouse → East Tunnel return (K5 hop 5)
- **Evidence:** `warehouse-to-east` dual pure **285f** ×2 exact, room `0xCF80`
  xy=(216,364) pose=26 from `post_ice_business_to_warehouse_pure` elev ~(37,139) p138
- Controller: elev-band unmorph + LEFT blue door (reverse of east_to_warehouse)
- Export: `scratch/post_ice_warehouse_to_east_pure.state` (+ dual)
- Parent **rr-dbu.8** stays IN_PROGRESS PARTIAL (hop0–5 GREEN; Alpha PB open)
- Next bead **rr-68ib**: East Tunnel → Glass reverse
- Units: kpdr_dev 8/8; no continuous / STATUS
- No push

### 2026-08-10T01:33 CDT — SM rr-68ib East→Glass pure dual GREEN
- Lane: super_metroid only (`snes/super_metroid/`)
- Bead **rr-68ib** CLOSED: Pure East Tunnel → Glass return (K5 hop 6)
- **Evidence:** `east-to-glass` dual pure **253f** ×2 exact, room `0xCEFB`
  xy=(216,395) pose=12 from `post_ice_warehouse_to_east_pure` ~(216,364) p26 crouch
- Controller: uncrouch residual + LEFT blue door (reverse of glass_to_east)
- Export: `scratch/post_ice_east_to_glass_pure.state` (+ dual)
- Parent **rr-dbu.8** stays IN_PROGRESS PARTIAL (hop0–6 GREEN; Alpha PB open)
- Next bead **rr-85c4**: Glass → West Tunnel reverse
- Units: kpdr_dev + source_states green; no continuous / STATUS
- No push

### 2026-08-10T~01:35 CDT — harvest sleep bed miss CLOSED
- Lane: harvest only (`snes/harvest/`)
- Bead **rr-m0wq** CLOSED: Sleep interaction miss at bed (70,86) D7
- Root cause: face-left + A walks into mattress; B after A cancels Yes/No sleep confirm
- Fix (`GoToSleepTask`): face-up only; B only before first A of attempt; A-only
  confirm/dismiss; toss held once; re-nav if off-stand mid-verify; longer verify
- **Evidence:** `recordings/rr_m0wq_sleep_days6.json` — `Y1_Inside_House` D2→D8,
  6/6 overnights first-try (incl D7→D8), Clean, mid_run_loads=0, ram_writes=0
- Units: SleepAndPlannerTests green
- Residual: parent **rr-5in** / **rr-20w** full power-on→Summer income still open
- No push

### 2026-08-10T01:30 CDT — zelda_i L4 post-ladder exit+west pure dual-green (rr-05fz partial)
- Lane: zelda_i only (`nes/zelda_i/`)
- Claimed **rr-05fz** (in_progress): post-Stepladder residual under epic **rr-q3n**
- **exit_60** pure **2/2** ~765f: `Level4Stepladder` idle 150f (item freeze) → clear 4× Keese → hold4 BFS exit mode-9 0x60 → **0x32** play → `Level4PostLadder`
- **west_31** pure **2/2** ~372f: free LEFT BFS around pushed 0x68 → **0x31** → `Level4Room31PostLadder`
- Live backtrack **0x31→0x30→0x40** with ladder; 0x30 N still sealed; map/Gleeok/TF residual
- Traps: pedestal freeze ~100–150 idle; BFS settle must wait mode 4/6/7 ~400f (180f false-negative)
- Segments: `exit_60`, `west_31` in `run_level4_rooms.py`. Units 9 passed.
- Child **rr-rvae** created for map+Gleeok+TF. **rr-05fz** left open until TF 0x08.
- Evidence: `l4_05fz_exit60_exit_60.json`, `l4_05fz_west31_west_31.json`
- No push. Next: map room from Level4PostLadder / rr-rvae

### 2026-08-10T03:15 CDT — zelda post-ladder pure exit (open residual)
- `rr-05fz` open: exit_60 2/2 ~765f; west_31 2/2 ~372f. Checkpoints Level4PostLadder.
- Map/Gleeok/TF residual tip `rr-rvae`. Traps: 150f item freeze; BFS settle ~400f.


### 2026-08-10T01:41 CDT — SM rr-abx5 West→Below pure dual GREEN
- Lane: super_metroid only (`snes/super_metroid/`)
- Bead **rr-abx5** CLOSED: Pure West Tunnel → Below Spazer return (K5 hop 8)
- **Evidence:** `west-to-below` dual pure **272f** ×2 exact, room `0xA408`
  xy=(472,393) pose=82 from `post_ice_glass_to_west_pure` ~(216,139) p10
- Controller: LEFT blue door run_shoot (reverse of below_spazer_floor_to_west)
- Export: `scratch/post_ice_west_to_below_pure.state` (+ dual)
- Parent **rr-dbu.8** stays IN_PROGRESS PARTIAL (hop0–8 GREEN; Alpha PB open)
- Next bead: Pure Below → Bat reverse (from new pin)
- Units: kpdr_dev + source_states green; no continuous / STATUS
- No push

### 2026-08-10T~02:00 CDT — SM rr-rp00 Below→Bat pure dual GREEN
- Lane: super_metroid only (`snes/super_metroid/`)
- Bead **rr-rp00** CLOSED: Pure Below Spazer → Bat Room return (K5 hop 9)
- **Evidence:** `below-to-bat` dual pure **485f** ×2 exact, room `0xA3DD`
  xy=(472,139) pose=12 from `post_ice_west_to_below_pure` ~(472,393) p82
- Controller: LEFT floor runner across Below (reverse of bat_to_below_spazer door)
- Export: `scratch/post_ice_below_to_bat_pure.state` (+ dual)
- Parent **rr-dbu.8** stays IN_PROGRESS PARTIAL (hop0–9 GREEN; Alpha PB open)
- Next bead: Pure Bat → Red reverse (from new right-sill pin)
- Units: kpdr_dev + source_states green; no continuous / STATUS
- No push

### 2026-08-10T02:01 CDT — hourly watcher tick
- **in_progress (busy lanes — no re-dispatch):**
  - super_metroid: `rr-av5s` Red→Hellway (K5 hop 12) + parent `rr-dbu.8` Alpha PB stack (hops 0–9+ GREEN)
  - zelda_i: `rr-rvae` map+Gleeok+TF from Level4PostLadder (post-ladder exit pure landed)
  - harvest: `rr-5in` Gate B full power-on Spring→Summer (`rr-3q27` CLOSED empty-can 3/3)
- **SKIP permanently:** smb / pure_hl (never spawn)
- **Idle priority action:** mega_man_2 `rr-54ui` OPEN (night4 residual: TAS/ROM LL spawn or nametable past 984)
- **Spawned manager (worktree):** mega_man_2 → claim `rr-54ui` (TAS/ROM analysis tip)
  - subagent_id: `019fea79-f501-7d23-94c8-cb6c96cca41a` (worktree isolation)
- **Not spawned (busy):** super_metroid, zelda_i, harvest
- **Not spawned:** solver rr-gbd.32 CLOSED; remaining P3/P4 sketches
- **Cap:** 3 busy + 1 MM2 = 4 / 6 lanes
- Never push.

### 2026-08-10T03:30 CDT — harvest rr-5in PARTIAL sparse water
- Plant + sparse water detect GREEN; empty-can after sparse plant still RED (rr-o00y).
- Next: post-charge y≥32 F0 fill path then re-run power-on end-of-spring.


### 2026-08-10T~02:15 CDT — mega_man_2 rr-54ui night5 residual (OPEN)
- Lane: mega_man_2 only (`nes/mega_man_2/`)
- Claimed **rr-54ui** (was open; no other MM2 in_progress)
- Pursued residual tip: ROM/TAS enemy placement + nametable past 984
- **Result: still OPEN** — no camera≥5 / boss door / grounded past prog 984
- Findings:
  - Map-match: prog~950 pre-LL (A/late); gap after 984 = expected B/LL sky
  - Type36 Air Tikki: indestructible; f420 64→128 teleport-hit; stands = tiles
  - Freefall grid: max feet=1 prog **980**; **0** tile hits prog>984
  - Types only {1,35,36} (shoot/edge/policy camp + WRAM novelty)
  - ROM: property rows 0x22–0x25; no spawn-list decode yet
- Smoke AirScreen2→4 GREEN 502f HP16; units 10/10
- Child **rr-fpd6**: decode LL spawn / TAS FM2 at prog≥1000
- Evidence: `nes/mega_man_2/recordings/air_post4_night5/` (+ RED_PIN.txt)
- Do not re-run: goblin-solid, pure-RIGHT grids, y84 edge LL camps
- No push
- Next: claim rr-fpd6 (Mesen/disasm spawn or TAS compare)

### 2026-08-10T02:16 CDT — SM rr-av5s Red→Hellway PARTIAL (K5 hop 12)
- Lane: super_metroid only (`snes/super_metroid/`)
- Claimed **rr-av5s** Pure Red Tower → Hellway return (K5 hop 12)
- **PARTIAL:** lower right-wall WJ from `post_ice_bat_to_red_pure` ~(216,2443)
  reaches right-pocket ledge ~(225,2091). Mid/upper not dual green.
- Wiring: `play_red_to_hellway`, `ROOM_HELLWAY=0xA2F7`, probe `red-to-hellway`,
  geometry + human final-ascent RLE recon (desyncs from pure pin — enemy state)
- Trap: pocket ceiling peaks ~y1964; apex LEFT-WJ free-falls; human RLE needs
  thrash-warmed Red enemy state
- Residual: `docs/tasks/rr-av5s-residual.md`; parent **rr-dbu.8** stays open
  hop0–11 GREEN, hop12 PARTIAL
- Units: kpdr_dev + controller_common 27 passed; no continuous / STATUS
- Bead **rr-av5s** left **in_progress** (mid shaft next)
- No push

### 2026-08-10T~night — zelda_i L4 map room assisted dual-green (rr-rvae partial)
- Lane: zelda_i only (`nes/zelda_i/`)
- Claimed **rr-rvae** (in_progress): map + Gleeok path + TF 0x08 from Level4PostLadder
- **Live recon:** KEY-UP 0x30 (ladder+key) → **0x20** (5× Vire) → RIGHT → **0x21**
  (5× Gel 0x15 + RoomItemId **0x17** map). Pickup `ADDR_MAP|0x08` @~(208,181).
- **map_21** assisted **2/2** ~17872f: recon key poke (post-ladder keys=0) → clear
  0x20 → state-BFS east → gel thrash → hold6 BFS map. Checkpoint **Level4Map**.
- Evidence: `recordings/l4_rvae_map21_map_21.json`
- Traps: KEY-UP 0x31 = isolated south pocket on 0x21 (not map path); 0x20 door
  bit R stays 0; gels block maze until thrash; natural key residual
- Epic **rr-q3n** left open (TF 0x08). Parent **rr-05fz** open residual.
- Units: test_level4_dungeon 10 passed. No push.
- Next: natural key for map natural-entry; Gleeok + TF 0x08

### 2026-08-10T04:00 CDT — zelda L4 map room PARTIAL
- `rr-rvae` open: map_21 assisted 2/2 ~17872f Level4Map ADDR_MAP|0x08.
- Path 0x31→0x30 KEY-UP→0x20→0x21. Next: Gleeok/TF from Level4Map.

### 2026-08-10T~02:35 CDT — harvest rr-o00y CLOSED (empty-can fill GREEN)
- Lane: harvest only (`snes/harvest/`)
- Bead **rr-o00y** CLOSED: Empty-can refill thrash after sparse plant
- **Evidence:**
  - Dry fixture: `can_peak=20`, `refill=1`, `watered=3` (~10k f)
  - Power-on continuous: D5 `REFILL OK can=20`, watered=2/2 sparse plant;
    D6–D7 water holds (`recordings/power_on_spring_to_summer.log`)
  - Residual JSON: `snes/harvest/recordings/rr_o00y_fill_green.json`
- **Code:** east→south past fence end x≥31; west→south-lip bands
  (soft/south_far/south/east_pond); south densify east-prefer + thrash bail
- Units: 68 passed (`test_water_refill` + `test_crop_planter_logic`)
- Parent **rr-5in** PARTIAL: fill GREEN; dies D7 `return_home` hands-clear
  (money $100). Residual **rr-6g7g**
- Next tip: rr-6g7g clear hands before house entry, re-run --end-of-spring
- No push

### 2026-08-10T04:15 CDT — harvest sparse plant fill CLOSED
- **CLOSED** `rr-o00y`: power-on D5 REFILL can=20 watered=2/2; dry fixture watered=3.
- rr-5in residual: hands-clear return_home D7 (rr-6g7g).

### 2026-08-10T03:01 CDT — hourly watcher tick
- **in_progress (busy lanes — no re-dispatch):**
  - harvest: `rr-5in` Gate B full + residual `rr-6g7g` return_home hands-clear
  - super_metroid: `rr-av5s` Red→Hellway hop12 + parent `rr-dbu.8` K5 Alpha stack
  - zelda_i: `rr-rvae` map+Gleeok+TF (map_21 assisted dual-green landed; Gleeok/TF open)
- **SKIP permanently:** smb / pure_hl (never spawn)
- **Idle priority action:** mega_man_2 — parent `rr-54ui` open; tip **rr-fpd6** LL spawn/TAS decode prog≥1000
- **Spawned manager (worktree):** mega_man_2 → claim `rr-fpd6`
  - subagent_id: `019feab0-ba14-7612-9b6d-9fb9dc417594` (worktree isolation)
- **Not spawned (busy):** harvest, super_metroid, zelda_i
- **Not spawned:** solver rr-gbd.32 CLOSED; P3/P4 sketches only
- **Cap:** 3 busy + 1 MM2 = 4 / 6 lanes
- Notes: harvest fill GREEN (rr-o00y closed); zelda Level4Map assisted. Never push.

### 2026-08-10T~03:05 CDT — SM rr-av5s Red→Hellway PARTIAL lower past pocket
- Lane: super_metroid only (`snes/super_metroid/`)
- Claimed **rr-av5s** Pure Red Tower → Hellway return (K5 hop 12)
- **PARTIAL advance:** lower climb past right-pocket dead-end
  - From `post_ice_bat_to_red_pure` ~(206,2443): right-wall WJ + re-catch
  - Lands open shaft **~(219,1942)** (~475f) — past pocket ~(225,2091) / A-ceiling ~y1964
  - Export: `scratch/dev_red_lower_past_pocket.state`
- **Still RED:** mid tunnel y≤1880 / bomb floor / upper / Hellway dual
  - Full pure pin: room=0xA253 x=219 y=1942 frames≈7716 (mid stall)
  - Traps: right-wall hard ceiling ~1942; shaft too wide for single-WJ left latch;
    IBJ mid not dual-stable; Bat door abort on floor RIGHT spins
- Controller: `routes/kpdr/k5/red_to_hellway.py` (lower recatch + mid IBJ shell)
- Residual: `docs/tasks/rr-av5s-residual.md`; parent **rr-dbu.8** open hop12 PARTIAL
- Units: kpdr_dev + controller_common 27 passed; no continuous / STATUS
- Bead **rr-av5s** left **in_progress** (mid from y1942 next)
- No push

### 2026-08-10T04:30 CDT — SM Red climb past pocket to y1942
- `rr-av5s` PARTIAL: past pocket ceiling ~(219,1942). Mid/upper/Hellway still RED.
- Next: freeze-ripper/HBJ from 1942 → tunnel y<=1880 → Hellway.

### 2026-08-10T04:45 CDT — zelda Gleeok approach live (partial)
- `rr-rvae` open: map dual-green; Gleeok room 0x13 type 0x43 entered once from 0x12.
- Stabilize 0x12 RIGHT→0x13 dual-green; fight + TF 0x08 residual.


### 2026-08-10T~03:20 CDT — harvest rr-6g7g CLOSED (return_home hands-clear)
- Lane: harvest only (`snes/harvest/`)
- Claimed **rr-6g7g** (rr-5in residual after rr-o00y fill GREEN)
- **Result: CLOSED** — clear hands / toss before house entry on power-on water days
- Fix:
  - `ReturnHomeTask`: always relocate to open drop spot south of house when
    hands full (was only near-door); multi-face stationary A-drop first
  - `toss_held_actions`: stationary face+A (fence_flow proven); optional short
    step-away only for wall-hug doors — long B-walk re-picked stones (`0x0D`)
  - `FarmClearTask`: more drop attempts + multi-face before handing off held
  - Door push: prefer walk into threshold (not blind B-hold) + mid-wall escape
- Evidence: `recordings/power_on_spring_to_summer.json`
  - **7 overnights** past former D7 hands-clear fail
  - reason no longer `could not clear hands…`
  - crop wet=2/alive; Clean mid_run_state_loads=0
  - Residual on **rr-5in**: D9 `return_home failed: nav_house_front multi_nav timeout`, money=$100, Summer not reached
- Units: `BuildDayPhasesTests` 65 OK (incl. hands-full drop-spot + toss tests)
- Bead **rr-6g7g** CLOSED; **rr-5in** still PARTIAL
- No push
- Next (rr-5in): stabilize house approach after water days + first potato ship for money>100

### 2026-08-10T05:00 CDT — harvest hands-clear CLOSED
- **CLOSED** `rr-6g7g`: open drop + multi-face toss; power-on 7 overnights past D7.
- rr-5in residual: D9 nav_house_front multi_nav timeout; money=$100 Summer open.

### 2026-08-10T~night — zelda_i L4 Gleeok enter dual-green (rr-rvae partial)
- Lane: zelda_i only (`nes/zelda_i/`)
- Claimed **rr-rvae** (in_progress): map dual-green already; stabilize 0x12→0x13
- **Live fix:** after 0x12 Vire clear doors raw=2 (L only). Bomb/KEY-RIGHT fail.
  Push block **0x68 LEFT** stand~(112,144) (96,144)→(80,144) opens R bit
  doors 2→3. Naive y141 hold-RIGHT still fails (maze) — hold4
  **PATH_12_TO_GLEEOK** plen31 → **0x13** Gleeok type **0x43** HP≈160 + HC 0x1a.
- **Dual-green 2/2** assisted from `Level4Room12`. Checkpoints
  **Level4Room12Cleared**, **Level4GleeokEnter**, **Level4Boss**.
- Code: `ROOM_12_SPEC`, push geometry, `PATH_12_TO_GLEEOK`, factories/predicates
  in `level4_dungeon.py`; tests + AGENTS/LEVEL4_ROUTE.
- Evidence: `recordings/l4_rvae_right13_dual.json`
- Residual: Gleeok fight + HC + TF `0x08` (epic **rr-q3n** open)
- Units: test_level4_dungeon 10 passed. No push.
- Next: Gleeok combat policy from Level4GleeokEnter dual-green

### 2026-08-10T05:45 CDT — zelda L4 Gleeok + TF 0x08 dual-green ★
- **CLOSED** `rr-rvae`: Gleeok fight + HC + TF 0x08 dual-green 2/2 ~4286f from Level4GleeokEnter.
- Checkpoint Level4Complete. Epic rr-q3n open for continuous residual (rr-05fz).


### 2026-08-10 — SMB DONE (human reaffirm)
- User: "drop the smb work.. lets call it done"
- All `smb`-labeled beads already CLOSED/PARKED (incl. rr-k0x pure HL, hybrid/TAS frame-cuts).
- Product stands: **M8 Clean** power-on natural_82. Pure HappyLee / stitchless / sub-5 tracks not resumed.
- Night watch: **never spawn smb** for remainder of program unless human reopens explicitly.
- No open smb processes; no open smb beads.

### 2026-08-10T04:01 CDT — hourly watcher tick
- **in_progress (busy lanes — no re-dispatch):**
  - super_metroid: `rr-av5s` Red→Hellway hop12 + parent `rr-dbu.8` K5 Alpha stack
  - zelda_i: `rr-05fz` L4 post-Stepladder continuous residual (Gleeok+TF dual-green CLOSED on rr-rvae)
  - harvest: `rr-5in` Gate B full (hands-clear rr-6g7g CLOSED; D9 house nav residual)
- **SKIP permanently:** smb / pure_hl (human reaffirmed DONE — never spawn)
- **Idle priority action:** mega_man_2 `rr-54ui` — **rr-fpd6 CLOSED** LL types 0x3D/0x3E spawn; tip = land Thunder Chariot (~28px X short)
- **Spawned manager (worktree):** mega_man_2 → claim `rr-54ui` (LL cloud land)
  - subagent_id: `019feae7-b435-7d00-8a01-56b88386361c` (worktree isolation)
- **Not spawned (busy):** super_metroid, zelda_i, harvest
- **Not spawned:** solver gbd.32 closed; P3/P4 only
- **Cap:** 3 busy + 1 MM2 = 4 / 6 lanes
- Never push.

### 2026-08-10T~night — zelda_i rr-05fz CLOSED (natural key + continuous TF)
- Lane: zelda_i only (`nes/zelda_i/`)
- Claimed **rr-05fz** residual after rr-rvae Gleeok/TF dual
- **Natural key:** compass path leaves keys=0 at PostLadder (recon poke only).
  **Skip-compass** from `Level4Room50Cleared` → 0x40 key (keys=2) → KEY-R 0x31
  (keys=1) → stepladder → `Level4Room31PostLadderNaturalKey` (ladder=1, keys≥1)
- **map_21 --no-key-poke** dual-green 2/2 ~16309f; `recon_poke=false`
  Evidence: `recordings/l4_05fz_map_natural_map_21.json`
- **Continuous PostLadder→TF** dual-green 2/2 ~34748f assisted, key_poke=false
  Path: map → BOMB_UP 0x21→0x11 → BOMB_RIGHT 0x11→0x12 → clear → push 0x68
  → PATH_12_TO_GLEEOK → Gleeok/HC/TF 0x08
  Runner: `scripts/run_level4_continuous_tf.py`
  Evidence: `recordings/l4_05fz_postladder_cont_tf.json` + `l4_05fz_map_to_tf.json`
- Code: `run_level4_rooms.py` module-level imports fix + `--no-key-poke`; continuous runner
- **CLOSED** `rr-05fz`. Epic **rr-q3n** still OPEN for Clean STATUS residual.
- Units: test_level4_dungeon 10 passed. **No push.**
- Next: Clean harden L4 continuous / parallel `rr-38p`


### 2026-08-10T04:40 CDT — zelda rr-05fz CLOSED ★
- **CLOSED** `rr-05fz`: natural KEY-UP (skip-compass NaturalKey) + continuous PostLadder→map→Gleeok→TF dual ~34.7kf assisted (`--no-key-poke`).
- Commit `2da4435`. Not Clean STATUS.
- Residual **rr-vdnc** Clean continuous without infinite-life → epic rr-q3n.
- **Spawned** zelda manager on **rr-vdnc** (worktree).
- SMB still permanently skipped.





### 2026-08-10T04:50 CDT — SM Red mid dual y1606 (rr-av5s PARTIAL)
- Commit `ea5ad62`: bottom IBJ → tunnel→midplat → midplat IBJ dual **temp floor y1606**.
- Do not climb_lower before IBJ. Frozen-ripper seats not solid.
- Upper/Hellway still RED. **Re-spawned** SM manager for upper → Hellway dual.
- Bead rr-av5s stays in_progress; parent rr-dbu.8 open.


### 2026-08-10T05:10 CDT — mega_man_2 rr-54ui PARTIAL (cloud land)
- Claimed `rr-54ui` (lane free). **Not closed** — no camera≥5 / no stand.
- **Rider kill GREEN:** type `0x3D` pulsed B (period 3–8) 20→13→6→despawn.
  Body `0x3E` remains; kill flash types 6+118.
- **X residual improved:** fpd6 dx≈28 → Y-meet after kill **dx≈5–10** (still
  freefall). Kill with dy≳20 co-sinks with cloud (gap frozen).
- **Deeper residual:** empty Thunder Chariot object-solid stand not achieved
  even at dx≈5, dy≈0. Next: `aobject_tsa` / flag 128→192 / feet-on-top.
- Probes: `scripts/cloud_land_probe.py` + recordings `air_post4_cloud*` +
  `RED_PIN.md`. Docs STATUS/plan/AGENTS/LL_SPAWN_DECODE updated.
- Units: `pytest nes/mega_man_2/tests`. **No push.** Bead stays in_progress.
- Next: object-solid decode then re-try stand → cam≥5.

### 2026-08-10T~04:40 CDT — harvest rr-5in house approach GREEN (Gate B still PARTIAL)
- Lane: harvest only (`snes/harvest/`)
- Claimed **rr-5in** (Gate B full power-on Spring→Summer income)
- **Advance:** house approach after water/CLEAR **GREEN** (former D9 multi_nav fail)
- Root: CLEAR finished south of **y=31 fence wall** (tiles x=11–29). Direct
  multi_nav to door and mid-field densify (x≈248) hit solid fence / SW rock
  pocket with no BFS path (~(102,726) / ~(118,678)).
- Fix (Clean):
  - `ReturnHomeTask`: densify east (x≥480) or west (x≤160) of fence, or through
    confirmed open gap; SW pre-escape + multi softlock recovery; timeout 11k
  - `FarmClearTask`: exit-staging B-run when finishing south of fence
  - `FarmClearer`: north-bias target sort (avoid deep-south softlock)
  - `MultiMapNavTask`: early softlock fail + stagnant thrash
- Evidence: `recordings/power_on_spring_to_summer.json`
  - **11 overnights** Spring D2→D13, money **$160** (`money_gt_100=true`)
  - Clean `mid_run_state_load=false`; reason **budget** (not nav_house_front)
  - Prior pin (D9 multi_nav / D8 SW softlock) cleared
- **Still RED for Gate B full:** Summer D1 not reached; CROP_WATER fails D7+
  (`refill exhausted` / dry=6 watered=0 densify stuck (23,28)→(27,28)); no
  harvest/ship; burns frames into max_frames budget
- Residual bead **rr-5go9** (blocks rr-5in): refill thrash after fence
- Units: `BuildDayPhasesTests` 70 OK
- Bead **rr-5in** left **in_progress** PARTIAL; no push
- Next: rr-5go9 stabilize empty-can/refill densify on continuous power-on, then
  re-run `--end-of-spring` for Summer D1 + gate_b_full

### 2026-08-10T04:50 CDT — zelda rr-vdnc CLOSED ★
- **CLOSED** `rr-vdnc`: Clean continuous PostLadderNaturalKey → TF `0x08` dual-green 2/2 (no `--infinite-life`, key_poke=false) ~33.9kf.
- Gleeok south-stand policy (`STAND_DY=22`, FB dodge≤14; no head chase). Evidence `l4_vdnc_clean_cont_tf.json`, `l4_vdnc_gleeok_clean_dual.json`.
- Not full-game Clean STATUS. Parent epic `rr-q3n` residual remains compose/natural-entry.
- SMB still permanently skipped.

### 2026-08-10T04:55 CDT — zelda rr-vdnc CLOSED ★ (merged to main)
- Cherry-picked worktree commits → `77ff000` / beads notes.
- Clean continuous PostLadder→TF dual ~33.9kf (south-stand Gleeok).
- Created **rr-zavx** Z4.5 natural-entry Clean compose; spawned manager.
- SMB permanently skipped.


### 2026-08-10T05:00 CDT — MM2 rider kill GREEN (rr-54ui PARTIAL)
- Commit `1220ebf`: pulsed-B rider kill; X closed; empty cloud object-solid residual.
- Re-spawned MM2 manager for feet-on-top / aobject_tsa stand → cam≥5.
- Zelda: rr-vdnc Clean dual CLOSED on main; tip rr-zavx natural-entry compose running.

### 2026-08-10T~session — MM2 solid decode PARTIAL (rr-54ui still open)
- `scripts/cloud_solid_decode.py` + `recordings/air_post4_cloud_solid/` (708 recipes).
- **Decoded:** `aobject_tsa` = AI timer (not solid); flag 128→192 = `objects_right`
  facing; type 6 = `objects_killed` (~12f); type 118 not seen; body stays `0x3E`.
- **feet_dy=0 @ dx≤2** after kill still freefall; co-sink locks feet_dy≈−3…−4.
- Diagnostic place-on-top poke still falls through (not Clean evidence).
- Cam≥5 **no**. Bead **rr-54ui** stays in_progress PARTIAL. No push.
- Next: disasm body AI solid-arm / TAS pin / screen-align.

### 2026-08-10T05:01 CDT — hourly watcher tick
- **in_progress (busy lanes — no re-dispatch):**
  - super_metroid: `rr-av5s` Red→Hellway hop12 + parent `rr-dbu.8` K5 Alpha stack
  - zelda_i: `rr-zavx` Z4.5 Clean compose dock→PostLadder→TF continuous
  - harvest: `rr-5in` Gate B full + residual `rr-5go9` CROP_WATER refill thrash
  - mega_man_2: `rr-54ui` LL cloud land (rider kill GREEN; empty chariot object-solid residual)
- **SKIP permanently:** smb / pure_hl (never spawn)
- **Spawned managers:** none (all priority game lanes busy)
- **Not spawned:** solver gbd.32 CLOSED; alttp P3 only; capacity 4/6 but rule = no dispatch when priority lanes full
- **Cap:** 4 / 6 concurrent game lanes
- Board notes: zelda rr-vdnc Clean TF dual CLOSED; harvest money>$100 / 11 overnights PARTIAL Summer; MM2 X closed residual feet-on-top.
- Never push. Next tick: only free lanes.

### 2026-08-10T05:15 CDT — MM2 solid decode PARTIAL (rr-54ui)
- Commit `6f847c7`: aobject_tsa=AI timer; feet_dy=0 still freefall; body stays 0x3E.
- Re-spawned MM2 for disasm kaminari_goro solid-arm + TAS pin + screen-align.
- Rider kill remains GREEN. Stand/cam≥5 RED.


### 2026-08-10T~session — MM2 disasm + screen-align PARTIAL (rr-54ui)
- Lane: mega_man_2 only. Claim kept **rr-54ui** (not closed — no stand/cam≥5).
- **Disasm (lsmmega/mm2 bank14 `14_19`):** body spawns rider `0x3D`; **no solid-arm
  rewrite** on child death. Flag `$08` = AI phase only. Appear solid needs `$10`
  (never on empty `0x3E`). Full PRG: 4× CMP #$3E AI-only; 0× CMP #$3D.
- **Screen-align:** kill window cam=3 / body scr=4; cam=4 arrives top_dy≈−19.
  247 recipes, 40 kills, 0 stand. Cloud top ≈ by−16 (OAM).
- **Diag:** fall_top poke achieves top_dy≈+1 still freefall → solid path inactive.
- Probe: `scripts/cloud_screen_align.py` + `recordings/air_post4_screen_align/`.
- Residual tip: human/TAS stick pin or alt path past s4 without cloud ride.
- Bead stays in_progress PARTIAL. No push.

### 2026-08-10T05:30 CDT — MM2 empty-cloud solid engine residual
- Commit `d0162b2`: disasm body AI no solid-arm; screen-align 0/40 stand; fall_top poke still freefall.
- Re-spawned MM2 for **alternate Clean path past s4** (not more solid grids).
- Pin: CLOUD_LAND_RED_PIN.md


### 2026-08-10T05:40 CDT — MM2 park Air s4 residual (rr-54ui PARTIAL)
- Landed `d423647`: no Clean Air-first past s4; cloud solid never arms; Item-1 needs Heat-first.
- Child **rr-f3nr** ready: FCEUX stick pin OR Heat→Air Item-1 Clean segment.
- Freeing Air-cloud thrash; spawn Heat→Air Item-1 exploration on rr-f3nr.


### 2026-08-10T05:50 CDT — MM2 Heat→Item-1 scaffold ★ PARTIAL
- **CLOSED** `rr-f3nr` PARTIAL: Heat1 + HeatScreen1 dual-green (`1f1cfb3`).
- Chain: rr-808 mid/late → rr-809 boss+Item-1 → rr-810 Air+Item-1 cam≥5.
- Spawned Heat mid/late manager on **rr-808**.


### 2026-08-10T05:55 CDT — zelda rr-zavx PARTIAL ★
- **CLOSED** `rr-zavx` PARTIAL: Clean dual Entrance→skip-compass NaturalKey 2/2 ~45.8kf (`459a61b`).
- Residual **rr-gjey**: Gleeok Clean needs hp≥~108; pure spine ends ~103 after map peel.
- Spawned zelda manager on rr-gjey (heart-safe / lower-HP Gleeok).

## 2026-08-10 rr-av5s PARTIAL — Red→Hellway past floor y1459

- Dual mid floor y1606 + human RLE@850 dual ~(122,1459)p81
- Hellway still RED; upper residual from p81 handoff
- Keep rr-av5s in_progress

### 2026-08-10T06:00 CDT — SM past temp floor dual y1459 (rr-av5s PARTIAL)
- Landed `0e46ed4`: human RLE 850f from y1606 → dual ~(122,1459)p81.
- Hellway still RED. Re-spawned upper WJ/Hellway manager.
- Do not bomb floor from below; do not force-unmorph from p81.


### 2026-08-10T06:05 CDT — harvest rr-5go9 CLOSED ★
- Commit `2cedeea`: CROP_WATER refill densify after fence GREEN (D9–D11 watered=6).
- power-on: wet=2 dry=4, money=$160, overnights=10; fails D12 return_home multi_nav timeout.
- Residual **rr-5in**: return_home after water + re-refill at (25,34) → Summer D1.
- Spawned harvest manager on rr-5in residual.


### 2026-08-10T06:10 CDT — harvest water + MM2 Heat mid ★
- **CLOSED** `rr-5go9` water refill densify (`2cedeea`); residual rr-5in return_home D12 → Summer.
- **CLOSED** `rr-808` PARTIAL: Heat dual-green screens 2–5 (`bc2d2fe`); residual rr-809 boss+Item-1.
- Spawned harvest return_home/Summer + MM2 Heat boss managers.

### 2026-08-10T06:01 CDT — hourly watcher tick
- **in_progress (busy lanes — no re-dispatch):**
  - super_metroid: `rr-av5s` Red→Hellway hop12 PARTIAL (y1459) + parent `rr-dbu.8`
  - zelda_i: `rr-gjey` Entrance→TF Clean residual (Gleeok hp≥108; rr-zavx PARTIAL closed)
  - harvest: `rr-5in` Gate B full (rr-5go9 water densify CLOSED; D12 return_home residual)
  - mega_man_2: `rr-54ui` Air s4 cloud residual + `rr-809` Heat boss+Item-1 (Heat screens 2–5 dual GREEN)
- **SKIP permanently:** smb / pure_hl (never spawn)
- **Spawned managers:** none (all priority game lanes busy)
- **Not spawned:** solver gbd.32 CLOSED; oracle P1 not over pure SM stack; capacity 4/6
- **Cap:** 4 / 6 concurrent game lanes
- Board: Heat→Item-1 chain scaffolding; Air cloud solid path falsified → Heat-first route.
- Never push. Next tick: only free lanes.

### 2026-08-10T06:20 CDT — MM2 Heat pre-boss cam≥7 (rr-809 PARTIAL)
- Commit `40d1a04`: HeatScreen5Ground→cam7 dual 3/3 prog1792; pins s5g/s6/s7.
- Residual: s7 alcove climb → boss + Item-1. Re-spawned boss climb manager.
- Do not tip rr-810 until Item-1.


### 2026-08-10T~session — SM rr-av5s PARTIAL upper y687
- Period WJ dual end (171,687)p25 f11802 exact; peak ~y420; Hellway still RED.
- Bead rr-av5s stays in_progress PARTIAL. No push.
- Next: y420→top door y180 → dual green Hellway + export.

### 2026-08-10T06:50 CDT — SM upper WJ dual y687 (rr-av5s PARTIAL)
- Commit `9caf235`: period WJ dual end (171,687)p25 f11802; peak ~y420.
- Hellway still RED. Re-spawned top-door→Hellway manager.
- Do not thrash after y687 pin; do not force-unmorph p81.


### 2026-08-10T06:55 CDT — MM2 Heat s7 wall-lock residual
- Commit `770466b`: HeatScreen7Mid sx152; ladder x192+ unreachable; no boss/Item-1.
- Re-spawned with **new route hypothesis only** (no hop spam).
- Do not tip rr-810 until Item-1.


## 2026-08-10 mega_man_2 rr-809 PARTIAL — s7 high-path cam≥8
- Dual-green HeatScreen7Mid → cam8 3/3 ~447f (ladder + scroll_down)
- Low alcove sx152 is dead-end (solid sy≥96); high path cam6 climb → cross above wall
- Pins: HeatLadder, HeatScreen8, HeatScreen7HighPast
- Residual: Sniper/Yoku from HeatScreen8 → boss → Item-1

### 2026-08-10T07:15 CDT — MM2 Heat high-path past wall ★
- Commits `9604f5a`+: dual HeatScreen7Mid→cam8 3/3; ladder+scroll_down; HeatScreen8 pin.
- Low alcove sx152 trap; human path climbs high on cam6 then over wall.
- Residual **rr-k1ea** Sniper/Yoku→boss; spawned manager. No Item-1 yet (rr-809 open).


### 2026-08-10T07:05 CDT — zelda rr-gjey CLOSED PARTIAL ★
- Landed `9fa264a`: post-boss fireball flee; GleeokEnter floor ≥106 dual; map@108 dual.
- Natural enter still ~98–100. Residual **rr-bxzj** heart-safe spine → spawned.
- Not full-game STATUS (rr-q3n open).

### 2026-08-10T07:01 CDT — hourly watcher tick
- **in_progress (busy lanes — no re-dispatch):**
  - super_metroid: `rr-av5s` Red→Hellway hop12 (upper WJ y687 PARTIAL) + parent `rr-dbu.8`
  - harvest: `rr-5in` Gate B full (water densify CLOSED; return_home/Summer residual)
  - mega_man_2: `rr-k1ea` HeatScreen8 Sniper/Yoku + `rr-809` boss+Item-1 + `rr-54ui` Air residual
- **SKIP permanently:** smb / pure_hl (never spawn)
- **Idle priority action:** zelda_i free (`rr-gjey` CLOSED PARTIAL) → tip **rr-bxzj** heart-safe spine Gleeok≥108
- **Spawned manager (worktree):** zelda_i → claim `rr-bxzj`
  - subagent_id: `019feb8c-b284-73c0-bd7b-c530890e312d` (worktree isolation)
- **Not spawned (busy):** super_metroid, harvest, mega_man_2
- **Not spawned:** solver gbd.32 CLOSED; SM oracle below pure stack tip
- **Cap:** 3 busy + 1 zelda = 4 / 6 lanes
- Never push.

### 2026-08-10T07:20 CDT — harvest return_home D12 GREEN (rr-5in PARTIAL)
- Commit `44ac361`: far-east pre-escape + east-of-pond densify; 14 overnights to D16.
- Terminal budget not house-nav. Residual **rr-qc9r** CROP_WATER thrash D13–15 → spawned.


### 2026-08-10T07:45 CDT — SM thin-seat dual y587 (rr-av5s PARTIAL)
- Commits `9f62553`/`c8fe24b`: WJ no phase8; dual (91,587)p3; ice ladder scaffold.
- Hellway RED. Re-spawned ice-ladder → Hellway dual manager.


### 2026-08-10T07:55 CDT — MM2 first Yoku dual-green (rr-k1ea PARTIAL)
- Commit `4a238be`: HeatScreen8 → first Yoku land 3/3; pin HeatScreen8Yoku.
- Residual multi-level Yoku → Sniper → boss. Re-spawned chain manager.


### 2026-08-10T07:35 CDT — harvest late water GREEN ★ (rr-qc9r CLOSED)
- Commit `0205c5b`: CROP_WATER D9/11/13 watered=6; money=$400 ship≥5; 21 ovn D23 Clean.
- Residual **rr-ws8h** return_home exit_to_farm timeout → spawned. Gate B still open.


### 2026-08-10T07:35 CDT — NIGHT WATCH STOPPED (human wrap)
- Human: wrap up and stop. Killed live managers + hourly scheduler `019fe966cf19`.
- **SMB** permanently parked (M8 Clean stands). No push this session.
- Product still **in_progress** (honest residual — do not invent closes):
  - super_metroid **rr-av5s** PARTIAL thin-seat (91,587) + ice ladder → Hellway RED; parent **rr-dbu.8**
  - zelda_i **rr-bxzj** heart-safe spine (after rr-gjey fireball floor; Entrance→TF residual)
  - harvest **rr-ws8h** / **rr-5in** Gate B: water GREEN $400 21 ovn; return_home exit_to_farm residual
  - mega_man_2 **rr-k1ea**/**rr-809** first Yoku dual; multi-Yoku→boss+Item-1 residual; **rr-54ui** Air cloud solid parked engine residual
- Next human session: `bd ready` → pick one product tip; no auto-dispatch.

