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

