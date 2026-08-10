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

