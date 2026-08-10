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
