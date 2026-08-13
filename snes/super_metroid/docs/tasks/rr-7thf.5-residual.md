## Residual — rr-7thf.5 Wave B pure green (Climb + Parlor + Landing Site)

### Result
**GREEN** — dual hop-replay (boot-settle **0**) for hops 5–7 of
`g4_tourian_human_mb`. Safe trim seeds exported. No STATUS promote.

### GREEN/RED table

| Hop | Room | Leave / goal | Dual | End pin (both runs) | Notes |
|----:|------|--------------|:----:|---------------------|-------|
| 5 | Climb `0x96BA` | → Parlor `0x92FD` | **GREEN** | room=`0x92FD` xy=`[367, 32]` pose=21 `ROOM_TRANSITION` | replay f16275..18150; anchor f016274 |
| 6 | Parlor `0x92FD` | → LS `0x91F8` | **GREEN** | room=`0x91F8` xy=`[1263, 139]` pose=11 `ROOM_TRANSITION` | replay f18274..19651; anchor f018273 |
| 7 | Landing Site `0x91F8` | ship / credits (no leave) | **GREEN** | room=`0x91F8` xy=`[1152, 126]` pose=155 **`ENDING_OR_CREDITS`** | replay f19807..24220; anchor f019806 |

Wave A (Escape 1–4, hops 1–4) already dual GREEN under `rr-7thf.4` (closed);
no residual rewrite needed.

### Landing Site acceptance predicate

Leave room is **None** (final hop). Green if **either**:

1. **Preferred:** end `phase` ∈ `{ending_or_credits}` (CLI `--accept-phase ending_or_credits`), **or**
2. **Geometry:** still in `0x91F8` with end xy band ≈ `[1152, 126]` (default `xy_tol=24`; ship pin pose 155 observed).

Verified this session: both runs hit **geometry + phase** simultaneously
(`ENDING_OR_CREDITS` at ship xy). Default hop-replay (no `--accept-phase`) is
already green via end_xy; phase accept is alternate for credits drift.

Library: `check_hop_green(..., accept_phases=...)` — phase is **OR** with room+xy.

### Commands (re-verify)

```bash
# Critical: --boot-settle 0 (default). Do NOT use settle=5 — desyncs Escape 4+.

uv run python snes/super_metroid/scripts/tools/replay_human_hop.py \
  snes/super_metroid/tasks/g4_tourian_human_mb.json --hop 5 --dual

uv run python snes/super_metroid/scripts/tools/replay_human_hop.py \
  snes/super_metroid/tasks/g4_tourian_human_mb.json --hop 6 --dual

# LS: default geometry, or explicit phase accept
uv run python snes/super_metroid/scripts/tools/replay_human_hop.py \
  snes/super_metroid/tasks/g4_tourian_human_mb.json --hop 7 --dual \
  --accept-phase ending_or_credits

# Safe open-loop seeds (leading+trailing only)
uv run python snes/super_metroid/scripts/tools/trim_human_hop.py \
  snes/super_metroid/tasks/g4_tourian_human_mb.json --hop 5 --mode safe \
  -o snes/super_metroid/tasks/g4_tourian_human_mb_seeds/climb_safe.json
uv run python snes/super_metroid/scripts/tools/trim_human_hop.py \
  snes/super_metroid/tasks/g4_tourian_human_mb.json --hop 6 --mode safe \
  -o snes/super_metroid/tasks/g4_tourian_human_mb_seeds/parlor_safe.json
uv run python snes/super_metroid/scripts/tools/trim_human_hop.py \
  snes/super_metroid/tasks/g4_tourian_human_mb.json --hop 7 --mode safe \
  -o snes/super_metroid/tasks/g4_tourian_human_mb_seeds/landing_site_safe.json
```

Unit tests (phase option):

```bash
uv run pytest snes/super_metroid/tests/test_human_tape_replay.py -q
```

### Seeds written

Under `snes/super_metroid/tasks/g4_tourian_human_mb_seeds/`:

| File | Mode | Frames after trim |
|------|------|------------------:|
| `climb_safe.json` | safe | 2025 |
| `climb_hint.json` | traversal (edit hints only) | 596 |
| `parlor_safe.json` | safe | 1381 |
| `parlor_hint.json` | traversal | 286 |
| `landing_site_safe.json` | safe | 4451 |
| `landing_site_hint.json` | traversal | 632 |

**Open-loop:** use `*_safe.json` only. Traversal hints drop mid-idle/retry
frames that still affect enemy RNG — not dual-green validated as seeds.

### Files touched (code)

- `snes/super_metroid/human_tape_replay.py` — `accept_phases` on `check_hop_green` / `run_hop_replay`
- `snes/super_metroid/scripts/tools/replay_human_hop.py` — `--accept-phase`
- `snes/super_metroid/tests/test_human_tape_replay.py` — phase OR geometry unit test

### Acceptance

- [x] Climb dual GREEN (leave `0x92FD`)
- [x] Parlor dual GREEN (leave `0x91F8`)
- [x] LS dual GREEN under documented predicate (phase + ship xy observed)
- [x] Safe seeds for hops 5–7
- [x] Residual with commands + results
- [ ] STATUS promote — **not done** (per card)

### Residual risks

1. Hop-replay greens are **anchor + frame slice**, not continuous power-on → credits.
2. `landing_site_safe` trims trailing idle (~119f); re-validate if used as open-loop skill seed past ship touch.
3. Climb/Parlor traversal hints cut large retry ranges — edit only, do not open-loop.
4. Escape 4+ still require **boot-settle 0**.

### Next action

- Close `rr-7thf.5` when human accepts this residual.
- Next wave: `rr-7thf.6` (G4 statues → Metroids → Big Boy) or promote Wave A/B safe seeds to pure controllers under `routes/`.
- Optional: dual-green validate `*_safe.json` frame lists (not just raw hop slices).

### Non-claims

- Did not STATUS-promote continuous credits
- Did not package pure controllers under `routes/kpdr`
- Did not re-run dual on trimmed safe seeds (only raw hop-replay)
- Did not force-push / commit (not requested)
