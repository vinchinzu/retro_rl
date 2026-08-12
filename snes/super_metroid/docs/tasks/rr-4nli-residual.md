## Residual — rr-4nli / rr-4nli.1 / rr-4nli.2

### Result
**GREEN** — product-chain inventory landed; first hop (Climb) hop-replay **dual GREEN**
from the archived `s2` live pin. Autopilot join contract documented: adapter
starts from exact live RAM (subpixels, door speed, enemy phase).

### GREEN/RED table

| Item | Result | Notes |
|------|--------|-------|
| Board | **GREEN** | `tasks/PRODUCT_CHAIN_HOP_BOARD.json` — **282** hops / 20 product segments; **282/282** live pins; **2/282** reactive policies (both Climb); **13** bank dual-green |
| `sN/tape.json` → `sN/anchors.json` | **GREEN** | `load_anchors_index` fallback; hop-replay `--list-hops` on `s2/tape.json` found 15 anchors |
| s2 hop 9 Climb `0x96BA:0x975C→0x92FD:0x0004` | **GREEN** ×2 | leave Parlor `0x92FD` xy=`[373, 33]` pose 21 `ROOM_TRANSITION`; pin `f006140_enter_0x96BA_0x96BA.state` (same pin the verified AP policy was compiled from) |
| Bank promote | **GREEN** | `dual_green ← 0x96BA:0x975C->0x92FD:0x0004` frames=2183 source=full_start_v1 |
| AP policy | already **verified_live_anchor** | `policies/reactive_rooms/room_96ba_from_975c_to_92fd.json` — `room_adapter.search_live_adapter` + takeover-sweep |

### Autopilot / subpixel / door / enemy RNG

Hop-replay dual-green is the **seed**, not the full-run player.

| Layer | What it absorbs |
|-------|-----------------|
| Live enter/boot pin | Settled ordinary frame in this room+items |
| `door_kinematics.DoorKinematics` | Leave/entry speed, subpixels, pose, shine timer — door tech is not "same buttons" |
| `room_adapter.search_live_adapter` | Pulse-search from **exact live RAM** (subpixels, vx/mom, pose, **enemy phase**) onto the policy trajectory |
| RoomAutopilot takeover-sweep | Human→AP join at 25/50/75% after idle perturb (timing + kinematics + enemy clock drift) |
| Tape splice / death-cut | **Not allowed** — skipped frames still tick enemy RNG |

Climb is the gold-standard room: open-loop body dual-green **and** a compiled
reactive policy that can rejoin when the door or a human handoff is not the
recorded pin.

### Files changed
- `human_tape/anchors.py` — `sN/tape.json` finds `sN/anchors.json`
- `human_tape/product_chain.py` — product-chain hop board + AP join contract
- `scripts/tools/build_product_chain_board.py` — CLI `--write --summary`
- `tests/test_product_chain_board.py`
- `tasks/PRODUCT_CHAIN_HOP_BOARD.json`
- `docs/tasks/HUMAN_TAPE_PIPELINE.md` — product-chain epic pointer

### Verify paste

```text
uv run pytest snes/super_metroid/tests/test_product_chain_board.py -q
# 4 passed

uv run python snes/super_metroid/scripts/tools/replay_human_hop.py \
  snes/super_metroid/tasks/full_start_v1_segments/s2/tape.json --hop 9 --dual
# GREEN dual  hop=9 0x96BA→0x92FD  xy=[373, 33]
```

### Acceptance
- [x] Board lists every product-chain hop with hop_key / pin / policy / bank
- [x] One hop dual-green from archived live pin
- [x] AP join (subpixel / door / enemy phase) documented; Climb policy cited
- [ ] Next hop (Parlor s2 hop 10) dual-green + policy compile — **rr-4nli.3**
- [ ] Natural-entry: Climb leave kinematics seed Parlor (rr-nzrg.2)

### Residual risks
- 280/282 hops still have **no** reactive policy — AP falls back to human
- 1-frame door-flicker hops exist; next-work filter is `dwell >= 60`
- Bank still has junk keys from title RAM (`0x5555`) on s1 hop 0–1
- Moat→WS human tape still missing (bot spark pin seam)

### Next action (required)
- **Next card ID:** rr-4nli.3
- **One change:** hop-replay `--dual` **s2 hop 10 Parlor** `0x92FD` (Climb leave → Flyway). Then compile `optimize_room_policy.py` for that hop with `--takeover-sweep` so AP can join off-pin.
- **Source state:** `tasks/full_start_v1_anchors/f008451_enter_0x92FD_0x92FD.state` (confirm via `--list-hops` on `s2/tape.json`)
### Non-claims
- Did not STATUS-promote / claim continuous power-on → credits
- Did not concat product-chain tapes into one movie
- Did not compile a new reactive policy this session (Climb already verified)
### Probe pin
`tasks/full_start_v1_anchors/f006140_enter_0x96BA_0x96BA.state`
