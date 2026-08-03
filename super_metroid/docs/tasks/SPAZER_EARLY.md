# Early Spazer epic — walljump path → continuous fold → 100%

**Status:** planned / first cards ready (parallel track)  
**Opened:** 2026-08-03  
**Does not block** K4 Bubble → Bat serial spine.

Human ref (red room, Sidehopper mid-left, floor plant / claw + spikes,
early loadout Energy 74 / 9M / 3S):

![early Spazer red-room walljump context](refs/early_spazer_red_room.png)

## Product goal

Collect **Spazer Beam** early (pre-Kraid window) via the classic Red Brinstar
detour, with **walljump-capable** movement where the geometry demands it.
Then:

1. Prefer Spazer in later combat / spray policies when held.
2. Keep this as a **named continuous tip** first (`--to spazer`).
3. After pure + dual continuous integrity, **fold** the detour into the default
   KPDR continuous spine (Below Spazer no longer skips the item room).
4. Treat the epic as the first concrete step toward a **100%** track (all
   major items / maps / bosses), parallel to any% KPDR until fold.

## Topology (rooms)

| Room | ID | Role |
|------|-----|------|
| Red Tower | `0xA253` | Continuous (K1/K2); descent already green |
| Bat Room | `0xA3DD` | Continuous; three-platform → Below Spazer |
| Below Spazer | `0xA408` | Continuous tip `below_spazer` (**skips** item today) |
| **Spazer Room** | **`0xA447`** | **Collect** pedestal; return left to `0xA408` |
| West Tunnel… | `0xCF54`… | Existing warehouse chain after return |

Today continuous path: `… → Bat → Below Spazer → West` (no `0xA447`).  
Insert: `Below Spazer → Spazer Room collect → Below Spazer → West`.

Practice policy already exists dual-track:
`policies/room_clears/room_a447_from_a408_to_a408.json` (collect_and_return).
That is **not** continuous evidence.

## Wall jumps

- Spazer room itself is a short collect/return; hard geometry is usually the
  **red-room approaches** (tall red shafts / ledges) and any pre–Hi-Jump climb
  used when inserting the detour early.
- Reuse walljump lessons from Bubble mid (`routes/kpdr/bubble_mountain_*.py`)
  — same pose-26 / fresh-A patterns, different room geometry.
- Walljump skill is also a **100% / hard-room** investment (later LN, some
  optional packs, clean-track climbs).

## Why this changes planning

| Effect | Detail |
|--------|--------|
| Beam loadout | `collected_beams` / `equipped_beams` gain Spazer; spray width improves |
| Boss policies | Kraid / Phantoon / later fights can prefer Spazer+missile mixes when held |
| Graph | New edges `below_spazer_to_spazer` / `spazer_collect` / return; item bit gate |
| Timing | Small detour frames on K2; often recovered via easier later combat |
| 100% | Spazer is mandatory for full item; epic seeds the 100% board |
| Continuous fold | After green tip, default spine stops skipping `0xA447` |

**Not** a ship-first / PRKD repath. Insertion stays on KPDR Red → Bat → Spazer.

## Priority vs spine

| Track | Priority | Blocks K4 Bubble? |
|-------|----------|-------------------|
| B spine (Bubble → Bat → Speed) | P0 | — |
| **Early Spazer** | **P2 parallel** | **No** |
| Continuous **fold** into default hops | P2 after pure green | Planner-serial only |
| 100% board | P3 (docs + item list) | No |

Serialize hot modules if touching `continuous.py` / `progression.py` /
`STATUS.md` (same rule as Clean fold).

## Ticket ladder

| ID | Kind | Goal | Status |
|----|------|------|--------|
| [`SM-SPAZER-SCAFFOLD`](SM-SPAZER-SCAFFOLD.md) | pure scaffold | Module + room const + stubs | **ready** |
| [`SM-SPAZER-SRC`](SM-SPAZER-SRC.md) | source | Continuous-like Below Spazer source | **ready** |
| [`SM-SPAZER-PURE`](SM-SPAZER-PURE.md) | pure | Collect + return pure green (walljump if needed) | open (after SRC) |
| [`SM-SPAZER-GRAPH`](SM-SPAZER-GRAPH.md) | graph | Progression edges + beam flag | open |
| [`SM-SPAZER-COMPOSE`](SM-SPAZER-COMPOSE.md) | compose | Catalog tip `--to spazer` | open |
| [`SM-SPAZER-STAB`](SM-SPAZER-STAB.md) | stabilize | Dual continuous integrity | open |
| [`SM-SPAZER-STATUS`](SM-SPAZER-STATUS.md) | status | Secondary STATUS / tracker promote | open |
| [`SM-SPAZER-POLICY`](SM-SPAZER-POLICY.md) | pure/docs | Later policies prefer Spazer when held | open (after pure) |
| [`SM-SPAZER-FOLD`](SM-SPAZER-FOLD.md) | compose | Fold detour into default continuous spine | open (after STAB) |
| [`SM-100-TRACK`](SM-100-TRACK.md) | docs | 100% milestone / item board scaffold | **ready** (parallel) |

Supersedes one-line parked `SM-OPT-SPAZER` in `BACKLOG.csv`.

## Acceptance (epic close)

- [ ] Pure green: continuous-like Below Spazer → Spazer collected → back in
      `0xA408` with Spazer beam bit set (no progression RAM writes)
- [ ] Continuous tip `--to spazer` dual integrity green (secondary tip OK)
- [ ] Graph edges marked and tested; natural-entry from real predecessor
- [ ] At least one later combat policy path *uses* Spazer when collected
- [ ] Fold decision recorded: default spine includes Spazer **or** explicit
      defer with reason
- [ ] 100% board lists Spazer as collected on that track

## Related

- Route: [`docs/routes/ROUTE_KPDR.md`](../routes/ROUTE_KPDR.md) (Spazer optional safety)
- Tracker row K2.2: [`KPDR_TRACKER.csv`](../routes/KPDR_TRACKER.csv)
- Continuous tip already: `below_spazer` (prefix)
- Practice: `room_a447_from_a408_to_a408`
- Process: [`PROCESS.md`](PROCESS.md) pure-first
- Queue: [`QUEUE.md`](QUEUE.md) parallel track table
