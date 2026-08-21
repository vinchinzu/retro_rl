# Queue — Zelda I

**Source of truth:** `bd ready -l zelda_i`

Human snapshot. Expand room beads when that dungeon is the tip (or for
parallel isolated pure from checkpoints).

## Agent priority (finish → tune)

1. **Pathfinding + puzzles** (hops, doors, keys, bomb walls, blocks, items)
2. **Assisted full route** (`--infinite-life`; damage heatmap in report)
3. **Clean harden later** — rank rooms by `assist.damage_by_location`; combat
   polish is residual, not tip-blocking

Infinite life is intentional for first-pass agents. Do not stall L2 tip on
sword kiting once geometry is known.

## Live tip (Survival spine — no new room leaf unless claimed)

| Gate | Status | Notes |
|------|--------|-------|
| Power-on → L1 Triforce | **Clean green** | M5; do not overwrite |
| Survival spine L1 | **green** `rr-4d53.1` | power-on → TF `0x01` → L2 `0x7d`; Aquamentus 877f |
| Survival spine L2 TF | **green** `rr-4d53.2.3` | `--through level2` TF `0x02` |
| Survival spine L3 entry | **green** `rr-4d53.3.0` | Manji `0x7c`; 53918f |
| Survival spine L3 west key | **green** `rr-4d53.3.1.1` | live `0x7c` → `0x7b` keys=5; 54589f |
| Survival spine L3 dest | **tip** `rr-4d53.3.1.2` | live occupancy dest `0x5b` |
| L3 Raft / TF | blocked | `.3.3.*` then `.3.4.*`; isolated pins do not close |
| L4 / L5 spine hops | **queued** | `rr-4d53.6` / `.7` after L3 TF |
| L9 backward suffix | **parked P4** | `rr-yxy6` blocked on `rr-4d53` |
| Hygiene | **parked P4** | `rr-ekwl` run_once leftover |

### Architecture (agent monitor)

```
tip spine:  rr-4d53.3.1.2 dest 0x5b (run_survival_spine --through level3)
parked:     L9 (rr-yxy6); hygiene rr-ekwl; isolated L4 rr-q3n
queued:     .3.3 Raft → .3.4 TF → .6 L4 → .7 L5 → .4 compose
process:    one session until fail → --infinite-life full clear → heatmap Clean
close rule: LEVEL3_ROUTE.md § Spine attach (isolated Level3* pins are not approval)
```

Claim one tip leaf: `bd update rr-4d53.3.1.2 --status in_progress`.

### All-night wave results (2026-08-06 night)

| Bead | Result |
|------|--------|
| **rr-lzk** | **2/2 Clean** bomb N 0x6f→0x5f; `run_level2_bomb_north.py` |
| **rr-etl** | **2/2 Clean** Goriya 0x5e; `run_level2_clear5e.py` |
| **rr-fvt** | 0x5f = 5× Gel; doors DOWN-only; clear ≠ open R/U |
| **rr-cjf** | **LIVE** 0x5f bomb-UP→**0x4f**; 0x5e UP→0x4e→0x4f |
| **rr-3pz** | `level2_puzzles.py` catalog |
| **rr-mhl** | `door_graph.py` + L2 seed |
| **rr-65w** | **2/2 Clean** L3 north to 0x5b Darknuts |
| **rr-vpl** tip | Raft **2/2** + Manhandla→TF **2/2 assisted** (`run_level3_to_boss.py` / `Level3Complete`) |
| **rr-miy** | **2/2 assisted** L6 west wizzrobes 0x78 |
| **rr-87a** / **rr-076** | L5 0x67 Bubble dead-end; 0x77 isolated combat 2/2; route east key is `rr-28p` |
| **rr-iri** / **rr-ccx** / **rr-q8a** | OW hops + candle shop 0x5E; burn residual |

## Parallel wave results (2026-08-06)

### Wave 1 — recon (OW door + entry)

| L | Door OW | Entry room | Checkpoint | Track |
|--:|---------|------------|------------|-------|
| 3 | **0x74** | **0x7c** | `Level3Entrance` | assisted enter |
| 4 | **0x45** (dock **0x55**) | **0x71** | `Level4Entrance` | assisted enter |
| 5 | **0x0B** (hills **0x1B**) | **0x76** | `Level5Entrance` | assisted enter |
| 6 | **0x22** | **0x79** | `Level6Entrance` | assisted enter |
| 7 | 0x42 hyp | — | plan (needs whistle) | source |
| 8 | **0x6D** bush | — | `Level8BushOW` | candle blocked |
| 9 | 0x05 hyp | — | plan (needs TF `0xFF`) | source |

Docs: `LEVEL{3–9}_ROUTE.md`, `OVERWORLD_DOORS.md`, `PARALLEL_RECON.md`.

### Wave 2 — isolated pure (closed)

| Bead | Result | Track | Checkpoint | Runner |
|------|--------|-------|------------|--------|
| **rr-g4p** L3 west key | **0x7c→0x7b** 6× Zol `0x13`, keys 0→1 | **Clean** 2/2 | `Level3WestKey` | `run_level3_west_key.py` |
| **rr-vqw** L5 clear 0x66 | 3× `0x30` Gibdo, doors east→**0x67** | **Clean** 2/2 | `Level5Cleared66` | `run_level5_clear66.py` |
| **rr-r9l** L6 east key | **0x79→0x7a** 5× `0x24`, keys 0→1 | **assisted** 2/2 | `Level6EastKey` | `run_level6_east_key.py --infinite-life` |
| **rr-q8a** L8 candle enter | PARTIAL — bush only | — | `Level8BushOW` | candle residual |

Door traps burned:

- L3 west: pure LEFT sticks x≈32 → use **LEFT+UP** @ y≈149
- L6 east: y≈157 → x≈208 → y≈144–149 RIGHT (no A while aligning)

## Next beads (tip + ready)

```bash
bd ready -l zelda_i   # tip: rr-4d53.3.1.2 L3 dest 0x5b
```

| Bead | Role | Start / notes |
|------|------|---------------|
| **`rr-4d53.2.3`** | **closed** Boom → TF `0x02` | `run_survival_spine.py --through level2` |
| **`rr-4d53.3.0`** | **closed** Manji entry `0x7c` | `run_survival_spine.py --through level3` (old stop) |
| **`rr-4d53.3.1.1`** | **closed** west key `0x7b` | `l3_west_key_spine.json` 54589f keys=5 |
| **`rr-4d53.3.1.2`** | **TIP** dest `0x5b` | spine-only close; isolated north-chain 2/2 is not approval |
| **`.3.3.*` / `.3.4.*`** | Raft → TF | blocked; see LEVEL3_ROUTE Spine attach |
| **`rr-4d53.6` / `.7` / `.4`** | later spine | L4 then L5 then one-session compose |
| **`rr-ekwl` / `rr-yxy6`** | parked | hygiene / L9; do not claim |
| **`rr-38p` ZOW.1** | parallel free | white sword + candle + bomb bag |
| Later | `rr-d6v` L6 TF, `rr-4oz` Clean residual, `rr-yhr` bracelet/mag sword | after assist tip |

### Closed wave history (keep for evidence)

L3 west key / Raft / Manhandla TF, L5 clear 0x66, L6 east key, L2 bomb/Goriya,
OW recon table — see closed beads + `LEVEL{2,3,5,6}_ROUTE.md`.

### Gated (after items)

| Epic | Gate |
|------|------|
| L4 Snake clear | live entry ✓; interior `rr-5lu` |
| L7 Demon | Whistle from L5 + bait shop |
| L9 Death Mountain | Full TF `0xFF` |
| Continuous dry run | assisted then Clean stack |

## Reactive splice (later)

For each pure green segment:

1. Isolated pure from checkpoint (done for L3 west / L5 0x66 / L6 east key)
2. Natural-entry from real predecessor (no mid-run state load)
3. NamedRoute / `routes_later.py` promote when tip arrives
4. Clean STATUS only after natural 2/2 — never from assist-only

Assisted L6 east key needs Clean combat harden before STATUS.

## Process

- [`PROCESS.md`](PROCESS.md) — dual track, pure-first
- [`PARALLEL_RECON.md`](PARALLEL_RECON.md) — wave 1
- [`PARALLEL_PURE.md`](PARALLEL_PURE.md) — wave 2
- [`ASSIST_CONTRACT.md`](../ASSIST_CONTRACT.md)
- [`STATUS.md`](../STATUS.md) — Clean claims only (do not invent promotes here)

## Commands (wave 2 green)

```bash
uv run python nes/zelda_i/scripts/run_level3_west_key.py --trials 2 --save-state
uv run python nes/zelda_i/scripts/run_level5_clear66.py --trials 2
uv run python nes/zelda_i/scripts/run_level5_clear66.py --from-entrance --save-state
uv run python nes/zelda_i/scripts/run_level6_east_key.py --infinite-life --trials 2 --save-state
```
