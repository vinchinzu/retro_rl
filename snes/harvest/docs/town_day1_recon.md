# Spring D1 town reconnaissance

Last verified: 2026-08-01

This note records the clean, controller-only D1 town work so the unfinished
part can be recorded from the same natural entry later. It is deliberately a
reconnaissance note, not a completion claim.

## Natural entry

`PowerOnStartTask` reaches Spring D1 07:00 in town map `0x04` at
`(712,424)`. The clean boot used no initial state load, no mid-run state load,
and no RAM writes. The existing evidence is
`recordings/power_on_boot_probe.json`.

## Required conversations

The D1 town handoff requires six conversations before the truck will accept
the “ready to leave” response:

| Bit | Person | Map / working approach |
|-----|--------|------------------------|
| `0x01` | Ann | Town lower road; approach the tool-shop-side NPC from the west, around `(388,924)`, face left, press A |
| `0x02` | Eve | Town lower-west road; approach from below around `(162,896)`, face up, press A |
| `0x04` | Nina | Flower-shop back room map `0x1D`; from the room spawn, route left/up/right to the NPC near `(74,113)`, face right, press A |
| `0x08` | Flower-shop owner | Flower shop map `0x1C`; the ROM event object targets `(40,360)` (the camera renders it in the upper-left counter area). Exact clean recording is still open |
| `0x10` | Livestock dealer | Animal shop map `0x24`; from the spawn, run right then up to approximately `(201,157)`, face right, press A |
| `0x20` | Maria | Church map `0x1B`; from the spawn, walk up about 60 frames to `(128,396)`, face left, press A |

The logical completion invariant is the six-bit mask `0x3F` at the live D1
town event field `0x11F74` (the live RAM snapshot mirrors WRAM at `+0x4000`).
The ROM event scripts assign the bits as shown above. Catalog name:
`d1_town_event_mask` (alias `town_day1_event_mask`).

## Verified transitions and routes

- Town → flower shop: from the clean gate, use town waypoints
  `(688,280) → (600,280) → (600,262)` and walk up. The transition is
  `0x04 → 0x1C`; the shop settles at `(144,456)`.
- Flower shop → back room: from the front-room spawn, move left about 20
  frames, then up. The transition is `0x1C → 0x1D` and settles near
  `(104,184)`.
- Town → church: segment through `(688,280) → (600,280) → (500,280) →
  (376,280) → (376,200) → (375,139)`, then walk up into map `0x1B`.
- Town → animal shop: use the lower-road route to `(688,888)`, then
  `(601,888) → (601,874)` and walk up into map `0x24`.
- Ann and Eve both set their expected bits in live probes. Nina, Maria, and
  the livestock dealer also have verified interaction stands and event text.

## Human recording (2026-08-01) — `tasks/town_day1_rest.json`

From `Y1_Spring_D1_AnnEve` (mask `0x03`) the rest of the handoff was recorded
controller-only (11 134 frames) → house sleep → Spring D2 06:00.

| Bit | Frame | Stand / face | Map |
|-----|------:|--------------|-----|
| `0x10` Livestock | 2976 | `(230,139)` face **down** + A | animal shop `0x24` |
| `0x04` Nina | 5159 | `(101,102)` face **left** + A | flower back `0x1D` |
| `0x08` Flower owner | 6599 | `(34,347)` face **down** + A | flower shop `0x1C` |
| `0x20` Maria | 8411 | `(103,405)` face **up** + A | church `0x1B` |
| mask `0x3F` complete | 8411 | — | — |
| Truck leave | ~9777 | path `0x0C` then **cutscene into house** `0x15` (no outdoor farm map) | |
| Sleep → D2 | 10788–10845 | bed → morning house `(136,120)` | house |

**Important:** `(201,157)` face-right in the animal shop is the later **buy-cow**
menu stand — it does **not** set D1 bit `0x10`. Use `(230,139)` face down.

### Starter tools (not picked in the recording)

New-game init already places free bags on the shed shelf
(`shed_items_row_2 = 0x88` = watering can `0x80` | grass seeds `0x08`).
`town_day1_rest` never visited the shed — end state still has empty carry.

ROM-verified shelf stands (face up + A):

| Item | Tool id | Stand px | Clears bit |
|------|---------|----------|------------|
| Grass seeds | `0x0C` | `(96,118)` | row2 `0x08` |
| Watering can | `0x10` | `(96,168)` | row2 `0x80` |

`TownDay1HandoffTask` picks **grass then can** after the town sequence (both
fit in the 2-slot carry pair when hands are empty). Verified from
`Y1_Inside_House` / `Y1_Front_House` (free-move `game_state & 0x4000`).

### Gate B blocker (2026-08-09, rr-bhr)

Pure Town_Gate / power-on path reaches peak mask `0x3F` and D2 morning bed
`(136,120)` via composed talks + truck leave + `GoToSleep`. **Shed still fails.**

#### Symptoms

1. `ExitToFarm` from that D2 bed reaches farm mid-warp `~(137,212)` then
   `(136,344)` but **clears free-move** (`game_state` `0x4001 → 0x0001` /
   `0x0081` during settle).
2. Player is auto-walked south into house-enter stand `~(133,425)` with no
   horizontal control (`gs` bit `0x1000` scripted walk).
3. Door dialogue (`text 0x0124/0x0125`) then soft-lock → tilemap `0x5F`.
4. `$0970` (`house_size`) INC's during Ann talk (0→2) is a **dialogue step
   counter**, not remodel — **not causal** for free-move loss.

#### Causal root (ROM + offline A/B on `town_day1_rest_end`)

| Pre-exit `event_flags_1f68` | Free-move after ExitToFarm | Notes |
|----------------------------|----------------------------|-------|
| `0x0011` (truck D2 bed) | **Lost** → softlock | Baseline pure/rest truck path |
| `0x0011` + `house_size=0` | **Lost** | house_size not causal |
| `0x0031` / `0x0091` | **Lost** | partial intro bits still fail |
| `0x00A1` / `0x00B1` (Y1) | **Kept** — can walk to shed | Min `0x00A1` = truck+intro+dog |

Bits (HM-Decomp `bank_83` `CODE_83CEAE`, `bank_84` dog whistle):

- `0x0001` — truck/day processing (present after truck leave)
- `0x0020` — first outdoor morning intro done (CC `0x0C/0`; sets mid-exit)
- `0x0080` — **dog owned**

With only `0x0011`, house→farm runs morning intro: ORA `0x0020`, clear free-move,
auto-walk to door. Controller-only recovery (neutral, mash A/B, name-entry
guesses, hold-down mid-warp, clock wait) **never restores free-move**; dog bit
`0x0080` never sets. Y1 fixtures already have `0x00B1` so intro is skipped.

Pure talks alone reach mask `0x3F` with free-move and flags still `0x0010`, but
`d1_town_to_farm` cannot leave town without the truck cutscene (east gate
stays closed). So shed cannot be completed on D1 without truck either.

#### Mitigations landed (still not acceptance)

- Gate B truck: rest leave-only slice `f9200:9800` + `GoToSleep` morning settle
- `ExitToFarm` / `ExitBuilding` push DOWN through farm mid-warp `y<330`
- `ShedFetchItemTask` fails fast with `farm_control_lost` (+ `f1f68` / intro_ok)
- Helpers: `farm_free_move_ready`, `outdoor_intro_flags_ready` (mask `0x00A1`)

#### Next fix (pure)

Human (or pure automation) must **complete D2 morning outdoor dog intro** so
`event_flags_1f68` reaches ≥ `0x00A1` **with free-move restored**, then shed
grass+can. Re-record from truck D2 bed through successful outdoor free-move
stand — not mid-warp y=212, and not house-front softlock. RAM-poke of `0x00B1`
is diagnosis-only (not Clean).

## Automation status (2026-08-01)

Precomputed controller automation lives in `harvest.tasks.town_day1_handoff`
and `uv run python -m harvest.scripts.town_day1_recon auto`.

When `tasks/town_day1_rest.json` is present, full-mask auto **replays that
recording** (livestock → nina → owner → maria → truck → house sleep → D2),
then optionally attempts shed starter tools.

| Bit | Person | Auto status |
|-----|--------|-------------|
| `0x01` | Ann | **Works** — outdoor route; rest recording assumes already set |
| `0x02` | Eve | **Works** — outdoor route; rest recording assumes already set |
| `0x04`–`0x20` | Nina/owner/livestock/Maria | **Works via rest recording** (human capture) |
| Shed pickups | grass + can | **Works** from `house_size=0`; soft-optional after rest (AnnEve is size2) |

Baseline verified run (Ann|Eve = `0x03`):

```bash
HEADLESS=1 uv run python -m harvest.scripts.town_day1_recon auto \
  --state Y1_Spring_D1_Town_Gate --no-sleep --no-require-full-mask \
  --out recordings/town_day1_auto.json
```

Full handoff via rest recording (peak mask `0x3F` → D2):

```bash
HEADLESS=1 uv run python -m harvest.scripts.town_day1_recon auto \
  --state Y1_Spring_D1_AnnEve \
  --out recordings/town_day1_rest_auto.json
# Verified 2026-08-01: success=True peak_mask=0x3F day=2
```

Replay the human capture:

```bash
HEADLESS=1 uv run python -m harvest.scripts.town_day1_recon replay \
  --task town_day1_rest --state Y1_Spring_D1_AnnEve --require-day2 \
  --out recordings/town_day1_rest_replay.json
```

Entry state from power-on: `Y1_Spring_D1_Town_Gate.state` (saved by `capture-entry`
or `auto --power-on`).

## Record → automate tooling

Use the recon CLI (or the shell wrapper) to capture a clean entry, record the
controller route with a live mask HUD, then headlessly replay for skill
extraction:

```bash
# Checklist / bit labels
uv run python -m harvest.scripts.town_day1_recon checklist
# or: ./scripts/record_town_day1_recon.sh --checklist

# Pin Spring D1 town gate from power-on (no state load)
HEADLESS=1 uv run python -m harvest.scripts.town_day1_recon capture-entry
# → custom_integrations/HarvestMoon-Snes/Y1_Spring_D1_Town_Gate.state
# → recordings/town_day1_entry.json

# Interactive record (auto-captures entry state if missing)
./scripts/record_town_day1_recon.sh
# Clean power-on + record in one session:
./scripts/record_town_day1_recon.sh --power-on
# F5 → tasks/town_day1_handoff.json (+ end states)

# Headless replay + mask assertion (for automation extraction)
HEADLESS=1 uv run python -m harvest.scripts.town_day1_recon replay \
  --task town_day1_handoff \
  --out recordings/town_day1_handoff_replay.json
# Full natural-entry replay:
HEADLESS=1 uv run python -m harvest.scripts.town_day1_recon replay \
  --task town_day1_handoff --power-on --require-day2
```

Suggested capture order from the gate: flower owner (`0x08`) + Nina (`0x04`),
church Maria (`0x20`), Ann (`0x01`) + Eve (`0x02`), livestock dealer (`0x10`),
truck leave at mask `0x3F`, path → farm → sleep → D2.

## Walkthrough references

The six-person checklist and truck handoff agree with these walkthroughs:

- [GameFAQs Admiral walkthrough](https://gamefaqs.gamespot.com/snes/562623-harvest-moon/faqs/52336)
- [GamerZenith first-day guide](https://gamerzenith.com/guides/first-day-hmsnes/)
- [Zerocool GameFAQs guide](https://gamefaqs.gamespot.com/snes/562623-harvest-moon/faqs/22320)

