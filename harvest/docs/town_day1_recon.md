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

## Still to record

1. Capture the flower-shop owner conversation from a clean D1 entry. The
   front room has a lower public floor and a counter/object trigger; the
   generic BFS route stops at the counter, so this needs a short recorded
   controller segment and a `0x08` assertion.
2. Return from the shop and visit the remaining town NPCs as needed, then
   walk to the truck/shipper object near town `(728,424)`.
3. Press the truck dialogue’s leave/ready response and record the resulting
   town → path → farm transition.
4. Continue to the farmhouse, sleep, and assert natural D2. Only after this
   succeeds should the D2→Summer soak be relabeled as a power-on replay.

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

