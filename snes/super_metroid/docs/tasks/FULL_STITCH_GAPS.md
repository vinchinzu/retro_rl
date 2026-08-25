# Full-stitch gap map — power-on → credits

Operator view of **what is not green** for a continuous assisted power-on →
ending/credits stitch. Not a STATUS promote. Product continuous tip remains
**Ice Beam** only until dual continuous past Ice lands.

Sources: [STATUS.md](../STATUS.md) · [plan.md](../plan.md) ·
[routes/KPDR_TRACKER.md](../routes/KPDR_TRACKER.md) ·
[HUMAN_TAPE_PIPELINE.md](HUMAN_TAPE_PIPELINE.md) · residuals under
`docs/tasks/rr-*.md` · beads `rr-dbu.8` / `rr-av5s` / `rr-dbu.9` / `rr-7thf.*`.

---

## Continuous product tip

| Field | Value |
|-------|-------|
| STATUS tip | **Ice Beam** dual green **148,167f** ×2 (`--to ice`) |
| Integrity | 0 loads / progression / capacity / deaths |
| Beams / items | `0x1007` / `0x3105` (Charge+Spazer+Wave+Ice; Speed) |
| Reports | `recordings/ice.json` + `ice_dual.json` |
| Next continuous | **K5** Alpha PB pure stack → natural Moat → … → credits |

Everything after Ice is **not** continuous-product yet, even when pure
one-hops or hop-replay skills are dual green.

---

## KPDR spine after Ice (not continuous yet)

| Seg / bead | Segment | Pure dual | Continuous | Notes |
|------------|---------|-----------|------------|-------|
| K4 Ice ✅ | Power-on → Ice PLM | compose dual | **GREEN** 148167f | STATUS tip |
| K5 return hops 0–11 | Ice → … → Bat → Red bottom | **11 dual GREEN** | ⬜ | `rr-dbu.8`; residual |
| **K5 hop 12** | Red Tower → Hellway | **PARTIAL** thin-seat ~(91,587); Hellway RED | ⬜ | **`rr-av5s`** ice ladder residual |
| K5 hops 13–14 | Hellway → Caterpillar → Alpha PB | ⬜ | ⬜ | first PB capacity |
| **K5 stack** | post-Ice → Alpha PB PLM | stack incomplete | ⬜ | **`rr-dbu.8`** in_progress |
| **K6.0 Moat** | approach to pre-spark pin | spark hop GREEN alone (`rr-hhj`); approach ⬜ | ⬜ | **`rr-dbu.9`** blocked on K5 |
| K6.1 West Ocean | after Moat spark | ⬜ | ⬜ | needs Moat natural entry |
| K6.2+ WS | WS entrance → Phantoon → power ship → Gravity | human tapes only | ⬜ | see missing anchors |
| K7 Maridia | Glass → Draygon → SJ | human tapes only | ⬜ | thrash queue |
| K8 LN | Bubble → Ridley | human tapes only | ⬜ | thrash queue |
| K9 Tourian | G4 → MB → escape → LS | hop-replay partial | ⬜ | Waves A–D below |

KPDR tracker snapshot (whole route, not only post-Ice): **41 continuous /
48 open / 2 controller_dev / 1 parked** (92 segments). Continuous set ends
at K4-class tips already STATUS-backed; K5–K9 rows remain `open` on the
tracker until continuous dual re-verify.

---

## Late spine human skills (hop-replay, not continuous compose)

Epic **`rr-7thf`**. Pipeline: extract → hop-replay from live anchor → trim →
dual green. **Not** continuous power-on stitch.

| Wave | Bead | Focus | Status |
|------|------|-------|--------|
| **A** | `rr-7thf.4` ✅ | Escape 1–4 (`g4_tourian_human_mb`) | **4/4 dual GREEN** |
| **B** | `rr-7thf.5` ✅ | Climb + Parlor + Landing Site ship | **3/3 dual GREEN** (`ENDING_OR_CREDITS` observed) |
| **C** | `rr-7thf.6` ◐ | G4 statues → Metroids → Big Boy | **15/16 GREEN**; **RED** Metroid 4 leave |
| **D** | `rr-7thf.7` ✅ closed | MB approach + MB fight | Approach 0–7 GREEN; leave-to-escape GREEN; **RED** bb hop8 mid→stun |
| thrash | `rr-7thf.9` ○ | Ridley / Worst / Metal Pirates / … | open queue |

### Known hop-replay RED (need mid F6 / pure rewrite)

| Residual | Hop | Detail |
|----------|-----|--------|
| `rr-7thf.6` | Metroid Room 4 `0xDBCD` → Hopper | enter→mid dual GREEN (`f020360` mid_lockstep); mid→Hopper **RED** |
| `rr-7thf.7` | MB room mid→stun (bb hop 8) | enter→mid lockstep pin OK; full mid-stun open-loop **RED**; escape leave via mb boot GREEN |

Wave A/B dual green does **not** make credits continuous from power-on.

---

## Human tapes missing anchors

Bead **`rr-7thf.8`**. Offline midpoints proposed; **cannot hop-replay** without
live enter/boot pins. Re-record short takes with anchors ON + F5 (full buttons).
Do not invent mid pins by multi-minute full-tape open-loop; use hop-compose
from live pins once tapes exist.

| Tape | Issue | Follow-up |
|------|-------|-----------|
| `ws_ship_human` | no live anchors | short re-record from ws-entrance / post-phantoon |
| `gravity_path_human` | legacy extract / snapshots only | not a hop board; re-record path |
| `maridia_grapple_human` | end state **LOST** (open-loop desync) | `--from post-grapple`; hops in extract only |

---

## Thrash queue not dual-green skills yet

Bead **`rr-7thf.9`**. Top dwells from `tasks/LATE_SPINE_HOP_BOARD.json`
`thrash_ranking` (hop-replay dual green still open outside Waves A/B/partial C/D):

| Rank | Room | Dwell (frames) | Tape |
|-----:|------|---------------:|------|
| 1 | Mother Brain's Room | 14721 | `g4_tourian_human_bb` (mid→stun RED) |
| 2 | Metal Pirates Room | 11657 | `post-main-hall` |
| 3 | Colosseum | 11459 | `maridia_botwoon_path_human` |
| 4 | Mother Brain's Room | 10805 | `g4_tourian_human_mb` |
| 5 | Red Tower | 8119 | `post-main-hall` |
| 6 | Ridley's Room | 7810 | `post-main-hall` |
| 7 | East Pants Room | 7687 | `post_sj_exit_human` |
| 8 | Mt. Everest | 7625 | `maridia_botwoon_path_human` |
| 9 | Aqueduct | 6721 | `maridia_botwoon_path_human` |
| 10 | Metroid Room 1 | 6460 | `g4_tourian_human` |
| 11 | Golden Torizo | 6355 | `post-main-hall` |
| 12 | Worst Room in the Game | 6012 | `post-main-hall` |

Acceptance for thrash queue: top rooms each get hop-replay dual green + trim
report (or pure skill residual) — not continuous compose.

---

## Compose / stitch blockers (real full-run gaps)

What must be **pure dual-green + natural-entry continuous** before STATUS can
claim power-on → credits:

```text
Ice (✅ continuous dual)
  → K5 pure stack complete (rr-dbu.8: hop12+ Hellway→Alpha PB still open)
  → continuous compose past Ice / Alpha PB dual re-verify
  → K6 Moat natural approach (rr-dbu.9) + spark chain already GREEN (rr-hhj)
  → West Ocean → WS → Phantoon → Gravity (anchors + pure/compose)
  → Maridia (Grapple path anchors; Botwoon/Draygon thrash → pure)
  → LN (Metal Pirates / Ridley thrash → pure)
  → G4 → Metroids (Metroid4 leave RED) → Big Boy → MB (mid→stun RED)
  → Escape (A GREEN) → Climb/Parlor/LS (B GREEN) as continuous tail
  → dual continuous full run → STATUS credits
```

| Blocker class | Why it blocks full stitch | Owner beads |
|---------------|---------------------------|-------------|
| **K5 incomplete** | Continuous tip stuck at Ice; no pure path to Alpha PB / elev out | `rr-dbu.8`, `rr-av5s` |
| **Moat approach** | Spark pin alone ≠ natural Moat entry from product chain | `rr-dbu.9` |
| **WS / Gravity / Grapple tapes** | No live anchors → no hop-replay pure path | `rr-7thf.8` |
| **Maridia / LN thrash** | Long human dwells not dual-green skills | `rr-7thf.9` |
| **Metroid 4 leave** | Wave C hole on G4→BB path | `rr-7thf.6` |
| **MB mid→stun** | Wave D phase-split residual | `rr-7thf.7` residual |
| **Continuous compose** | Hop-replay green ≠ power-on chain; each segment needs natural-entry continuous dual | plan spine / STATUS process |

### Explicit non-claims

- Wave A/B dual green ≠ continuous credits from power-on.
- Pure K5 hops 0–11 ≠ continuous past Ice.
- Moat spark GREEN ≠ Moat approach continuous.
- Closing a bead ≠ STATUS promote.

---

## Live work

```bash
bd ready -l super_metroid -l spine
# Product next: rr-8g2u power-on --to phantoon dual (scratch)
```

Pipeline: [HUMAN_TAPE_PIPELINE.md](HUMAN_TAPE_PIPELINE.md).
Process: `.grok/skills/sm-session/`. Ready work: `bd ready -l super_metroid -l spine`.
