# Speed → Wave → Ice → Moat — human tape map + pure beads

**Epic tracker:** `rr-dbu`  
**Full tape (rr-dbu.12 GREEN):** `tasks/speed_to_wave_ice_moat_human.json`  
**End state:** `tasks/speed_to_wave_ice_moat_human_end.state` (+ scratch copy)  
**Start used:** `scratch/post_speed_collected.state` (Speed `0xAD1B` ~(169,123))  
**Legacy partial (ignore for Ice):** `tasks/speed_to_ice_moat_human.json` — stops Double Chamber only (5373f)

| Metric | Value |
|--------|------:|
| Frames | **39,711** (~662 s @ 60 fps) |
| Recorded | 2026-08-06T20:09 |
| Route | `speed-to-ice-moat` |
| Assist | ON (energy+ammo; practice only — **not pure**) |
| Deaths | 0 |
| Progression / capacity writes | 0 |
| Wave first entry | f**4942** `0xADDE` |
| Ice first entry | f**15626** `0xA890` |
| Moat first entry | f**39559** `0x95FF` |
| End | f39710 `0x95FF` ~(52,144) missiles 20 supers 5 pbs 5 |

**Accept checklist (rr-dbu.12):** Wave ✓ · Ice ✓ · Moat stretch ✓

---

## Operator notes (human, 2026-08-06 take)

Recorded take is **mostly clean** with known flaws. Keep these for pure recon
and any re-record — do **not** treat this tape as perfect open-loop RLE.

| Note | Detail | Follow-up |
|------|--------|-----------|
| **No Spazer / Charge at start** | `--from speed` loads `post_speed_collected` (pure Speed handoff). Spazer mainline continuous uses beams **`0x1004`** (Charge+Spazer). Operator expected Spazer+Charge. | Prefer `scratch/post_speed_collected_with_spazer.state` for re-record / future `--from speed` default. Pure Ice can still use continuous Wave tip handoff later. Fix when convenient — **not blocking** room recon. |
| **Climb freeze bad → use 2WJ** | Freeze-climb segment felt wrong; intended technique is **two wall jumps**, not ice-freeze ladder. | When pure-ing that room, prefer 2WJ geometry over freeze-platform thrash. Likely Ice Snake and/or Red Tower vertical (long dwells in tape). |
| **Some mistakes** | Stasis / thrash windows (see longest below). Geometry still readable. | Pure one-hops should re-solve cleanly; do not clone thrash RLE. |
| **“2nd” take** | Session printed empty `[CHECKPOINT 1]` ×4, then checkpoint f22504, final save f39711. One saved product file only — treat this as the **kept take**. Empty checkpoints are F1/F5 noise, not a second JSON. | Re-record only if Spazer loadout or climb technique must be product-grade for human RLE. |

### Longest stasis / thrash (avoid cloning)

| frames | len | room | buttons (held) |
|--------|----:|------|----------------|
| 5468–5700 | 233 | Wave `0xADDE` | A B LEFT X |
| 3942–4144 | 203 | Double Chamber | A B LEFT |
| 34123–34225 | 103 | Alpha PB `0xA3AE` | A B RIGHT UP |
| 16277–16365 | 89 | Ice `0xA890` | A LEFT X |

---

## Full hop table (this tape)

Room transitions only (first frame in each room). Names from `room_ids` + board.

### Phase A — Speed → Wave

| # | frames | room_hex | name | entry xy | notes |
|---|--------|----------|------|----------|-------|
| 1 | 0–284 | `0xAD1B` | Speed | (169,123) | start pin |
| 2 | 285–887 | `0xACF0` | Speed Hall | (19,139) | |
| 3 | 888–1666 | `0xB07A` | Bat Cave | (18,139) | return shelf path |
| 4 | 1667–2029 | `0xACB3` | Bubble | (20,395) | Wave branch |
| 5 | 2030–2899 | `0xAD5E` | Single Chamber | (493,395) | |
| 6 | 2900–4941 | `0xADAD` | Double Chamber | (236,395) | gate + Super; thrash ~f3942 |
| 7 | 4942–5908 | `0xADDE` | **Wave Beam** | (1005,139) | collect; thrash ~f5468 |

### Phase B — Wave return → Ice (★ pure recon)

Outbound Ice path **skips Ice Tutorial on entry**. Uses **Acid Room** (`0xA75D`)
between Ice Gate and Ice Snake — do not invent Tutorial-only pure without need.

| # | frames | room_hex | name | entry xy | notes |
|---|--------|----------|------|----------|-------|
| 8 | 5909–6751 | `0xADAD` | Double Chamber | (18,139) | Wave exit left |
| 9 | 6752–7496 | `0xAD5E` | Single Chamber | (20,395) | |
| 10 | 7497–8951 | `0xACB3` | Bubble | (19,139) | leave left → Farm |
| 11 | 8952–9282 | `0xAF72` | Upper Norfair Farm | (19,907) | post-Speed shortcut |
| 12 | 9283–9707 | `0xB106` | Frog Speedway | (18,139) | |
| 13 | 9708–9987 | `0xB167` | Frog Save | (18,139) | |
| 14 | 9988–10816 | `0xA7DE` | Business | (20,139) | Super left → Ice |
| 15 | 10817–11230 | `0xA815` | Ice Beam Gate | (18,907) | |
| 16 | 11231–11963 | `0xA75D` | **Ice Beam Acid Room** | (786,651) | not Tutorial |
| 17 | 11964–15625 | `0xA8B9` | Ice Snake | (20,139) | long; **prefer 2WJ** |
| 18 | 15626–16490 | `0xA890` | **Ice Beam** | (494,395) | PLM collect |

### Phase B return (Ice → Business)

| # | frames | room_hex | name | entry xy | notes |
|---|--------|----------|------|----------|-------|
| 19 | 16491–17886 | `0xA8B9` | Ice Snake | (18,139) | exit |
| 20 | 17887–19151 | `0xA865` | Ice Tutorial | (236,146) | **return path only** |
| 21 | 19152–20020 | `0xA815` | Ice Gate | (494,139) | |
| 22 | 20021–20837 | `0xA7DE` | Business | (1772,651) | bottom / elev band |

### Phase C — K5 Alpha PB + Crateria → Moat (stretch)

| # | frames | room_hex | name | entry xy | notes |
|---|--------|----------|------|----------|-------|
| 23 | 20838–21200 | `0xA6A1` | Warehouse | elev | |
| 24 | 21201–21437 | `0xCF80` | East Tunnel | (19,139) | reverse Spazer tunnel |
| 25 | 21438–21647 | `0xCEFB` | Glass Tunnel | (17,395) | |
| 26 | 21648–21857 | `0xCF54` | West Tunnel | (16,395) | |
| 27 | 21858–22366 | `0xA408` | Below Spazer | (18,139) | |
| 28 | 22367–23077 | `0xA3DD` | Bat Room | (20,395) | |
| 29 | 23078–29946 | `0xA253` | Red Tower | (19,136) | **~7k frames** — thrash / freeze-climb candidate |
| 30 | 29947–31449 | `0xA2F7` | Hellway | (239,139) | |
| 31 | 31450–32172 | `0xA322` | Caterpillar | (751,139) | |
| 32 | 32173–34439 | `0xA3AE` | **Alpha PB** | (18,1931) | PB unlock; thrash ~f34123 |
| 33 | 34440–35969 | `0xA322` | Caterpillar | (750,139) | return |
| 34 | 35970–36550 | `0x962A` | Elevator to Caterpillar | elev | up to Crateria |
| 35 | 36551–38129 | `0x948C` | Crateria Kihunter | (130,33) | pre-Moat class |
| 36 | 38130–38326 | `0x95D4` | Crateria Tube | (20,139) | brief |
| 37 | 38327–39558 | `0x948C` | Crateria Kihunter | (238,139) | re-enter |
| 38 | 39559–39710 | `0x95FF` | **The Moat** | (749,139) | end ~(52,144) |

Pure Moat **spark** remains pin-only GREEN (`rr-hhj`). This tape documents
**natural approach loadout** after Ice + Alpha PB — not spark pure.

---

## Pure stack map (from this tape)

### Ice pure (`rr-dbu.11`) — package `routes/kpdr/ice/`

Split one-hops **after** Wave continuous tip / Business natural entry:

```text
Business 0xA7DE
  → Ice Gate 0xA815
    → Acid Room 0xA75D          ★ tape path (not Tutorial-first)
      → Ice Snake 0xA8B9        ★ prefer 2WJ over freeze climb
        → Ice 0xA890 PLM
```

Return path (optional later): Ice → Snake → **Tutorial** `0xA865` → Gate → Business.

Do **not** invent hops that skip Acid if that was the natural door sequence.

### K5 / Moat approach (later)

```text
Business → Warehouse → East → Glass → West → Below Spazer → Bat
  → Red Tower → Hellway → Caterpillar → Alpha PB 0xA3AE
  → Caterpillar → elev 0x962A → Kihunter 0x948C → (Tube) → Moat 0x95FF
```

Beads: `rr-dbu.8` (K5), `rr-dbu.9` (Moat approach to pre-spark pin).

---

## Start loadout caveat

| State | Path | Loadout |
|-------|------|---------|
| Used this take | `scratch/post_speed_collected.state` | pure Speed handoff — **no Spazer/Charge** (operator complaint) |
| Prefer next human | `scratch/post_speed_collected_with_spazer.state` | Spazer mainline parity |
| Product continuous | `--to wave` dual | beams `0x1005` Charge+Spazer+Wave at tip |

When pure Ice starts from continuous Wave / Business pins, loadout follows the
spine (Spazer mainline). Human tape geometry still valid; combat difficulty
was harder without Charge/Spazer.

---

## Legacy partial tape (historical only)

`tasks/speed_to_ice_moat_human.json` — **misnamed**; 5373f; ends Double Chamber
past blue gate after missile pack. Never Wave/Ice/Moat. Kept for gate human
RLE history only.

| # | frames | room_hex | name |
|---|--------|----------|------|
| 1–6 | 0–5372 | Speed…Double | see git history / older section |

---

## Re-record runbook (optional)

Only if Spazer loadout or clean 2WJ climb is required for human RLE product.

```bash
# Prefer Spazer start when wiring exists:
uv run python snes/super_metroid/scripts/record/guided_human.py \
  --from speed --route speed-to-ice-moat \
  --name speed_to_wave_ice_moat_human_spazer
# (after default start points at post_speed_collected_with_spazer)
```

Controls: **F5/F1** save · **ESC/Q** cancel. Assist ON = practice, not pure.

### Why not continuous / pure from this file

- Human assist tape ≠ pure controller evidence
- Do not claim continuous `--to ice` until pure stack greens (`rr-dbu.7`)
- Package new Ice rooms under `routes/kpdr/ice/` — never extend Wave megafile

---

## Bead board (post-tape 2026-08-06)

| Id | Title | Status |
|----|-------|--------|
| **rr-dbu.12** | Full human Speed→Wave→Ice→Moat | **GREEN** — this tape |
| **rr-fg3** | Pure Business → Ice Gate | **GREEN** dual 894f ×2 `0xA815` ~(1752,651); `routes/kpdr/ice/` |
| **rr-dbu.11** | Ice pure stack recon → one-hops | **partial** — Gate done; Acid→Snake→PLM open |
| **rr-9t4** | Pure Ice Gate → Acid Room | **ready** (after rr-fg3) |
| rr-dbu.7 | continuous `--to ice` | blocked on Ice pure |
| rr-dbu.8 | K5 Alpha PB pure | blocked on Ice |
| rr-dbu.9 | Moat approach pure | blocked on K5; spark pin `rr-hhj` GREEN |
| Wave stack | gate + Super + tip wave | **done** (`rr-re9` / `rr-l0u`) |

### Done pure spine (pre-Ice)

`rr-g4i` Speed→Bubble → Bubble→Single → `rr-g1b` Single→Double → gate → Wave PLM → continuous **wave** 136,361f dual.

## Non-claims

- This tape is **not** pure / continuous evidence
- No STATUS tip past Wave until Ice pure + compose greens
- Start loadout without Spazer is a known tape flaw, not a route veto
- Climb freeze thrash is not the intended pure technique (2WJ)
