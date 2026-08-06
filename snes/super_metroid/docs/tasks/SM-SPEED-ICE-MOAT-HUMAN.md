# Speed → Ice → Moat — human tape map + pure beads

**Epic tracker:** `rr-dbu` (P3 doc epic; not a work card)  
**Primary tape:** `tasks/speed_to_ice_moat_human.json`  
**Start:** `scratch/post_speed_collected.state` (Speed `0xAD1B`)  
**End state:** `scratch/speed_to_ice_moat_human_end.state`  
**Frames:** 5373 · assist unlimited energy+ammo (not pure)

## Tape truth

Filename is **aspirational**. Recording **never** reaches Wave (`0xADDE`),
Ice (`0xA890`), or Moat (`0x95FF`). Stops in **Double Chamber** past blue gate
after a missile pack.

Need full path: bead **`rr-dbu.12`** (human re-record).

## Human hop table (on tape only)

| # | frames | room_hex | name | notes |
|---|--------|----------|------|-------|
| 1 | 0–389 | `0xAD1B` | Speed | start ~(169,123) |
| 2 | 390–955 | `0xACF0` | Speed Hall | |
| 3 | 956–2130 | `0xB07A` | Bat Cave | pause f1631–1911 |
| 4 | 2131–2636 | `0xACB3` | Bubble | exit ~(492,395) |
| 5 | 2637–3302 | `0xAD5E` | Single | missiles @f3031 |
| 6 | 3303–5372 | `0xADAD` | Double | **gate f5126 x413**; end ~(494,139) |

### Double Chamber gate (human) → `rr-dbu.10`

| window | frames | geometry / inputs |
|--------|--------|-------------------|
| seat | ~4650–4710 | ~(378,139) missiles; R / R+X |
| peak | ~4714–4731 | A+R+X peak y≈108–111 |
| fall X | ~4834–4848 | pure X |
| **past gate** | **f5126** | **x=413 y=135** |
| missile pack | f5206 | ~(494,139) |
| tape end | f5372 | still ADAD — **no Wave** |

## Bead board (reeval 2026-08-06)

### Ready pure (do these)

| Id | Title | Notes |
|----|-------|-------|
| **rr-dbu.1** | Bubble→Single natural-entry re-verify | smoke RED from `post_speed_return_to_bubble_pure` |
| **rr-dbu.10** | Double Chamber **blue gate open only** | one-knob; not Wave PLM |

### Serial after gate

```
rr-dbu.10 gate open
    → rr-re9 Wave Super door + PLM (0x0001)
        → rr-l0u continuous --to wave
        → rr-dbu.11 post-Wave Ice recon tracker (split one-hops then)
            → rr-dbu.7 tip ice / rr-dbu.8 K5 / rr-dbu.9 Moat approach
```

### Parallel meta

| Id | Title |
|----|-------|
| **rr-dbu.12** | Record human Speed→Wave→Ice→Moat (full, not aspirational) |

### Closed / folded

| Id | Fate |
|----|------|
| rr-dbu.2–.6 | **closed** — premature Ice serial invent without tape; use `rr-dbu.11` |
| rr-g4i, rr-g1b, rr-hhj | pure GREEN done |
| rr-yzv | residual GREEN; natural-entry recheck is `rr-dbu.1` |

### Done spine (product pure)

`rr-g4i` Speed→Bubble → (`rr-yzv` Bubble→Single) → `rr-g1b` Single→Double → **gate/Wave open**  
Moat spark hop: `rr-hhj` GREEN (pin-only). Continuous tip: **`--to speed` 130388f** dual.

## Probe smoke (2026-08-06 session)

| Segment | Result |
|---------|--------|
| `speed-return-to-bubble` | GREEN |
| `bubble-to-single` from return pin | **GREEN dual 421f** (rr-dbu.1) — fixed `_DOOR_X` shadow |
| `single-to-double` | GREEN 700f (refreshed successor pin) |
| `double-chamber-to-wave` | **RED** gate ~(475,409) → rr-dbu.10 open |
| Moat spark pure | GREEN (rr-hhj) |

### rr-dbu.1 root cause

`k4_wave.py` module constants: Bubble `_DOOR_X = 470` was overwritten by
Double Chamber `_DOOR_X = 920` after K4.10 scaffold. Sill push never shot.
Fixed: `_BSC_DOOR_*` / `_DC_DOOR_*`.

## Non-claims

- No continuous ice/moat/wave tip  
- No STATUS promote past Speed  
- Human tape ≠ pure evidence for gate  
- Stopped multi-agent invent on gate without PLM truth  
