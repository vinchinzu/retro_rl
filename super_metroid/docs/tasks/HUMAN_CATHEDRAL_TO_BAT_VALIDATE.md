# Validation — human `cathedral_to_bat_human` (2026-08-03)

Source: `super_metroid/tasks/cathedral_to_bat_human.json`  
End state: `tasks/cathedral_to_bat_human_end.state` / `scratch/cathedral_to_bat_human_end.state`  
Start: `scratch/post_cathedral_entrance_to_cathedral_pure.state`  
Loadout: Morph, Bombs, Varia, Hi-Jump — **no beams** (no Ice / Wave / Charge).

## Verdict

| Segment | Room(s) | Human | Pure controller | Guide was… |
|---------|---------|-------|-----------------|------------|
| Cathedral → Rising Tide | `0xA788` → `0xAFA3` | **Easy** (~1082f), lower-right Super door `(748,395)` | GREEN (~1194f) | Roughly right door band; path OK |
| Rising Tide → Bubble | `0xAFA3` → `0xACB3` | **Easy** (~1827f), right exit `x≈1262` | GREEN (~5350f, slower) | **Door X wrong** (guide had ~1008; room is 5 screens → exit ~1262) |
| Bubble → Bat | `0xACB3` → `0xB07A` | **Not completed** | RED (known) | **Wrong product emphasis** for this recording |

**Last ordinary room in the recording is Bubble Mountain (`0xACB3`), not Bat Cave.**  
End pin: `(70, 427)` pose 2 — **save-door lip** (left mid), not top Super door.

## Room sequence (trace)

```text
0xA788 Cathedral          f=0..1081     exit (748, 395)
0xAFA3 Rising Tide        f=1082..2908  exit (1262, 139)
0xACB3 Bubble Mountain    f=2909..3338
0xB0DD Bubble Save Room   f=3339..3544  ← side trip (wrong door)
0xACB3 Bubble Mountain    f=3545..9589  end (70, 427)
```

`0xB0DD` = **Bubble Mountain Save Room** (graph connection `acb3_2 → b0dd_1`).  
Already listed as a **wrong-door trap** at left y≈368 in `SM-K4.4-PURE-residual.md`. Not progress toward Speed/Bat.

## Freeze ray / Ice

- Recording loadout has **beams = 0** (no Ice).
- Freeze-enemy platforms are **not available** on the first Bubble visit (KPDR collects Ice **after** Speed).
- Any “land on frozen enemy” climb is a **different / later** strat — not this segment’s contract.
- MapRando first Bubble top path is still **Hi-Jump + walljump** cavity climb (not Ice).

## Bubble metrics (human, clean coords)

| Metric | Value |
|--------|-------|
| min_y | ~220 (left column peaks) |
| max_x | 363 |
| Phase C hits (`x≥300`, `y∈[200,430]`) | few / marginal (best y≈420) |
| Phase D / top-right door band | **0** |
| Ordinary `0xB07A` | **never** |
| Save-lip band frames | high (~1221) |
| Floor time | high (~1811) including morph |

So the human validated **Cathedral + Rising Tide as easy, correct rooms**, then spent the remaining ~6.5k frames **inside Bubble** (plus a Save detour) without solving the top Super door. That matches the pure residual: lower/mid reachable, **y≈350→top** is the hard gap.

## What was wrong in our guided path

1. **Rising Tide exit waypoint** used ~`(1008,112)` — short of the real right door (~`1262,139`).
2. **Bubble guide** jumped straight to “top Super → Bat” walljump spine; it did **not** mark Save `0xB0DD` as a trap, so the line could read as “go left mid” toward Save.
3. Recording name/route `cathedral-to-bat` oversold the tip: human evidence only closed **through Bubble entry**, not Bat.

## What is still correct

- First Bubble visit **is** Cathedral climb (not Frog Speedway) — human took CATH path.
- Target for KPDR Speed remains Bubble top-right → **Bat Cave `0xB07A`** → Speed Hall (graph `acb3_7 → b07a`).
- Pure CATH-03 / CATH-04 still GREEN; human does not overturn that.

## Follow-ups

1. Retune `guide_paths.py` from this trace (Cathedral ridge→lower door, Rising Tide full cross, Bubble: avoid Save, mid lip → cavity, not left-high missile door).
2. Next human record: start `--from bubble` with trap callouts; stop only on ordinary `0xB07A`.
3. Pure work stays R14: raise right contact into shelf/air, then top — **not** Ice freeze platforms.
