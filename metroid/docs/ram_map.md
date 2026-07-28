# RAM map — Metroid (NES)

Sources: Data Crystal, Dirty McDingus disassembly, live fceumm probes
(2026-07-27).

## System RAM (`env.get_ram()`, 2 KiB)

| Addr | Name | Notes |
|------|------|-------|
| `$1D` | Engine mode | 0=game, 1=title/password |
| `$1E` | Game mode | 3=playing, 5=paused, 8=intro settle |
| `$31` | Paused | 1=yes |
| `$4D` | Samus direction | 0=right, 1=left |
| `$4F` | Map Y | |
| `$50` | Map X | Start=3, morph room=1 (Y=14) |
| `$51/$52` | Screen X/Y | Often stale; prefer object coords |
| `$56` | In door | 0=no |
| `$74` | Area | `$10` Brinstar when set; may be 0 early |
| `$106/$107` | Health BCD | fixed-point ###.# |
| `$109` | Item pause | not reliable for morph detect |
| `$10E` | Missiles enabled | 1 after first missiles |
| `$300` | Samus status | object status |
| `$30D` | Samus room Y | |
| `$30E` | Samus room X | |

## Cartridge WRAM (`env.data.memory.extract`)

| Addr | Name | Notes |
|------|------|-------|
| `$6877` | Energy tanks | |
| `$6878` | Equipment | bit4=Maru Mari, bit0=Bombs, … |
| `$6879` | Missiles | current |
| `$687A` | Missile capacity | |

Equipment bits: bombs `$01`, hi-jump `$02`, long `$04`, screw `$08`, morph
`$10`, varia `$20`, wave `$40`, ice `$80`.

## Readiness / stop predicates

`is_level1_ready`: engine=game, mode=playing, not paused, map in range, health
initialized.

`is_morph_obtained`: equipment `$6878 & 0x10`.

`is_missiles_obtained`: missile capacity `$687A > 0` (first expansion).
