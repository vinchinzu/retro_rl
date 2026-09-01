# RNG — Super Metroid `$05E5`

Power-on WRAM does **not** seed the PRNG. Boot clears bank `$7E`, then writes
`$05E5 = $0061`. Same button tape → same seed at first Ceres / Zebes control.
Skip the intro by pinning first movement and poking `$05E5`.

Helpers: `super_metroid.ram.rng1` / `rng1_advance` / `set_rng`. Probe:
`scripts/tools/probe_rng.py`. Address table: [ram_map.md](ram_map.md).

## When it rolls

`$80:8111` (`GenerateRandomNumber`) runs **once at the top of the main game
loop**, before the game-state handler — not in NMI, not every PPU frame:

```
$82:894F  JSL $808111   ; roll $05E5
$82:8963  LDA $0998     ; then dispatch game state
```

Nested `Wait for NMI` during fades / loads / cinematics can burn emulator
frames with **no** roll. Enemy AI and drops add extra `JSL $808111` calls on
the same seed. Pause still hits `$82:894F` (the seed ticks); enemy extra-rolls
freeze. Lava/acid rooms XBA the seed every frame. Beetoms / sidehoppers
**overwrite** `$05E5` on room entry (`$0017` / `$0025`).

Formula (PJBoy `$80:8111`; ~`5·r + $111` with an 8-bit carry):

```python
from super_metroid.ram import rng1, rng1_advance, set_rng

rng1(0x0061)            # → 0x02F6
rng1_advance(0x5705, 10)
set_rng(env, 0x5705)    # future rolls only
```

Main cycle after reset is 2280 values ([wiki](https://wiki.supermetroid.run/Random_Number_Generator)).

## Live power-on (TAS mash, this core)

`scripts/tools/probe_rng.py` START/A mash, snes9x fill `$55`:

| Event | Emulator frame | `$05E5` | `$0998` |
|-------|----------------|---------|---------|
| `env.reset()` | 0 | `$5555` (fill, not random) | garbage |
| Boot writes seed | **260** | **`$0061`** | 0 |
| First main-loop roll | **269** | `$0061 → $02F6` | 1 opening |
| Intro cinematic | 505 | `$0E7E` | 30 |
| Ceres fade-in | 8449 | `$B766` | 7, room `$DF45` |
| **First Ceres control** | **8479** | **`$5705`** | **8** |

Zero extra rolls through first control (empty elevator). `$5705` is `$0061`
after 7712 rolls (also after 3152 — already in the 2280-cycle). Product
legacy `_boot_spans` first control is the **10860**-frame open-loop, same
seed path, more title/intro ticks.

Zebes first movement: first `gs=8` in Landing Site `$91F8` after cinematic
gs `$20–$22`. Same main-loop tick through Ceres boom / Zebes load.

## Skip boot

Pin at first `gs=8` (Ceres `$DF45` or Landing `$91F8`). Do **not** re-power-on
for “new WRAM.”

```python
from super_metroid.ram import rng1_advance, set_rng

# k extra title/intro frames (or k empty-room idles)
set_rng(env, rng1_advance(0x5705, k))  # TAS-mash Ceres-control seed
```

Ceres elevator and Landing Site have no enemies, so a poke matches a
different boot wait. Extra idle on those rooms is ~1 roll per main-loop
frame (not per PPU frame during lag/fade).

Poke **after** enemies spawned does not re-init them. Enemy rooms: poke then
re-enter the door (or `boot_idle_frames` doorway bootstrap).

## Probe

```bash
# Offline: next k seeds (no ROM)
uv run python snes/super_metroid/scripts/tools/probe_rng.py --advance 5 --seed 0x5705

# Live: power-on mash → first gs=8 Ceres Elevator
uv run python snes/super_metroid/scripts/tools/probe_rng.py
uv run python snes/super_metroid/scripts/tools/probe_rng.py \
  --json snes/super_metroid/scratch/rng_boot.json
```
