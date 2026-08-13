# Spore Spawn no-assist policy — residual

**Status:** windows 1–3 are 2/2 from the human enter pin; late parks are 1-hit;
60 HP left with 1 missile, Samus at 11 energy. Full 22k still RED.
**Pin:** `tasks/full_start_v1_anchors/f015374_enter_0x9DC7_0x9DC7.state`
**760 HP pin:** `scratch/spore_hp760.state` (after window 1)
**Probe:** `uv run python scripts/probe/spore_spawn_combat.py window --windows 7`
**Report:** `scratch/spore_window_v28.json`
**Human tape:** `full_start_v1` s6 hop 11, room `0x9DC7`, morph seat `(21, 697)`.

Do **not** overwrite the assisted continuous bounce in `routes/kpdr/spore_spawn.py`
until this policy dual-greens from the human enter pin with zero resource writes.
Do **not** start another 22k: after window 7 Samus has 11 energy (one contact
kills) and the last missile did not spend.

## What works (verified this pass)

- **Seat** and **window 1 2/2** unchanged from v13.
- **`--windows N`** on the probe: seat → wait-closed → next open → fire.
  Logs each spend as `{xy, pose, eye, hp, ms}`.
- **Live-eye fire band** (the v13 `x=180–195` band was window-1-only):

  | Window | Park | Shots | Result |
  |--------|------|-------|--------|
  | 1 | (185, 586) high right | (197, 614), (194, 605) | 960→760 2/2 |
  | 2 | (142, 604) mid | (154, 631), (153, 611) | 760→560 2/2 |
  | 3 | (185, 586) high right | same as w1 | 560→360 2/2 |
  | 4 | (96, 666) floor | (95, 680) | 360→260 1/1 |
  | 5 | (191, 615) mid-right | (203, 642) | 260→160 1/1 |
  | 6 | (96, 666) floor | (93, 680) | 160→60 1/1 |
  | 7 | (191, 615) mid-right | none (peaked at 144, 614) | 60 HP, 1 missile |

- **Hit rule (updated):** missiles go straight up. Fire when
  `-4 ≤ samus_x − eye_x ≤ 12` (asymmetric: +11 after the flinch hits;
  −11 at x=174 misses) **and** `eye_y ≤ samus_y ≤ eye_y + 30` (36px below
  is a stalk miss). No LEFT/RIGHT on the spend frame. Tap X, ≥10f apart.
- **Floor park** (eye y≥650): short hop, not dash-bounce. Dash-bounce
  peaks *above* (96, 666) and `fire_y_max=624` used to reject the legal
  y=680–692 band.
- **760 HP pin** saved after window 1 2/2.

## What fails

1. **Late parks die after the first hit.** 10f missile cooldown is longer
   than the floor/mid-right park survives. Second tap is into a moving eye
   (96,666 → 119,637 in 14f). Do not chase 2/2 on windows 4+ — take the
   1-hit and sit.
2. **Last 60 HP unspent.** Window 7 is the window-5 park again, 1 missile
   left. Bounce peaked at **(144, 614)** vs eye **(191, 615)** — right
   height, 47px left. Health 35→11 on that approach (contact). Next
   contact kills.
3. **Farm still unsolved** (same as last residual). Do not jump-farm.
   `$1997` / `$F337` layout is still correct and still unseen live.
4. 9 of 10 missile hits landed. The 10th is a **contact-free close to
   (191, 615) at 11 energy**, not a 22k farm climb.

## Next actions (do not start another 22k first)

1. From `scratch/spore_hp760.state` or a new 60-HP pin, land the last
   missile on the (191, 615) park **without contact**. Window 5's spend
   `(203, 642)` vs `(191, 615)` is the measured hit. Window 7 drifted
   left — close with less dash, or wait until the eye is parked at 191
   before leaving the seat.
2. Optional: save a 60-HP / 1-missile pin at the post-window-6 seat
   (`(21, 697)`, 35 energy) so the last shot is a one-window probe.
3. Farm only if the last shot is impossible at 11 energy: stand, shoot a
   `$DE7A` spore, stop at `$1997==0xF337` or a missile/energy increment.
   Energy drops are useful now (11 HP).
4. `strategy --max-frames 22000` only after a kill from the enter pin
   with `energy_writes 0`, `missile_writes 0`, not dead. Then swap
   `routes/kpdr/spore_spawn.py`.

```bash
# From game dir. 3-window check (should still be 2/2 + 2/2 + 2/2):
uv run python scripts/probe/spore_spawn_combat.py window --windows 3 \
  --report scratch/spore_window.json

# Later parks from the 760 pin:
uv run python scripts/probe/spore_spawn_combat.py window \
  --state scratch/spore_hp760.state --windows 2
```
