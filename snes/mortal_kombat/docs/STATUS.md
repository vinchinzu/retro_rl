# Status: Mortal Kombat (SNES)

## Program gate

| Field | Value |
|-------|-------|
| Maturity | M3 |
| Best verified result | Model-free power-on tape through Match 7 to the Endurance 1 transition |
| Runtime class | Bronze |
| Intervention class | Clean |

Isolated save-state wins are not this result. The tape is not a reactive
policy and not credits.

## Verified

- Power-on boot reaches Liu Kang vs Johnny Cage at 161/161, timer 153.
- `Fight_LiuKang` is vs Sub-Zero, not Cage.
- One model-free input tape, cold-booted 5/5 on the documented HUD and pose
  signatures, reaches the next transition at these frames: Match 2 at 7,863,
  Match 3 at 12,918, Match 4 at 18,077, Match 5 at 25,164, Match 6 at 29,783,
  Match 7 at 36,752, Endurance 1 at 41,503. Runtime loads no model.
- Live opponents on that tape: Johnny Cage, Sonya, Sub-Zero, Raiden (2-1),
  Kano (2-0), Johnny Cage again (2-1), Liu Kang mirror (2-0). The character
  byte left on a pin is the previous fight's HUD.
- Endurance 1 from the Fight 7 pin is courtyard Kano. There is no Endurance 1
  tape. Throne-room `Match5_LiuKang` is a different Kano. `Endurance1_LiuKang`
  is Sub-Zero.
- v3 specialist zips finished an overnight 4M train (2026-08-23). N=5
  save-state rates are noisy and are not a continuous clear.
- Health max is 161. Liu Kang id is 3. Win means two rounds and more rounds
  than the opponent. Addresses are in `docs/ram_map.md`.

## Not done

- A reactive policy from power-on.
- Endurance, Goro, Shang Tsung, or credits.
