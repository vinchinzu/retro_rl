# MK1 SNES RAM map

Read-only WRAM via `get_ram()`. HUD bytes: `data.json` + GameHacking.org USA
(CRC32 `DEF42945`). Pose bytes were re-probed on `Fight_LiuKang` (2026-08-23):
`0x00DA` / `0x0174` are animation noise, not screen X.

| Name | Addr | Hex | Notes |
|------|-----:|-----|-------|
| game_mode | 34 | `0x0022` | Title/fight codes use 07/08/09 |
| match_counter | 10 | `0x000A` | 0 = M1 … 11 = Shang |
| p2_character | 36 | `0x0024` | 0–6 roster, 7 Goro, 8 Shang Tsung |
| timer | 290 | `0x0122` | Round timer, max ~154 |
| continue_timer | 999 | `0x03E7` | Continue screen |
| p2_rounds | 1207 | `0x04B7` | Best-of-3 0–2. Reused on VS / timer=1 cheats; ignore >2 |
| p1_health | 1209 | `0x04B9` | Max **161** |
| p2_health | 1211 | `0x04BB` | Max **161** |
| p1_rounds | 6510 | `0x196E` | |
| p1_character | 6514 | `0x1972` | Liu Kang = **3** |
| p1_x / p1_y (v3 obs) | 218 / 219 | `0x00DA` / `0x00DB` | Object-stride guess. **Animation noise** — overnight v3 zips were trained on these bytes; do not retarget obs without retraining |
| p2_x / p2_y (v3 obs) | 372 / 373 | `0x0174` / `0x0175` | Same, P2. P2 Y sticks ~24 |
| p1_x / p1_y (pose) | 6502 / 6504 | `0x1966` / `0x1968` | Live screen pose. Start ~68/144. Used by scripted policy |
| p2_x / p2_y (pose) | 783 / 815 | `0x030F` / `0x032F` | P2 X starts ~180 then walks in. Y standing ~144 |
| p1_state | 274 | `0x0112` | Object +`0x38`; often 0 |
| p2_state | 430 | `0x01AE` | Same field on P2; noisy |

High WRAM sprite tables `0x7688` / `0x7788` (Hacc) are optional if `get_ram()`
is long enough — not required for v3 obs.

Hitboxes in `ram.py` are **derived** AABBs from X/Y + facing (hurt 28×80
stand / 28×48 crouch; attack 40×24 when state ≠ 0). Policies see overlap /
in-range bits plus raw state bytes — not pixels.

v3 observation is 20 floats (`snapshot_features`). Incompatible with v1
(9-dim) and v2 (13-dim) MLP zips.

Round / match notes:

- Match 1 keeps `match_counter=0`. Timer-down after round 1 is **not** char select.
- Natural Liu Kang arcade (this power-on tape): Match 1 Johnny Cage,
  Match 2 Sonya (id 6), Match 3 Sub-Zero (id 5), Match 4 Raiden (id 2),
  Match 5 Kano (id 1), Match 6 Johnny Cage again (id 0), Match 7 Liu Kang
  mirror (id 3). M1–M6 are not a unique remaining-roster; Scorpion never
  appeared. `Fight_LiuKang` timeout-KO (vs Sub-Zero) still loads Scorpion.
  Pin HUD is leftover from the previous win (Scorpion at Fight 1, Sonya
  at Fight 2, Sub-Zero at Fight 3, Raiden at Fight 4, Kano at Fight 5,
  Cage at Fight 6), not the fighter you actually play. First fight-ready
  after VS can be a black fade with `p2_character=0`; identify on a
  visible frame.
- For scripted replay scoring, count health transitions `>0 → 0` for each
  fighter. These settle before the delayed HUD round bytes and avoid the noisy
  P2 byte producing a false loss at the final KO.
- Intra-match KO / "FIGHT!" intro auto-advance; START pauses.
- After a match win: ~900 frames of no START (KO → FINISH HIM → pose → VS), then
  pulse START. `match_counter` increments on the VS / load into the next fight.
- Timeout-KO probe: set `health=161`, `enemy_health=1`, `timer=1` (HP=0 does
  not trip SNES KO). Do not hold timer at 1 every frame.

Character select cursor (hold ~8f, then release): `Cage --DOWN--> Kano
--DOWN--> Raiden --RIGHT--> Liu Kang`. RIGHT from Cage drops to the bottom
row (Sub-Zero / Sonya). Confirm with Y or A. `p1_character` tracks the cursor.
