## Residual — rr-8g2u power-on `--to phantoon` dual

**Status:** Dual GREEN. STATUS living tip (`rr-b926`, 2026-08-25). Pin
compose **13823f** ×2. Power-on `--to phantoon` **195336f** ×2 exact match
(video + `--no-video`). Default CLI is `phantoon`. `--to ws` still ends
`0xCA08`. Next rung: Gravity `rr-kw8t`.

**Pin in:** `scratch/post_ws_poweron.state` (`0xCA08` ~(57,139) p1 gs=8)
**Pin out:** `scratch/post_phantoon_leave.state` (`0xCC6F` ~(1240,139) p10
gs=8, `$D82B` bit 0). Do not clobber `post_phantoon_poweron.state` /
`post_phantoon_defeated.state`.

### Power-on video (2026-08-24)

`recordings/phantoon.mp4` — 511 MiB, 3255.6s (54m16s) @60fps, 512×480
h264 + AAC, `--video-start power_on` (Ceres on tape). Report:
`scratch/phantoon.json` (not `recordings/phantoon.json`).

| | |
|-|-|
| CLI | **GREEN** `phantoon_defeated` **195336f** ×2 exact |
| Leave | `0xCC6F` (1240,139) p10 gs=8 dt=0 health 299 |
| Beams / items | `0x1007` / `0x3105` |
| Integrity | loads=0, prog=0, deaths=0, video frames match |
| Tail | WO→WS @175967 `0xCA08`; Entrance @176402 `0xCAF6`; Main @177636 `0xCC6F`; Basement→room @178300 `0xCD13`; fight @195000 `0xCD13`; loot-exit @195168 `0xCC6F` |

Glance (hop_glance PHANTOON_LEAVE): **no misses** on the mapped still.
Live hop `after` peeked `$D82B` bit 0. Serialized
`final_state.boss_bits[3]=254` is low-WRAM open-bus, not a bank-7E peek.

### Already green (do not re-prove)

| Layer | Dual | Leave |
|-------|-----:|-------|
| Entrance → Main | **403f** ×2 | `0xCAF6` (1063,907) p9 gs=8 |
| Main → basement | **1208f** ×2 | `0xCC6F` (657,92) p24 gs=8 |
| Basement → room | **718f** ×2 | `0xCD13` (39,124) p81 gs=8 |
| Doppler fight | **12118f** ×2 | `0xCD13` (37,187) p1 HP 0 + bit 0 |
| Loot + left-door | **337f** ×2 | `0xCC6F` (1240,139) p10 gs=8 |
| Fight+leave compose | **12455f** ×2 | same basement |
| Pin compose `ws-to-phantoon` | **13823f** ×2 | `0xCC6F` (1240,139) p10 gs=8 boss 1 HP 299 |
| Power-on `--to phantoon` | **195336f** ×2 exact | same basement |

Charge-only / charge+missiles / Ice-on X-Factor stay research.

### This window

- **Settle=5 (red):** pin compose fight `halt_miss` at f=3958 in `0xCD13`.
- **Settle=0 (green, knob):** `zero_settle_segments` += `"ws-to-phantoon"`.
  Dual **13823f** ×2 from `post_ws_poweron.state`.
  `scratch/ws_to_phantoon_dual.json`.
- **Power-on video GREEN** 195336f, then `--no-video` twin **195336f**
  (exact). hop_glance misses []. `scratch/phantoon_dual.json`.

### Next action

- Closed into STATUS living tip (`rr-b926`). Gravity is `rr-kw8t`.
- Tape: `recordings/phantoon.mp4` (Ceres → post-Phantoon).

### Non-claims (compose window; STATUS is `rr-b926`)

- Did not rewrite Entrance / Main / Basement / fight bodies
- Did not clobber `post_phantoon_leave.state` /
  `post_phantoon_poweron.state` / `post_phantoon_defeated.state`
