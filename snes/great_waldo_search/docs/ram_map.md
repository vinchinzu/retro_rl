# Great Waldo Search — RAM map

Segment-first discovery notes from Scene1–5 (Flying Carpets →
Land of Waldos / five-scrolls ending).

## Confirmed cursor pair

| Signal | Address | Type | Notes |
|--------|---------|------|-------|
| Cursor X | `0x0215` (533) | `\|u1` | Moves with LEFT/RIGHT; mirrors at `0x0091`, `0x2216` |
| Cursor Y | `0x0217` (535) | `\|u1` | Moves with UP/DOWN; mirrors at `0x0093`, `0x2218` |

Wired into `custom_integrations/GreatWaldoSearch-Snes/data.json`.

Hold **Y** while moving for faster cursor travel (TASVideos / in-game).

## Score / find signals

| Signal | Address | Type | Notes |
|--------|---------|------|-------|
| Score lo | `0x0047` (71) | `\|u1` | LE u16 with `0x0048` |
| Score hi | `0x0048` (72) | `\|u1` | Scene1~2500 … Scene5 ending ~18575–18725 |
| Found-ish | `0x01BD` (445) | `\|u1` | `0 → 2` after scroll +1000; stays 2 on Waldo |

Score bytes are **noisy during click animations**. Prefer a long settle
(≥80–100 frames; Scene5 Waldo ≥200f warm) + stable samples, or visual
SCORE / congratulations / five-scrolls ending.

## Scene / mode

Not isolated. Scenes are **multi-screen horizontal panoramas**. Camera-
related motion observed at `0x00C3` while holding edge directions — useful
as a pan indicator, not a clean scene-id.

## Parental assist (controller 2)

| Input | Effect |
|-------|--------|
| P2 **A** hold (Scene1 start) | Lands **(32, 100)** scroll |
| P2 **A** after Scene1 scroll | Lands ~(206, 100); need RIGHT pan for Waldo |
| P2 **A** after Scene2 scroll | Lands ~(32, 100) while camera pans; **≥500f** then click Waldo |
| P2 **A** Scene3 (good RNG) | ~300f → scroll ~(160,100); ~200f → Waldo ~(198,100) |
| P2 **A** Scene4 (good RNG) | ~500f → scroll ~(34,100); ~500f pan → Waldo ~(196,140) |
| P2 **A** Scene5 (good RNG) | ~300f → scroll ~(32,100); ~500f → Waldo ~(180,60) |
| P2 **B** | Clock freeze / RNG (TASVideos); not used here |

Requires `players=2`. **Do not** hold P2-A while pressing P1-A.

## Scene1 target coords (RAM cursor space)

| Label | (x, y) | Status | Evidence |
|-------|--------|--------|----------|
| `p2a_primary_1000` | (32, 100) | **confirmed** | P2-A ~300f → P1-A → +1000 |
| `waldo_pan_right80` | (36, 28) | **confirmed** | AfterFind → RIGHT+Y×80 → A → ≥2500 |
| Waldo hitbox | x=32–46, y=18–42 | mapped | settle-gated grid after pan |

Clear gate: settled score **≥2500**. Script: `scripts/clear_scene1.py`.

## Scene2 target coords (cave)

| Label | (x, y) | Status | Evidence |
|-------|--------|--------|----------|
| `scene2_scroll_right` | (224, 100) | **confirmed** | Drive/A → +1000 (hitbox ~x≥218, y~90–110) |
| `scene2_waldo_p2a500` | (32, 120) | **confirmed** | AfterFind → P2-A≥500f → A → ≥5125 |
| `scene2_p2a_pre_scroll` | ~(206, 100) | assist only | short P2-A; not scroll click |

Manual LEFT+Y pan does **not** substitute for post-scroll P2-A. Clear gate:
settled score **≥5125** (carry ~2625 + 1000 + 1500; bonus → 5275–5450).
Script: `scripts/clear_scene2.py`.

## Scene3 target coords (Battling Monks)

| Label | (x, y) | Status | Evidence |
|-------|--------|--------|----------|
| `scene3_scroll_p2a300` | (160, 100) | **confirmed** | P2-A~300f → A → +1000 (~6450) |
| `scene3_waldo_p2a200` | (198, 100) | **confirmed** | AfterFind → P2-A~200f → A → ≥7850 |

Favorable layout from `Scene2_Cleared` idle~5f then ~7× A. Clear gate:
settled **≥7850**. Script: `scripts/clear_scene3.py`.

## Scene4 target coords (Unfriendly Giants)

| Label | (x, y) | Status | Evidence |
|-------|--------|--------|----------|
| `scene4_scroll_p2a500` | (34, 100) | **confirmed** | P2-A~500f → A → +1000 (~8950) |
| `scene4_waldo_p2a500` | (196, 140) | **confirmed** | AfterFind → P2-A~500f → A → ≥10450 |
| Waldo hitbox | x≈180–204, y≈132–148 | mapped | settle-gated grid after P2-A pan |

Favorable layout from `Scene3_Cleared` idle~5f then ~7× A. Soft layouts
send P2-A to ~(206,100) instead of left scroll. Clear gate: settled
**≥10450** (often ~10650). Script: `scripts/clear_scene4.py`.

## Scene5 target coords (Land of Waldos — final search)

| Label | (x, y) | Status | Evidence |
|-------|--------|--------|----------|
| `scene5_scroll_p2a300` | (32, 100) | **confirmed** | P2-A~300f → A → often +3000 (~13650) |
| `scene5_waldo_p2a500` | (180, 60) | **confirmed** | AfterFind → P2-A~500f → A → ≥15150 |
| Waldo hitbox | x≈168–208, y≈40–60 | mapped | long-settle grid after P2-A |

Favorable layout from `Scene4_Cleared` idle~5f then ~7× A. Soft layouts
send P2-A to ~(206,100). Clear gate: settled **≥15150** (often
~18575–18725) → yellow five-scrolls ending. Script: `scripts/clear_scene5.py`.

## Workflow (Scene5)

1. `Scene5.state` Land of Waldos+HUD + `players=2` (rebuild from `Scene4_Cleared` if soft)
2. Hold P2-A **300** frames; click `(32, 100)` → +1000/+3000; save `Scene5_AfterFind1000`
3. Hold P2-A **500** frames (do not click during assist)
4. Click `(180, 60)` → total ≥15150; save `Scene5_Cleared` (ending screen)
