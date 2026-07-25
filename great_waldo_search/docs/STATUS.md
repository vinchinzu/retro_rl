# Great Waldo Search — Status

**Approach:** save-state + scene-segment scripts first; continuous
title-to-credits later. Retries and mid-scene `.state` files are expected.

| Item | State |
|------|--------|
| Integration `GreatWaldoSearch-Snes` | done |
| ROM from shared zip | done |
| Dev save past title | `Scene1.state` (Flying Carpets + HUD) |
| RAM cursor X/Y | confirmed `0x0215` / `0x0217` |
| Score u16 + found-ish | `0x0047`/`0x0048`, `0x01BD` (after +1000) |
| Segment policy | `scene_policy` → shared `snes_oneshot.cursor` |
| P2-A assist seek | Scene1: (32,100); Scene2 Waldo: P2-A≥500f |
| Scene1 clear | **done** (scroll+Waldo; `Scene1_Cleared.state`) |
| Scene2 cave | **done** (scroll+Waldo; `Scene2_Cleared.state`) |
| Scene3 Battling Monks | **done** (`clear_scene3.py`; `Scene3_Cleared.state`) |
| Scene4 Unfriendly Giants | **done** (`clear_scene4.py`; `Scene4_Cleared.state`) |
| Scene5 Land of Waldos | **done** (`clear_scene5.py`; `Scene5_Cleared.state`) |
| Ending (five scrolls) | **reached** from Scene5 Waldo clear |
| Full game / continuous run | later |

## Current milestone

### Scene1 Flying Carpets (from `Scene1.state`, `players=2`)

1. **Scroll:** P2-A ~300f → cursor `(32, 100)` → P1-A → **+1000**,
   `0x01BD=2`. Save `Scene1_AfterFind1000.state`.
2. **Pan:** drive to right edge, hold **RIGHT+Y** ~**80** frames.
3. **Waldo:** P1-A at **`(36, 28)`** → settled total **≥2500**.

### Scene2 Underground Hunters / cave (from `Scene2.state`)

`Scene2.state` must be cave+HUD (not congrats). Rebuild from
`Scene1_Cleared` with ~8× (A hold 6 + idle 60) if soft.

1. **Scroll:** drive/click **`(224, 100)`** → **+1000** (carry ~2625 →
   3625), `0x01BD=2`. Save `Scene2_AfterFind1000.state`.
2. **Assist pan:** hold **P2-A ≥500f** (cursor ~(32,100); camera pans).
   Manual LEFT+Y alone does **not** open the Waldo window.
3. **Waldo:** P1-A at **`(32, 120)`** → settled total **≥5125** (often
   5275–5450). Save `Scene2_Cleared.state`. Gate on settle, not mid-anim.

### Scene3 Battling Monks (from `Scene3.state`, `players=2`)

`Scene3.state` must be Monks+HUD with a favorable layout RNG. Rebuild from
`Scene2_Cleared`: idle ~5f, then ~7× (A hold 6 + idle 60). Soft layouts
make P2-A seek clocks instead of scroll.

1. **Scroll:** P2-A ~300f → ~(160, 100) → P1-A → **+1000** (~6450).
2. **Waldo:** P2-A ~200f → ~(206, 100) → P1-A at ~(198, 100) → settled
   **≥7850** (often ~8300–8350). Save `Scene3_Cleared.state`.

### Scene4 Unfriendly Giants (from `Scene4.state`, `players=2`)

`Scene4.state` must be Giants+HUD with a favorable layout RNG. Rebuild from
`Scene3_Cleared`: idle ~5f, then ~7× (A hold 6 + idle 60). Soft layouts
make P2-A seek right (~206,100) instead of left scroll.

1. **Scroll:** P2-A ~500f → ~(34, 100) → P1-A → **+1000** (~8950).
2. **Waldo:** P2-A ~500f (camera pans) → P1-A at ~(196, 140) → settled
   **≥10450** (often ~10650). Save `Scene4_Cleared.state`.

### Scene5 Land of Waldos (from `Scene5.state`, `players=2`) — final search

`Scene5.state` must be Land of Waldos + HUD with favorable layout RNG.
Rebuild from `Scene4_Cleared`: idle ~5f, then ~7× (A hold 6 + idle 60).
Soft layouts send P2-A to ~(206,100) instead of left scroll.

1. **Scroll:** P2-A ~300f → ~(32, 100) → P1-A → often **+3000**
   (~13650), `0x01BD=2`. Save `Scene5_AfterFind1000.state`.
2. **Waldo:** P2-A ~500f → P1-A at ~(180, 60) → settled **≥15150**
   (often ~18575–18725). Save `Scene5_Cleared.state`.
3. **Ending:** yellow “HOORAY! … FIVE SCROLLS” screen. Further A
   advances into post-game / another-challenge flow (score resets).

Use a **longer settle** (≥200f warm) for the Waldo bonus animation.

## Commands

```bash
uv run python -m snes_oneshot.setup_all_roms great_waldo_search

SDL_VIDEODRIVER=dummy uv run python \
  great_waldo_search/scripts/clear_scene1.py

SDL_VIDEODRIVER=dummy uv run python \
  great_waldo_search/scripts/clear_scene2.py

SDL_VIDEODRIVER=dummy uv run python \
  great_waldo_search/scripts/clear_scene3.py

SDL_VIDEODRIVER=dummy uv run python \
  great_waldo_search/scripts/clear_scene4.py

SDL_VIDEODRIVER=dummy uv run python \
  great_waldo_search/scripts/clear_scene5.py

uv run --frozen pytest great_waldo_search/tests snes_oneshot/tests/test_cursor.py -q
```

## Blockers / open

- Scene-complete / scene-id bytes not isolated (`0x00C3` moves with camera).
- Score RAM still noisy mid-animation; Scene5 Waldo needs longer settle.
- Scene3–5 layout RNG can soft-lock assist seeks — rebuild state if needed.
- Next: optional continuous title → five-scrolls ending without mid-run saves.
