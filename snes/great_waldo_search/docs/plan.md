# Great Waldo Search — Plan

Ladder #1 (tier 0). Pipeline proof: boot, menus, cursor select, scene advance.

## Working rule

Not gated on uninterrupted evaluation early. Prefer:

1. Dev `.state` files under `custom_integrations/GreatWaldoSearch-Snes/`
2. Scene-segment policies (title → Scene1; cursor → known Waldo/Woof/etc.)
3. Retries from save on miss / wrong click
4. Chain scenes once one search clear is reliable
5. Continuous title-to-credits only as later hardening

## Milestones

1. **Scaffold** — done (ROM, integration, docs, cursor policy stub)
2. **Scene1.state** — done (title START @~560 → NORMAL → START into search)
3. **RAM** — cursor X/Y `0x0215`/`0x0217`; score `0x0047`/`0x0048`;
   found-ish `0x01BD`; scene id still open
4. **Segment clear Scene1** — done: scroll (32,100) + RIGHT×80 + Waldo (36,28)
5. **Segment clear Scene2** — done: scroll (224,100) + P2-A×500 + Waldo (32,120)
6. **Scene chain** — Scene5 Land of Waldos **done** (`clear_scene5.py`;
   five-scrolls ending from Waldo clear).
7. **Continuous run** — **done** (`scripts/record_full_run.py` →
   `recordings/great_waldo_search_full_credits.mp4`)

## Segment policy (Scene1)

```text
# players=2
hold P2-A until cursor settles          # lands (32,100) scroll
P1-A → settle score >= 1000, 0x01BD=2
# do NOT re-hold P2-A
drive to right edge; RIGHT+Y ~80 frames  # panorama pan
P1-A at (36,28) → settle score >= 2500
# success: congrats / next scene (cave)
```

## Segment policy (Scene2 cave)

```text
# players=2, Scene2.state = cave+HUD (not congrats)
drive/click (224,100) → settle score >= 3625, 0x01BD=2
hold P2-A >= 500 frames                 # required; LEFT pan alone fails
P1-A at (32,120) → settle score >= 5125
# success: congrats / next scene (monks)
```

Open-loop coordinate tables are acceptable for this tier-0 game. Shared cursor
math lives in `retro_harness.cursor`. Scripts: `clear_scene1.py`,
`clear_scene2.py`.
