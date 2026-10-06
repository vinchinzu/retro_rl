# ALTTP plan

Future work only. C1 evidence is in [STATUS.md](STATUS.md).

The Clean Sanctuary-to-Kakariko extension and replay commands are in
[tasks/residual.md](tasks/residual.md). C2 remains the next maturity gate.

1. Gather early Kakariko gear from the Clean village arrival (`rr-7cdd`):
   - Kakariko Well (room `0x2F`): 3 chests (60 rupees) + 1 chest (3 bombs),
     plus bombable wall to Piece of Heart #1.
   - Blind's Hideout (room `0x11D`): basement chests (80 rupees), plus
     bombable wall to Piece of Heart #2.
   - Tavern back door (room `0x103`, screen `0x21`): Bottle #1 chest.
   - Sick Kid's House (room `0x102`, screen `0x21`): Bug-Catching Net (show bottle).
   - Bottle Merchant (screen `0x18`): Bottle #2 (buy for 100 rupees).
2. With 3+ bombs, 4+ hearts, 2 bottles, and bug net in hand, proceed to
   Eastern Palace approach and lobby transition.
3. Finish Eastern Palace lobby transition (`rr-eog9.6`), compose the CLI
   and replay (`rr-eog9.7`), and verify the complete natural-entry route (`.8`).
4. Repair the fresh no-assist controller rescue regression (`rr-ihr8`).
   The C1 tape still replays exactly, but the changed courtyard arrival
   leads the current controller to stall in room `0x71` before Zelda.
   Preserve the published C1 proof while repairing that predecessor.
5. Reduce remaining Clean sewer damage only from replay-derived predecessors.
   The accepted C1 escort takes 14HP in `0x22`, `0x21`, and `0x11`, but survives
   with no refill assistance. Forecast the full pickup/door/block handoff,
   evaluate offsets, then require CRC forward replay. Keep the exact mantle
   walker and carrier-targeted rat-key fight.
6. Keep dungeon entry pins separate from continuous claims. See
   [PORT_PLAN.md](PORT_PLAN.md) for C2–C15. Alternate `0x55` key/shutter work
   remains practice; gauntlet and romhack packages remain outside this route.

Use the Zelda I process: inspect a real predecessor and hit trail, change
one transit or required-item combat leg, forecast its complete policy, test
multiple offsets and successor acceptance, then replay committed inputs from
power-on without loads or writes. The living handoff is
[tasks/residual.md](tasks/residual.md).
