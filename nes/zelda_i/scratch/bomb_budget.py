"""How many kills the 20R bomb pack costs, from the ROM drop tables.

Sources: ``scratch/drop_mechanics_rom.md`` (aldonunez ``Z_04.asm``
``DropItemRates`` / ``Types0..3`` / ``SetUpDroppedItem``), the ROM spawn table
via ``overworld.locations.q1_farms``, and the live pre-L1 walks in
``docs/PRE_L1.md``. No emulator: this is the arithmetic that says whether a
corridor can pay for the bombs at all.

    uv run python nes/zelda_i/scratch/bomb_budget.py

Letters are a trap. ``drop_mechanics_rom.md`` follows Baxter (row 1 = B, the
two-5-rupee table); ``overworld/locations.py`` calls that same table ``C``
and the bomb table ``B``. Both agree on contents and rates, so this module
keys on the ROM row and prints both aliases.
"""

from __future__ import annotations

from zelda_i.overworld.locations import Q1_SPAWNS, grid_name
from zelda_i.overworld.shop_p7 import SHOP_P7_PRICE, shop_p7_screens
from zelda_i.overworld.prey import (
    BOMB,
    CLOCK,
    DROP_ROWS,
    FAIRY,
    FIVE_RUPEE,
    HEART,
    RUPEE,
    drop_row,
    random_rupees,
)

BOMB_PACK_PRICE = SHOP_P7_PRICE  # coast shop_p7, not the later 0x4A cave

VALUE = {RUPEE: 1, FIVE_RUPEE: 5}
NAME = {BOMB: "bomb", FIVE_RUPEE: "5R", RUPEE: "1R", CLOCK: "clock", HEART: "heart",
        FAIRY: "fairy"}

# The tables live in ``overworld.prey`` now, because policy reads them too and
# a reporting CLI is the wrong owner for a number the hunt targets on. This
# only adds the two letter aliases, which are a documentation trap rather than
# ROM: ``drop_mechanics_rom.md`` follows Baxter (row 1 = B, the two-5-rupee
# table) and ``overworld/locations.py`` calls that same table C.
_LETTERS = {0: ("A", "A"), 1: ("B", "C"), 2: ("C", "B"), 3: ("D", "D")}
ROWS = {
    row: (rate, table, _LETTERS[row][0], _LETTERS[row][1], types)
    for row, (rate, table, types) in DROP_ROWS.items()
}

# South coast 0x77 → 0x6F. Not the inland 0x68 / 0x4A join.
WALK = shop_p7_screens()
NEAR = (0x5F,)  # north of the shop; the shortfall hunt, not inland 0x48


row_of = drop_row


def per_kill(row: int) -> tuple[float, float]:
    """(expected rupees, expected bomb packs) for one random-table kill."""
    rate, table = ROWS[row][0], ROWS[row][1]
    bombs = sum(1 for item in table if item == BOMB) / len(table)
    return random_rupees(row), (rate / 256) * bombs


def clean_run(kills: int, row: int = 0) -> list[tuple[int, str, float]]:
    """Per-kill outcome of ``kills`` consecutive kills with no Link_BeHarmed.

    ``$0627`` counts every kill and only ``Link_BeHarmed`` zeroes it;
    ``$0050`` caps at 10 and is zeroed by any forced drop. ``$0627 == 16``
    forces a fairy *before* the ``$0050 >= 10`` test, so a clean streak pays
    at kills 10, 26, 36, 46 ... not 10, 20, 30: the fairy spends six kills of
    5-rupee progress.
    """
    e_rupees, _ = per_kill(row)
    out: list[tuple[int, str, float]] = []
    help_count = 0
    total = 0.0
    for world in range(1, kills + 1):
        help_count = min(help_count + 1, 10)
        if world == 16:
            kind, gain = "forced fairy", 0.0
            help_count = 0
        elif help_count >= 10:
            kind, gain = "forced 5R", 5.0
            help_count = 0
        else:
            kind, gain = "random", e_rupees
        total += gain
        out.append((world, kind, total))
    return out


def _supply(screens: tuple[int, ...]) -> tuple[float, float, int, list[str]]:
    rupees = bombs = 0.0
    bodies = 0
    lines = []
    by_screen = {s.screen: s for s in Q1_SPAWNS}
    for screen in screens:
        spawn = by_screen.get(screen)
        if spawn is None:
            lines.append(f"  0x{screen:02x} {grid_name(screen):4} -- no spawn")
            continue
        # A grouped byte is a group index, not an ObjType. Pricing it as a
        # drop row credits rupees the screen does not have.
        if spawn.grouped:
            lines.append(
                f"  0x{screen:02x} {grid_name(screen):4} {spawn.prey:14} "
                "-- grouped spawn, no type census"
            )
            continue
        row = row_of(spawn.monster_id)
        r, b = per_kill(row)
        rupees += r * spawn.count
        bombs += b * spawn.count
        bodies += spawn.count
        letter = f"{ROWS[row][2]}/{ROWS[row][3]}"
        lines.append(
            f"  0x{screen:02x} {grid_name(screen):4} {spawn.prey:14} x{spawn.count} "
            f"row {row} ({letter:3}) {r * spawn.count:5.2f}R"
            + (f"  {b * spawn.count:4.2f} bomb packs" if b else "")
        )
    return rupees, bombs, bodies, lines


def main() -> int:
    print("Per-kill expectation by ROM drop row (Z_04.asm Types0..3 / DropItemRates)\n")
    print(f"{'row':>3} {'letters':8} {'P(drop)':>8} {'R/kill':>8} {'kills/R':>8} {'packs/kill':>11}")
    for row, (rate, _, baxter, loc, _types) in ROWS.items():
        r, b = per_kill(row)
        per_r = f"{1 / r:8.1f}" if r else "     inf"
        print(f"{row:>3} {baxter}/{loc:<6} {rate / 256:8.3f} {r:8.3f} {per_r} {b:11.3f}")

    walk_r, walk_b, walk_n, lines = _supply(WALK)
    print(f"\nOne pass of the 0x77 -> 0x6F coast walk ({walk_n} bodies)")
    print("\n".join(lines))
    print(f"  random drops, whole walk: {walk_r:.1f}R" + (f" + {walk_b:.2f} bomb packs" if walk_b else ""))

    near_r, near_b, near_n, near_lines = _supply(NEAR)
    print(f"\nOne hop off the walk ({near_n} bodies)")
    print("\n".join(near_lines))
    print(f"  random drops: {near_r:.1f}R")

    a, _ = per_kill(0)
    print(f"\nRandom drops only, row 0 octoroks: {BOMB_PACK_PRICE / a:.0f} kills for {BOMB_PACK_PRICE}R")
    b1, _ = per_kill(1)
    print(f"Random drops only, row 1 tektite/leever: {BOMB_PACK_PRICE / b1:.0f} kills for {BOMB_PACK_PRICE}R")

    print("\nOne unbroken streak on row-0 prey (forced drops + random)")
    print(f"{'kill':>5}  {'drop':<13} {'E[rupees]':>10}")
    run = clean_run(50)
    for world, kind, total in run:
        if kind != "random" or world in (1, 5):
            print(f"{world:5}  {kind:<13} {total:10.2f}")
    need = next(w for w, _, t in run if t >= BOMB_PACK_PRICE)
    forced = [w for w, k, _ in run if k == "forced 5R"]
    guaranteed = forced[BOMB_PACK_PRICE // 5 - 1]
    print(f"\n  E[rupees] reaches {BOMB_PACK_PRICE}R at kill {need}")
    print(f"  forced drops alone reach {BOMB_PACK_PRICE}R at kill {guaranteed}")

    print(f"\nWhat the walk can actually pay ({walk_n + near_n} bodies, one pass)")
    total_random = walk_r + near_r
    for streak in (0, 10, 26):
        forced_paid = 5 * sum(1 for w in forced if w <= streak)
        print(
            f"  best unbroken streak {streak:2}: "
            f"{forced_paid:2}R forced + {total_random:.1f}R random "
            f"= {forced_paid + total_random:.1f}R"
            + ("  <- pays for the pack" if forced_paid + total_random >= BOMB_PACK_PRICE else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
