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

BOMB_PACK_PRICE = 20  # overworld.bomb_shop.BOMB_SHOP_PRICE

BOMB, FIVE_RUPEE, RUPEE, CLOCK, HEART, FAIRY = 0x00, 0x0F, 0x18, 0x21, 0x22, 0x23
VALUE = {RUPEE: 1, FIVE_RUPEE: 5}
NAME = {BOMB: "bomb", FIVE_RUPEE: "5R", RUPEE: "1R", CLOCK: "clock", HEART: "heart",
        FAIRY: "fairy"}

# row -> (DropItemRates byte, Types<row> 10-column table, Baxter letter,
#         locations.py letter, ObjTypes that select the row)
ROWS = {
    0: (0x50, (HEART, RUPEE, HEART, RUPEE, FAIRY, RUPEE, HEART, HEART, RUPEE, RUPEE),
        "A", "A", (0x07, 0x08, 0x0E, 0x04, 0x0F)),
    1: (0x98, (FIVE_RUPEE, RUPEE, HEART, RUPEE, FIVE_RUPEE, HEART, CLOCK, RUPEE, RUPEE, RUPEE),
        "B", "C", (0x0D, 0x10, 0x21, 0x22, 0x13, 0x28, 0x2A)),
    2: (0x68, (HEART, BOMB, RUPEE, CLOCK, RUPEE, HEART, BOMB, RUPEE, BOMB, HEART),
        "C", "B", (0x09, 0x0A, 0x03, 0x01)),
    3: (0x68, (HEART, HEART, FAIRY, RUPEE, HEART, FAIRY, HEART, HEART, HEART, RUPEE),
        "D", "D", ()),
}
_ROW_BY_TYPE = {t: row for row, (_, _, _, _, types) in ROWS.items() for t in types}

# A "grouped" spawn byte carries a group index, not an ObjType: 0x49 reads
# ``group_28`` and 0x28 is also Rope (row 1), which would credit the screen
# 5.3R it does not have. These screens come from the live census instead
# (``scratch/contact1.json`` / ``kill_streak_reach6.json``), as {type: count}.
MEASURED: dict[int, dict[int, int]] = {
    0x58: {0x07: 2, 0x08: 2},           # octorok / octorok_fast
    0x49: {0x08: 5, 0x09: 1},           # octorok_fast x5 + one blue
}

# The 0x77 -> 0x4A bomb walk, plus the screen one hop off it.
WALK = (0x78, 0x68, 0x58, 0x59, 0x49, 0x4A)
NEAR = (0x48,)


def row_of(type_id: int) -> int:
    """``Types0..3`` row for an ObjType. Anything unlisted falls to row 3."""
    return _ROW_BY_TYPE.get(int(type_id), 3)


def per_kill(row: int) -> tuple[float, float]:
    """(expected rupees, expected bomb packs) for one random-table kill."""
    rate, table = ROWS[row][0], ROWS[row][1]
    p = rate / 256
    rupees = sum(VALUE.get(item, 0) for item in table) / len(table)
    bombs = sum(1 for item in table if item == BOMB) / len(table)
    return p * rupees, p * bombs


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
        census = MEASURED.get(screen)
        if census is not None:
            r = b = 0.0
            for type_id, count in census.items():
                rr, bb = per_kill(row_of(type_id))
                r += rr * count
                b += bb * count
            n = sum(census.values())
            rupees += r
            bombs += b
            bodies += n
            rows = "+".join(
                f"{ROWS[row_of(t)][2]}/{ROWS[row_of(t)][3]}x{c}" for t, c in census.items()
            )
            lines.append(
                f"  0x{screen:02x} {grid_name(screen):4} {'(live census)':14} x{n} "
                f"{rows:14} {r:5.2f}R"
                + (f"  {b:4.2f} bomb packs" if b else "")
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
    print(f"\nOne pass of the 0x77 -> 0x4A bomb walk ({walk_n} bodies)")
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
