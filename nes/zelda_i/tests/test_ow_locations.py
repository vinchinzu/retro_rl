"""First-quest overworld location catalog. ROM check is optional."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zelda_i.anchors import (
    SCREEN_BRACELET_ARMOS,
    SCREEN_CANDLE_SHOP,
    SCREEN_LADDER_HEART,
    SCREEN_LEVEL3_ENTRANCE,
    SCREEN_LEVEL7_BAIT_SHOP,
    SCREEN_LEVEL7_ENTRANCE,
    SCREEN_LEVEL8_BUSH,
    SCREEN_MAGICAL_SWORD_GRAVE,
)
from zelda_i.overworld.locations import (
    CAVE_DUNGEON_1,
    CAVE_SHOP_ARROWS,
    CAVE_SHOP_SPECIAL,
    CAVE_WHITE_SWORD,
    CAVE_WOOD_SWORD,
    Q1_VANILLA,
    SECRET_PAYOUT,
    SECRET_RUPEE_CAVES,
    bomb_farms,
    cave_keeper,
    decode_cave_wares,
    decode_ow_attrs,
    easy_farms,
    farm_at,
    five_rupee_farms,
    grid_name,
    location,
    location_at,
    locations_from_rom,
    q1_farms,
    q1_item_locations,
    q1_shops,
    restock_for,
    rupee_farms,
    secret_payout,
    spawns_from_rom,
    worth_heart_farm,
    worth_rupee_farm,
)
from zelda_i.ram import (
    ADDR_WORLD_FLAGS,
    SCREEN_LEVEL1_ENTRANCE,
    SCREEN_LEVEL2_ENTRANCE,
    SCREEN_START,
    WORLD_FLAG_ITEM,
    ow_secret_taken,
    world_flag,
)

_ROM = (
    Path(__file__).resolve().parents[1] / "roms" / "Legend of Zelda, The.nes"
)


def test_grid_name_start_and_dungeons() -> None:
    assert grid_name(SCREEN_START) == "H8"
    assert grid_name(SCREEN_LEVEL1_ENTRANCE) == "H4"
    assert grid_name(0x0A) == "K1"
    assert grid_name(0x0B) == "L1"


def test_unique_names_and_screens() -> None:
    names = [loc.name for loc in Q1_VANILLA]
    screens = [loc.screen for loc in Q1_VANILLA]
    assert len(names) == len(set(names))
    assert len(screens) == len(set(screens))


def test_verified_screens_match_anchors() -> None:
    assert location("wooden_sword").screen == SCREEN_START
    assert location("wooden_sword").cave_id == CAVE_WOOD_SWORD
    assert location("white_sword").cave_id == CAVE_WHITE_SWORD
    assert location("dungeon_1").screen == SCREEN_LEVEL1_ENTRANCE
    assert location("dungeon_1").cave_id == CAVE_DUNGEON_1
    assert location("dungeon_2").screen == SCREEN_LEVEL2_ENTRANCE
    assert location("dungeon_3").screen == SCREEN_LEVEL3_ENTRANCE
    assert location("dungeon_7").screen == SCREEN_LEVEL7_ENTRANCE
    assert location("dungeon_8").screen == SCREEN_LEVEL8_BUSH
    assert location("arrow_shop").cave_id == CAVE_SHOP_ARROWS
    assert location("arrow_shop").vanilla == "arrows_80"
    assert location("candle_shop").screen == SCREEN_CANDLE_SHOP
    assert location("special_shop_e4").screen == SCREEN_LEVEL7_BAIT_SHOP
    assert location("special_shop_e4").cave_id == CAVE_SHOP_SPECIAL
    assert location("magical_sword").screen == SCREEN_MAGICAL_SWORD_GRAVE
    assert location("bracelet_armos").screen == SCREEN_BRACELET_ARMOS
    assert location("ladder_heart").screen == SCREEN_LADDER_HEART


def test_shops_have_three_slots() -> None:
    shops = q1_shops()
    assert location("arrow_shop") in shops
    assert location("candle_shop") in shops
    assert all(shop.shop_slots == 3 for shop in shops)


def test_item_locations_exclude_dungeons() -> None:
    items = q1_item_locations()
    assert location("wooden_sword") in items
    assert location("arrow_shop") in items
    assert location("letter") in items
    assert all(loc.kind != "dungeon" for loc in items)


def test_world_flags_index_by_screen() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_WORLD_FLAGS + 0x77] = WORLD_FLAG_ITEM
    assert world_flag(ram, 0x77) == WORLD_FLAG_ITEM
    assert ow_secret_taken(ram, 0x77)
    assert not ow_secret_taken(ram, 0x37)


def test_catalog_covers_q1_rom_caves() -> None:
    if not _ROM.is_file():
        return
    attrs = decode_ow_attrs(_ROM.read_bytes())
    q1 = {row.screen for row in attrs if row.cave_id and not row.q1_ignore}
    catalog = {loc.screen for loc in Q1_VANILLA if loc.cave_id}
    assert q1 == catalog, f"missing {sorted(q1 - catalog)} extra {sorted(catalog - q1)}"


def test_vanilla_rom_decoder_keeps_known_names() -> None:
    if not _ROM.is_file():
        return
    decoded = locations_from_rom(_ROM.read_bytes())
    by_name = {loc.name: loc for loc in decoded}
    assert by_name["wooden_sword"].cave_id == CAVE_WOOD_SWORD
    assert by_name["arrow_shop"].screen == 0x4A
    assert location_at(0x77) is not None


def test_start_screen_has_no_farm() -> None:
    assert farm_at(SCREEN_START) is None


def test_arrow_shop_farm_is_blue_tektite_group_c() -> None:
    """ROM $18500 0x4A is type 0x0D (blue tektite), drop group C (5-rupees)."""
    spot = farm_at(0x4A)
    assert spot is not None
    assert spot.prey == "tektite_blue"
    assert spot.drop_group == "C"
    assert "rupee_5" in spot.drops
    assert spot.restock_neighbor == 0x49
    assert spot.restock_direction == "LEFT"
    assert spot.evidence == "verified"
    assert spot in five_rupee_farms()
    assert spot in rupee_farms()


def test_east_of_start_is_easy_octorok_farm() -> None:
    spot = farm_at(0x78)
    assert spot is not None
    assert spot.prey == "octorok"
    assert spot.drop_group == "A"
    assert spot.drops == ("rupee", "heart", "fairy")
    assert spot.restock_neighbor == 0x77
    assert spot.restock_direction == "LEFT"
    assert spot.evidence == "verified"
    assert spot in easy_farms()


def test_restock_for_catalog_pairs() -> None:
    assert restock_for(0x4A) == (0x49, "LEFT")
    assert restock_for(0x77) is None


def test_worth_rupee_farm_route_screens() -> None:
    i8 = worth_rupee_farm(0x78)
    assert i8 is not None
    assert i8.prey == "octorok"
    assert i8.drop_group == "A"
    assert i8 in easy_farms()
    k5 = worth_rupee_farm(0x4A)
    assert k5 is not None
    assert k5.prey == "tektite_blue"
    assert k5.drop_group == "C"


def test_worth_heart_farm_skips_leevers_keeps_4a() -> None:
    """Heart-farm hook must not chase 0x48 leevers; 0x4A tektites stay legal."""
    assert farm_at(0x48) is not None and farm_at(0x48).prey == "leever"
    assert worth_heart_farm(0x48) is None
    assert worth_rupee_farm(0x48) is not None
    k5 = worth_heart_farm(0x4A)
    assert k5 is not None
    assert k5.prey == "tektite_blue"
    assert worth_heart_farm(0x78) is not None
    assert worth_heart_farm(0x59) is None  # peahat on the L2 door path
    assert farm_at(0x58) is not None and farm_at(0x58).prey.startswith("group_")
    assert worth_heart_farm(0x58) is None


def test_lynel_screens_are_not_worth_rupee_farm() -> None:
    lynels = [spot for spot in q1_farms() if spot.prey in ("lynel", "lynel_blue")]
    assert lynels
    for spot in lynels:
        assert worth_rupee_farm(spot.screen) is None


def test_bomb_farms_are_drop_group_b() -> None:
    bombs = bomb_farms()
    assert bombs
    assert all(spot.drop_group == "B" for spot in bombs)
    assert all("bomb" in spot.drops for spot in bombs)


def test_spawn_table_matches_rom() -> None:
    if not _ROM.is_file():
        return
    from_rom = {row.screen: row for row in spawns_from_rom(_ROM.read_bytes())}
    spot = farm_at(0x4A)
    assert spot is not None
    assert from_rom[0x4A].prey == spot.prey
    assert from_rom[0x4A].grouped is False
    assert 0x77 not in from_rom


_need_rom = pytest.mark.skipif(not _ROM.is_file(), reason="local Zelda I ROM not present")


@_need_rom
def test_catalog_rupee_caves_are_the_rom_secrets() -> None:
    """Every cave type whose wares are one middle rupee is a "rupees" row, and
    the payout the route counts on is the ROM's price (10R caves were once
    catalogued as a gamble, so no walk ever visited them)."""
    wares = decode_cave_wares(_ROM.read_bytes())
    secret_types = {t: secret_payout(w) for t, w in wares.items() if secret_payout(w)}
    assert secret_types == SECRET_PAYOUT
    for loc in Q1_VANILLA:
        if loc.cave_id in secret_types:
            assert loc.kind == "rupees", loc
            assert loc.vanilla == f"rupees_{secret_types[loc.cave_id]}", loc
        else:
            assert loc.kind != "rupees", loc


@_need_rom
def test_catalog_shops_sell_something() -> None:
    """A "shop" row's cave type has at least one non-rupee ware (a paid hint
    shows three rupee slots and sells nothing)."""
    wares = decode_cave_wares(_ROM.read_bytes())
    for loc in q1_shops():
        items = {w.item for w in wares[loc.cave_id] if not w.empty}
        assert items - {0x18}, loc


@_need_rom
def test_secret_spots_are_catalogued_rupee_caves() -> None:
    """The measured secret objects sit on screens the ROM routes to a rupee
    cave, and a bomb stand faces a rock while a burn stand faces a tree."""
    attrs = {row.screen: row for row in decode_ow_attrs(_ROM.read_bytes())}
    for screen, spot in SECRET_RUPEE_CAVES.items():
        assert attrs[screen].cave_id == spot.cave_id, hex(screen)
        assert spot.rupees == SECRET_PAYOUT[spot.cave_id]
        want = "bomb" if spot.uses_bomb else "burn"
        assert location_at(screen).open == want, hex(screen)
        # Stand on the ROM turn lattice's axis for the facing: a vertical
        # face needs a lattice column, a horizontal one a lattice row.
        sx, sy = spot.stand
        assert (sx % 8 == 0) if spot.face in ("UP", "DOWN") else (sy % 8 == 5), hex(screen)


def test_cave_keeper_matches_measured_sprites() -> None:
    """Keeper object type = cave type + $5A, from four live caves."""
    assert cave_keeper(0x11) == 0x6B  # take-any old man (0x7B, 0x2C)
    assert cave_keeper(0x18) == 0x72  # 0x0E letter
    assert cave_keeper(0x21) == 0x7B  # 0x48 30R moblin
    assert cave_keeper(0x22) == 0x7C  # 0x0F 100R moblin
