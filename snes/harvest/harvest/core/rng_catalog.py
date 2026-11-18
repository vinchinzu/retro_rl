"""Authoritative catalog of ROM random number generation (RNG) mechanics in Harvest Moon SNES.

Every constant, probability denominator, memory address, and state flag in this module
is extracted directly from the ROM disassembly (HM-Decomp). This module defines all
game RNG mechanics that change flags or alter outcomes so planning/search trees
can easily handle, simulate, and plan around them.

Overview of SNES Harvest Moon RNG Architecture:
-----------------------------------------------
1. PRNG Implementation:
   - Routine: GetRNG at bank_83.asm:195 ($838138).
   - Internal State: 3 bytes in direct page / low WRAM:
     - !RNG_mem_1 = $0100
     - !RNG_mem_2 = $0101
     - !RNG_mem_3 = $0102
   - Frame Advance: bank_80.asm:745 calls GetRNG once per frame during ReadJoypad.

2. Bounded Integer Uniform Generator:
   - Routine: RNGReturn0toA at bank_80.asm:1152 ($8089F9).
   - Calling Convention: Enter with accumulator A = denominator N (where N divides 256).
   - Return Value: Integer in range [0, N - 1] in register A.
   - Probability of rolling 0 (success): exactly 1 / N.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Weather Generation & Forecasting
# ═══════════════════════════════════════════════════════════════════════════════
# ROM Routines:
#   - WeatherTomorrow: bank_82.asm:1355 ($828CF9)
#   - SetWeatherFlags: bank_82.asm:1244 ($828C09)
# Addresses:
#   - !weather_tomorrow: $098C (8-bit code for tomorrow)
#   - Weather flags: $0196 (16-bit bitmask for today)

# Tomorrow Weather Codes ($098C)
WEATHER_CODE_SUNNY = 0x00
WEATHER_CODE_RAIN = 0x01
WEATHER_CODE_SNOW = 0x02
WEATHER_CODE_HURRICANE = 0x03
WEATHER_CODE_BLIZZARD = 0x04       # Called 'fairclimate' in decomp; destructive blizzard
WEATHER_CODE_THUNDER = 0x05        # Thunderstorm / calm
WEATHER_CODE_FLOWER_FESTIVAL = 0x06
WEATHER_CODE_HARVEST_FESTIVAL = 0x07
WEATHER_CODE_THANKSGIVING = 0x08
WEATHER_CODE_STAR_NIGHT = 0x09
WEATHER_CODE_NEW_YEAR = 0x0A
WEATHER_CODE_EGG_FESTIVAL = 0x0B

# Today Weather Flags ($0196 bitmask from Climate_Flag_Table, bank_82.asm:1351)
WEATHER_FLAG_RAIN = 0x0002         # Set in rain (or spring snow)
WEATHER_FLAG_SUNNY_SUMMER = 0x0004 # Set in summer sunny (or winter rain)
WEATHER_FLAG_SNOW = 0x0008         # Set in winter snow
WEATHER_FLAG_HURRICANE = 0x0010    # Set in hurricane
WEATHER_FLAG_THUNDER = 0x0100      # Set in thunderstorm
WEATHER_FLAG_BLIZZARD = 0x0200     # Set in blizzard/snowstorm
WEATHER_FLAG_WILD_DOG_BREACH = 0x0400 # Set overnight by broken fence wild dog breach

# Weather roll tables (1 in N roll denominators by season: 0=Spring, 1=Summer, 2=Fall, 3=Winter)
# Exact ROM arrays from bank_82.asm:1580-1585:
HURRICANE_CHANCE_TABLE = (0, 30, 0, 0)     # Hurricane_Chance_Table: db $00,$1E,$00,$00
RAIN_CHANCE_TABLE = (6, 10, 10, 0)         # Rain_Chance_Table:      db $06,$0A,$0A,$00
SNOW_CHANCE_TABLE = (0, 0, 0, 3)           # Snow_Chance_Table:      db $00,$00,$00,$03
THUNDER_CHANCE_TABLE = (0, 30, 0, 0)       # Thunder_Chance_Table:   db $00,$1E,$00,$00
SNOWSTORM_CHANCE_TABLE = (0, 0, 0, 8)      # Snowstorm_Chance_Table: db $00,$00,$00,$08

# Turtle shell item modifier on hurricane denominator (bank_82.asm:1525: ASL A -> doubles denominator)
TURTLE_SHELL_FLAG_BIT = 0x1000             # $7F1F6C bit 0x1000
HURRICANE_DENOMINATOR_BASE = 30
HURRICANE_DENOMINATOR_TURTLE = 60          # 30 * 2 = 60 (halves hurricane probability)

# Year 1 Guaranteed Weather Dates:
#   - Winter D7: Guaranteed Blizzard (code 4) if Year 1 and FLAG64 bit 0x0002 is clear (bank_82.asm:1449)
#   - Summer D29: Guaranteed Thunderstorm (code 5) if Year 1 and FLAG64 bit 0x0001 is clear (bank_82.asm:1483)
YEAR1_BLIZZARD_DAY = 7
YEAR1_THUNDER_DAY = 29

# Persistent Flags Set by Weather Events:
# When Blizzard (code 4) activates in SetWeatherFlags (bank_82.asm:1327):
FLAG64_BLIZZARD_SEEN = 0x0002              # $7F1F64 bit 0x0002
FLAG6E_BLIZZARD_EVENT = 0x0100             # $7F1F6E bit 0x0100
# When Thunderstorm (code 5) activates in SetWeatherFlags (bank_82.asm:1340):
FLAG64_THUNDER_SEEN = 0x0001               # $7F1F64 bit 0x0001
FLAG6E_THUNDER_EVENT = 0x0080              # $7F1F6E bit 0x0080


def forecast_weather_distribution(
    season: int,
    day: int,
    year: int = 0,
    *,
    turtle_shell_owned: bool = False,
    blizzard_seen: bool = False,
    thunder_seen: bool = False,
) -> Dict[int, float]:
    """Compute the exact probability distribution over tomorrow's weather code ($098C).

    Follows the exact priority branches in bank_82.asm:1355 WeatherTomorrow:
    1. Fixed festival checks (Flower Spring 22, Harvest Fall 11, Egg Fall 19,
       Thanksgiving Winter 9, Star Night Winter 23, New Year Winter 30).
    2. Winter special case: Year 1 & !blizzard_seen:
       - Day 7: 100% Blizzard (code 4).
       - Other days: 1/8 Blizzard roll.
    3. Thunderstorm check (Summer, Year 1 & !thunder_seen):
       - Day 29: 100% Thunderstorm (code 5).
       - Other days: 1/30 Thunderstorm roll.
    4. Hurricane check (Summer, day != 30):
       - 1/30 (or 1/60 with turtle shell) Hurricane (code 3).
    5. Normal weather rolls:
       - Rain check (Spring 1/6, Summer 1/10, Fall 1/10).
       - Snow check (Winter 1/3).
       - Fallback: Sunny (code 0).
    """
    # 1. Festival days have 100% fixed weather
    if season == 0 and day == 22:
        return {WEATHER_CODE_FLOWER_FESTIVAL: 1.0}
    if season == 2:
        if day == 11:
            return {WEATHER_CODE_HARVEST_FESTIVAL: 1.0}
        if day == 19:
            return {WEATHER_CODE_EGG_FESTIVAL: 1.0}
    if season == 3:
        if day == 9:
            return {WEATHER_CODE_THANKSGIVING: 1.0}
        if day == 23:
            return {WEATHER_CODE_STAR_NIGHT: 1.0}
        if day == 30:
            return {WEATHER_CODE_NEW_YEAR: 1.0}

    probs: Dict[int, float] = {}
    p_remaining = 1.0

    # 2. Winter snowstorm / blizzard
    if season == 3:
        if year == 0 and not blizzard_seen:
            if day == YEAR1_BLIZZARD_DAY:
                return {WEATHER_CODE_BLIZZARD: 1.0}
            p_blizzard = 1.0 / SNOWSTORM_CHANCE_TABLE[season]
            probs[WEATHER_CODE_BLIZZARD] = p_blizzard
            p_remaining -= p_blizzard
        # Normal snow check
        p_snow = (1.0 / SNOW_CHANCE_TABLE[season]) * p_remaining
        probs[WEATHER_CODE_SNOW] = p_snow
        p_remaining -= p_snow
        probs[WEATHER_CODE_SUNNY] = p_remaining
        return probs

    # 3. Summer special events
    if season == 1:
        # Thunder check
        if year == 0 and not thunder_seen:
            if day == YEAR1_THUNDER_DAY:
                return {WEATHER_CODE_THUNDER: 1.0}
            p_thunder = 1.0 / THUNDER_CHANCE_TABLE[season]
            probs[WEATHER_CODE_THUNDER] = p_thunder
            p_remaining -= p_thunder

        # Hurricane check (not on day 30)
        if day != 30:
            denom = HURRICANE_DENOMINATOR_TURTLE if turtle_shell_owned else HURRICANE_DENOMINATOR_BASE
            p_hurricane = (1.0 / denom) * p_remaining
            probs[WEATHER_CODE_HURRICANE] = p_hurricane
            p_remaining -= p_hurricane

    # 4. Normal rain check
    rain_denom = RAIN_CHANCE_TABLE[season]
    if rain_denom > 0:
        p_rain = (1.0 / rain_denom) * p_remaining
        probs[WEATHER_CODE_RAIN] = p_rain
        p_remaining -= p_rain

    # Fallback sunny
    probs[WEATHER_CODE_SUNNY] = probs.get(WEATHER_CODE_SUNNY, 0.0) + p_remaining
    return probs


def weather_code_to_today_flags(weather_code: int, season: int) -> Tuple[int, int, int]:
    """Convert tomorrow's weather code to today's flags set by bank_82.asm:1244 SetWeatherFlags.

    Returns:
        (weather_flags_0196, upgrade_flags_or_1f64, world_event_flags_or_1f6e)
    """
    flag_196 = 0x0000
    flag_64 = 0x0000
    flag_6e = 0x0000

    if weather_code == WEATHER_CODE_SUNNY:
        if season == 1:  # Summer
            flag_196 = WEATHER_FLAG_SUNNY_SUMMER
    elif weather_code == WEATHER_CODE_RAIN:
        if season == 3:  # Winter
            flag_196 = WEATHER_FLAG_SUNNY_SUMMER
        else:
            flag_196 = WEATHER_FLAG_RAIN
    elif weather_code == WEATHER_CODE_SNOW:
        if season == 0:  # Spring
            flag_196 = WEATHER_FLAG_RAIN
        else:
            flag_196 = WEATHER_FLAG_SNOW
    elif weather_code == WEATHER_CODE_HURRICANE:
        flag_196 = WEATHER_FLAG_HURRICANE
    elif weather_code == WEATHER_CODE_BLIZZARD:
        flag_196 = WEATHER_FLAG_BLIZZARD
        flag_64 = FLAG64_BLIZZARD_SEEN
        flag_6e = FLAG6E_BLIZZARD_EVENT
    elif weather_code == WEATHER_CODE_THUNDER:
        flag_196 = WEATHER_FLAG_THUNDER
        flag_64 = FLAG64_THUNDER_SEEN
        flag_6e = FLAG6E_THUNDER_EVENT

    return flag_196, flag_64, flag_6e


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Overnight Storm / Climate Farm Damage Check
# ═══════════════════════════════════════════════════════════════════════════════
# ROM Routine:
#   - ClimateFarmDamageCheck: bank_82.asm:2164 ($82A713)
#   - LoadsDamageProbabilities: bank_82.asm:316 ($828270)
#   - DamageProbabilityTable: bank_82.asm:343 ($828298)

# Table row entries: (fence_roll_denominator, grass_roll_denominator, crop_roll_denominator)
# Roll denominator N means 1 in N chance per tile of being damaged (0 means never damaged).
DAMAGE_PROBABILITY_TABLE = {
    "rain": (0x60, 0, 0),         # db $60,$00,$00,$00 -> 1/96 fence (~1.04%), no grass/crops
    "snow": (0x40, 0, 0),         # db $40,$00,$00,$00 -> 1/64 fence (~1.56%), no grass/crops
    "hurricane": (0x08, 0x10, 0x04), # db $08,$10,$04,$00 -> 1/8 fence (12.5%), 1/16 grass (6.25%), 1/4 crops (25%)
    "blizzard": (0x08, 0, 0x04),  # db $08,$00,$04,$00 -> 1/8 fence (12.5%), no grass, 1/4 crops (25%)
    "new_year": (0, 0x40, 0x20),  # db $00,$40,$20,$00 -> 1/64 grass (1.56%), 1/32 crops (3.125%)
}


def storm_damage_probabilities(weather_flags: int, *, is_new_year: bool = False) -> Tuple[float, float, float]:
    """Return per-tile (p_fence_break, p_grass_loss, p_crop_loss) from overnight weather."""
    if is_new_year:
        f_d, g_d, c_d = DAMAGE_PROBABILITY_TABLE["new_year"]
    elif weather_flags & WEATHER_FLAG_HURRICANE:
        f_d, g_d, c_d = DAMAGE_PROBABILITY_TABLE["hurricane"]
    elif weather_flags & WEATHER_FLAG_BLIZZARD:
        f_d, g_d, c_d = DAMAGE_PROBABILITY_TABLE["blizzard"]
    elif weather_flags & WEATHER_FLAG_RAIN:
        f_d, g_d, c_d = DAMAGE_PROBABILITY_TABLE["rain"]
    elif weather_flags & WEATHER_FLAG_SNOW:
        f_d, g_d, c_d = DAMAGE_PROBABILITY_TABLE["snow"]
    else:
        return 0.0, 0.0, 0.0

    p_fence = 1.0 / f_d if f_d > 0 else 0.0
    p_grass = 1.0 / g_d if g_d > 0 else 0.0
    p_crop = 1.0 / c_d if c_d > 0 else 0.0
    return p_fence, p_grass, p_crop


def expected_farm_damage(
    fences: int,
    grass_tiles: int,
    crop_tiles: int,
    weather_flags: int,
    *,
    is_new_year: bool = False,
) -> Tuple[float, float, float]:
    """Compute expected count of (broken_fences, destroyed_grass, destroyed_crops)."""
    p_f, p_g, p_c = storm_damage_probabilities(weather_flags, is_new_year=is_new_year)
    return fences * p_f, grass_tiles * p_g, crop_tiles * p_c


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Overnight Farm Decay & Broken Fence Wild Dog Breach
# ═══════════════════════════════════════════════════════════════════════════════
# ROM Routine:
#   - DailyFarmUpdate: bank_82.asm:2470-2504
# Checks:
#   - .brokenfence (bank_82.asm:2492): for each tile with broken fence ($06),
#     rolls RNGReturn0toA with A = 8 (1 in 8 chance).
#     Success sets:
#       - $0196 |= 0x0400 (WEATHER_FLAG_WILD_DOG_BREACH)
#       - $7F1F6E |= 0x0200 (EVENT_FLAG_WILD_DOG_BREACH)
#   - .addrandomtrash (bank_82.asm:2470): on empty tiles ($00..$02), if Spring or Summer
#     and day % 4 == 0, rolls GetRNG == 0 (1 in 256 chance) to sprout weed ($03).

WILD_DOG_BREACH_ROLL_DENOMINATOR = 8
WEED_SPROUT_ROLL_DENOMINATOR = 256


def wild_dog_breach_probability(num_broken_fences: int) -> float:
    """Probability that at least one broken fence tile triggers a wild dog breach overnight.

    P(breach) = 1 - (1 - 1/8)^N = 1 - (7/8)^N
    """
    if num_broken_fences <= 0:
        return 0.0
    return 1.0 - (7.0 / 8.0) ** num_broken_fences


def expected_weed_sprouts(empty_tiles: int, season: int, day: int) -> float:
    """Expected new weeds sprouting overnight on empty farm tiles."""
    if season not in (0, 1):  # Spring or Summer only
        return 0.0
    if (day & 0x03) != 0:     # Only every 4th day
        return 0.0
    return empty_tiles / WEED_SPROUT_ROLL_DENOMINATOR


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Livestock Overnight Sickness & Cranky Mechanics
# ═══════════════════════════════════════════════════════════════════════════════
# ROM Routine:
#   - CowFeedingandStatus: bank_83.asm:6723-6940
# Sickness:
#   - Unfed adult cow rolls RNGReturn0toA with A = 4 (1 in 4 chance).
#   - Sets status bit 0x20, timer = 7. Happiness -= 12, player happiness -= 30.
# Cranky:
#   - Adult cow left outdoors (map != 0x27):
#     - If $0196 & 0x0400 (wild dog breach active): 100% chance of cranky!
#     - Else if $7F1F5A & 0x1000: rolls RNGReturn0toA with A = 16 (1 in 16 chance).
#     - In .cowgetscranky (bank_83.asm:6917): sets status bit 0x10, timer = 3,
#       cow happiness -= 8.

COW_SICK_ROLL_DENOMINATOR = 4
COW_SICKNESS_START_TIMER = 7
COW_UNFED_HAPPINESS_PENALTY = -8
COW_SICK_HAPPINESS_PENALTY = -12
COW_UNFED_SICK_OVERNIGHT_DELTA = -20
COW_SICK_PLAYER_HAPPINESS_PENALTY = -30 # bank_83.asm:6878 LDA #$FFE2 (-30)

COW_CRANKY_OUTSIDE_ROLL_DENOMINATOR = 16
COW_CRANKY_START_TIMER = 3
COW_CRANKY_HAPPINESS_PENALTY = -8


def cow_outside_cranky_probability(*, wild_dog_breach: bool = False, flag_1f5a_1000: bool = True) -> float:
    """Probability that an adult cow left in the pasture overnight becomes cranky/mad.

    - If a wild dog breach occurs ($0196 & 0x0400), probability is 100% (bank_83.asm:6908).
    - Otherwise, if flag $7F1F5A:1000 is set, probability is 1/16 (6.25%).
    - Barn cows (map == 0x27) are never subject to this roll.
    """
    if wild_dog_breach:
        return 1.0
    if flag_1f5a_1000:
        return 1.0 / COW_CRANKY_OUTSIDE_ROLL_DENOMINATOR
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Tool Power Berry & Drop Drops
# ═══════════════════════════════════════════════════════════════════════════════
# ROM Routines:
#   - ToolUsedHoe / GoldHoe: bank_82_toolused_subrutines.asm:157 & 1377
#   - ToolUsedHammer / GoldHammer: bank_82_toolused_subrutines.asm:477 & 1700
#   - ToolUsedAxe / GoldAxe: bank_82_toolused_subrutines.asm:528 & 1732
#   - ToolUsedSickle / GoldSickle: bank_82_toolused_subrutines.asm:120 & 1340

# Roll denominators (1 in N chance)
HOE_POWER_BERRY_ROLL_DENOMINATOR = 64   # LDA.B #$40; JSL RNGReturn0toA (line 251)
HOE_MOLE_ROLL_DENOMINATOR = 32          # LDA.B #$20; JSL RNGReturn0toA (line 181)
HOE_COIN_ROLL_DENOMINATOR = 16          # LDA.B #$10; JSL RNGReturn0toA (line 206) -> 1G
HOE_MONEY_BAG_ROLL_DENOMINATOR = 16     # LDA.B #$10; JSL RNGReturn0toA (line 228) -> 5G
HAMMER_CAVE_BERRY_ROLL_DENOMINATOR = 4  # LDA.B #$04; JSL RNGReturn0toA (line 497)
AXE_STUMP_BERRY_ROLL_DENOMINATOR = 16   # LDA.B #$10; JSL RNGReturn0toA (line 606)
SICKLE_FROG_ROLL_DENOMINATOR = 16       # LDA.B #$10; JSL RNGReturn0toA (line 130)

# Power Berry Permanent Acquisition Bitflags ($7F1F64 / 0x11F64 upgrade_flags)
POWER_BERRY_CAVE_STATUE = 0x0200
POWER_BERRY_MOUNTAIN_STUMP = 0x0400
POWER_BERRY_FARM_HOE_1 = 0x0800
POWER_BERRY_FARM_HOE_2 = 0x1000

# Daily Lock Bitflags ($7F1F5C / 0x11F5C event_flags_1f5c)
DAILY_LOCK_CAVE_STATUE = 0x0080         # Set when statue struck, clears daily retry
DAILY_LOCK_FARM_BERRY = 0x0100          # Set when farm berry dug up today
DAILY_LOCK_FARM_GOLD = 0x0200           # Set when coin or money bag dug up today


def can_dig_farm_power_berry(upgrade_flags: int, daily_flags_1f5c: int, *, on_farm: bool = True) -> bool:
    """True if hoeing on the farm can roll for a Power Berry."""
    if not on_farm:
        return False
    if daily_flags_1f5c & DAILY_LOCK_FARM_BERRY:
        return False
    # If both farm hoe berries (0x0800 and 0x1000) are already owned, cannot get another
    if (upgrade_flags & POWER_BERRY_FARM_HOE_1) and (upgrade_flags & POWER_BERRY_FARM_HOE_2):
        return False
    return True


def can_hammer_cave_statue_berry(upgrade_flags: int, daily_flags_1f5c: int, *, in_cave: bool = True) -> bool:
    """True if striking the cave statue with hammer can roll for a Power Berry."""
    if not in_cave:
        return False
    if daily_flags_1f5c & DAILY_LOCK_CAVE_STATUE:
        return False
    return not bool(upgrade_flags & POWER_BERRY_CAVE_STATUE)


def can_axe_mountain_stump_berry(upgrade_flags: int, *, in_mountain: bool = True) -> bool:
    """True if chopping a mountain stump with axe can roll for a Power Berry."""
    if not in_mountain:
        return False
    return not bool(upgrade_flags & POWER_BERRY_MOUNTAIN_STUMP)


def hoe_expected_yield_per_till(
    upgrade_flags: int,
    daily_flags_1f5c: int,
    *,
    on_farm: bool = True,
    season: int = 0,
    flags_1f5a: int = 0,
) -> Dict[str, float]:
    """Compute per-till event probabilities and expected gold for strategy trees.

    Execution order in ToolUsedHoe (bank_82_toolused_subrutines.asm:163-265):
    1. Mole check (Spring/Summer, if $7F1F5A:1000 clear): 1/32. If hit, returns early.
    2. Coin check (if $7F1F5C:0200 clear): 1/16 -> +1G, returns early.
    3. Money bag check (if coin missed): 1/16 -> +5G, returns early.
    4. Berry check (if $7F1F5C:0100 clear & farm berries remaining): 1/64.
    """
    if not on_farm:
        return {"mole": 0.0, "coin": 0.0, "money_bag": 0.0, "power_berry": 0.0, "expected_gold": 0.0}

    p_continue = 1.0

    # 1. Mole
    p_mole = 0.0
    if season in (0, 1) and not (flags_1f5a & 0x1000):
        p_mole = 1.0 / HOE_MOLE_ROLL_DENOMINATOR
        p_continue -= p_mole

    # 2. Coin / Money bag
    p_coin = 0.0
    p_bag = 0.0
    if not (daily_flags_1f5c & DAILY_LOCK_FARM_GOLD):
        p_coin = p_continue * (1.0 / HOE_COIN_ROLL_DENOMINATOR)
        p_miss_coin = p_continue * (1.0 - 1.0 / HOE_COIN_ROLL_DENOMINATOR)
        p_bag = p_miss_coin * (1.0 / HOE_MONEY_BAG_ROLL_DENOMINATOR)
        p_continue = p_miss_coin * (1.0 - 1.0 / HOE_MONEY_BAG_ROLL_DENOMINATOR)

    # 3. Power berry
    p_berry = 0.0
    if can_dig_farm_power_berry(upgrade_flags, daily_flags_1f5c, on_farm=True):
        p_berry = p_continue * (1.0 / HOE_POWER_BERRY_ROLL_DENOMINATOR)

    exp_gold = p_coin * 1.0 + p_bag * 5.0
    return {
        "mole": p_mole,
        "coin": p_coin,
        "money_bag": p_bag,
        "power_berry": p_berry,
        "expected_gold": exp_gold,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Mountain Wild Animal Encounters
# ═══════════════════════════════════════════════════════════════════════════════
# ROM Routine:
#   - SUB_809D0B: bank_80.asm:3806-3896
# Triggered when player faces wild animal spawn tiles on mountain maps (0x10..0x13):
#   - Spring facing tile 0x00F9: 1/16 roll -> sets $7F1F5C |= 0x2000
#   - Fall facing tile 0x00FA:   1/16 roll -> sets $7F1F5C |= 0x4000
#   - Winter facing tile 0x00FB: 1/16 roll -> sets $7F1F5C |= 0x8000

MOUNTAIN_ANIMAL_ROLL_DENOMINATOR = 16
FLAG5C_SPRING_ANIMAL = 0x2000
FLAG5C_FALL_ANIMAL = 0x4000
FLAG5C_WINTER_ANIMAL = 0x8000


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Morning Happiness Bird Event
# ═══════════════════════════════════════════════════════════════════════════════
# ROM Routine:
#   - bank_83.asm:9433-9456
# Conditions:
#   - Weather is clear ($0196 & 0x001A == 0)
#   - Time is exactly 6:00:00 AM (hour=6, min=0, sec=0)
#   - Roll: RNGReturn0toA with A = 8 (1 in 8 chance)
# Outcome:
#   - Runs CC script 0x0023, adds +2 to player happiness.

MORNING_BIRD_ROLL_DENOMINATOR = 8
MORNING_BIRD_HAPPINESS_BONUS = 2


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Egg Festival Egg Hiding Spots Shuffle
# ═══════════════════════════════════════════════════════════════════════════════
# ROM Routine:
#   - bank_83.asm:9476-9550
# Array $09A4..$09AB (8 bytes) is initialized to [0, 1, 2, 3, 4, 5, 0, 0].
# Then 7 iterations (Y = 0..6) swap element Y with a random element in 0..7
# using RNGReturn0toA with A = 8.

EGG_FESTIVAL_INITIAL_SLOTS = (0, 1, 2, 3, 4, 5, 0, 0)


def shuffle_egg_festival_slots(rng_sequence: Optional[Sequence[int]] = None) -> List[int]:
    """Simulate the Egg Festival egg hiding spots shuffle (bank_83.asm:9522).

    If rng_sequence is provided, uses those 7 random target indices (each 0..7).
    Otherwise generates standard Fisher-Yates random indices.
    """
    slots = list(EGG_FESTIVAL_INITIAL_SLOTS)
    for y in range(7):
        target = rng_sequence[y] if rng_sequence is not None else int(np.random.randint(0, 8))
        slots[y], slots[target] = slots[target], slots[y]
    return slots


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Fishing Catch & Married Breakfast Variants
# ═══════════════════════════════════════════════════════════════════════════════
# ROM Routines:
#   - FishingCatchSomething: bank_84.asm:10621 (RNGReturn0toA with A = 3 -> 0, 1, 2)
#   - Breakfast food when married: bank_80.asm:46, bank_82.asm:144, 829
#     (RNGReturn0toA with A = 3 -> !what_to_eat in $098D = 0, 1, 2)

FISHING_CATCH_VARIANTS = 3
MARRIED_BREAKFAST_VARIANTS = 3


__all__ = [
    # Weather
    "WEATHER_CODE_SUNNY",
    "WEATHER_CODE_RAIN",
    "WEATHER_CODE_SNOW",
    "WEATHER_CODE_HURRICANE",
    "WEATHER_CODE_BLIZZARD",
    "WEATHER_CODE_THUNDER",
    "WEATHER_CODE_FLOWER_FESTIVAL",
    "WEATHER_CODE_HARVEST_FESTIVAL",
    "WEATHER_CODE_THANKSGIVING",
    "WEATHER_CODE_STAR_NIGHT",
    "WEATHER_CODE_NEW_YEAR",
    "WEATHER_CODE_EGG_FESTIVAL",
    "WEATHER_FLAG_RAIN",
    "WEATHER_FLAG_SUNNY_SUMMER",
    "WEATHER_FLAG_SNOW",
    "WEATHER_FLAG_HURRICANE",
    "WEATHER_FLAG_THUNDER",
    "WEATHER_FLAG_BLIZZARD",
    "WEATHER_FLAG_WILD_DOG_BREACH",
    "HURRICANE_CHANCE_TABLE",
    "RAIN_CHANCE_TABLE",
    "SNOW_CHANCE_TABLE",
    "THUNDER_CHANCE_TABLE",
    "SNOWSTORM_CHANCE_TABLE",
    "TURTLE_SHELL_FLAG_BIT",
    "HURRICANE_DENOMINATOR_BASE",
    "HURRICANE_DENOMINATOR_TURTLE",
    "YEAR1_BLIZZARD_DAY",
    "YEAR1_THUNDER_DAY",
    "FLAG64_BLIZZARD_SEEN",
    "FLAG6E_BLIZZARD_EVENT",
    "FLAG64_THUNDER_SEEN",
    "FLAG6E_THUNDER_EVENT",
    "forecast_weather_distribution",
    "weather_code_to_today_flags",
    # Storm Farm Damage
    "DAMAGE_PROBABILITY_TABLE",
    "storm_damage_probabilities",
    "expected_farm_damage",
    # Farm Decay & Wild Dogs
    "WILD_DOG_BREACH_ROLL_DENOMINATOR",
    "WEED_SPROUT_ROLL_DENOMINATOR",
    "wild_dog_breach_probability",
    "expected_weed_sprouts",
    # Livestock Overnight
    "COW_SICK_ROLL_DENOMINATOR",
    "COW_SICKNESS_START_TIMER",
    "COW_UNFED_HAPPINESS_PENALTY",
    "COW_SICK_HAPPINESS_PENALTY",
    "COW_UNFED_SICK_OVERNIGHT_DELTA",
    "COW_SICK_PLAYER_HAPPINESS_PENALTY",
    "COW_CRANKY_OUTSIDE_ROLL_DENOMINATOR",
    "COW_CRANKY_START_TIMER",
    "COW_CRANKY_HAPPINESS_PENALTY",
    "cow_outside_cranky_probability",
    # Tools & Drops
    "HOE_POWER_BERRY_ROLL_DENOMINATOR",
    "HOE_MOLE_ROLL_DENOMINATOR",
    "HOE_COIN_ROLL_DENOMINATOR",
    "HOE_MONEY_BAG_ROLL_DENOMINATOR",
    "HAMMER_CAVE_BERRY_ROLL_DENOMINATOR",
    "AXE_STUMP_BERRY_ROLL_DENOMINATOR",
    "SICKLE_FROG_ROLL_DENOMINATOR",
    "POWER_BERRY_CAVE_STATUE",
    "POWER_BERRY_MOUNTAIN_STUMP",
    "POWER_BERRY_FARM_HOE_1",
    "POWER_BERRY_FARM_HOE_2",
    "DAILY_LOCK_CAVE_STATUE",
    "DAILY_LOCK_FARM_BERRY",
    "DAILY_LOCK_FARM_GOLD",
    "can_dig_farm_power_berry",
    "can_hammer_cave_statue_berry",
    "can_axe_mountain_stump_berry",
    "hoe_expected_yield_per_till",
    # Mountain Animals
    "MOUNTAIN_ANIMAL_ROLL_DENOMINATOR",
    "FLAG5C_SPRING_ANIMAL",
    "FLAG5C_FALL_ANIMAL",
    "FLAG5C_WINTER_ANIMAL",
    # Morning Bird
    "MORNING_BIRD_ROLL_DENOMINATOR",
    "MORNING_BIRD_HAPPINESS_BONUS",
    # Egg Festival
    "EGG_FESTIVAL_INITIAL_SLOTS",
    "shuffle_egg_festival_slots",
    # Fishing & Breakfast
    "FISHING_CATCH_VARIANTS",
    "MARRIED_BREAKFAST_VARIANTS",
]
