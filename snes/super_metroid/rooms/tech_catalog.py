"""Map Rando / sm-json-data tech tree catalog + bot builder coverage.

Source of truth for tech *definitions* and dependency tree:
``refs/sm-json-data/tech.json`` (same corpus Map Rando builds on —
https://maprando.com/logic, example node
https://maprando.com/logic/tech/23 = ``canStopOnADime``).

Difficulty tiers (Basic / Medium / …) live on the Map Rando logic UI, not in
sm-json-data. They are embedded here (scraped from ``/logic``) so builders can
filter by skill budget without a network call.

Rebuild the on-disk index::

    uv run python snes/super_metroid/scripts/export/maprando_tech_catalog.py
    uv run python snes/super_metroid/scripts/export/maprando_tech_catalog.py --summary
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from super_metroid.paths import GAME_DIR, MAPS_DIR

SM_JSON_TECH_PATH = GAME_DIR / "refs" / "sm-json-data" / "tech.json"
MAPRANDO_TECH_CATALOG_PATH = MAPS_DIR / "maprando_tech_catalog.json"
MAPRANDO_LOGIC_URL = "https://maprando.com/logic"

Difficulty = Literal[
    "Implicit",
    "Basic",
    "Medium",
    "Hard",
    "Very Hard",
    "Expert",
    "Expert+",
    "Extreme",
    "Extreme+",
    "Insane",
    "Insane+",
    "Beyond",
    "Ignored",
    "Unlisted",
]

BuilderPriority = Literal["core", "try", "later", "out_of_scope"]
BotStatus = Literal["green", "partial", "missing", "unassessed"]

DIFFICULTY_ORDER: tuple[str, ...] = (
    "Implicit",
    "Basic",
    "Medium",
    "Hard",
    "Very Hard",
    "Expert",
    "Expert+",
    "Extreme",
    "Extreme+",
    "Insane",
    "Insane+",
    "Beyond",
    "Ignored",
    "Unlisted",
)

# Map Rando UI difficulty for each tech name (from https://maprando.com/logic).
# Format: difficulty|name|id  — id is the /logic/tech/<id> slug.
_MAPRANDO_DIFFICULTY_TABLE = """
Implicit|canStopOnADime|23
Implicit|canTrivialMidAirMorph|31
Implicit|canUseGrapple|50
Implicit|canTurnaroundSpinJump|59
Implicit|canUseEnemies|101
Implicit|canTrivialUseFrozenEnemies|108
Implicit|canEscapeEnemyGrab|119
Implicit|canSpecialBeamAttack|125
Implicit|canDash|211
Basic|canHeatRun|6
Basic|canMidAirMorph|32
Basic|canWallJump|76
Basic|canUseFrozenEnemies|109
Basic|canShinespark|132
Medium|canSuitlessMaridia|7
Medium|canSpaceJumpWaterBounce|10
Medium|canDisableEquipment|12
Medium|canDownGrab|21
Medium|canCrouchJump|58
Medium|canGravityJump|61
Medium|canSpringBallJumpMidAir|64
Medium|canCarefulJump|71
Medium|canPreciseWallJump|77
Medium|canConsecutiveWallJump|80
Medium|canBombHorizontally|87
Medium|canIBJ|89
Medium|canSpeedyJump|212
Medium|canHorizontalShinespark|133
Medium|canShinechargeMovement|136
Medium|canMidairShinespark|134
Medium|canAwakenZebes|167
Hard|canSuitlessLavaDive|5
Hard|canSunkenTileWideWallClimb|8
Hard|canCrossRoomJumpIntoWater|9
Hard|canPlayInSand|13
Hard|canManageReserves|18
Hard|canMoonwalk|24
Hard|canResetFallSpeed|30
Hard|canWallJumpInstantMorph|33
Hard|canLateralMidAirMorph|37
Hard|canMockball|41
Hard|canTwoTileSqueeze|44
Hard|canBufferedMomentumConservingTurnaround|240
Hard|canPreciseGrapple|51
Hard|canMorphTurnaround|185
Hard|canUseIFrames|62
Hard|canTrickySpringBallJump|65
Hard|canTrickyWallJump|78
Hard|canLongIBJ|193
Hard|canJumpIntoIBJ|90
Hard|canPowerBombMidIBJ|98
Hard|canSnailClimb|102
Hard|canNeutralDamageBoost|103
Hard|canMochtroidIceClimb|110
Hard|canTrickyUseFrozenEnemies|111
Hard|canDodgeWhileShooting|116
Hard|canHitbox|122
Hard|canPseudoScrew|123
Hard|canOffScreenSuperShot|126
Hard|canGateGlitch|127
Hard|canHeroShot|130
Hard|canHyperGateShot|10001
Hard|canHorizontalMidairShinespark|244
Hard|canWaterShineCharge|150
Hard|canXRayWaitForIFrames|152
Hard|canXRayStandUp|155
Hard|canCeilingClip|169
Hard|canXRayCeilingClip|170
Hard|canCrystalFlash|174
Hard|canCameraManip|182
Very Hard|canPrepareForNextRoom|4
Very Hard|canEscapeSand|14
Very Hard|canDownBack|22
Very Hard|canMoonfall|25
Very Hard|can4HighMidAirMorph|35
Very Hard|canSpringBallBounce|38
Very Hard|canBounceBall|43
Very Hard|canSpeedball|42
Very Hard|canTunnelCrawl|46
Very Hard|canMomentumConservingTurnaround|180
Very Hard|canQuickDrop|47
Very Hard|canMidairWiggle|48
Very Hard|canFlatleyJump|60
Very Hard|canStationarySpinJump|63
Very Hard|canSpringwall|66
Very Hard|canCrumbleJump|70
Very Hard|canTrickyJump|72
Very Hard|canCWJ|75
Very Hard|canSpringBallBombJump|99
Very Hard|canBombJumpWaterEscape|100
Very Hard|canHorizontalDamageBoost|104
Very Hard|canManipulateMellas|105
Very Hard|canSamusEaterStandUp|106
Very Hard|canKago|107
Very Hard|canMetroidAvoid|114
Very Hard|canWallJumpWithCharge|124
Very Hard|canAutoCancelWeapon|131
Very Hard|canSidePlatformCrossRoomJump|197
Very Hard|canShinechargeMovementComplex|137
Very Hard|canUseSpeedEchoes|139
Very Hard|canBlueSpaceJump|145
Very Hard|canTemporaryBlue|146
Very Hard|canStutterWaterShineCharge|151
Very Hard|canXRayTurnaround|153
Very Hard|canXRayClimb|156
Very Hard|canPartialFloorClip|173
Very Hard|canHeatedCrystalFlash|191
Very Hard|canRefillStation10PowerBombCrystalFlash|238
Very Hard|canTurnaroundAimCancel|179
Expert|canSpaceJumpWaterEscape|11
Expert|canPauseAbuse|19
Expert|can3HighWallMidAirMorph|34
Expert|canRJump|36
Expert|canTrickySpringBallBounce|39
Expert|canStationaryLateralMidAirMorph|40
Expert|canMomentumConservingMorph|181
Expert|canGrappleJump|52
Expert|canDoubleSpringBallJumpMidAir|67
Expert|canSpringFling|69
Expert|canFastWallJumpClimb|81
Expert|canJumpIntoRespawningBlock|84
Expert|canWallJumpBombBoost|86
Expert|canHBJ|88
Expert|canBabyMetroidAvoid|115
Expert|canFarmWhileShooting|117
Expert|canTrickyDodgeEnemies|118
Expert|canTrickyDashJump|73
Expert|canShinechargeMovementTricky|138
Expert|canSlowShortCharge|140
Expert|canPreciseSpaceJump|144
Expert|canChainTemporaryBlue|147
Expert|canPreciseStutterWaterShineCharge|199
Expert|canRightSideDoorStuck|157
Expert|canRMode|161
Expert|canPreciseCeilingClip|171
Expert|can10PowerBombCrystalFlash|177
Expert|canUsePowerBombLag|190
Expert+|canSandGrappleBoost|17
Expert+|canPreciseReserveRefill|188
Expert+|canGrappleTeleport|55
Expert+|canGrappleBombHang|56
Expert+|canUnmorphBombBoost|85
Expert+|canDoubleBombJump|96
Expert+|canCrazyCrabClimb|113
Expert+|canEnemyExtendRunway|120
Expert+|canControlShinesparkEnd|189
Expert+|canShinesparkSlopeClip|203
Expert+|canCarryFlashSuit|207
Expert+|canSpikeSuit|141
Expert+|canRModeCrystalFlashInterrupt|221
Expert+|canCarryBlueSuit|215
Expert+|canRModeSparkInterrupt|216
Expert+|canCrystalSpark|217
Expert+|canXRayCancelShinecharge|154
Expert+|canLongXRayClimb|202
Expert+|canXMode|159
Expert+|canGMode|162
Expert+|canBombIntoCrystalFlashClip|175
Expert+|canDeepTransition|183
Expert+|canSkipDoorLock|184
Extreme|canBePatient|1
Extreme|canSandfallBounce|15
Extreme|canSandBombBoost|16
Extreme|canEnemyStuckMoonfall|28
Extreme|canOffScreenMovement|49
Extreme|canPreciseGrappleJump|195
Extreme|canUnmorphGrappleHang|57
Extreme|canFrozenEnemyGrappleHang|192
Extreme|canInsaneJump|74
Extreme|canInsaneWallJump|79
Extreme|canUnderwaterWallJump|83
Extreme|canBombAboveIBJ|91
Extreme|canStaggeredIBJ|97
Extreme|canWallIceClip|112
Extreme|canGrappleClip|129
Extreme|canShinesparkDeepStuck|135
Extreme|canPatientSpikeSuit|234
Extreme|canComplexCarryFlashSuit|208
Extreme|canBlueSuitSpikeJump|242
Extreme|canUnderwaterCrystalSpark|227
Extreme|canXModeBlueSuit|223
Extreme|canSpeedKeep|143
Extreme|canReserveTriggerBufferXRay|230
Extreme|canRightSideDoorStuckFromWater|158
Extreme|canXModeMovement|243
Extreme|canRModeStandupClip|228
Extreme|canComplexGMode|205
Extreme|canGModeXRayClimb|235
Extreme|canGModeImmobile|163
Extreme|canArtificialMorph|164
Extreme|canRemoteAcquire|236
Extreme|canPowerBombItemOverloadPLMs|206
Extreme|canUpwardGModeSetup|166
Extreme|canBlueSuitGModeSetup|218
Extreme|canJumpIntoCrystalFlashClip|176
Extreme|canHeated10PowerBombCrystalFlash|239
Extreme+|canBeLucky|231
Extreme+|canReserveDoubleDamageBoost|20
Extreme+|canMoondance|26
Extreme+|canCrouchGateClip|45
Extreme+|canGrappleTeleportWallEscape|209
Extreme+|canUnderwaterBombIntoSpringBallJump|68
Extreme+|canDiagonalBombJump|94
Extreme+|canTrickyEnemyExtendRunway|121
Extreme+|canTrickySpikeSuit|233
Extreme+|canComplexRModeCrystalFlashInterrupt|226
Extreme+|canUseFlashSuitInitialSpark|214
Extreme+|canRModePauseAbuseSparkInterrupt|222
Extreme+|canDoubleXModeBlueSuit|224
Extreme+|canPauseRemorphTemporaryBlue|149
Extreme+|canHeatedGMode|198
Extreme+|canSamusEaterTeleport|194
Extreme+|canHighPixelCeilingClip|172
Insane|canBeVeryPatient|2
Insane|canInsaneMidAirMorph|196
Insane|canTrickyGrappleJump|53
Insane|canLongUnderwaterWallJump|187
Insane|canElevatorCrystalFlash|178
Insane|canSlopeSpark|210
Insane|canTrickyRModeCrystalFlashInterrupt|229
Insane|canXModeSpikeSuit|219
Insane|canTrickyCarryFlashSuit|142
Insane|canSuperjump|160
Insane|canLongChainTemporaryBlue|148
Insane|canSlopeXMode|225
Insane|canTrickyGMode|200
Insane|canDownwardGModeSetup|165
Insane|canCount|237
Insane+|canBeVeryLucky|232
Insane+|canExtendedMoondance|27
Insane+|canFreeFallClip|29
Insane+|canBombGrappleJump|54
Insane+|canSuperSink|204
Insane+|canBootless2WideUWJ|201
Insane+|canCeilingBombJump|92
Insane+|canRiskySpikeSuit|241
Insane+|canRModeKnockbackSpark|213
Insane+|canRightSideDashlessDoorStuck|220
Beyond|canBeExtremelyPatient|3
Beyond|canUnderwaterWallJumpBreakFree|186
Beyond|canLongCeilingBombJump|93
Ignored|canWrapAroundShot|128
Ignored|canRiskPermanentLossOfAccess|168
"""

# Bot execution status for builder work (Implicit / Basic / Medium scored).
# status: green = reusable skill API; partial = in controllers/tapes only;
# missing = no bot path yet.
_BOT_ASSESSMENTS: dict[str, tuple[str, str, str]] = {
    # Implicit
    "canStopOnADime": (
        "partial",
        "Angle-hold stop builder (elevators / door seat)",
        "skills/basic_moves.stop_on_a_dime",
    ),
    "canTrivialMidAirMorph": (
        "green",
        "ensure_morph mid-air",
        "routes/controller_common.ensure_morph",
    ),
    "canUseGrapple": (
        "partial",
        "Item capability only; no grapple swing skill",
        "rooms/capabilities",
    ),
    "canTurnaroundSpinJump": (
        "partial",
        "Spin + face flip inside WJ/runway",
        "skills/walljump, runway",
    ),
    "canUseEnemies": (
        "partial",
        "Enemy slot read + damage-boost hold",
        "skills/runway, walljump.damage_boost_hold",
    ),
    "canTrivialUseFrozenEnemies": (
        "missing",
        "No freeze-enemy builder",
        "",
    ),
    "canEscapeEnemyGrab": (
        "partial",
        "Knockback escape helpers",
        "skills/knockback",
    ),
    "canSpecialBeamAttack": (
        "partial",
        "Charge / combat primitives",
        "combat/*",
    ),
    "canDash": (
        "green",
        "runway_dash + dash charge",
        "skills/runway, shinespark",
    ),
    # Basic
    "canHeatRun": (
        "partial",
        "Heat frame budgets in K4 paths; not generic skill",
        "routes/kpdr heat hops",
    ),
    "canMidAirMorph": (
        "green",
        "ensure_morph / morph bomb climbs",
        "controller_common, skills/morph_bomb",
    ),
    "canWallJump": (
        "green",
        "walljump_once / consecutive",
        "skills/walljump",
    ),
    "canUseFrozenEnemies": (
        "missing",
        "No ice-freeze platform builder",
        "",
    ),
    "canShinespark": (
        "green",
        "charge_store_activate dual-track",
        "skills/shinespark",
    ),
    # Medium
    "canSuitlessMaridia": (
        "partial",
        "Human-tape Maridia spine; not a skill primitive",
        "human_tape / guided_human",
    ),
    "canSpaceJumpWaterBounce": ("missing", "", ""),
    "canDisableEquipment": (
        "missing",
        "Pause equip strip not implemented as skill",
        "",
    ),
    "canDownGrab": (
        "partial",
        "Ledge grab builder (needs room-specific success gate)",
        "skills/basic_moves.down_grab",
    ),
    "canCrouchJump": (
        "partial",
        "DOWN→A crouch-jump builder",
        "skills/basic_moves.crouch_jump",
    ),
    "canGravityJump": ("missing", "", ""),
    "canSpringBallJumpMidAir": ("missing", "", ""),
    "canCarefulJump": (
        "partial",
        "Geometry bands + lip timing",
        "skills/geometry, door_exit",
    ),
    "canPreciseWallJump": (
        "green",
        "WallJumpTiming tight windows",
        "skills/walljump",
    ),
    "canConsecutiveWallJump": (
        "green",
        "consecutive_walljumps / period climb",
        "skills/walljump",
    ),
    "canBombHorizontally": (
        "partial",
        "Morph bomb climb is vertical-biased",
        "skills/morph_bomb",
    ),
    "canIBJ": (
        "partial",
        "Double-bomb IBJ in K5 / Kraid return; not general height IBJ",
        "skills/morph_bomb, kraid_return",
    ),
    "canSpeedyJump": (
        "partial",
        "runway_dash + jump patterns",
        "skills/runway, basic_moves.speedy_jump",
    ),
    "canHorizontalShinespark": (
        "green",
        "activate_shinespark LEFT/RIGHT",
        "skills/shinespark",
    ),
    "canShinechargeMovement": (
        "green",
        "charge_until_boost while moving",
        "skills/shinespark",
    ),
    "canMidairShinespark": (
        "green",
        "spin unspin activate",
        "skills/shinespark.store_then_spin_unspin_activate",
    ),
    "canAwakenZebes": (
        "partial",
        "Product continuous boot path",
        "routes/continuous",
    ),
}


@dataclass(frozen=True)
class TechNode:
    """One Map Rando / sm-json-data tech with bot builder metadata."""

    id: int | None
    name: str
    difficulty: str
    category: str
    parent: str | None
    depth: int
    tech_requires: tuple[Any, ...]
    other_requires: tuple[Any, ...]
    note: str
    builder_priority: str
    bot_status: str
    bot_notes: str
    bot_module: str

    @property
    def logic_url(self) -> str | None:
        if self.id is None:
            return None
        return f"{MAPRANDO_LOGIC_URL}/tech/{self.id}"

    @property
    def is_builder_target(self) -> bool:
        return self.builder_priority in ("core", "try")


def parse_maprando_difficulties(
    table: str = _MAPRANDO_DIFFICULTY_TABLE,
) -> dict[str, tuple[str, int]]:
    """Return ``{tech_name: (difficulty, maprando_tech_id)}``."""
    out: dict[str, tuple[str, int]] = {}
    for line in table.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        difficulty, name, tid = line.split("|")
        out[name] = (difficulty, int(tid))
    return out


def builder_priority_for(difficulty: str) -> str:
    if difficulty in ("Implicit", "Basic"):
        return "core"
    if difficulty == "Medium":
        return "try"
    if difficulty in ("Hard", "Very Hard"):
        return "later"
    return "out_of_scope"


def _note_str(note: Any) -> str:
    if note is None:
        return ""
    if isinstance(note, list):
        return " ".join(str(x) for x in note)
    return str(note)


def _walk_tech(
    tech: Mapping[str, Any],
    category: str,
    difficulties: Mapping[str, tuple[str, int]],
    *,
    parent: str | None = None,
    depth: int = 0,
) -> list[dict[str, Any]]:
    name = str(tech["name"])
    sm_id = tech.get("id")
    diff_pair = difficulties.get(name)
    difficulty = diff_pair[0] if diff_pair else "Unlisted"
    # Prefer sm-json-data id; fall back to Map Rando slug id.
    tech_id = int(sm_id) if sm_id is not None else (diff_pair[1] if diff_pair else None)
    bot = _BOT_ASSESSMENTS.get(name)
    if bot is None:
        status, notes, module = "unassessed", "", ""
    else:
        status, notes, module = bot
    entry: dict[str, Any] = {
        "id": tech_id,
        "name": name,
        "maprandoDifficulty": difficulty,
        "category": category,
        "parent": parent,
        "depth": depth,
        "techRequires": list(tech.get("techRequires") or []),
        "otherRequires": list(tech.get("otherRequires") or []),
        "note": _note_str(tech.get("note")),
        "url": f"{MAPRANDO_LOGIC_URL}/tech/{tech_id}" if tech_id is not None else None,
        "builderPriority": builder_priority_for(difficulty),
        "bot": {
            "status": status,
            "notes": notes,
            "module": module,
        },
    }
    rows = [entry]
    for ext in tech.get("extensionTechs") or []:
        rows.extend(
            _walk_tech(ext, category, difficulties, parent=name, depth=depth + 1)
        )
    return rows


def parse_techs_from_sm_json_data(
    tech_path: Path | None = None,
    *,
    difficulties: Mapping[str, tuple[str, int]] | None = None,
) -> list[dict[str, Any]]:
    """Flatten ``tech.json`` into catalog rows (includes extension techs)."""
    path = tech_path or SM_JSON_TECH_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    diffs = dict(difficulties or parse_maprando_difficulties())
    rows: list[dict[str, Any]] = []
    for cat in data.get("techCategories") or []:
        category = str(cat.get("name") or "")
        for tech in cat.get("techs") or []:
            rows.extend(_walk_tech(tech, category, diffs))

    # Map Rando-only techs not present in vendored sm-json-data.
    known = {r["name"] for r in rows}
    for name, (difficulty, tid) in diffs.items():
        if name in known:
            continue
        bot = _BOT_ASSESSMENTS.get(name)
        status, notes, module = bot if bot else ("unassessed", "", "")
        rows.append(
            {
                "id": tid,
                "name": name,
                "maprandoDifficulty": difficulty,
                "category": "MapRandoExtra",
                "parent": None,
                "depth": 0,
                "techRequires": [],
                "otherRequires": [],
                "note": "Present on maprando.com/logic but not in vendored sm-json-data.",
                "url": f"{MAPRANDO_LOGIC_URL}/tech/{tid}",
                "builderPriority": builder_priority_for(difficulty),
                "bot": {"status": status, "notes": notes, "module": module},
            }
        )
    return rows


def build_catalog_payload(
    tech_path: Path | None = None,
) -> dict[str, Any]:
    """Full on-disk catalog payload (JSON-serializable)."""
    path = tech_path or SM_JSON_TECH_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    techs = parse_techs_from_sm_json_data(path)
    by_diff: dict[str, int] = {}
    for t in techs:
        d = str(t["maprandoDifficulty"])
        by_diff[d] = by_diff.get(d, 0) + 1
    return {
        "kind": "super_metroid_maprando_tech_catalog",
        "source": {
            "smJsonData": "refs/sm-json-data/tech.json",
            "mapRandoLogic": MAPRANDO_LOGIC_URL,
            "mapRandoTechExample": f"{MAPRANDO_LOGIC_URL}/tech/23",
            "difficultyNote": (
                "Map Rando difficulty tiers are UI/settings labels on "
                f"{MAPRANDO_LOGIC_URL}; sm-json-data stores the tech tree "
                "(categories + extensionTechs) without difficulty. "
                "Difficulties embedded from maprando.com/logic."
            ),
        },
        "difficultyOrder": list(DIFFICULTY_ORDER),
        "builderPolicy": {
            "core": "Implicit + Basic — bot must execute as reusable skills",
            "try": "Medium — build as room-optimization builders when useful",
            "later": "Hard / Very Hard — only when a route demands it",
            "out_of_scope": (
                "Expert+ and above — human tape / TAS import first, "
                "not reactive builders"
            ),
        },
        "statusLegend": {
            "green": "Reusable skill API exists and is used in product/practice",
            "partial": "Present in controllers/tapes but not a clean builder skill yet",
            "missing": "No bot implementation",
            "unassessed": "Higher tier; not scored for builder work yet",
        },
        "counts": {
            "total": len(techs),
            "byDifficulty": by_diff,
            "builderCore": sum(1 for t in techs if t["builderPriority"] == "core"),
            "builderTry": sum(1 for t in techs if t["builderPriority"] == "try"),
            "botGreen": sum(1 for t in techs if t["bot"]["status"] == "green"),
            "botPartial": sum(1 for t in techs if t["bot"]["status"] == "partial"),
            "botMissing": sum(1 for t in techs if t["bot"]["status"] == "missing"),
        },
        "categories": [str(c.get("name") or "") for c in data.get("techCategories") or []],
        "nextTechId": data.get("nextTechId"),
        "techs": techs,
    }


def write_catalog(
    catalog_path: Path | None = None,
    *,
    tech_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Write ``maps/maprando_tech_catalog.json`` and return path + payload."""
    out = catalog_path or MAPRANDO_TECH_CATALOG_PATH
    payload = build_catalog_payload(tech_path=tech_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    load_tech_catalog.cache_clear()
    return out, payload


def _row_to_node(row: Mapping[str, Any]) -> TechNode:
    bot = row.get("bot") or {}
    return TechNode(
        id=row.get("id"),
        name=str(row["name"]),
        difficulty=str(row.get("maprandoDifficulty") or "Unlisted"),
        category=str(row.get("category") or ""),
        parent=row.get("parent"),
        depth=int(row.get("depth") or 0),
        tech_requires=tuple(row.get("techRequires") or ()),
        other_requires=tuple(row.get("otherRequires") or ()),
        note=str(row.get("note") or ""),
        builder_priority=str(row.get("builderPriority") or "out_of_scope"),
        bot_status=str(bot.get("status") or "unassessed"),
        bot_notes=str(bot.get("notes") or ""),
        bot_module=str(bot.get("module") or ""),
    )


@lru_cache(maxsize=1)
def load_tech_catalog(
    catalog_path: str | None = None,
) -> tuple[TechNode, ...]:
    """Load on-disk catalog (rebuild from sm-json-data if missing)."""
    path = Path(catalog_path) if catalog_path else MAPRANDO_TECH_CATALOG_PATH
    if not path.is_file():
        write_catalog(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return tuple(_row_to_node(r) for r in data.get("techs") or [])


def clear_tech_cache() -> None:
    load_tech_catalog.cache_clear()


def tech_by_name(name: str) -> TechNode | None:
    for node in load_tech_catalog():
        if node.name == name:
            return node
    return None


def tech_by_id(tech_id: int) -> TechNode | None:
    for node in load_tech_catalog():
        if node.id == tech_id:
            return node
    return None


def techs_at_difficulty(*difficulties: str) -> list[TechNode]:
    want = {d for d in difficulties}
    return [t for t in load_tech_catalog() if t.difficulty in want]


def builder_targets(
    *,
    priorities: Iterable[str] = ("core", "try"),
    max_status: BotStatus | None = None,
) -> list[TechNode]:
    """Techs the bot should treat as room-optimization builders.

    ``max_status`` filters to that status only when set (e.g. ``\"green\"``).
    """
    pri = set(priorities)
    rows = [t for t in load_tech_catalog() if t.builder_priority in pri]
    if max_status is not None:
        rows = [t for t in rows if t.bot_status == max_status]
    return rows


def builder_coverage_summary() -> dict[str, Any]:
    """Counts for Implicit/Basic/Medium builder readiness."""
    targets = builder_targets()
    by_status: dict[str, list[str]] = {
        "green": [],
        "partial": [],
        "missing": [],
        "unassessed": [],
    }
    for t in targets:
        by_status.setdefault(t.bot_status, []).append(t.name)
    return {
        "total": len(targets),
        "green": by_status["green"],
        "partial": by_status["partial"],
        "missing": by_status["missing"],
        "counts": {k: len(v) for k, v in by_status.items()},
    }


__all__ = [
    "DIFFICULTY_ORDER",
    "MAPRANDO_LOGIC_URL",
    "MAPRANDO_TECH_CATALOG_PATH",
    "SM_JSON_TECH_PATH",
    "TechNode",
    "build_catalog_payload",
    "builder_coverage_summary",
    "builder_priority_for",
    "builder_targets",
    "clear_tech_cache",
    "load_tech_catalog",
    "parse_maprando_difficulties",
    "parse_techs_from_sm_json_data",
    "tech_by_id",
    "tech_by_name",
    "techs_at_difficulty",
    "write_catalog",
]
