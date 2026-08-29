"""Vanilla Super Metroid TAS movie catalog (TASVideos game 121).

Fetch + convert button-press movies we can later parse into skills
(mockball, shinespark, walljump, x-ray climb, crystal flash, moonfall, …).
ROM hacks and non-input files stay listed as skips so we do not re-download
them.

Movies are gitignored; re-fetch with ``python -m super_metroid.tas.fetch_refs``.

Sources: https://tasvideos.org/121G · https://tasvideos.org/UserFiles/Game/121
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from super_metroid.paths import TAS_REF_DIR

MovieKind = Literal["lsmv", "bk2", "smv"]
Postprocess = Literal["gunzip_if_needed", "unwrap_nested_lsmv"]

REF_DIR = TAS_REF_DIR

_TV = "https://tasvideos.org"
_UF = f"{_TV}/UserFiles/Info"


@dataclass(frozen=True)
class MovieRef:
    """One vendored TASVideos movie (or an explicit skip)."""

    filename: str
    url: str
    kind: MovieKind
    source: str
    notes: str
    category: str
    vanilla: bool = True
    fetch: bool = True
    postprocess: Postprocess = "gunzip_if_needed"
    tags: tuple[str, ...] = ()
    full_slice_id: str | None = None
    expected_frames: int | None = None
    skip_reason: str | None = None
    emulator: str = ""

    @property
    def path(self) -> Path:
        return REF_DIR / self.filename

    @property
    def stem(self) -> str:
        return Path(self.filename).stem


# --- fetch (vanilla button streams, plus the existing contest smoke BK2) ---

MOVIES: tuple[MovieRef, ...] = (
    MovieRef(
        filename="sniq_any_3653M.lsmv",
        url=f"{_TV}/3653M?handler=Download",
        kind="lsmv",
        postprocess="unwrap_nested_lsmv",
        source="Sniq any% #3653M",
        notes="Current published any%. Arm-pump, mockball, door speed, shine.",
        category="any%",
        tags=("any%", "published", "movement"),
        expected_frames=129_712,
        emulator="lsnes rr2-β23",
    ),
    MovieRef(
        filename="sniq_100p.bk2",
        url=f"{_UF}/55928342467251616?handler=Download",
        kind="bk2",
        source="Sniq 100% BK2 (feos converter of #4010M)",
        notes="BizHawk-native 100% button log. Item-route encyclopedia.",
        category="100%",
        tags=("100%", "published", "bizhawk"),
        expected_frames=222_789,
        emulator="BizHawk BSNES (TASConverter)",
    ),
    MovieRef(
        filename="sniq_any_wip.lsmv",
        url=f"{_UF}/36208532992045040?handler=Download",
        kind="lsmv",
        source="Sniq any% WIP userfile (to Red Brinstar 2nd visit)",
        notes="Shorter early/mid route experiment.",
        category="any% wip",
        tags=("any%", "wip", "early"),
        expected_frames=55_037,
        emulator="lsnes",
    ),
    MovieRef(
        filename="moozooh_smtc4.bk2",
        url=f"{_UF}/638502075337523909?handler=Download",
        kind="bk2",
        source="moozooh SM TAS Contest Round 4 final",
        notes="Map Rando contest — not vanilla. Kept as a short BK2 smoke test.",
        category="contest",
        vanilla=False,
        tags=("contest", "short"),
        expected_frames=5_384,
        emulator="BizHawk",
        skip_reason="Map Randomizer contest ROM; not vanilla Zebes.",
    ),
    MovieRef(
        filename="sniq_low_3273M.lsmv",
        url=f"{_TV}/3273M?handler=Download",
        kind="lsmv",
        postprocess="unwrap_nested_lsmv",
        source="Sniq low% #3273M (13%)",
        notes="Shinespark-without-Gravity encyclopedia. Skill-dense low%.",
        category="low%",
        tags=("low%", "published", "shinespark", "full"),
        full_slice_id="sniq_low_full",
        expected_frames=167_797,
        emulator="lsnes rr2-β23",
    ),
    MovieRef(
        filename="sniq_100_4010M.lsmv",
        url=f"{_TV}/4010M?handler=Download",
        kind="lsmv",
        postprocess="unwrap_nested_lsmv",
        source="Sniq 100% #4010M (native lsnes)",
        notes="Authoring-core 100%. Pair with sniq_100p.bk2 for BizHawk.",
        category="100%",
        tags=("100%", "published", "lsnes", "full"),
        full_slice_id="sniq_100_lsmv_full",
        expected_frames=222_788,
        emulator="lsnes rr2-β23",
    ),
    MovieRef(
        filename="sniq_geg_5238M.lsmv",
        url=f"{_TV}/5238M?handler=Download",
        kind="lsmv",
        postprocess="unwrap_nested_lsmv",
        source="Sniq & NobodyNada game-end-glitch #5238M",
        notes="OOB + ACE ending. PAL ROM. Not a movement-skill source.",
        category="game end glitch",
        tags=("glitch", "ace", "published", "full"),
        full_slice_id="sniq_geg_full",
        expected_frames=18_640,
        emulator="lsnes RR2-B25",
    ),
    MovieRef(
        filename="sniq_any_3362M.lsmv",
        url=f"{_TV}/3362M?handler=Download",
        kind="lsmv",
        postprocess="unwrap_nested_lsmv",
        source="Sniq any% #3362M (obsolete)",
        notes="Prior any%; includes the Draygon ammo glitch current any% dropped.",
        category="any%",
        tags=("any%", "obsolete", "draygon", "full"),
        full_slice_id="sniq_any_3362_full",
        expected_frames=135_769,
        emulator="lsnes rr2-β23",
    ),
    MovieRef(
        filename="total_13pct_charge_speed.lsmv",
        url=f"{_UF}/30904919119106655?handler=Download",
        kind="lsmv",
        source="total 13% Charge/Speed route test (userfile)",
        notes="Charge+Speed 13% route test. Alternate low% item path.",
        category="13%",
        tags=("low%", "wip", "charge", "speed", "full"),
        full_slice_id="total_13pct_full",
        expected_frames=182_797,
        emulator="lsnes",
    ),
    MovieRef(
        filename="hero_bubbleroom.smv",
        url=f"{_UF}/6285130127679659?handler=Download",
        kind="smv",
        source="hero_of_the_day Norfair bubble-room alt (userfile)",
        notes="Isolated vanilla room strategy. Convert to BK2 for BizHawk.",
        category="room",
        tags=("room", "norfair", "isolated", "full"),
        full_slice_id="hero_bubbleroom_full",
        expected_frames=407,
        emulator="Snes9x",
    ),
    MovieRef(
        filename="hero_kraid_entry.smv",
        url=f"{_UF}/6261053699896510?handler=Download",
        kind="smv",
        source="hero_of_the_day Kraid entry idea (userfile)",
        notes="Isolated vanilla Kraid-entry idea. Convert to BK2 for BizHawk.",
        category="room",
        tags=("room", "kraid", "isolated", "full"),
        full_slice_id="hero_kraid_entry_full",
        expected_frames=422,
        emulator="Snes9x",
    ),
    MovieRef(
        filename="saturn_rbo_2078M.smv",
        url=f"{_TV}/2078M?handler=Download",
        kind="smv",
        source="Saturn Reverse Boss Order #2078M",
        notes="Crystal flash, suitless heat/water, gravity-jump class tech.",
        category="rbo",
        tags=("rbo", "crystal-flash", "suitless", "full"),
        full_slice_id="saturn_rbo_full",
        expected_frames=168_144,
        emulator="Snes9x 1.43 v12",
    ),
    MovieRef(
        filename="taco_kriole_any_1368M.smv",
        url=f"{_TV}/1368M?handler=Download",
        kind="smv",
        source="Taco & Kriole any% #1368M",
        notes="Classic realtime any% movement (CWJ, mockball era).",
        category="any%",
        tags=("any%", "obsolete", "movement", "full"),
        full_slice_id="taco_kriole_any_full",
        expected_frames=139_292,
        emulator="Snes9x 1.43 v9",
    ),
    MovieRef(
        filename="cpadolf_xray_1978M.smv",
        url=f"{_TV}/1978M?handler=Download",
        kind="smv",
        source="Cpadolf X-Ray glitch #1978M",
        notes="X-ray climb / door-glitch encyclopedia. Convert to BK2.",
        category="x-ray glitch",
        tags=("glitch", "x-ray", "climb", "full"),
        full_slice_id="cpadolf_xray_full",
        expected_frames=77_108,
        emulator="Snes9x 1.43 v12",
    ),
    MovieRef(
        filename="cpadolf_gt_2558M.lsmv",
        url=f"{_TV}/2558M?handler=Download",
        kind="lsmv",
        postprocess="unwrap_nested_lsmv",
        source="amaurea, Cpadolf & total GT-code GEG #2558M",
        notes="GT debug code + charged spacetime ACE. Best Glitch% lineage.",
        category="glitch%",
        tags=("glitch", "gt-code", "ace", "full"),
        full_slice_id="cpadolf_gt_full",
        expected_frames=53_661,
        emulator="lsnes rr2-β17",
    ),
    MovieRef(
        filename="nymx_sniq_sporespawn_4481S.lsmv",
        url=f"{_TV}/4481S?handler=Download",
        kind="lsmv",
        postprocess="unwrap_nested_lsmv",
        source="nymx & Sniq Spore Spawn playground #4481S",
        notes="Isolated Spore Spawn (skipped in any%/low%/100%). Walljump catalog.",
        category="spore spawn",
        tags=("boss", "spore-spawn", "playground", "full"),
        full_slice_id="nymx_sporespawn_full",
        expected_frames=33_342,
        emulator="lsnes rr2-b23",
    ),
    MovieRef(
        filename="nymx_ed_100map_5110S.bk2",
        url=f"{_TV}/5110S?handler=Download",
        kind="bk2",
        source="nymx & EternisedDragon 100% map #5110S",
        notes="Spike spark, flash suit, boomerang. Zip-wrapped BKM parsed as BK2.",
        category="100% map",
        tags=("100%", "map", "flash-suit", "spike-spark", "playground", "full"),
        full_slice_id="nymx_100map_full",
        expected_frames=261_130,
        emulator="BizHawk 1.7.0 (BKM)",
    ),
    MovieRef(
        filename="sniq_geg_3768M.lsmv",
        url=f"{_TV}/3768M?handler=Download",
        kind="lsmv",
        postprocess="unwrap_nested_lsmv",
        source="Sniq game-end-glitch #3768M (NTSC, console-verified)",
        notes="JU/NTSC ACE (5238M is PAL/EU). Better harness ROM match.",
        category="game end glitch",
        tags=("glitch", "ace", "ntsc", "full"),
        full_slice_id="sniq_geg_ntsc_full",
        expected_frames=24_192,
        emulator="lsnes rr2-β23",
    ),
    MovieRef(
        filename="saturn_low_ice_2202M.smv",
        url=f"{_TV}/2202M?handler=Download",
        kind="smv",
        source="Saturn low% Ice Beam #2202M",
        notes="14% Ice route (no Speed). Frozen-enemy CWJ, ice-only bosses.",
        category="low% ice",
        tags=("low%", "ice", "full"),
        full_slice_id="saturn_low_ice_full",
        expected_frames=153_429,
        emulator="Snes9x 1.43 v9",
    ),
    MovieRef(
        filename="namespoofer_low_speed_2220M.smv",
        url=f"{_TV}/2220M?handler=Download",
        kind="smv",
        source="NameSpoofer low% Speedbooster #2220M",
        notes="14% Speed route (no Ice). Ancestor of 3273 Charge/Speed.",
        category="low% speed",
        tags=("low%", "speed", "full"),
        full_slice_id="namespoofer_low_speed_full",
        expected_frames=159_518,
        emulator="Snes9x 1.43 v9",
    ),
)


SKIPPED: tuple[MovieRef, ...] = (
    MovieRef(
        filename="project_base.bk2",
        url=f"{_UF}/639031328143430101?handler=Download",
        kind="bk2",
        source="jpgtg19 Project Base TAS",
        notes="ROM hack.",
        category="hack",
        vanilla=False,
        fetch=False,
        skip_reason="Project Base ROM hack, not vanilla Super Metroid.",
    ),
    MovieRef(
        filename="ancient_chozo.bk2",
        url=f"{_UF}/638299586796990572?handler=Download",
        kind="bk2",
        source="RaXx Ancient Chozo 104%",
        notes="ROM hack (Project Base descendant).",
        category="hack",
        vanilla=False,
        fetch=False,
        skip_reason="Ancient Chozo hack.",
    ),
    MovieRef(
        filename="moozooh_smtc3.bk2",
        url=f"{_UF}/638470598619477523?handler=Download",
        kind="bk2",
        source="moozooh SM TAS Contest Round 3",
        notes="Map Rando contest.",
        category="contest",
        vanilla=False,
        fetch=False,
        skip_reason="Map Randomizer contest ROM.",
    ),
    MovieRef(
        filename="drakeekard_100.bk2",
        url=f"{_UF}/637823197715887081?handler=Download",
        kind="bk2",
        source="DrakeekarD casual 100%",
        notes="153 rerecords — not TAS-quality skill source.",
        category="casual",
        vanilla=True,
        fetch=False,
        skip_reason="Casual 100% (153 rerecords); Sniq 100% already covers the route.",
    ),
    MovieRef(
        filename="amaurea_ace_test.smv",
        url=f"{_UF}/13214447004475931?handler=Download",
        kind="smv",
        source="amaurea total-control ACE test",
        notes="64-frame ACE payload, not a movement movie.",
        category="ace",
        vanilla=True,
        fetch=False,
        skip_reason="64-frame ACE payload; published GEG movie covers ACE inputs.",
    ),
    MovieRef(
        filename="SuperMetroid-20.wch",
        url=f"{_UF}/637936906469269359?handler=Download",
        kind="bk2",
        source="BizHawk RAM watch",
        notes="Not a movie.",
        category="watch",
        vanilla=False,
        fetch=False,
        skip_reason="RAM watch, not button presses.",
    ),
    MovieRef(
        filename="mission_rescue.smv",
        url=f"{_UF}/40702924990967495?handler=Download",
        kind="smv",
        source="Hoandjzj Mission Rescue",
        notes="ROM hack.",
        category="hack",
        vanilla=False,
        fetch=False,
        skip_reason="Mission Rescue ROM hack.",
    ),
    MovieRef(
        filename="golden_dawn.smv",
        url=f"{_UF}/38718817282369305?handler=Download",
        kind="smv",
        source="Hoandjzj Golden Dawn 100%",
        notes="ROM hack.",
        category="hack",
        vanilla=False,
        fetch=False,
        skip_reason="Golden Dawn ROM hack.",
    ),
    MovieRef(
        filename="cliffhanger.smv",
        url=f"{_UF}/37851263100062324?handler=Download",
        kind="smv",
        source="Hoandjzj Cliffhanger any%",
        notes="ROM hack.",
        category="hack",
        vanilla=False,
        fetch=False,
        skip_reason="Cliffhanger ROM hack.",
    ),
)


def fetchable() -> tuple[MovieRef, ...]:
    return tuple(m for m in MOVIES if m.fetch)


def vanilla_fetchable() -> tuple[MovieRef, ...]:
    return tuple(m for m in MOVIES if m.fetch and m.vanilla)


def by_filename(name: str) -> MovieRef:
    for movie in MOVIES:
        if movie.filename == name:
            return movie
    raise KeyError(name)


def catalog_full_slice_ids() -> list[str]:
    return [m.full_slice_id for m in MOVIES if m.full_slice_id]
