"""Continuous-like pure entry source catalog (code twin of SOURCE_STATES.md).

Pure geometry cards must load a named state whose room (and optional pose/x/y
band) matches the hop. This module validates fingerprints and suggests sources
so executors spend less time hand-picking scratch states.

Binary states live under ``custom_integrations/SuperMetroid-Snes/`` (often
gitignored). The **index** here is source of truth for room expectations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from super_metroid.paths import INTEGRATION_DIR
from super_metroid.ram import SuperMetroidState, probe_pin


@dataclass(frozen=True)
class SourceFingerprint:
    """Expected live RAM fingerprint for a pure entry state."""

    source_id: str
    """Stable id (matches SOURCE_STATES.md row)."""

    relative_path: str
    """Path under SuperMetroid-Snes/ (e.g. ``scratch/post_varia_collected.state``)."""

    room_id: int
    """Expected ``room_id`` after boot (+ short settle)."""

    use_for: str = ""
    """Segment ids / probe names this source feeds (free text)."""

    x_min: int | None = None
    x_max: int | None = None
    y_min: int | None = None
    y_max: int | None = None
    poses: frozenset[int] | None = None
    """When set, pose must be in this set."""

    continuous_like: bool = True
    """False for dev/topology anchors (not continuous evidence)."""

    @property
    def path(self) -> Path:
        return INTEGRATION_DIR / self.relative_path

    def room_hex(self) -> str:
        return f"0x{self.room_id:04X}"


# Spine / KPDR continuous-like (keep in sync with docs/SOURCE_STATES.md).
SOURCE_CATALOG: tuple[SourceFingerprint, ...] = (
    SourceFingerprint(
        "natural_post_spore_spawn",
        "natural_post_spore_spawn.state",
        0x9B5B,
        use_for="K0–K3 practice; not a continuous tip",
        continuous_like=True,
    ),
    SourceFingerprint(
        "post_varia_collected",
        "scratch/post_varia_collected.state",
        0xA6E2,
        use_for="varia-to-kraid / varia_to_kraid",
    ),
    SourceFingerprint(
        "post_varia_continuous",
        "scratch/post_varia_continuous.state",
        0xA6E2,
        use_for="full-continuous reverse-chain revalidation",
        x_min=100,
        x_max=140,
        y_min=110,
        y_max=145,
        poses=frozenset({81}),
    ),
    SourceFingerprint(
        "post_varia_continuous_to_kraid",
        "scratch/post_varia_continuous_to_kraid.state",
        0xA59F,
        use_for="natural Varia reverse-chain predecessor",
    ),
    SourceFingerprint(
        "post_varia_continuous_to_eye",
        "scratch/post_varia_continuous_to_eye.state",
        0xA56B,
        use_for="natural Varia reverse-chain predecessor",
    ),
    SourceFingerprint(
        "post_varia_continuous_to_baby",
        "scratch/post_varia_continuous_to_baby.state",
        0xA521,
        use_for="natural Varia reverse-chain predecessor",
    ),
    SourceFingerprint(
        "post_varia_to_kraid",
        "scratch/post_varia_to_kraid_pure.state",
        0xA59F,
        use_for="kraid-to-eye-return / kraid_to_eye_return",
    ),
    SourceFingerprint(
        "post_kraid_to_eye",
        "scratch/post_kraid_to_eye_return.state",
        0xA56B,
        use_for="eye-to-baby-return / eye_to_baby_return",
    ),
    SourceFingerprint(
        "post_eye_to_baby",
        "scratch/post_eye_to_baby_return.state",
        0xA521,
        use_for="baby-to-kihunter-return / baby_to_kihunter_return",
    ),
    SourceFingerprint(
        "post_baby_to_kihunter",
        "scratch/post_baby_to_kihunter_return.state",
        0xA4DA,
        use_for="kihunter-to-zeela-return / kihunter_to_zeela_return",
    ),
    SourceFingerprint(
        "post_varia_continuous_to_kihunter",
        "scratch/post_varia_continuous_to_kihunter.state",
        0xA4DA,
        use_for="kihunter-to-zeela-return natural predecessor revalidation",
        x_min=450,
        x_max=480,
        y_min=390,
        y_max=400,
        poses=frozenset({165}),
    ),
    SourceFingerprint(
        "post_varia_continuous_to_zeela",
        "scratch/post_varia_continuous_to_zeela.state",
        0xA471,
        use_for="zeela-to-warehouse-return natural predecessor revalidation",
    ),
    SourceFingerprint(
        "post_varia_continuous_to_warehouse",
        "scratch/post_varia_continuous_to_warehouse.state",
        0xA6A1,
        use_for="warehouse-to-business natural predecessor revalidation",
    ),
    SourceFingerprint(
        "post_business_continuous",
        "scratch/post_business_continuous.state",
        0xA7DE,
        use_for="business-to-frog-save / business-to-ice-gate natural continuous predecessor",
    ),
    SourceFingerprint(
        "post_business_to_ice_gate_pure",
        "scratch/post_business_to_ice_gate_pure.state",
        0xA815,
        use_for="Ice Gate pin from pre-Speed Business (geometry only; Gate→Acid needs Speed)",
        continuous_like=False,
        x_min=1700,
        x_max=1800,
        y_min=620,
        y_max=680,
        poses=frozenset({1, 2, 9, 10}),
    ),
    SourceFingerprint(
        "post_business_to_ice_gate_wave_speed_pure",
        "scratch/post_business_to_ice_gate_wave_speed_pure.state",
        0xA815,
        use_for="ice-gate-to-acid pure predecessor (Wave+Speed loadout; rr-9t4)",
        continuous_like=False,
        x_min=1700,
        x_max=1800,
        y_min=620,
        y_max=680,
        poses=frozenset({1, 2, 9, 10}),
    ),
    SourceFingerprint(
        "post_ice_gate_to_acid_pure",
        "scratch/post_ice_gate_to_acid_pure.state",
        0xA75D,
        use_for="acid-to-snake pure predecessor (rr-9t4 dual handoff)",
        continuous_like=False,
        x_min=400,
        x_max=520,
        y_min=120,
        y_max=160,
        poses=frozenset({1, 2, 9, 10}),
    ),
    SourceFingerprint(
        "post_ice_acid_to_snake_pure",
        "scratch/post_ice_acid_to_snake_pure.state",
        0xA8B9,
        use_for="snake-to-ice pure predecessor (rr-5cf dual handoff)",
        continuous_like=False,
        # Room-clear / pure settle is mid-shaft ~y650 (multi-screen Snake).
        x_min=80,
        x_max=250,
        y_min=600,
        y_max=720,
        poses=frozenset({1, 2, 9, 10}),
    ),
    SourceFingerprint(
        "post_ice_snake_to_ice_pure",
        "scratch/post_ice_snake_to_ice_pure.state",
        0xA890,
        use_for="Ice PLM dual pure handoff (rr-5if, beams include Ice 0x0002)",
        continuous_like=False,
        x_min=160,
        x_max=220,
        y_min=100,
        y_max=160,
        poses=frozenset({1, 2, 9, 10, 75, 77, 81}),
    ),
    SourceFingerprint(
        "post_frog_continuous",
        "scratch/post_frog_continuous.state",
        0xB167,
        use_for="frog-save-to-speedway natural continuous predecessor",
        x_min=50,
        x_max=70,
        y_min=130,
        y_max=145,
        poses=frozenset({1}),
    ),
    SourceFingerprint(
        "post_business_to_frog_save_pure",
        "scratch/post_business_to_frog_save_pure.state",
        0xB167,
        use_for="record of the Business-to-Frog pure handoff; superseded by post_frog_continuous",
        x_min=50,
        x_max=70,
        y_min=130,
        y_max=145,
        poses=frozenset({1}),
    ),
    SourceFingerprint(
        "post_bat_cave_to_speed_hall_pure",
        "scratch/post_bat_cave_to_speed_hall_pure.state",
        0xACF0,
        use_for="speed-hall-to-speed pure predecessor (Bat→Hall GREEN successor)",
        continuous_like=False,
    ),
    SourceFingerprint(
        "post_speed_hall_to_speed_pure",
        "scratch/post_speed_hall_to_speed_pure.state",
        0xAD1B,
        use_for="post Speed Booster pure collect; human Wave/Ice/Moat record",
        continuous_like=False,
        x_min=150,
        x_max=200,
        y_min=100,
        y_max=150,
    ),
    SourceFingerprint(
        "post_speed_collected",
        "scratch/post_speed_collected.state",
        0xAD1B,
        use_for="speed-return-to-bubble pure / human Wave/Ice/Moat record (standing handoff)",
        continuous_like=False,
        x_min=150,
        x_max=200,
        y_min=100,
        y_max=150,
    ),
    SourceFingerprint(
        "post_speed_return_to_bubble_pure",
        "scratch/post_speed_return_to_bubble_pure.state",
        0xACB3,
        use_for="bubble-to-single-chamber pure / Wave branch (post Speed return) predecessor",
        continuous_like=False,
        x_min=450,
        x_max=500,
        y_min=90,
        y_max=160,
    ),
    SourceFingerprint(
        "post_bubble_to_single_chamber_pure",
        "scratch/post_bubble_to_single_chamber_pure.state",
        0xAD5E,
        use_for="single-to-double-chamber pure / Wave branch predecessor",
        continuous_like=False,
        x_min=20,
        x_max=80,
        y_min=100,
        y_max=180,
    ),
    SourceFingerprint(
        "post_single_to_double_chamber_pure",
        "scratch/post_single_to_double_chamber_pure.state",
        0xADAD,
        use_for="double-chamber-to-wave / Wave PLM pure predecessor",
        continuous_like=False,
        x_min=20,
        x_max=80,
        y_min=100,
        y_max=180,
    ),
    SourceFingerprint(
        "post_double_chamber_to_wave_pure",
        "scratch/post_double_chamber_to_wave_pure.state",
        0xADDE,
        use_for=(
            "Wave pure successor / Ice branch predecessor "
            "(STALE 2026-08-09: loads 0xADAD ~(923,311) — use dev_wave_collected)"
        ),
        continuous_like=False,
        x_min=140,
        x_max=220,
        y_min=90,
        y_max=180,
    ),
    SourceFingerprint(
        "dev_wave_collected",
        "scratch/dev_wave_collected.state",
        0xADDE,
        use_for="wave-to-double-chamber pure return / Wave tip handoff (chozo pin)",
        continuous_like=False,
        x_min=150,
        x_max=200,
        y_min=90,
        y_max=150,
    ),
    SourceFingerprint(
        "post_wave_to_double_chamber_pure",
        "scratch/post_wave_to_double_chamber_pure.state",
        0xADAD,
        use_for="double-to-single return pure predecessor (Wave return stack)",
        continuous_like=False,
        x_min=900,
        x_max=1050,
        y_min=100,
        y_max=180,
    ),
    SourceFingerprint(
        "post_double_to_single_chamber_pure",
        "scratch/post_double_to_single_chamber_pure.state",
        0xAD5E,
        use_for="single-to-bubble return pure predecessor (Wave return stack)",
        continuous_like=False,
        # Live dual pin (rr-qpkd): ~(216,630) after bottom-left door settle.
        x_min=180,
        x_max=260,
        y_min=580,
        y_max=680,
    ),
    SourceFingerprint(
        "post_single_to_bubble_pure",
        "scratch/post_single_to_bubble_pure.state",
        0xACB3,
        use_for="bubble-to-farm return pure predecessor (Wave return stack)",
        continuous_like=False,
        x_min=400,
        x_max=560,
        y_min=350,
        y_max=450,
    ),
    SourceFingerprint(
        "post_bubble_to_farm_pure",
        "scratch/post_bubble_to_farm_pure.state",
        0xAF72,
        use_for="farm-to-speedway return pure predecessor (Wave return stack; needs Speed)",
        continuous_like=False,
        # Farm right-top settle after Bubble bottom-left leave ~(472–523,139).
        x_min=400,
        x_max=560,
        y_min=100,
        y_max=180,
    ),
    SourceFingerprint(
        "post_kihunter_to_zeela",
        "scratch/post_kihunter_to_zeela_return.state",
        0xA471,
        use_for="zeela-to-warehouse-return / zeela_to_warehouse_return",
    ),
    SourceFingerprint(
        "post_zeela_to_warehouse",
        "scratch/post_zeela_to_warehouse_return.state",
        0xA6A1,
        use_for="warehouse-to-business reverse / warehouse_to_business",
    ),
    SourceFingerprint(
        "business_climb_entry",
        "scratch/continuous_like_business_climb_entry.state",
        0xA7DE,
        use_for="business-to-warehouse",
    ),
    SourceFingerprint(
        "continuous_like_bat",
        "scratch/continuous_like_bat.state",
        0xA3DD,
        use_for="bat pure / dwell isolation",
    ),
    SourceFingerprint(
        "post_below_spazer_with_charge_continuous",
        "scratch/post_below_spazer_with_charge_continuous.state",
        0xA408,
        use_for="Spazer climb pure / below_spazer continuous with Charge",
        continuous_like=True,
        x_min=40,
        x_max=70,
        y_min=380,
        y_max=410,
    ),
    SourceFingerprint(
        "post_warehouse_with_spazer_continuous",
        "scratch/post_warehouse_with_spazer_continuous.state",
        0xA6A1,
        use_for=(
            "warehouse/hijump pure with Charge+Spazer; continuous-like "
            "Charge pin + pure Spazer detour + West→Warehouse"
        ),
        continuous_like=True,
        x_min=30,
        x_max=80,
        y_min=100,
        y_max=150,
    ),
    SourceFingerprint(
        "post_speed_hall_pre_speed_with_spazer",
        "scratch/post_speed_hall_pre_speed_with_spazer.state",
        0xACF0,
        use_for="speed-hall-to-speed / human pre-Speed with Charge+Spazer",
        continuous_like=False,  # beams OR'd onto pure Hall geometry
        x_min=40,
        x_max=90,
        y_min=110,
        y_max=150,
    ),
    SourceFingerprint(
        "pre_spazer_door_with_charge",
        "scratch/pre_spazer_door_with_charge.state",
        0xA408,
        use_for="below-spazer-to-spazer / below_spazer_to_spazer",
        continuous_like=False,  # geometry place; inventory continuous-legal
        x_min=440,
        x_max=490,
        y_min=120,
        y_max=160,
    ),
    SourceFingerprint(
        "post_spazer_entry_pure",
        "scratch/post_spazer_entry_pure.state",
        0xA447,
        use_for="spazer-collect / spazer_collect",
    ),
    SourceFingerprint(
        "post_spazer_collect_pure",
        "scratch/post_spazer_collect_pure.state",
        0xA447,
        use_for="spazer-return-to-below / spazer_return_to_below",
        x_min=150,
        x_max=200,
        y_min=150,
        y_max=200,
    ),
    SourceFingerprint(
        "post_spazer_return_pure",
        "scratch/post_spazer_return_pure.state",
        0xA408,
        use_for="spazer-top-to-west / spazer_top_to_west / top→mid→West",
        x_min=350,
        x_max=420,
        y_min=140,
        y_max=170,
    ),
    SourceFingerprint(
        "post_spazer_west_pure",
        "scratch/post_spazer_west_pure.state",
        0xCF54,
        use_for="post Spazer pure West handoff after top→west",
        x_min=20,
        x_max=80,
        y_min=80,
        y_max=160,
    ),
    SourceFingerprint(
        "red_to_warehouse",
        "scratch/red_to_warehouse_controller.state",
        0xA6A1,
        use_for="warehouse-hijump-kraid",
    ),
    # Dev / topology (not continuous evidence)
    SourceFingerprint(
        "dev_kpdr_kraid_entry",
        "dev_kpdr_kraid_entry.state",
        0xA59F,
        use_for="kraid fight pure",
        continuous_like=False,
    ),
    SourceFingerprint(
        "dev_kpdr_varia",
        "dev_kpdr_varia.state",
        0xA6E2,
        use_for="varia probes",
        continuous_like=False,
    ),
)


_BY_ID: dict[str, SourceFingerprint] = {s.source_id: s for s in SOURCE_CATALOG}


def get_source(source_id: str) -> SourceFingerprint:
    """Return catalog row or raise ``KeyError``."""
    return _BY_ID[source_id]


def list_sources(
    *, continuous_like_only: bool = False
) -> tuple[SourceFingerprint, ...]:
    if continuous_like_only:
        return tuple(s for s in SOURCE_CATALOG if s.continuous_like)
    return SOURCE_CATALOG


def suggest_sources_for_room(
    room_id: int,
    *,
    continuous_like_only: bool = True,
    segment_hint: str = "",
) -> tuple[SourceFingerprint, ...]:
    """Rank catalog rows for a pure hop entry room.

    Prefers continuous-like rows whose ``use_for`` mentions ``segment_hint``,
    then any continuous-like match, then dev anchors.
    """
    hint = segment_hint.replace("_", "-").lower()
    matches = [s for s in SOURCE_CATALOG if s.room_id == room_id]
    if continuous_like_only:
        matches = [s for s in matches if s.continuous_like]

    def rank(s: SourceFingerprint) -> tuple[int, int, str]:
        use = s.use_for.replace("_", "-").lower()
        hint_hit = 0 if hint and hint in use else 1
        cont = 0 if s.continuous_like else 1
        return (hint_hit, cont, s.source_id)

    return tuple(sorted(matches, key=rank))


def suggest_source_path(
    room_id: int,
    *,
    segment_hint: str = "",
    continuous_like_only: bool = True,
) -> Path | None:
    """Best catalog path for ``room_id``, or None if unknown."""
    ranked = suggest_sources_for_room(
        room_id,
        continuous_like_only=continuous_like_only,
        segment_hint=segment_hint,
    )
    return ranked[0].path if ranked else None


@dataclass(frozen=True)
class FingerprintCheck:
    ok: bool
    failures: tuple[str, ...]
    pin: dict[str, object]
    source_id: str | None = None


def validate_fingerprint(
    state: SuperMetroidState,
    *,
    expected_room: int | None = None,
    source: SourceFingerprint | None = None,
    source_id: str | None = None,
) -> FingerprintCheck:
    """Validate live state against room / optional catalog fingerprint bands."""
    fp = source
    if fp is None and source_id is not None:
        fp = get_source(source_id)
    room = expected_room if expected_room is not None else (fp.room_id if fp else None)
    failures: list[str] = []
    if room is not None and state.room_id != room:
        failures.append(f"room: expected 0x{room:04X}, got 0x{state.room_id:04X}")
    if fp is not None:
        if fp.x_min is not None and state.samus_x < fp.x_min:
            failures.append(f"x<{fp.x_min}: got {state.samus_x}")
        if fp.x_max is not None and state.samus_x > fp.x_max:
            failures.append(f"x>{fp.x_max}: got {state.samus_x}")
        if fp.y_min is not None and state.samus_y < fp.y_min:
            failures.append(f"y<{fp.y_min}: got {state.samus_y}")
        if fp.y_max is not None and state.samus_y > fp.y_max:
            failures.append(f"y>{fp.y_max}: got {state.samus_y}")
        if fp.poses is not None and state.pose not in fp.poses:
            failures.append(f"pose {state.pose} not in {sorted(fp.poses)}")
    return FingerprintCheck(
        ok=not failures,
        failures=tuple(failures),
        pin=probe_pin(state),
        source_id=fp.source_id if fp else None,
    )


def match_source_by_path(path: Path) -> SourceFingerprint | None:
    """Resolve a filesystem path to a catalog row when possible."""
    resolved = path.resolve()
    for row in SOURCE_CATALOG:
        try:
            if row.path.resolve() == resolved:
                return row
        except OSError:
            continue
        # Also match by suffix relative path string.
        if path.as_posix().endswith(row.relative_path):
            return row
        if path.name == Path(row.relative_path).name:
            # Ambiguous name-only match only if unique.
            same = [
                s for s in SOURCE_CATALOG if Path(s.relative_path).name == path.name
            ]
            if len(same) == 1:
                return same[0]
    return None
