"""Multi-seed opening tip (house→uncle) S/T campaign for ALTTP Rando (rr-gbd.26).

Thin consumer of :class:`retro_harness.seed_campaign.SeedCampaignRunner`.
Published seeds are offline fixture packages under ``seeds/``; the integration
ROM is the documented **JP 1.0 vanilla FirstPlay** substrate until a real
ALTTPR generator/patch lands. Path is seed-agnostic (no spoiler oracle).

Modes
-----
* ``dry`` (default): audited synthetic envs for harness S/T dry-run. Fixture
  seeds with uncle sword at Link's Uncle (standard mode) clear; used for CI
  and the published dry-run report.
* ``live``: one real ``ALTTPRando-Snes`` house→uncle tip per seed
  (FirstPlay → fighter sword). Missing ROM/setup is fail-closed
  ``INFRA_ERROR`` (non-claimable).

Substrate honesty
-----------------
Fixture packages are **not** shuffled multi-seed ROMs. The dry report is
seed-abstract evidence for the *harness + tip contract*, labeled
``substrate=vanilla`` / ``seed_source=fixture`` (JP vanilla FirstPlay demo
substrate). Not shuffled-seed robustness until a generator/patch is wired
per seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from retro_harness.audit import (
    AuditCapabilities,
    InterventionClass,
    RuntimeObservationClass,
)
from retro_harness.benchmark import (
    BenchmarkCase,
    BenchmarkTier,
    EvaluationContract,
    PolicyIdentity,
    SeedCampaignResult,
    SeedCampaignRunner,
    SeedRobustnessConfig,
    SeedValue,
    StartIdentity,
    write_seed_robustness_report,
    zero_action_for_env,
)
from retro_harness.identity import sha256_file
from retro_harness.seed_campaign import SeedCampaignInfraError
from alttp_rando.paths import (
    FIRST_PLAY_STATE,
    GAME,
    INTEGRATION_DIR,
    RECORDINGS_DIR,
    SEEDS_DIR,
    SHARED_Z3_JP_ROM,
)
from alttp_rando.seed import SeedPackage, write_fixture_seed
from alttp_rando.solver_bindings import HOUSE_TO_UNCLE_SPEC

CampaignMode = Literal["dry", "live"]

# Published campaign contract (deterministic dry-run defaults).
DEFAULT_SEEDS: tuple[str, ...] = ("1337", "1338", "1339")
DEFAULT_SUCCESS_THRESHOLD = 2
# Live house→uncle ~3,662 frames on retained natural_entry evidence; headroom.
DEFAULT_LIVE_BUDGET = 15_000
# Dry-run uses a short budget so the harness path is cheap in CI.
DEFAULT_DRY_BUDGET = 8
DEFAULT_GENERATOR = "alttp_rando.fixture"
DEFAULT_GENERATOR_VERSION = "1"
DEFAULT_LOGIC = "noglitches"
DEFAULT_GOAL = "house_to_uncle"

POLICY_NAME = HOUSE_TO_UNCLE_SPEC.skill_id

CAMPAIGN_REPORT = RECORDINGS_DIR / "opening_tip_seed_campaign.json"
CAMPAIGN_LEDGER = RECORDINGS_DIR / "opening_tip_seed_campaign.ledger.json"
CLASSIC_REPORT = RECORDINGS_DIR / "opening_tip_seed_robustness.json"
# Committed published dry artifact (recordings/ is gitignored).
DOCS_PUBLISHED_REPORT = Path(__file__).resolve().parent / "docs" / "opening_tip_seed_campaign_dry.json"

# Retained house→uncle natural_entry frames (product evidence) for dry report realism.
HOUSE_TO_UNCLE_BASELINE_FRAMES = 3_662
TERMINAL_MILESTONE = "fighter_sword"

_SWORD_ITEMS = frozenset(
    {
        "progressive sword",
        "fighter sword",
        "master sword",
        "tempered sword",
        "golden sword",
        "sword",
    }
)


class HouseToUncleCampaignPolicy:
    """Display policy for the seed-agnostic house→uncle opening tip.

    Dry mode: idle actions while the audited synthetic env advances.
    Live mode: tip execution happens inside the env factory; act is idle.
    """

    name = POLICY_NAME

    def reset(self, env: Any, case: BenchmarkCase) -> None:
        del env, case

    def act(self, obs: Any, info: dict[str, Any], env: Any, case: BenchmarkCase) -> Any:
        del obs, info, case
        return zero_action_for_env(env)


@dataclass(frozen=True)
class OpeningTipCampaignSpec:
    """Resolved campaign knobs."""

    seeds: tuple[str, ...]
    success_threshold: int
    budget: int
    mode: CampaignMode
    seeds_root: Path
    ledger_path: Path
    report_path: Path
    classic_report_path: Path | None
    spoiler_oracle: bool = False


class _DiscreteActionSpace:
    n = 12

    def sample(self) -> int:
        return 0


class _AuditedTipEnv:
    """Minimal audited env that reaches a terminal milestone after N steps."""

    def __init__(
        self,
        *,
        success_after: int | None,
        terminal_milestone: str | None,
        failure_mode: str | None = None,
        audit_provider: str = "alttp_rando.opening_tip_campaign.dry",
    ) -> None:
        self.success_after = success_after
        self.terminal_milestone = terminal_milestone
        self.failure_mode = failure_mode
        self.action_space = _DiscreteActionSpace()
        self.step_count = 0
        self._audit = {
            "ram_writes": 0,
            "mid_run_loads": 0,
            "assists": {},
            "audit_capabilities": AuditCapabilities.all(audit_provider).to_record(),
        }

    def _info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "count": self.step_count,
            **self._audit,
        }
        if self.terminal_milestone is not None:
            info["terminal_milestone"] = self.terminal_milestone
        if self.failure_mode is not None and (
            self.success_after is None or self.step_count >= (self.success_after or 0)
        ):
            info["failure_mode"] = self.failure_mode
        return info

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        del args, kwargs
        self.step_count = 0
        return np.zeros((2, 2), dtype=np.uint8), self._info()

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        del action
        self.step_count += 1
        info = self._info()
        if self.success_after is not None and self.step_count >= self.success_after:
            info["terminal_milestone"] = self.terminal_milestone or TERMINAL_MILESTONE
            return np.zeros((2, 2), dtype=np.uint8), 1.0, True, False, info
        return np.zeros((2, 2), dtype=np.uint8), 0.0, False, False, info

    def close(self) -> None:
        return None


def fixture_seed_dir(seed: SeedValue, seeds_root: Path = SEEDS_DIR) -> Path:
    return Path(seeds_root) / f"fixture_{seed}"


def ensure_campaign_seed(
    seed: SeedValue,
    *,
    seeds_root: Path = SEEDS_DIR,
) -> SeedPackage:
    """Load or write a deterministic offline fixture package for ``seed``."""
    directory = fixture_seed_dir(seed, seeds_root)
    if (directory / "meta.json").is_file():
        return SeedPackage.load(directory)
    return write_fixture_seed(
        seed_number=str(seed),
        name=f"fixture_{seed}",
        directory=directory,
    )


def ensure_campaign_seeds(
    seeds: Sequence[SeedValue],
    *,
    seeds_root: Path = SEEDS_DIR,
) -> tuple[SeedPackage, ...]:
    return tuple(ensure_campaign_seed(seed, seeds_root=seeds_root) for seed in seeds)


def opening_clearable_without_spoiler(package: SeedPackage) -> bool:
    """Return True when the seed-agnostic house→uncle tip is expected to clear.

    Does **not** consult the spoiler log for routing. Fixture packages place
    Progressive Sword at Link's Uncle under standard mode — that is the opening
    tip precondition. Explicit ``opening_clearable: false`` meta opts out.
    """
    if package.meta.get("opening_clearable") is False:
        return False
    settings = package.settings or {}
    mode = str(settings.get("mode", "standard")).strip().casefold()
    if mode not in ("standard", "vanilla", ""):
        return False
    for loc in package.locations:
        if str(loc.get("location", "")).strip().casefold() != "link's uncle":
            continue
        item = str(loc.get("item", "")).strip().casefold()
        if item in _SWORD_ITEMS or "sword" in item:
            return True
    return False


def default_spec(
    *,
    mode: CampaignMode = "dry",
    seeds: Sequence[SeedValue] | None = None,
    success_threshold: int | None = None,
    budget: int | None = None,
    seeds_root: Path | None = None,
    ledger_path: Path | None = None,
    report_path: Path | None = None,
    classic_report_path: Path | None = None,
    include_classic_report: bool = True,
) -> OpeningTipCampaignSpec:
    seed_tuple = tuple(str(s) for s in (seeds if seeds is not None else DEFAULT_SEEDS))
    if budget is None:
        budget = DEFAULT_DRY_BUDGET if mode == "dry" else DEFAULT_LIVE_BUDGET
    if success_threshold is None:
        success_threshold = min(DEFAULT_SUCCESS_THRESHOLD, len(seed_tuple))
    if include_classic_report:
        classic_path: Path | None = Path(
            classic_report_path if classic_report_path is not None else CLASSIC_REPORT
        )
    else:
        classic_path = None
    return OpeningTipCampaignSpec(
        seeds=seed_tuple,
        success_threshold=success_threshold,
        budget=int(budget),
        mode=mode,
        seeds_root=Path(seeds_root or SEEDS_DIR),
        ledger_path=Path(ledger_path or CAMPAIGN_LEDGER),
        report_path=Path(report_path or CAMPAIGN_REPORT),
        classic_report_path=classic_path,
        spoiler_oracle=False,
    )


def build_campaign_config(spec: OpeningTipCampaignSpec) -> SeedRobustnessConfig:
    """Published S/T contract for the opening tip campaign."""
    rom_sha = None
    if SHARED_Z3_JP_ROM.is_file():
        try:
            rom_sha = sha256_file(SHARED_Z3_JP_ROM)
        except OSError:
            rom_sha = None

    metadata: dict[str, Any] = {
        "package": "alttp_rando",
        "edge": "house_to_uncle",
        "tip": "fighter_sword",
        "mode": spec.mode,
        "substrate": "vanilla",
        "spoiler_oracle": False,
        "seed_source": "fixture",
        "skill_id": POLICY_NAME,
        "rom_variant": "japanese_1.0",
        "start_state": FIRST_PLAY_STATE,
        "note": (
            "Fixture seeds on the documented JP 1.0 vanilla FirstPlay substrate. "
            "Uncle sword at Link's Uncle (standard mode) is the seed-agnostic "
            "opening tip precondition. Path does not consult a spoiler oracle. "
            "Not shuffled-seed robustness until an ALTTPR generator/patch is wired."
        ),
    }
    if rom_sha is not None:
        metadata["rom_sha256"] = rom_sha

    return SeedRobustnessConfig(
        generator=DEFAULT_GENERATOR,
        generator_version=DEFAULT_GENERATOR_VERSION,
        logic=DEFAULT_LOGIC,
        goal=DEFAULT_GOAL,
        seeds=spec.seeds,
        budget=spec.budget,
        success_threshold=spec.success_threshold,
        runtime_observation_class=RuntimeObservationClass.BRONZE,
        intervention_class=InterventionClass.CLEAN,
        metadata=metadata,
    )


def _start_identity_for_seed(seed: SeedValue, package: SeedPackage) -> StartIdentity:
    rom_sha = None
    if SHARED_Z3_JP_ROM.is_file():
        try:
            rom_sha = sha256_file(SHARED_Z3_JP_ROM)
        except OSError:
            rom_sha = None
    return StartIdentity(
        f"first_play_fixture_{seed}",
        rom_sha256=rom_sha,
        metadata={
            "seed_number": package.seed_number,
            "seed_name": package.name,
            "source": package.source,
            "integration": GAME,
            "start_state": FIRST_PLAY_STATE,
        },
    )


def _is_success(info: dict[str, Any], terminated: bool, truncated: bool) -> bool:
    del truncated
    if info.get("tip_success") is True:
        return True
    if terminated and info.get("terminal_milestone") == TERMINAL_MILESTONE:
        return True
    return False


def _require_live_rom() -> Path:
    integration_rom = INTEGRATION_DIR / "rom.sfc"
    if not integration_rom.is_file():
        raise SeedCampaignInfraError(
            "ALTTPRando-Snes ROM is not configured; run "
            "`uv run python -m alttp_rando.scripts.setup_rom`"
        )
    if not SHARED_Z3_JP_ROM.is_file() and not integration_rom.is_file():
        raise SeedCampaignInfraError(
            "ALttP JP 1.0 ROM missing for live multi-seed opening tip"
        )
    first_play = INTEGRATION_DIR / f"{FIRST_PLAY_STATE}.state"
    if not first_play.is_file():
        raise SeedCampaignInfraError(
            f"{FIRST_PLAY_STATE}.state missing; run "
            "`SDL_VIDEODRIVER=dummy uv run python -m alttp_rando.scripts.make_boot`"
        )
    return integration_rom


def _live_tip_result(seed: SeedValue, *, report_dir: Path) -> Any:
    """Run the real FirstPlay → uncle fighter sword tip once for a seed."""
    from alttp_rando.house_to_uncle import run_house_to_uncle_from_first_play

    _require_live_rom()
    report_dir.mkdir(parents=True, exist_ok=True)
    return run_house_to_uncle_from_first_play(
        report_path=report_dir / f"live_tip_{seed}.json",
        close=True,
    )


def build_case_for_seed(
    seed: SeedValue,
    *,
    spec: OpeningTipCampaignSpec,
    config: SeedRobustnessConfig,
) -> BenchmarkCase:
    """Build one seed's benchmark case (fixture setup + success predicate)."""
    package = ensure_campaign_seed(seed, seeds_root=spec.seeds_root)
    start_identity = _start_identity_for_seed(seed, package)
    contract = EvaluationContract(
        runtime_observation_class=RuntimeObservationClass.BRONZE,
        intervention_class=InterventionClass.CLEAN,
        start_identity=start_identity,
        policy_identity=PolicyIdentity("unbound-policy"),
        benchmark_id=f"alttp_rando_opening_tip_{seed}",
        objective=config.goal,
    )

    if spec.mode == "dry":
        clearable = opening_clearable_without_spoiler(package)
        clear_frames = max(1, min(config.budget - 1, 3)) if clearable else None
        failure_mode = None if clearable else "uncle_sword_not_at_vanilla_location"

        def build_env() -> _AuditedTipEnv:
            return _AuditedTipEnv(
                success_after=clear_frames,
                terminal_milestone=TERMINAL_MILESTONE if clearable else "links_house",
                failure_mode=failure_mode,
            )

        notes = (
            f"dry fixture seed {seed}; clearable={clearable}; no spoiler oracle"
        )
    else:

        def build_env() -> _AuditedTipEnv:
            try:
                tip = _live_tip_result(
                    seed,
                    report_dir=spec.report_path.parent / "opening_tip_live",
                )
            except SeedCampaignInfraError:
                raise
            except FileNotFoundError as exc:
                raise SeedCampaignInfraError(str(exc)) from exc
            except OSError as exc:
                raise SeedCampaignInfraError(f"OSError: {exc}") from exc
            except RuntimeError as exc:
                raise SeedCampaignInfraError(f"RuntimeError: {exc}") from exc

            success = bool(getattr(tip, "success", False)) and (
                str(getattr(tip, "outcome", "")) == "fighter_sword_acquired"
                or bool(
                    getattr(getattr(tip, "segment", None), "snapshot", None)
                    and getattr(tip.segment.snapshot, "has_fighter_sword", False)
                )
            )
            frames = int(getattr(tip, "total_frames", 0) or 0)
            if frames <= 0:
                frames = config.budget
            frames = min(frames, config.budget)
            if success:
                return _AuditedTipEnv(
                    success_after=max(1, frames),
                    terminal_milestone=TERMINAL_MILESTONE,
                    audit_provider="alttp_rando.opening_tip_campaign.live",
                )
            outcome = str(getattr(tip, "outcome", "tip_failed"))
            return _AuditedTipEnv(
                success_after=None,
                terminal_milestone=outcome[:64],
                failure_mode=outcome[:128],
                audit_provider="alttp_rando.opening_tip_campaign.live",
            )

        notes = f"live ALTTPRando-Snes house→uncle tip for fixture seed {seed}"

    return BenchmarkCase(
        benchmark_id=f"alttp_rando_opening_tip_{seed}",
        display_name=f"ALTTP Rando opening tip seed {seed}",
        game=GAME,
        start_state=f"first_play_fixture_{seed}",
        tier=BenchmarkTier.BRONZE,
        objective=config.goal,
        max_steps=config.budget,
        build_env=build_env,
        is_success=_is_success,
        stop_on_success=True,
        tags=("alttp_rando", "opening_tip", "house_to_uncle", spec.mode),
        notes=notes,
        metadata={
            "seed": str(seed),
            "seed_package": str(package.directory),
            "mode": spec.mode,
            "spoiler_oracle": False,
        },
        contract=contract,
    )


def policy_factory(_seed: SeedValue) -> HouseToUncleCampaignPolicy:
    """Fresh policy instance per seed (SeedCampaignRunner contract)."""
    return HouseToUncleCampaignPolicy()


def run_opening_tip_campaign(
    *,
    mode: CampaignMode = "dry",
    seeds: Sequence[SeedValue] | None = None,
    success_threshold: int | None = None,
    budget: int | None = None,
    seeds_root: Path | None = None,
    ledger_path: Path | None = None,
    report_path: Path | None = None,
    classic_report_path: Path | None = None,
    write_classic: bool = True,
    publish_docs_report: bool = False,
) -> SeedCampaignResult:
    """Run the multi-seed opening tip campaign and write package-owned reports."""
    spec = default_spec(
        mode=mode,
        seeds=seeds,
        success_threshold=success_threshold,
        budget=budget,
        seeds_root=seeds_root,
        ledger_path=ledger_path,
        report_path=report_path,
        classic_report_path=classic_report_path,
        include_classic_report=write_classic,
    )
    ensure_campaign_seeds(spec.seeds, seeds_root=spec.seeds_root)
    config = build_campaign_config(spec)

    def build_case(seed: SeedValue) -> BenchmarkCase:
        return build_case_for_seed(seed, spec=spec, config=config)

    result = SeedCampaignRunner(
        config=config,
        build_case=build_case,
        policy_factory=policy_factory,
        ledger_path=spec.ledger_path,
        report_path=spec.report_path,
        stop_on_infra_error=False,
    ).run()

    if write_classic and result.claimable and spec.classic_report_path is not None:
        classic = result.to_seed_robustness_report()
        write_seed_robustness_report(spec.classic_report_path, classic)

    if publish_docs_report and result.claimable:
        from retro_harness.seed_campaign import atomic_write_json

        atomic_write_json(DOCS_PUBLISHED_REPORT, result.to_record())

    return result


def campaign_summary(result: SeedCampaignResult) -> dict[str, Any]:
    return {
        "claimable": result.claimable,
        "threshold_met": result.threshold_met,
        "successes": result.successes,
        "seeds_total": result.config.seed_count,
        "required_successes": result.config.success_threshold,
        "infra_errors": result.infra_error_count,
        "success_rate": result.success_rate,
        "report_path": str(result.report_path) if result.report_path else None,
        "ledger_path": str(result.ledger_path) if result.ledger_path else None,
        "mode": (result.config.metadata or {}).get("mode"),
        "spoiler_oracle": (result.config.metadata or {}).get("spoiler_oracle", False),
        "substrate": (result.config.metadata or {}).get("substrate"),
    }


__all__ = [
    "CAMPAIGN_LEDGER",
    "CAMPAIGN_REPORT",
    "CLASSIC_REPORT",
    "DEFAULT_DRY_BUDGET",
    "DEFAULT_LIVE_BUDGET",
    "DEFAULT_SEEDS",
    "DEFAULT_SUCCESS_THRESHOLD",
    "DOCS_PUBLISHED_REPORT",
    "HOUSE_TO_UNCLE_BASELINE_FRAMES",
    "HouseToUncleCampaignPolicy",
    "OpeningTipCampaignSpec",
    "POLICY_NAME",
    "TERMINAL_MILESTONE",
    "build_campaign_config",
    "build_case_for_seed",
    "campaign_summary",
    "default_spec",
    "ensure_campaign_seed",
    "ensure_campaign_seeds",
    "opening_clearable_without_spoiler",
    "policy_factory",
    "run_opening_tip_campaign",
]
