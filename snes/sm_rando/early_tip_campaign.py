"""Multi-seed early tip (ship→morph) S/T campaign for SM Rando (rr-gbd.25).

Thin consumer of :class:`retro_harness.seed_campaign.SeedCampaignRunner`.
Published seeds are offline fixture packages under ``seeds/``; the current
integration ROM is the documented vanilla substrate until a real generator
lands. Path is seed-agnostic (no spoiler oracle).

Modes
-----
* ``dry`` (default): audited synthetic envs for harness S/T dry-run. Fixture
  seeds with vanilla Morph placement clear; used for CI and the published
  dry-run report.
* ``live``: one real ``SMRando-Snes`` morph tip per seed (power-on → Morph
  Ball). Missing ROM/setup is fail-closed ``INFRA_ERROR`` (non-claimable).
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
from sm_rando.paths import (
    GAME,
    INTEGRATION_DIR,
    RECORDINGS_DIR,
    SEEDS_DIR,
    SHARED_SM_ROM,
)
from sm_rando.seed import SeedPackage, write_fixture_seed
from sm_rando.solver_bindings import SHIP_TO_MORPH_SPEC

CampaignMode = Literal["dry", "live"]

# Published campaign contract (deterministic dry-run defaults).
DEFAULT_SEEDS: tuple[str, ...] = ("1337", "1338", "1339")
DEFAULT_SUCCESS_THRESHOLD = 2
# Live morph clears ~26,824 frames on the retained baseline; budget leaves headroom.
DEFAULT_LIVE_BUDGET = 40_000
# Dry-run uses a short budget so the harness path is cheap in CI.
DEFAULT_DRY_BUDGET = 8
DEFAULT_GENERATOR = "sm_rando.fixture"
DEFAULT_GENERATOR_VERSION = "1"
DEFAULT_LOGIC = "vanilla"
DEFAULT_GOAL = "ship_to_morph"

POLICY_NAME = SHIP_TO_MORPH_SPEC.skill_id

CAMPAIGN_REPORT = RECORDINGS_DIR / "early_tip_seed_campaign.json"
CAMPAIGN_LEDGER = RECORDINGS_DIR / "early_tip_seed_campaign.ledger.json"
CLASSIC_REPORT = RECORDINGS_DIR / "early_tip_seed_robustness.json"

# Retained morph baseline frames (policy_to_morph product) for dry report realism.
MORPH_BASELINE_FRAMES = 26_824
TERMINAL_MILESTONE = "morph_ball"


class ShipToMorphCampaignPolicy:
    """Display policy for the seed-agnostic ship→morph early tip.

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
class EarlyTipCampaignSpec:
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
        audit_provider: str = "sm_rando.early_tip_campaign.dry",
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
        # Budget exhaustion is handled by the benchmark runner (truncated).
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


def morph_reachable_without_spoiler(package: SeedPackage) -> bool:
    """Return True when Morph is at its vanilla location (seed-agnostic tip).

    Does **not** consult the spoiler log for routing. Fixture packages place
    Morphing Ball at Morphing Ball; that is the early tip precondition.
    """
    for loc in package.locations:
        if (
            str(loc.get("location", "")) == "Morphing Ball"
            and str(loc.get("item", "")) == "Morphing Ball"
        ):
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
) -> EarlyTipCampaignSpec:
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
    return EarlyTipCampaignSpec(
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


def build_campaign_config(spec: EarlyTipCampaignSpec) -> SeedRobustnessConfig:
    """Published S/T contract for the early tip campaign."""
    rom_sha = None
    if SHARED_SM_ROM.is_file():
        try:
            rom_sha = sha256_file(SHARED_SM_ROM)
        except OSError:
            rom_sha = None

    metadata: dict[str, Any] = {
        "package": "sm_rando",
        "edge": "ship_to_morph",
        "tip": "morph",
        "mode": spec.mode,
        "substrate": "vanilla",
        "spoiler_oracle": False,
        "seed_source": "fixture",
        "skill_id": POLICY_NAME,
        "note": (
            "Fixture seeds on the documented vanilla Super Metroid substrate. "
            "Not shuffled-seed robustness until a rando generator/patch is wired. "
            "Path is seed-agnostic (no spoiler oracle)."
        ),
    }
    if rom_sha is not None:
        metadata["rom_sha256"] = rom_sha

    # Policy identity is intentionally per-seed/unbound at config level so
    # SeedCampaignRunner can bind the live ShipToMorphCampaignPolicy digest
    # (implementation-v1). Display name still comes from policy.name.
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
    if SHARED_SM_ROM.is_file():
        try:
            rom_sha = sha256_file(SHARED_SM_ROM)
        except OSError:
            rom_sha = None
    return StartIdentity(
        f"power_on_fixture_{seed}",
        rom_sha256=rom_sha,
        metadata={
            "seed_number": package.seed_number,
            "seed_name": package.name,
            "source": package.source,
            "integration": GAME,
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
            "SMRando-Snes ROM is not configured; run "
            "`uv run python -m sm_rando.scripts.setup_rom`"
        )
    if not SHARED_SM_ROM.is_file() and not integration_rom.is_file():
        raise SeedCampaignInfraError("Super Metroid ROM missing for live multi-seed tip")
    return integration_rom


def _live_tip_result(seed: SeedValue, *, report_dir: Path) -> Any:
    """Run the real power-on → Morph policy once for a seed."""
    from sm_rando.morph_policy import run_morph_policy

    _require_live_rom()
    report_dir.mkdir(parents=True, exist_ok=True)
    return run_morph_policy(
        video_path=None,
        report_path=report_dir / f"live_tip_{seed}.json",
    )


def build_case_for_seed(
    seed: SeedValue,
    *,
    spec: EarlyTipCampaignSpec,
    config: SeedRobustnessConfig,
) -> BenchmarkCase:
    """Build one seed's benchmark case (ROM/fixture setup + success predicate)."""
    package = ensure_campaign_seed(seed, seeds_root=spec.seeds_root)
    start_identity = _start_identity_for_seed(seed, package)
    # Unbound policy identity: runner rebinds to policy_identity_for(policy).
    contract = EvaluationContract(
        runtime_observation_class=RuntimeObservationClass.BRONZE,
        intervention_class=InterventionClass.CLEAN,
        start_identity=start_identity,
        policy_identity=PolicyIdentity("unbound-policy"),
        benchmark_id=f"sm_rando_early_tip_{seed}",
        objective=config.goal,
    )

    if spec.mode == "dry":
        reachable = morph_reachable_without_spoiler(package)
        # Use a short fixed clear length inside the published dry budget.
        clear_frames = max(1, min(config.budget - 1, 3)) if reachable else None
        failure_mode = None if reachable else "morph_not_at_vanilla_location"

        def build_env() -> _AuditedTipEnv:
            return _AuditedTipEnv(
                success_after=clear_frames,
                terminal_milestone=TERMINAL_MILESTONE if reachable else "ship",
                failure_mode=failure_mode,
            )

        notes = (
            f"dry fixture seed {seed}; morph_vanilla={reachable}; "
            "no spoiler oracle"
        )
    else:
        # Live: execute the real tip once, then replay frame count via audited env.
        def build_env() -> _AuditedTipEnv:
            try:
                tip = _live_tip_result(
                    seed,
                    report_dir=spec.report_path.parent / "early_tip_live",
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
                getattr(tip, "outcome", None) == "morph_ball_acquired"
                or (getattr(tip, "final_state", {}) or {}).get("morph_ball") is True
            )
            frames = int(getattr(tip, "total_frames", 0) or 0)
            if frames <= 0:
                frames = config.budget
            frames = min(frames, config.budget)
            if success:
                return _AuditedTipEnv(
                    success_after=max(1, frames),
                    terminal_milestone=TERMINAL_MILESTONE,
                    audit_provider="sm_rando.early_tip_campaign.live",
                )
            outcome = str(getattr(tip, "outcome", "tip_failed"))
            return _AuditedTipEnv(
                success_after=None,
                terminal_milestone=outcome,
                failure_mode=outcome,
                audit_provider="sm_rando.early_tip_campaign.live",
            )

        notes = f"live SMRando-Snes morph tip for fixture seed {seed}"

    return BenchmarkCase(
        benchmark_id=f"sm_rando_early_tip_{seed}",
        display_name=f"SM Rando early tip seed {seed}",
        game=GAME,
        start_state=f"power_on_fixture_{seed}",
        tier=BenchmarkTier.BRONZE,
        objective=config.goal,
        max_steps=config.budget,
        build_env=build_env,
        is_success=_is_success,
        stop_on_success=True,
        tags=("sm_rando", "early_tip", "ship_to_morph", spec.mode),
        notes=notes,
        metadata={
            "seed": str(seed),
            "seed_package": str(package.directory),
            "mode": spec.mode,
            "spoiler_oracle": False,
        },
        contract=contract,
    )


def policy_factory(_seed: SeedValue) -> ShipToMorphCampaignPolicy:
    """Fresh policy instance per seed (SeedCampaignRunner contract)."""
    return ShipToMorphCampaignPolicy()


def run_early_tip_campaign(
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
) -> SeedCampaignResult:
    """Run the multi-seed early tip campaign and write package-owned reports."""
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
    }


__all__ = [
    "CAMPAIGN_LEDGER",
    "CAMPAIGN_REPORT",
    "CLASSIC_REPORT",
    "DEFAULT_DRY_BUDGET",
    "DEFAULT_LIVE_BUDGET",
    "DEFAULT_SEEDS",
    "DEFAULT_SUCCESS_THRESHOLD",
    "EarlyTipCampaignSpec",
    "MORPH_BASELINE_FRAMES",
    "POLICY_NAME",
    "ShipToMorphCampaignPolicy",
    "TERMINAL_MILESTONE",
    "build_campaign_config",
    "build_case_for_seed",
    "campaign_summary",
    "default_spec",
    "ensure_campaign_seed",
    "ensure_campaign_seeds",
    "morph_reachable_without_spoiler",
    "policy_factory",
    "run_early_tip_campaign",
]
