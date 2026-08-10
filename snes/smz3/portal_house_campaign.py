"""Multi-seed portal→house S/T campaign for SMZ3 (rr-gbd.13).

Thin consumer of :class:`retro_harness.seed_campaign.SeedCampaignRunner`.
Published seeds are offline fixture packages under ``seeds/``; the early tip
(PortalSettled → Fortune Teller OW → Link's House chest) is **layout-fixed**
on the combo substrate and does not consult a spoiler oracle.

Modes
-----
* ``dry`` (default): audited synthetic envs for harness S/T dry-run. Fixture
  seeds with morph-original / uncle-sword settings clear; used for CI and the
  published dry-run report.
* ``live``: real ``SMZ3-Snes`` early quest tip per seed (portal settle → house
  chest). Missing ROM/setup is fail-closed ``INFRA_ERROR`` (non-claimable).
  Live path historically uses the missile red-door **resource assist** until
  natural morph→missiles is on the combo path — labeled in metadata.

Substrate honesty
-----------------
Fixture packages are **not** shuffled multi-seed ROMs. The dry report is
seed-abstract evidence for the *harness + tip contract*, labeled
``substrate=fixture`` / ``seed_source=fixture``. Not shuffled-seed robustness
until a generator/patch is wired per seed.
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
from smz3.paths import (
    GAME,
    INTEGRATION_DIR,
    RECORDINGS_DIR,
    SEEDS_DIR,
    SHARED_SM_ROM,
    SHARED_Z3_JP_ROM,
)
from smz3.seed import SeedPackage, ensure_fixture_seed

CampaignMode = Literal["dry", "live"]

# Published campaign contract (deterministic dry-run defaults).
DEFAULT_SEEDS: tuple[str, ...] = ("1337", "1338", "1339")
DEFAULT_SUCCESS_THRESHOLD = 2
# Live portal→house on seed 1337 is multi-minute with settle wait; headroom.
DEFAULT_LIVE_BUDGET = 120_000
# Dry-run uses a short budget so the harness path is cheap in CI.
DEFAULT_DRY_BUDGET = 8
DEFAULT_GENERATOR = "smz3.fixture"
DEFAULT_GENERATOR_VERSION = "1"
DEFAULT_LOGIC = "normal"
DEFAULT_GOAL = "portal_to_house"

POLICY_NAME = "smz3.portal_to_house"

CAMPAIGN_REPORT = RECORDINGS_DIR / "portal_house_seed_campaign.json"
CAMPAIGN_LEDGER = RECORDINGS_DIR / "portal_house_seed_campaign.ledger.json"
CLASSIC_REPORT = RECORDINGS_DIR / "portal_house_seed_robustness.json"
# Committed published dry artifact (recordings/ is gitignored).
DOCS_PUBLISHED_REPORT = Path(__file__).resolve().parent / "docs" / "portal_house_seed_campaign_dry.json"

TERMINAL_MILESTONE = "links_house_chest"


class PortalToHouseCampaignPolicy:
    """Display policy for the seed-agnostic portal→house early tip.

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
class PortalHouseCampaignSpec:
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
        audit_provider: str = "smz3.portal_house_campaign.dry",
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
    return ensure_fixture_seed(str(seed), directory=directory)


def ensure_campaign_seeds(
    seeds: Sequence[SeedValue],
    *,
    seeds_root: Path = SEEDS_DIR,
) -> tuple[SeedPackage, ...]:
    return tuple(ensure_campaign_seed(seed, seeds_root=seeds_root) for seed in seeds)


def portal_house_clearable_without_spoiler(package: SeedPackage) -> bool:
    """Return True when the seed-agnostic portal→house tip is expected to clear.

    Does **not** consult the spoiler log for routing. Fixture packages with
    morph at original + uncle sword are the dry-run precondition (matches the
    verified single-seed tip settings). Outdoor Fortune Teller → Link's House
    is layout-fixed on the combo; item spoiler is irrelevant for the flee path.
    """
    settings = package.settings or {}
    morph = str(settings.get("morphlocation", "original")).strip().casefold()
    sword = str(settings.get("swordlocation", "uncle")).strip().casefold()
    if morph not in ("original", "vanilla", ""):
        return False
    if sword not in ("uncle", "vanilla", ""):
        return False
    # Explicit opt-out for failure-mode fixtures.
    if package.meta.get("portal_house_clearable") is False:
        return False
    return True


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
) -> PortalHouseCampaignSpec:
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
    return PortalHouseCampaignSpec(
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


def build_campaign_config(spec: PortalHouseCampaignSpec) -> SeedRobustnessConfig:
    """Published S/T contract for the portal→house campaign."""
    rom_sha = None
    if SHARED_SM_ROM.is_file():
        try:
            rom_sha = sha256_file(SHARED_SM_ROM)
        except OSError:
            rom_sha = None
    z3_sha = None
    if SHARED_Z3_JP_ROM.is_file():
        try:
            z3_sha = sha256_file(SHARED_Z3_JP_ROM)
        except OSError:
            z3_sha = None

    # Dry is synthetic Clean; live tip path uses missile red-door resource assist.
    intervention = (
        InterventionClass.CLEAN
        if spec.mode == "dry"
        else InterventionClass.RESOURCE_ASSISTED
    )

    metadata: dict[str, Any] = {
        "package": "smz3",
        "edge": "portal_to_house",
        "tip": "links_house_chest",
        "mode": spec.mode,
        "substrate": "fixture",
        "spoiler_oracle": False,
        "seed_source": "fixture",
        "skill_id": POLICY_NAME,
        "live_assist": "missile_red_door" if spec.mode == "live" else None,
        "note": (
            "Fixture seeds on the documented combo early-tip substrate "
            "(morph original, uncle sword). Outdoor Fortune Teller → Link's "
            "House is layout-fixed; path is seed-agnostic (no spoiler oracle). "
            "Not shuffled-seed robustness until a rando generator/patch is wired. "
            "Live path uses missile red-door resource assist until natural "
            "morph→missiles lands on the combo ROM."
        ),
    }
    if rom_sha is not None:
        metadata["sm_rom_sha256"] = rom_sha
    if z3_sha is not None:
        metadata["z3_jp_rom_sha256"] = z3_sha

    return SeedRobustnessConfig(
        generator=DEFAULT_GENERATOR,
        generator_version=DEFAULT_GENERATOR_VERSION,
        logic=DEFAULT_LOGIC,
        goal=DEFAULT_GOAL,
        seeds=spec.seeds,
        budget=spec.budget,
        success_threshold=spec.success_threshold,
        runtime_observation_class=RuntimeObservationClass.BRONZE,
        intervention_class=intervention,
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
            "source": package.meta.get("source", "fixture"),
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
            "SMZ3-Snes ROM is not configured; run "
            "`uv run python -m smz3.scripts.wire_integration_rom` "
            "(and setup_roms / generate_seed --test first)"
        )
    return integration_rom


def _live_tip_result(seed: SeedValue, *, report_dir: Path) -> Any:
    """Run the real portal settle → house chest early quest once for a seed."""
    from smz3.quest import STOP_LINKS_HOUSE_CHEST, run_early_quest

    _require_live_rom()
    report_dir.mkdir(parents=True, exist_ok=True)
    result = run_early_quest(stop=STOP_LINKS_HOUSE_CHEST, close=True)
    report_path = report_dir / f"live_tip_{seed}.json"
    try:
        import json

        report_path.write_text(
            json.dumps(result.to_dict(), indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass
    return result


def build_case_for_seed(
    seed: SeedValue,
    *,
    spec: PortalHouseCampaignSpec,
    config: SeedRobustnessConfig,
) -> BenchmarkCase:
    """Build one seed's benchmark case (fixture setup + success predicate)."""
    package = ensure_campaign_seed(seed, seeds_root=spec.seeds_root)
    start_identity = _start_identity_for_seed(seed, package)
    contract = EvaluationContract(
        runtime_observation_class=RuntimeObservationClass.BRONZE,
        intervention_class=config.intervention_class,
        start_identity=start_identity,
        policy_identity=PolicyIdentity("unbound-policy"),
        benchmark_id=f"smz3_portal_house_{seed}",
        objective=config.goal,
    )

    if spec.mode == "dry":
        clearable = portal_house_clearable_without_spoiler(package)
        clear_frames = max(1, min(config.budget - 1, 3)) if clearable else None
        failure_mode = None if clearable else "portal_house_settings_not_seed_agnostic"

        def build_env() -> _AuditedTipEnv:
            return _AuditedTipEnv(
                success_after=clear_frames,
                terminal_milestone=TERMINAL_MILESTONE if clearable else "portal_settled",
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
                    report_dir=spec.report_path.parent / "portal_house_live",
                )
            except SeedCampaignInfraError:
                raise
            except FileNotFoundError as exc:
                raise SeedCampaignInfraError(str(exc)) from exc
            except OSError as exc:
                raise SeedCampaignInfraError(f"OSError: {exc}") from exc
            except RuntimeError as exc:
                raise SeedCampaignInfraError(f"RuntimeError: {exc}") from exc

            success = bool(getattr(tip, "ok", False))
            frames = int(getattr(tip, "frames", 0) or 0)
            if frames <= 0:
                frames = config.budget
            frames = min(frames, config.budget)
            if success:
                return _AuditedTipEnv(
                    success_after=max(1, frames),
                    terminal_milestone=TERMINAL_MILESTONE,
                    audit_provider="smz3.portal_house_campaign.live",
                )
            detail = str(getattr(tip, "detail", "tip_failed"))
            return _AuditedTipEnv(
                success_after=None,
                terminal_milestone=detail[:64],
                failure_mode=detail[:128],
                audit_provider="smz3.portal_house_campaign.live",
            )

        notes = f"live SMZ3-Snes portal→house tip for fixture seed {seed}"

    return BenchmarkCase(
        benchmark_id=f"smz3_portal_house_{seed}",
        display_name=f"SMZ3 portal→house seed {seed}",
        game=GAME,
        start_state=f"power_on_fixture_{seed}",
        tier=BenchmarkTier.BRONZE,
        objective=config.goal,
        max_steps=config.budget,
        build_env=build_env,
        is_success=_is_success,
        stop_on_success=True,
        tags=("smz3", "portal_to_house", "early_tip", spec.mode),
        notes=notes,
        metadata={
            "seed": str(seed),
            "seed_package": str(package.directory),
            "mode": spec.mode,
            "spoiler_oracle": False,
        },
        contract=contract,
    )


def policy_factory(_seed: SeedValue) -> PortalToHouseCampaignPolicy:
    """Fresh policy instance per seed (SeedCampaignRunner contract)."""
    return PortalToHouseCampaignPolicy()


def run_portal_house_campaign(
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
    """Run the multi-seed portal→house campaign and write package-owned reports."""
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
    "POLICY_NAME",
    "PortalHouseCampaignSpec",
    "PortalToHouseCampaignPolicy",
    "TERMINAL_MILESTONE",
    "build_campaign_config",
    "build_case_for_seed",
    "campaign_summary",
    "default_spec",
    "ensure_campaign_seed",
    "ensure_campaign_seeds",
    "policy_factory",
    "portal_house_clearable_without_spoiler",
    "run_portal_house_campaign",
]
