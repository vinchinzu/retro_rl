"""Fail-closed resumable multi-seed campaign runner.

Builds on :class:`~retro_harness.benchmark_runner.SeedRobustnessConfig` and
produces the same deterministic S/T report shape, with an atomic checkpoint
ledger so long campaigns can resume mid-seed-list.

Design rules (rr-gbd.33):

* ``policy_factory(seed)`` owns policy construction per seed.
* Infrastructure failures yield ordered ``INFRA_ERROR`` rows and force the
  campaign non-claimable; game success/failure still records normally.
* Missing or incomplete audits never become Clean by default.
* Resume requires an exact contract/config digest match.
* Ledger and final report writes use temp-file + ``os.replace``.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from retro_harness.audit import AttemptAudit, InterventionClass
from retro_harness.benchmark_claims import (
    ClaimValidationError,
    EvaluationContract,
    PolicyIdentity,
    policy_identity_for,
    validate_claim,
)
from retro_harness.benchmark_runner import (
    SEED_ROBUSTNESS_SCHEMA_VERSION,
    BenchmarkAttemptResult,
    BenchmarkCase,
    BenchmarkPolicy,
    SeedAttemptResult,
    SeedRobustnessConfig,
    SeedRobustnessReport,
    SeedValue,
    _contract_for_seed_case,
    _policy_name,
    _to_jsonable,
    _validate_seed_result_budget,
    run_benchmark,
)


SEED_CAMPAIGN_SCHEMA_VERSION = 1
LEDGER_EVENT = "seed_campaign_ledger"
CAMPAIGN_EVENT = "seed_campaign_report"


class SeedExecutionStatus(str, Enum):
    """Per-seed execution status recorded in the campaign ledger."""

    SUCCESS = "success"
    FAILURE = "failure"
    INFRA_ERROR = "infra_error"

    @classmethod
    def from_value(cls, value: Any) -> "SeedExecutionStatus":
        if isinstance(value, cls):
            return value
        if isinstance(value, Enum):
            value = value.value
        if not isinstance(value, str):
            raise TypeError("seed execution status must be a string")
        normalized = value.strip().casefold().replace("-", "_")
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(
            f"invalid seed execution status {value!r}; "
            "expected success, failure, or infra_error"
        )


class SeedCampaignError(ValueError):
    """Base error for seed-campaign runner failures."""


class SeedCampaignContractMismatch(SeedCampaignError):
    """Raised when an existing ledger cannot resume under the current contract."""


class SeedCampaignInfraError(SeedCampaignError):
    """Infrastructure failure while executing a seed (ROM/env/policy setup)."""


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Write ``text`` to ``path`` via temp file + ``os.replace``."""
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{report_path.name}.",
        suffix=".tmp",
        dir=str(report_path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, report_path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return report_path


def atomic_write_json(path: str | Path, record: Any) -> Path:
    """Serialize ``record`` as canonical JSON and write it atomically."""
    serialized = (
        json.dumps(record, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    return atomic_write_text(path, serialized)


def config_contract_digest(config: SeedRobustnessConfig) -> str:
    """Stable digest of the published campaign contract (config record)."""
    if not isinstance(config, SeedRobustnessConfig):
        raise TypeError("config must be a SeedRobustnessConfig")
    payload = json.dumps(
        config.to_record(),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SeedExecutionRow:
    """One ordered seed outcome in a campaign ledger / report.

    ``record`` freezes the exact JSON-safe payload written at execution time so
    resume re-emits byte-identical campaign reports without re-deriving fields.
    """

    seed: SeedValue
    status: SeedExecutionStatus
    result: SeedAttemptResult | None = None
    error: str | None = None
    record: dict[str, Any] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        status = SeedExecutionStatus.from_value(self.status)
        object.__setattr__(self, "status", status)
        if status is SeedExecutionStatus.INFRA_ERROR:
            if self.result is not None:
                raise ValueError("INFRA_ERROR rows cannot carry a SeedAttemptResult")
            if self.error is not None and not isinstance(self.error, str):
                raise TypeError("error must be a string or None")
            if self.record is None:
                object.__setattr__(self, "record", self._build_infra_record())
            else:
                object.__setattr__(self, "record", dict(self.record))
            return
        if self.result is None:
            raise ValueError(f"{status.value} rows require a SeedAttemptResult")
        if not isinstance(self.result, SeedAttemptResult):
            raise TypeError("result must be a SeedAttemptResult or None")
        if self.result.seed != self.seed:
            raise ValueError("execution row seed does not match result seed")
        if status is SeedExecutionStatus.SUCCESS and not self.result.success:
            raise ValueError("SUCCESS rows require result.success is True")
        if status is SeedExecutionStatus.FAILURE and self.result.success:
            raise ValueError("FAILURE rows require result.success is False")
        if self.record is None:
            body = self.result.to_record()
            body["execution_status"] = status.value
            object.__setattr__(self, "record", body)
        else:
            object.__setattr__(self, "record", dict(self.record))

    def _build_infra_record(self) -> dict[str, Any]:
        return {
            "seed": _to_jsonable(self.seed),
            "execution_status": SeedExecutionStatus.INFRA_ERROR.value,
            "outcome": "infra_error",
            "success": False,
            "frames": 0,
            "terminal_milestone": None,
            "failure_mode": "INFRA_ERROR",
            "error": self.error,
            "assists": {},
            "ram_writes": None,
            "mid_run_loads": None,
            "attempt_audit": None,
        }

    @property
    def success(self) -> bool:
        return self.status is SeedExecutionStatus.SUCCESS

    @property
    def is_infra_error(self) -> bool:
        return self.status is SeedExecutionStatus.INFRA_ERROR

    def to_record(self) -> dict[str, Any]:
        if self.record is not None:
            return dict(self.record)
        if self.status is SeedExecutionStatus.INFRA_ERROR:
            return self._build_infra_record()
        assert self.result is not None
        body = self.result.to_record()
        body["execution_status"] = self.status.value
        return body

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "SeedExecutionRow":
        if not isinstance(record, dict):
            raise TypeError("execution row record must be a mapping")
        seed = record.get("seed")
        status = SeedExecutionStatus.from_value(
            record.get("execution_status", record.get("status"))
        )
        frozen = dict(record)
        if status is SeedExecutionStatus.INFRA_ERROR:
            return cls(
                seed=seed,
                status=status,
                error=record.get("error"),
                record=frozen,
            )
        result = _seed_result_from_record(record)
        return cls(seed=seed, status=status, result=result, record=frozen)


@dataclass(frozen=True)
class SeedCampaignLedger:
    """Atomic checkpoint of completed seed rows for resume."""

    config: SeedRobustnessConfig
    policy_name: str
    rows: tuple[SeedExecutionRow, ...] = ()
    contract_digest: str = ""
    schema_version: int = SEED_CAMPAIGN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.config, SeedRobustnessConfig):
            raise TypeError("config must be a SeedRobustnessConfig")
        if not isinstance(self.policy_name, str) or not self.policy_name.strip():
            raise ValueError("policy_name must be a non-empty string")
        digest = self.contract_digest or config_contract_digest(self.config)
        if digest != config_contract_digest(self.config):
            raise SeedCampaignContractMismatch(
                "ledger contract digest does not match the supplied config"
            )
        rows = tuple(self.rows)
        expected = self.config.seeds
        if len(rows) > len(expected):
            raise ValueError("ledger has more rows than published seeds")
        for index, row in enumerate(rows):
            if not isinstance(row, SeedExecutionRow):
                raise TypeError("ledger rows must be SeedExecutionRow values")
            if row.seed != expected[index]:
                raise ValueError(
                    "ledger rows must follow published seed order: "
                    f"expected {expected[index]!r} at index {index}, got {row.seed!r}"
                )
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "contract_digest", digest)
        object.__setattr__(self, "policy_name", self.policy_name.strip())

    @property
    def completed_seeds(self) -> tuple[SeedValue, ...]:
        return tuple(row.seed for row in self.rows)

    @property
    def next_seed_index(self) -> int:
        return len(self.rows)

    @property
    def is_complete(self) -> bool:
        return len(self.rows) == self.config.seed_count

    def with_row(self, row: SeedExecutionRow) -> "SeedCampaignLedger":
        expected_index = self.next_seed_index
        if expected_index >= self.config.seed_count:
            raise ValueError("cannot append row to a complete ledger")
        expected_seed = self.config.seeds[expected_index]
        if row.seed != expected_seed:
            raise ValueError(
                f"next seed must be {expected_seed!r}, got {row.seed!r}"
            )
        return SeedCampaignLedger(
            config=self.config,
            policy_name=self.policy_name,
            rows=self.rows + (row,),
            contract_digest=self.contract_digest,
            schema_version=self.schema_version,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "event": LEDGER_EVENT,
            "schema_version": self.schema_version,
            "contract_digest": self.contract_digest,
            "policy": self.policy_name,
            "config": self.config.to_record(),
            "completed_count": len(self.rows),
            "seed_results": [row.to_record() for row in self.rows],
        }

    def write(self, path: str | Path) -> Path:
        return atomic_write_json(path, self.to_record())

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_config: SeedRobustnessConfig | None = None,
    ) -> "SeedCampaignLedger":
        ledger_path = Path(path)
        if not ledger_path.exists():
            raise FileNotFoundError(f"seed campaign ledger not found: {ledger_path}")
        try:
            record = json.loads(ledger_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SeedCampaignError(f"corrupt seed campaign ledger: {exc}") from exc
        if not isinstance(record, dict):
            raise SeedCampaignError("seed campaign ledger root must be an object")
        if record.get("event") not in (LEDGER_EVENT, None):
            raise SeedCampaignError(
                f"unexpected ledger event {record.get('event')!r}"
            )
        schema_version = record.get("schema_version", SEED_CAMPAIGN_SCHEMA_VERSION)
        if schema_version != SEED_CAMPAIGN_SCHEMA_VERSION:
            raise SeedCampaignError(
                f"unsupported seed campaign ledger schema_version {schema_version!r}"
            )
        config_record = record.get("config")
        if not isinstance(config_record, dict):
            raise SeedCampaignError("ledger config must be an object")
        config = _config_from_record(config_record)
        digest = record.get("contract_digest")
        if not isinstance(digest, str) or not digest.strip():
            raise SeedCampaignError("ledger contract_digest is required")
        actual_digest = config_contract_digest(config)
        if digest != actual_digest:
            raise SeedCampaignContractMismatch(
                "ledger contract_digest does not match embedded config"
            )
        if expected_config is not None:
            expected_digest = config_contract_digest(expected_config)
            if digest != expected_digest:
                raise SeedCampaignContractMismatch(
                    "existing campaign ledger contract does not match current config; "
                    "cannot resume"
                )
            # Prefer the caller's config object (same digest, may carry call-side identity).
            config = expected_config
        policy_name = record.get("policy", "")
        seed_results = record.get("seed_results", [])
        if not isinstance(seed_results, list):
            raise SeedCampaignError("ledger seed_results must be a list")
        rows = tuple(SeedExecutionRow.from_record(item) for item in seed_results)
        return cls(
            config=config,
            policy_name=str(policy_name),
            rows=rows,
            contract_digest=digest,
            schema_version=int(schema_version),
        )


@dataclass(frozen=True)
class SeedCampaignResult:
    """Completed (or partially completed after infra failure) campaign outcome."""

    config: SeedRobustnessConfig
    policy_name: str
    rows: tuple[SeedExecutionRow, ...]
    claimable: bool
    ledger_path: Path | None = None
    report_path: Path | None = None

    def __post_init__(self) -> None:
        if len(self.rows) != self.config.seed_count:
            raise ValueError("campaign result must contain one row per published seed")
        for expected, row in zip(self.config.seeds, self.rows, strict=True):
            if row.seed != expected:
                raise ValueError("campaign rows must follow published seed order")
        has_infra = any(row.is_infra_error for row in self.rows)
        if self.claimable and has_infra:
            raise ValueError("campaign with INFRA_ERROR rows cannot be claimable")

    @property
    def successes(self) -> int:
        return sum(1 for row in self.rows if row.success)

    @property
    def success_rate(self) -> float:
        return self.successes / self.config.seed_count

    @property
    def threshold_met(self) -> bool:
        return self.claimable and self.successes >= self.config.success_threshold

    @property
    def infra_error_count(self) -> int:
        return sum(1 for row in self.rows if row.is_infra_error)

    def seed_attempt_results(self) -> tuple[SeedAttemptResult, ...]:
        """Return per-seed results when every seed finished without infra error."""
        if any(row.is_infra_error for row in self.rows):
            raise SeedCampaignError(
                "cannot project SeedAttemptResult sequence while INFRA_ERROR rows exist"
            )
        return tuple(row.result for row in self.rows)  # type: ignore[misc]

    def to_seed_robustness_report(self) -> SeedRobustnessReport:
        if not self.claimable:
            raise SeedCampaignError(
                "non-claimable campaign cannot produce a publishable SeedRobustnessReport"
            )
        return SeedRobustnessReport(
            config=self.config,
            policy_name=self.policy_name,
            seed_results=self.seed_attempt_results(),
        )

    def to_record(self) -> dict[str, Any]:
        summary = {
            "seeds_total": self.config.seed_count,
            "seeds_successful": self.successes,
            "success_rate": self.success_rate,
            "required_successes": self.config.success_threshold,
            "threshold_met": self.threshold_met,
            "infra_errors": self.infra_error_count,
            "claimable": self.claimable,
        }
        return {
            "event": CAMPAIGN_EVENT,
            "schema_version": SEED_CAMPAIGN_SCHEMA_VERSION,
            "seed_robustness_schema_version": SEED_ROBUSTNESS_SCHEMA_VERSION,
            "policy": self.policy_name,
            "claimable": self.claimable,
            "config": self.config.to_record(),
            "summary": summary,
            "seed_results": [row.to_record() for row in self.rows],
        }

    def write_report(self, path: str | Path) -> Path:
        report_path = atomic_write_json(path, self.to_record())
        object.__setattr__(self, "report_path", Path(report_path))
        return Path(report_path)


@dataclass
class SeedCampaignRunner:
    """Resumable fail-closed multi-seed campaign executor.

    Parameters
    ----------
    config:
        Published S/T seed-robustness contract.
    build_case:
        Seed → :class:`BenchmarkCase` factory (ROM/env setup).
    policy_factory:
        Seed → policy factory. Required so each seed gets a fresh policy
        instance (or a deliberately seed-conditioned policy).
    ledger_path:
        Atomic checkpoint path. Resume loads from here when present.
    report_path:
        Optional final campaign report path (atomic replace on completion).
    result_extractor:
        Optional override converting a benchmark attempt into a seed result.
    """

    config: SeedRobustnessConfig
    build_case: Callable[[SeedValue], BenchmarkCase]
    policy_factory: Callable[[SeedValue], BenchmarkPolicy | Callable[..., Any]]
    ledger_path: str | Path
    report_path: str | Path | None = None
    result_extractor: (
        Callable[[SeedValue, BenchmarkAttemptResult], SeedAttemptResult] | None
    ) = None
    stop_on_infra_error: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.config, SeedRobustnessConfig):
            raise TypeError("config must be a SeedRobustnessConfig")
        if not callable(self.build_case):
            raise TypeError("build_case must be callable")
        if not callable(self.policy_factory):
            raise TypeError("policy_factory must be callable")
        self.ledger_path = Path(self.ledger_path)
        if self.report_path is not None:
            self.report_path = Path(self.report_path)

    def run(self) -> SeedCampaignResult:
        ledger = self._load_or_create_ledger()
        for seed in self.config.seeds[ledger.next_seed_index :]:
            row = self._execute_seed(seed)
            ledger = ledger.with_row(row)
            ledger.write(self.ledger_path)
            if row.is_infra_error and self.stop_on_infra_error:
                # Pad remaining seeds as ordered INFRA_ERROR so the campaign
                # always materializes one row per published seed.
                for remaining in self.config.seeds[ledger.next_seed_index :]:
                    pad = SeedExecutionRow(
                        seed=remaining,
                        status=SeedExecutionStatus.INFRA_ERROR,
                        error=f"skipped after infra error on seed {seed!r}",
                    )
                    ledger = ledger.with_row(pad)
                    ledger.write(self.ledger_path)
                break

        claimable = self._is_claimable(ledger.rows)
        result = SeedCampaignResult(
            config=self.config,
            policy_name=ledger.policy_name,
            rows=ledger.rows,
            claimable=claimable,
            ledger_path=Path(self.ledger_path),
        )
        if self.report_path is not None:
            report_path = result.write_report(self.report_path)
            object.__setattr__(result, "report_path", report_path)
        return result

    def _load_or_create_ledger(self) -> SeedCampaignLedger:
        path = Path(self.ledger_path)
        if not path.exists():
            # Peek first policy for a stable display name; factory is re-called per seed.
            first_policy = self.policy_factory(self.config.seeds[0])
            return SeedCampaignLedger(
                config=self.config,
                policy_name=_policy_name(first_policy),
                rows=(),
                contract_digest=config_contract_digest(self.config),
            )
        return SeedCampaignLedger.load(path, expected_config=self.config)

    def _execute_seed(self, seed: SeedValue) -> SeedExecutionRow:
        try:
            policy = self.policy_factory(seed)
            case = self.build_case(seed)
            if not isinstance(case, BenchmarkCase):
                raise TypeError("build_case must return a BenchmarkCase")
            if case.max_steps != self.config.budget:
                raise ValueError(
                    f"benchmark case for seed {seed!r} must use exactly the published "
                    f"frame budget ({self.config.budget})"
                )
            actual_policy_identity = policy_identity_for(policy)
            contract = _contract_for_seed_case(
                self.config, case, actual_policy_identity
            )
            run_result = run_benchmark(case, policy, contract=contract)
            attempt = run_result.attempts[0]
            if self.result_extractor is None:
                seed_result = SeedAttemptResult.from_benchmark_attempt(
                    seed,
                    attempt,
                    contract=contract,
                )
            else:
                seed_result = self.result_extractor(seed, attempt)
                if not isinstance(seed_result, SeedAttemptResult):
                    raise TypeError("result_extractor must return a SeedAttemptResult")
                seed_result = seed_result.bind_contract(contract)
            _validate_seed_result_budget(self.config, seed_result)
            # Fail closed: incomplete instrumentation cannot publish Clean
            # (or any claim). Game outcome is still recorded, but the
            # campaign claimability gate uses validate_claim.
            status = (
                SeedExecutionStatus.SUCCESS
                if seed_result.success
                else SeedExecutionStatus.FAILURE
            )
            return SeedExecutionRow(seed=seed, status=status, result=seed_result)
        except SeedCampaignInfraError as exc:
            return SeedExecutionRow(
                seed=seed,
                status=SeedExecutionStatus.INFRA_ERROR,
                error=str(exc) or exc.__class__.__name__,
            )
        except ClaimValidationError as exc:
            # Missing/incomplete audit or identity contradiction: do not
            # invent a Clean success. Record as infra-level claim failure.
            return SeedExecutionRow(
                seed=seed,
                status=SeedExecutionStatus.INFRA_ERROR,
                error=f"claim validation: {exc}",
            )
        except (OSError, RuntimeError, MemoryError) as exc:
            return SeedExecutionRow(
                seed=seed,
                status=SeedExecutionStatus.INFRA_ERROR,
                error=f"{exc.__class__.__name__}: {exc}",
            )
        except Exception as exc:
            # build_case / policy_factory / env construction bugs surface here.
            # Treat unexpected setup/runtime exceptions as infrastructure so a
            # flaky backend cannot look like a legitimate game failure.
            if isinstance(exc, (TypeError, ValueError)) and "budget" in str(exc):
                raise
            if isinstance(exc, (TypeError, ValueError)) and "must return" in str(exc):
                raise
            return SeedExecutionRow(
                seed=seed,
                status=SeedExecutionStatus.INFRA_ERROR,
                error=f"{exc.__class__.__name__}: {exc}",
            )

    @staticmethod
    def _is_claimable(rows: tuple[SeedExecutionRow, ...]) -> bool:
        if any(row.is_infra_error for row in rows):
            return False
        for row in rows:
            if row.result is None:
                return False
            try:
                # validate_claim fails closed on missing instrumentation.
                validate_claim(row.result.to_record())
            except (ClaimValidationError, TypeError, ValueError):
                return False
        return True


def run_seed_campaign(
    config: SeedRobustnessConfig,
    build_case: Callable[[SeedValue], BenchmarkCase],
    policy_factory: Callable[[SeedValue], BenchmarkPolicy | Callable[..., Any]],
    *,
    ledger_path: str | Path,
    report_path: str | Path | None = None,
    result_extractor: (
        Callable[[SeedValue, BenchmarkAttemptResult], SeedAttemptResult] | None
    ) = None,
    stop_on_infra_error: bool = False,
) -> SeedCampaignResult:
    """Convenience wrapper around :class:`SeedCampaignRunner`."""
    return SeedCampaignRunner(
        config=config,
        build_case=build_case,
        policy_factory=policy_factory,
        ledger_path=ledger_path,
        report_path=report_path,
        result_extractor=result_extractor,
        stop_on_infra_error=stop_on_infra_error,
    ).run()


def _config_from_record(record: dict[str, Any]) -> SeedRobustnessConfig:
    """Rebuild a SeedRobustnessConfig from its serialized record.

    Delegates to :meth:`SeedRobustnessConfig.from_record`, which prefers an
    embedded ``contract`` and only falls back to flat identity fields for
    legacy ledgers that omit it.
    """
    try:
        return SeedRobustnessConfig.from_record(record)
    except (TypeError, ValueError, KeyError) as exc:
        raise SeedCampaignError(str(exc)) from exc


def _seed_result_from_record(record: dict[str, Any]) -> SeedAttemptResult:
    """Rehydrate a SeedAttemptResult from a ledger/campaign row."""
    try:
        return SeedAttemptResult.from_record(record)
    except (TypeError, ValueError, KeyError) as exc:
        raise SeedCampaignError(str(exc)) from exc



__all__ = [
    "CAMPAIGN_EVENT",
    "LEDGER_EVENT",
    "SEED_CAMPAIGN_SCHEMA_VERSION",
    "SeedCampaignContractMismatch",
    "SeedCampaignError",
    "SeedCampaignInfraError",
    "SeedCampaignLedger",
    "SeedCampaignResult",
    "SeedCampaignRunner",
    "SeedExecutionRow",
    "SeedExecutionStatus",
    "atomic_write_json",
    "atomic_write_text",
    "config_contract_digest",
    "run_seed_campaign",
]
