"""Seed-report serialized-record validation helpers.

Owned separately from :mod:`retro_harness.benchmark.claims` so claim-core
stays under the soft LOC ceiling while seed-report aggregate checks remain
fail-closed and importable via the historical shim path.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from retro_harness.audit import InterventionClass, RuntimeObservationClass


def _validate_seed_report_record(
    config: Mapping[str, Any],
    seed_results: list[Any],
) -> None:
    # Lazy imports avoid an import cycle with claims.validate_claim.
    from retro_harness.benchmark.claims import (
        ClaimValidationError,
        PolicyIdentity,
        StartIdentity,
        _record_identity_digest,
    )

    config_contract = config.get("contract")
    if config_contract is not None and not isinstance(config_contract, Mapping):
        raise TypeError("seed report config contract must be a mapping")

    config_start = _record_identity_digest(
        config,
        "start_identity_digest",
        label="seed report config",
    )
    config_policy = _record_identity_digest(
        config,
        "policy_identity_digest",
        label="seed report config",
    )
    if isinstance(config_contract, Mapping):
        for field_name, normalizer in (
            (
                "runtime_observation_class",
                RuntimeObservationClass.from_value,
            ),
            ("intervention_class", InterventionClass.from_value),
        ):
            config_value = config.get(field_name)
            contract_value = config_contract.get(field_name)
            if config_value is None or contract_value is None:
                continue
            try:
                config_class = normalizer(config_value)
                contract_class = normalizer(contract_value)
            except (TypeError, ValueError) as exc:
                raise ClaimValidationError(
                    f"seed report config contract has an invalid {field_name}"
                ) from exc
            if config_class is not contract_class:
                raise ClaimValidationError(
                    f"seed report config {field_name} contradicts its evaluation contract"
                )
        for field_name in (
            "assist_contract_path",
            "assist_contract_digest",
        ):
            config_value = config.get(field_name)
            contract_value = config_contract.get(field_name)
            if (
                config_value is not None
                and contract_value is not None
                and config_value != contract_value
            ):
                raise ClaimValidationError(
                    f"seed report config {field_name} contradicts its evaluation contract"
                )
        if config.get("assist_mode") != config_contract.get("assist_mode"):
            raise ClaimValidationError(
                "seed report config assist_mode contradicts its evaluation contract"
            )
        contract_start = _record_identity_digest(
            config_contract,
            "start_identity_digest",
            label="seed report config contract",
        )
        contract_policy = _record_identity_digest(
            config_contract,
            "policy_identity_digest",
            label="seed report config contract",
        )
        if config_start is not None and contract_start is not None and config_start != contract_start:
            raise ClaimValidationError(
                "seed report config start identity contradicts its evaluation contract"
            )
        if (
            config_policy is not None
            and contract_policy is not None
            and config_policy != contract_policy
        ):
            raise ClaimValidationError(
                "seed report config policy identity contradicts its evaluation contract"
            )
        config_start = contract_start if contract_start is not None else config_start
        config_policy = contract_policy if contract_policy is not None else config_policy

    for field_name, normalizer in (
        (
            "runtime_observation_class",
            RuntimeObservationClass.from_value,
        ),
        ("intervention_class", InterventionClass.from_value),
    ):
        config_value = config.get(field_name)
        if config_value is None:
            continue
        try:
            config_value = normalizer(config_value)
        except (TypeError, ValueError) as exc:
            raise ClaimValidationError(
                f"seed report config has an invalid {field_name}"
            ) from exc
        for seed_record in seed_results:
            if not isinstance(seed_record, Mapping):
                raise TypeError("seed_results must contain mapping records")
            seed_value = seed_record.get(field_name)
            if seed_value is None:
                continue
            try:
                seed_value = normalizer(seed_value)
            except (TypeError, ValueError) as exc:
                raise ClaimValidationError(
                    f"seed result has an invalid {field_name}"
                ) from exc
            if seed_value is not config_value:
                raise ClaimValidationError(
                    f"seed result {field_name} does not match the report config"
                )

    for field_name in (
        "assist_contract_path",
        "assist_contract_digest",
        "assist_mode",
    ):
        config_value = config.get(field_name)
        for seed_record in seed_results:
            if not isinstance(seed_record, Mapping):
                raise TypeError("seed_results must contain mapping records")
            if (
                config_value is not None
                and seed_record.get(field_name) != config_value
            ):
                raise ClaimValidationError(
                    f"seed result {field_name} does not match the report config"
                )

    scopes = {
        "start_identity_digest": config.get("start_identity_scope", "shared"),
        "policy_identity_digest": config.get("policy_identity_scope", "shared"),
    }
    for field_name, scope in scopes.items():
        if scope not in {"shared", "per-seed"}:
            raise ClaimValidationError(f"invalid seed report {field_name} scope")
        if scope == "shared":
            if field_name == "start_identity_digest":
                expected = config_start
            else:
                expected = config_policy
            if expected is None:
                continue
            for seed_record in seed_results:
                seed_digest = _record_identity_digest(
                    seed_record,
                    field_name,
                    label="seed result",
                )
                if seed_digest != expected:
                    raise ClaimValidationError(
                        f"seed result {field_name} contradicts the shared report identity"
                    )
        else:
            if config_contract is not None:
                raise ClaimValidationError(
                    f"seed report config contract cannot use {field_name} per-seed scope"
                )
            if field_name == "start_identity_digest" and config_start is not None:
                generator = config.get("generator")
                generator_version = config.get("generator_version")
                seeds = config.get("seeds")
                if (
                    isinstance(generator, str)
                    and isinstance(generator_version, str)
                    and isinstance(seeds, list)
                ):
                    expected_start = StartIdentity(
                        f"seed-set:{generator}:{generator_version}:{','.join(map(str, seeds))}"
                    )
                    if config_start != expected_start.identity_digest:
                        raise ClaimValidationError(
                            "per-seed report config start identity is not its seed-set identity"
                        )
            if field_name == "policy_identity_digest" and config_policy is not None:
                expected_policy = PolicyIdentity("unbound-policy")
                if config_policy != expected_policy.identity_digest:
                    raise ClaimValidationError(
                        "per-seed report config policy identity must be unbound"
                    )


__all__ = ["_validate_seed_report_record"]
