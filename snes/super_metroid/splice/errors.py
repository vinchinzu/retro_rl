"""Fail-closed splice errors with structured fields."""

from __future__ import annotations

from typing import Any, Mapping


class SpliceError(Exception):
    """Base fail-closed error for ``super_metroid.splice``."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "splice",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.details: dict[str, Any] = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": type(self).__name__,
            "message": str(self),
            "code": self.code,
            "details": dict(self.details),
        }


class PreflightError(SpliceError):
    """Selected artifact missing, empty, corrupt, or unresolved by digest."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "preflight.missing",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class SchemaError(SpliceError):
    """Route/task/candidate schema rejected (invalid room, kind, or path)."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "schema.invalid",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class PrepareError(SpliceError):
    """Task cannot be prepared: missing or mismatched digest, fingerprint, or artifact."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "prepare.invalid",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class GradeError(SpliceError):
    """Replay/Join grade failed closed: digest mismatch, missing runner, or invalid candidate."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "grade.invalid",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class SelectError(SpliceError):
    """Planner selection rejected: unknown profile, missing candidate, or invalid offer."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "select.invalid",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class AssembleError(SpliceError):
    """Assembly failed closed: profile mismatch, missing session, or mid-run state load."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "assemble.invalid",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)
