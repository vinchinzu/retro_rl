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
