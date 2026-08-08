"""Offline tests for shared opening-route phase/segment runner."""

from __future__ import annotations

from alttp.opening_route.runner import run_from_state, run_phases
from alttp.ram import AlttpSnapshot
from alttp.route_report import RoutePhaseResult, SegmentResult


def _snap(**kwargs: object) -> AlttpSnapshot:
    # Minimal fake: SegmentResult only needs a snapshot object; use a simple
    # namespace-like stand-in via AlttpSnapshot fields when possible is heavy.
    # Tests use a lightweight object with attributes success_when reads.
    class _S:  # noqa: N801
        pass

    s = _S()
    for k, v in kwargs.items():
        setattr(s, k, v)
    return s  # type: ignore[return-value]


def test_run_phases_success_midway() -> None:
    calls: list[str] = []

    def phase_a(_env: object) -> RoutePhaseResult:
        calls.append("a")
        return RoutePhaseResult(
            phase="a", ok=True, frames=1, snapshot=_snap(done=False)
        )

    def phase_b(_env: object) -> RoutePhaseResult:
        calls.append("b")
        return RoutePhaseResult(
            phase="b", ok=True, frames=2, snapshot=_snap(done=True)
        )

    def phase_c(_env: object) -> RoutePhaseResult:
        calls.append("c")
        return RoutePhaseResult(
            phase="c", ok=True, frames=3, snapshot=_snap(done=True)
        )

    result = run_phases(
        object(),
        [phase_a, phase_b, phase_c],
        evaluate_acceptance=lambda s: {"done": bool(getattr(s, "done", False))},
        success_when=lambda s: bool(getattr(s, "done", False)),
        success_phase="finished",
        notes=["n1"],
    )
    assert result.ok is True
    assert result.phase == "finished"
    assert result.frames == 3
    assert calls == ["a", "b"]  # stopped after success
    assert len(result.phases) == 2


def test_run_phases_fail_stops() -> None:
    def bad(_env: object) -> RoutePhaseResult:
        return RoutePhaseResult(
            phase="bad",
            ok=False,
            frames=5,
            snapshot=_snap(done=False),
            detail="broke",
        )

    def never(_env: object) -> RoutePhaseResult:
        raise AssertionError("should not run")

    result = run_phases(
        object(),
        [bad, never],
        evaluate_acceptance=lambda s: {"done": False},
        success_when=lambda s: False,
    )
    assert result.ok is False
    assert result.phase == "bad"
    assert result.blocker == "broke"
    assert result.frames == 5


def test_run_from_state_calls_play(monkeypatch: object) -> None:
    closed: list[bool] = []

    class _Env:
        def reset(self) -> None:
            return None

        def close(self) -> None:
            closed.append(True)

    def fake_boot(_name: str) -> _Env:
        return _Env()

    monkeypatch.setattr(  # type: ignore[attr-defined]
        "alttp.startup.build_boot_env", fake_boot
    )

    def play(env: object, *, source: str = "x") -> SegmentResult:
        assert source == "state_load_dev"
        return SegmentResult(
            ok=True,
            phase="ok",
            frames=0,
            snapshot=_snap(),
            source=source,
        )

    result = run_from_state("FakeState", play, settle=False)
    assert result.ok is True
    assert result.phase == "ok"
    assert closed == [True]
