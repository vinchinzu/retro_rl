"""Per-hole score / match tally used by ``StrokePlayMission``."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScorecardBook:
    """Peak-stroke scores plus VS HAL hole-by-hole comparison."""

    holes: list[int] = field(default_factory=list)
    hole_numbers: list[int] = field(default_factory=list)
    opponent_holes: list[int] = field(default_factory=list)
    holes_won: int = 0
    holes_lost: int = 0
    holes_tied: int = 0

    @property
    def total(self) -> int:
        return int(sum(self.holes))

    @property
    def match_lead(self) -> int:
        """Holes up (positive) or down (negative) versus Hal."""
        return int(self.holes_won - self.holes_lost)

    def clear(self) -> None:
        self.holes.clear()
        self.hole_numbers.clear()
        self.opponent_holes.clear()
        self.holes_won = 0
        self.holes_lost = 0
        self.holes_tied = 0

    def record(
        self,
        scored: int,
        hole: int,
        *,
        opponent: int | None = None,
    ) -> None:
        """Append one finished hole. ``opponent`` is Hal's score in VS HAL."""
        if scored <= 0:
            return
        self.holes.append(scored)
        self.hole_numbers.append(hole)
        if opponent is None:
            return
        if opponent <= 0:
            opponent = scored
        self.opponent_holes.append(opponent)
        if scored < opponent:
            self.holes_won += 1
        elif scored > opponent:
            self.holes_lost += 1
        else:
            self.holes_tied += 1

    def as_dict(self, pars: list[int]) -> dict[str, int | list[int]]:
        over_par = [
            number
            for number, score, par in zip(self.hole_numbers, self.holes, pars)
            if score > par
        ]
        return {
            "holes": list(self.holes),
            "hole_numbers": list(self.hole_numbers),
            "pars": pars,
            "total": self.total,
            "to_par": int(sum(self.holes) - sum(pars)),
            "over_par_holes": over_par,
            "holes_completed": int(len(self.holes)),
            "holes_won": int(self.holes_won),
            "holes_lost": int(self.holes_lost),
            "holes_tied": int(self.holes_tied),
            "opponent_holes": list(self.opponent_holes),
        }
