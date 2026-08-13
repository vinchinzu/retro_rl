"""Pure short-charge / stutter plan builders for shinespark.

No emulator session dependency — only frame masks and button tuples.
Runtime execution lives in :mod:`super_metroid.routes.skills.shinespark`
(:func:`~super_metroid.routes.skills.shinespark.charge_by_plan`,
:func:`~super_metroid.routes.skills.shinespark.short_charge_until_boost`).
"""

from __future__ import annotations

from typing import Literal

Direction = Literal["LEFT", "RIGHT"]
Region = Literal["NTSC", "PAL"]

# --- Short charge (wiki / SM speed tech; 0-based charge-local frames) ---
# Boost counter ticks only on these frames if dash+forward are held.
NTSC_MAGIC_DASH_FRAMES: tuple[int, ...] = (25, 50, 70, 85)
PAL_MAGIC_DASH_FRAMES: tuple[int, ...] = (20, 40, 60, 70)

# Stutter short-charge minimum distance (pixels) — wiki measured.
NTSC_STUTTER_MIN_PX: float = 163.1875
PAL_STUTTER_MIN_PX: float = 157.668
# Full stop after charge is ~1 px more on NTSC wiki note.
NTSC_STUTTER_FULL_STOP_PX: float = 164.1875

# Short charge without stutter still needs the four magic frames (last index).
NTSC_SHORT_CHARGE_FRAMES: int = NTSC_MAGIC_DASH_FRAMES[-1] + 1  # 86
PAL_SHORT_CHARGE_FRAMES: int = PAL_MAGIC_DASH_FRAMES[-1] + 1  # 71


def magic_dash_frames(region: Region = "NTSC") -> tuple[int, ...]:
    """0-based charge-local frames where dash must be held for boost ticks."""
    if region == "PAL":
        return PAL_MAGIC_DASH_FRAMES
    if region == "NTSC":
        return NTSC_MAGIC_DASH_FRAMES
    raise ValueError(f"unknown region {region!r}; expected NTSC or PAL")


def stutter_forward_mask(region: Region = "NTSC") -> list[bool]:
    """Per-frame forward hold for the stutter prefix (frames before first magic).

    NTSC (3-4-4-4-2+3B): 25 frames (0–24) ending with 3× forward+dash then
    1× dash-only; first magic at 25 is *not* included.
    PAL (3-4-3-2-3-): 20 frames (0–19) of forward/release only.
    """
    if region == "NTSC":
        # 3 F, rel, 4 F, rel, 4 F, rel, 4 F, rel, 2 F, 3 F+B, 1 B — dash bits
        # are applied in :func:`short_charge_plan`; here only forward.
        segs: list[tuple[int, bool]] = [
            (3, True),
            (1, False),
            (4, True),
            (1, False),
            (4, True),
            (1, False),
            (4, True),
            (1, False),
            (2, True),
            (3, True),  # F+B stretch
            (1, False),  # release forward, keep dash (plan adds B)
        ]
    elif region == "PAL":
        segs = [
            (3, True),
            (1, False),
            (4, True),
            (1, False),
            (3, True),
            (1, False),
            (2, True),
            (1, False),
            (3, True),
            (1, False),
        ]
    else:
        raise ValueError(f"unknown region {region!r}; expected NTSC or PAL")
    out: list[bool] = []
    for n, hold_f in segs:
        out.extend([hold_f] * n)
    return out


def stutter_dash_mask(region: Region = "NTSC") -> list[bool]:
    """Dash hold bits for the stutter prefix (same length as forward mask)."""
    if region == "NTSC":
        # Only the trailing 3× F+B and 1× B-only use dash before magic 25.
        mask = [False] * 25
        # frames 21–23: F+B; frame 24: B only (indices)
        for i in (21, 22, 23, 24):
            mask[i] = True
        return mask
    if region == "PAL":
        return [False] * len(stutter_forward_mask("PAL"))
    raise ValueError(f"unknown region {region!r}; expected NTSC or PAL")


def short_charge_plan(
    region: Region = "NTSC",
    *,
    stutter: bool = False,
    store_on_last: bool = False,
    direction: Direction = "RIGHT",
    dash_button: str = "B",
) -> list[tuple[str, ...]]:
    """Build per-frame button tuples for a short charge (length = last magic+1).

    Frame 0 is the first charge frame with forward held (simple) or the start
    of the stutter prefix. Magic frames press ``dash_button``; non-magic
    frames hold forward only (except stutter releases / dash-only ticks).

    When ``store_on_last`` is True, the final magic frame also presses DOWN
    (wiki: 4th boost tick + crouch-store in one frame).
    """
    dir_btn = "LEFT" if direction == "LEFT" else "RIGHT"
    magics = magic_dash_frames(region)
    last = magics[-1]
    magic_set = set(magics)

    # Build forward/dash bool lanes, then map to harness buttons.
    n = last + 1
    hold_fwd = [True] * n
    hold_dash = [False] * n

    if stutter:
        fwd_prefix = stutter_forward_mask(region)
        dash_prefix = stutter_dash_mask(region)
        if len(fwd_prefix) != magics[0]:
            raise RuntimeError(
                f"stutter prefix length {len(fwd_prefix)} != first magic "
                f"{magics[0]} for {region}"
            )
        for i, (hf, hd) in enumerate(zip(fwd_prefix, dash_prefix)):
            hold_fwd[i] = hf
            hold_dash[i] = hd

    for f in magics:
        hold_fwd[f] = True
        hold_dash[f] = True

    plan: list[tuple[str, ...]] = []
    for f in range(n):
        buttons: list[str] = []
        if hold_fwd[f]:
            buttons.append(dir_btn)
        if hold_dash[f]:
            buttons.append(dash_button)
        if store_on_last and f == last:
            buttons.append("DOWN")
        plan.append(tuple(buttons))
    return plan


__all__ = [
    "Direction",
    "Region",
    "NTSC_MAGIC_DASH_FRAMES",
    "PAL_MAGIC_DASH_FRAMES",
    "NTSC_STUTTER_MIN_PX",
    "PAL_STUTTER_MIN_PX",
    "NTSC_STUTTER_FULL_STOP_PX",
    "NTSC_SHORT_CHARGE_FRAMES",
    "PAL_SHORT_CHARGE_FRAMES",
    "magic_dash_frames",
    "stutter_forward_mask",
    "stutter_dash_mask",
    "short_charge_plan",
]
