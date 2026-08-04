"""Thin Super+ ``play_<tip>`` / ``run_<tip>`` bindings for the segment registry.

Not a second tip runner. Tip ids come from Super+ rows
(:data:`~super_metroid.routes.kpdr.hops.SUPER_TIP_BY_ID`). Callables forward
to injected ``play_tip`` / ``run_tip`` (passed in to avoid circular imports
with :mod:`super_metroid.routes.continuous`).

Prefer ``run_to(tip_id)`` / ``play_tip`` for new code. New Super+ tips
auto-gain aliases when added to the tip-spec table — do not hand-write
extra runners here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

__all__ = [
    "build_post_supers_aliases",
    "install_post_supers_aliases",
]


def build_post_supers_aliases(
    tip_ids: Iterable[str],
    *,
    play_spec: Callable[..., Any],
    run_spec: Callable[..., Any],
) -> dict[str, Callable[..., Any]]:
    """Return ``play_<id>`` / ``run_<id>`` callables for each post-Supers tip id.

    Closures capture ``tip_id`` via default args so the loop is safe. Names and
    docs match the historical continuous module attributes.
    """
    aliases: dict[str, Callable[..., Any]] = {}
    for tip_id in tip_ids:
        play_name = f"play_{tip_id}"
        run_name = f"run_{tip_id}"

        def _play(
            session: object,
            splits: list[object],
            segments_list: list[object],
            *,
            _tip: str = tip_id,
            _play_spec: Callable[..., Any] = play_spec,
        ) -> object:
            return _play_spec(_tip, session, splits, segments_list)

        _play.__name__ = play_name
        _play.__qualname__ = play_name
        _play.__doc__ = f"Play continuous tip {tip_id!r} via TipSpec."
        aliases[play_name] = _play

        def _run(
            *,
            _tip: str = tip_id,
            _run_spec: Callable[..., Any] = run_spec,
            **kwargs: object,
        ) -> object:
            return _run_spec(_tip, **kwargs)

        _run.__name__ = run_name
        _run.__qualname__ = run_name
        _run.__doc__ = f"Power-on continuous tip {tip_id!r} via TipSpec."
        aliases[run_name] = _run

    return aliases


def install_post_supers_aliases(
    namespace: dict[str, Any],
    tip_ids: Iterable[str] | Mapping[str, object],
    *,
    play_spec: Callable[..., Any],
    run_spec: Callable[..., Any],
    overwrite: bool = False,
) -> dict[str, Callable[..., Any]]:
    """Build aliases and install into ``namespace`` (e.g. continuous ``globals()``).

    When ``overwrite`` is False (default), existing names are left alone
    (``setdefault``). Returns the built alias map for segment-registry wiring.
    """
    ids = tip_ids.keys() if isinstance(tip_ids, Mapping) else tip_ids
    aliases = build_post_supers_aliases(
        ids,
        play_spec=play_spec,
        run_spec=run_spec,
    )
    for name, fn in aliases.items():
        if overwrite:
            namespace[name] = fn
        else:
            namespace.setdefault(name, fn)
    return aliases
