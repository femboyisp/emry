"""Coerce framework scalar values to plain Python floats.

Training loops hand Emry metric values that may be Python ``int``/``float``,
0-dim ``torch.Tensor``s, or numpy scalars. This module reduces any of those to a
``float`` for emission.

It deliberately imports neither ``torch`` nor ``numpy`` — both are optional
dependencies — and instead duck-types on ``.item()`` and ``float()``. Non-finite
values (``nan``/``inf``) pass through unchanged; detecting them is the engine's
job (the anomaly processor), not the coercion layer's.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = ["to_float", "coerce_metrics"]


def to_float(value: Any) -> float:
    """Coerce a single scalar ``value`` to a ``float``.

    Accepts Python numbers (``int``/``float``/``bool``), any object exposing a
    scalar ``.item()`` (0-dim ``torch.Tensor``, numpy scalar/0-dim array), and
    anything convertible via ``float()``.

    Raises ``TypeError`` if the value is not a scalar (e.g. a multi-element
    tensor/array, ``None``, or a string).
    """
    # bool is a subclass of int; both convert cleanly.
    if isinstance(value, (int, float)):
        return float(value)

    # torch.Tensor / numpy scalar / 0-dim array expose .item().
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar: Any = item()
        except (ValueError, RuntimeError) as exc:
            # e.g. tensor.item() on a multi-element tensor.
            raise TypeError(
                f"cannot coerce non-scalar {type(value).__name__} to float: {exc}"
            ) from exc
        try:
            return float(scalar)
        except (TypeError, ValueError) as exc:
            # e.g. a 0-dim complex tensor: .item() succeeds but isn't a real.
            raise TypeError(
                f"cannot coerce {type(value).__name__}.item() to float: {exc}"
            ) from exc

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"cannot coerce {type(value).__name__} to a metric float"
        ) from exc


def coerce_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Coerce every value in a name → value mapping with :func:`to_float`.

    Raises ``TypeError`` (naming the offending key) if any value is not a scalar.
    """
    out: dict[str, float] = {}
    for name, value in metrics.items():
        try:
            out[name] = to_float(value)
        except TypeError as exc:
            raise TypeError(f"metric {name!r}: {exc}") from exc
    return out
