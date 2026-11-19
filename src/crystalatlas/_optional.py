"""Small optional-dependency boundary used by the neural modules."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

from .errors import OptionalDependencyError


def require_module(name: str, *, extra: str) -> ModuleType:
    """Import *name* or raise an actionable, package-specific error."""

    try:
        return import_module(name)
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise OptionalDependencyError(
            f"Optional dependency {name!r} is required; install crystalatlas[{extra}]."
        ) from exc


def require_torch() -> ModuleType:
    """Return :mod:`torch` when the ML extra is installed."""

    return require_module("torch", extra="ml")

