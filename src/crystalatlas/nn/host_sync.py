"""Device-side failure flags read back in one host synchronization (review 2026-10-29, H1).

Every finiteness, handedness and rank check of the geometry field used to end in
its own ``.item()``: 25 host synchronizations per packed field call on a
four-atom P1 cover, before the Transformer ran, and 2 more per RK4 stage in the
integrator.  On a GPU each read drains the stream.  :class:`DeviceChecks`
keeps the checks and their errors on the device and reads them once: the flags
are stacked and read with a single ``.item()``; only when that read reports a
failure are the flags read one by one, so the first failing check raises
exactly the error it always raised.  Nothing is skipped or waived; the strict
messages and exception types are unchanged, and a function called without a
``checks`` argument still reads its own flags before returning.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

from .._optional import require_torch

torch = require_torch()
Tensor = torch.Tensor


class DeviceChecks:
    """Failure flags gathered on the device and raised after one host read."""

    __slots__ = ("_errors", "_flags")

    def __init__(self) -> None:
        self._flags: list[Tensor] = []
        self._errors: list[Callable[[], Exception]] = []

    def __len__(self) -> int:
        return len(self._flags)

    def add(self, failed: Tensor, error: Callable[[], Exception]) -> None:
        """Record ``failed`` (one boolean, true on failure) and the error it raises.

        ``error`` is called only when the flag is set, so a message may read
        device values (that read happens on the failure path alone).
        """

        if not isinstance(failed, Tensor) or failed.dtype != torch.bool or failed.numel() != 1:
            raise ValueError("a device check flag must be a single boolean tensor")
        self._flags.append(failed.reshape(()))
        self._errors.append(error)

    def raise_first(self) -> None:
        """Read every flag with one synchronization; raise the first failure's error."""

        if not self._flags:
            return
        device = self._flags[0].device
        stacked = torch.stack([flag.to(device=device) for flag in self._flags])
        if not bool(stacked.any().item()):
            return
        for flag, error in zip(self._flags, self._errors, strict=True):
            if bool(flag.item()):
                raise error()
        raise AssertionError("a set device check flag disappeared")  # pragma: no cover

    @contextmanager
    def deferred(self) -> Iterator[DeviceChecks]:
        """Run a block whose flags are read once at its end.

        A ``RuntimeError`` raised inside the block (a singular solve, a failed
        SVD) yields to any check that failed before it, so invalid inputs still
        report their own error rather than the first operation they broke.
        """

        try:
            yield self
        except RuntimeError:
            self.raise_first()
            raise
        self.raise_first()


def check(checks: DeviceChecks | None, failed: Tensor, error: Callable[[], Exception]) -> None:
    """Defer ``failed`` to ``checks``, or read it now when no collector is given."""

    if checks is None:
        if bool(failed.item()):
            raise error()
    else:
        checks.add(failed, error)


__all__ = ["DeviceChecks", "check"]
