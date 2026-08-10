"""Stable VisualRL exception hierarchy (single owner).

Frozen by the v0.7 master plan (stage 2/3). Exactly six public exception
types are exposed to API callers; every internal failure either raises one of
them directly or is converted at the ``Experiment.run()``/``validate()``
boundary with ``raise ... from exc`` preserving the cause:

- configuration/selection problems -> :class:`ConfigError`/:class:`ComponentError`
- validation failures -> :class:`ValidationError`
- run-time reward/adapter/update/distributed failures -> :class:`RunError`
- resume locator/mechanical restore failures -> :class:`ResumeError`
- artifact write/commit failures -> :class:`ArtifactError`

No seventh sibling exception leaks to users, and no same-named wrapper
hierarchy is kept. Existing call sites migrate to these types in the later
atomic cutover stages; this module only owns the definitions.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ArtifactError",
    "ComponentError",
    "ConfigError",
    "ResumeError",
    "RunError",
    "UnknownComponentError",
    "ValidationError",
    "VisualRLError",
]


def attach_cleanup_notes(
    primary: BaseException,
    cleanup_errors: tuple[BaseException, ...],
) -> None:
    """Attach stable, non-sensitive cleanup diagnostics to a primary error.

    Cleanup must never replace the exception that stopped training.  Only the
    exception type and the fixed owner label assigned by the Runner are
    retained; messages, paths, configuration values, and credentials are
    deliberately excluded.
    """

    if not isinstance(primary, BaseException):
        raise TypeError("primary must be an exception")
    if type(cleanup_errors) is not tuple or any(
        not isinstance(error, BaseException) for error in cleanup_errors
    ):
        raise TypeError("cleanup_errors must be a tuple of exceptions")
    projected = tuple(
        (
            type(error).__name__,
            _cleanup_owner(error),
        )
        for error in cleanup_errors
    )
    if not projected:
        return
    note = "VisualRL cleanup failures: " + ", ".join(
        f"{error_type}@{owner}" for error_type, owner in projected
    )
    try:
        add_note = getattr(primary, "add_note", None)
        if callable(add_note):
            add_note(note)
            return
    except BaseException:  # noqa: BLE001,S110
        pass
    try:
        primary._visual_rl_cleanup_notes = projected
    except BaseException:  # noqa: BLE001
        return


def _cleanup_owner(error: BaseException) -> str:
    owner = getattr(error, "_visual_rl_cleanup_owner", "unknown")
    if owner not in {
        "coordinator",
        "components",
        "progress",
        "artifact_manager",
        "unknown",
    }:
        return "unknown"
    return owner


class VisualRLError(Exception):
    """Internal base class for every stable VisualRL exception.

    This base itself is not part of the public allowlist; only the six
    concrete subclasses below are. Catch it only inside VisualRL boundary
    code that re-raises a stable public type.
    """


class ConfigError(VisualRLError):
    """YAML/schema/parameter error detected before any model load.

    Carries the offending ``key`` (dotted config key, when known) and
    ``path`` (filesystem path of the YAML file or referenced file, when
    known) as structured optional fields.
    """

    def __init__(
        self,
        message: str,
        *,
        key: str | None = None,
        path: str | None = None,
    ) -> None:
        super().__init__(message)
        self.key = key
        self.path = path


class ComponentError(VisualRLError):
    """Component selection/definition error (unknown or unqualified builtin).

    Carries the component ``kind``/``name`` as structured optional fields.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: str | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.name = name


class UnknownComponentError(ComponentError):
    """Unknown builtin component name; lists the available same-kind names.

    This subclass is deliberately absent from the top-level public
    ``__all__``; callers see it as :class:`ComponentError`.
    """

    def __init__(
        self,
        kind: str,
        name: str,
        available: tuple[str, ...] = (),
    ) -> None:
        listing = ", ".join(available) if available else "<none>"
        super().__init__(
            f"Unknown {kind} component {name!r}; available {kind} components: {listing}",
            kind=kind,
            name=name,
        )
        self.available = tuple(available)


class ValidationError(VisualRLError):
    """Merged validation report failed before run directory/GPU/checkpoint.

    ``checks`` carries the structured ``ValidationCheck`` items (a tuple) of
    the failing report when the failure is expressible as checks; it is
    ``None`` for failures that cannot be represented as checks.
    """

    def __init__(
        self,
        message: str,
        *,
        checks: tuple[Any, ...] | None = None,
    ) -> None:
        super().__init__(message)
        self.checks = checks


class ExecutionTransformCompatibilityError(ValueError):
    """An immutable execution-transform plan is statically unsafe."""


class RunError(VisualRLError):
    """Run-time reward/adapter/update/distributed failure.

    ``step`` carries the logical step being executed when the failure
    surfaced, when known. Causes are preserved via ``raise ... from exc``.
    """

    def __init__(self, message: str, *, step: int | None = None) -> None:
        super().__init__(message)
        self.step = step


class ResumeError(VisualRLError):
    """Resume locator or mechanical training-state restore failure.

    ``path`` carries the resume/checkpoint path involved, when known.
    """

    def __init__(self, message: str, *, path: str | None = None) -> None:
        super().__init__(message)
        self.path = path


class ArtifactError(VisualRLError):
    """Artifact staging/commit failure.

    ``path`` carries the artifact path involved, when known.
    """

    def __init__(self, message: str, *, path: str | None = None) -> None:
        super().__init__(message)
        self.path = path
