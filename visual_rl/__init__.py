"""VisualRL v0.7 public Python API."""

from visual_rl.api import audit_run, inspect_run, load
from visual_rl.api_types import AuditReport, RunResult, RunStatus, ValidationReport
from visual_rl.callbacks import Callback, CallbackEvent
from visual_rl.errors import (
    ArtifactError,
    ComponentError,
    ConfigError,
    ResumeError,
    RunError,
    ValidationError,
)

__version__ = "0.7.0"

__all__ = (  # noqa: RUF022 - preserve the documented public API order
    "__version__",
    "load",
    "inspect_run",
    "audit_run",
    "Callback",
    "CallbackEvent",
    "ValidationReport",
    "RunResult",
    "RunStatus",
    "AuditReport",
    "ConfigError",
    "ComponentError",
    "ValidationError",
    "RunError",
    "ResumeError",
    "ArtifactError",
)
