"""Layered schema-v2 preflight and runtime binding contracts."""

from __future__ import annotations

from visual_rl.composition.preflight.artifacts import (
    ArtifactIdentityError,
    FilesystemArtifactIdentityResolver,
)
from visual_rl.composition.preflight.environment import run_environment_preflight
from visual_rl.composition.preflight.runtime import bind_runtime, bind_runtime_graph
from visual_rl.composition.preflight.static import run_static_preflight
from visual_rl.composition.preflight.types import (
    ArtifactIdentityRequest,
    ArtifactIdentityResolution,
    ArtifactIdentityResolver,
    EnvironmentPreflightResult,
    RuntimeBindInput,
    RuntimeBindResult,
    RuntimeFacts,
    RuntimeGraphBindInput,
    RuntimeGraphBindResult,
    StaticPreflightResult,
    runtime_launch_payload_id,
)

__all__ = (
    "ArtifactIdentityError",
    "ArtifactIdentityRequest",
    "ArtifactIdentityResolution",
    "ArtifactIdentityResolver",
    "EnvironmentPreflightResult",
    "FilesystemArtifactIdentityResolver",
    "RuntimeBindInput",
    "RuntimeBindResult",
    "RuntimeFacts",
    "RuntimeGraphBindInput",
    "RuntimeGraphBindResult",
    "StaticPreflightResult",
    "bind_runtime",
    "bind_runtime_graph",
    "run_environment_preflight",
    "run_static_preflight",
    "runtime_launch_payload_id",
)
