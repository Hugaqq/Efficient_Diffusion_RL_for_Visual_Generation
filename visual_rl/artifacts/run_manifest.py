"""Canonical recipe and launch manifests for the v0.8 artifact boundary."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from visual_rl.core.serialization import canonical_json_text, strict_json_load
from visual_rl.composition.recipes.schema import MaterializedRecipe
from visual_rl.composition.config.specs import LaunchSpec
from visual_rl.core.contracts.composition import ComponentArtifactBindingSet
from visual_rl.core.serialization import to_plain_dict
from visual_rl.errors import ArtifactError
from visual_rl.composition.preflight.types import RuntimeBindResult

__all__ = (
    "RECIPE_MANIFEST_SCHEMA_VERSION",
    "assert_launch_manifest_resume_compatible",
    "assert_recipe_manifest_resume_compatible",
    "launch_manifest_payload",
    "recipe_manifest_payload",
    "write_launch_manifest",
    "write_recipe_manifest",
)

RECIPE_MANIFEST_SCHEMA_VERSION = 2


def recipe_manifest_payload(
    recipe: MaterializedRecipe,
    artifact_binding_set: ComponentArtifactBindingSet,
) -> dict[str, Any]:
    """Project one typed materialized recipe and its separately derived G1 set."""

    if not isinstance(recipe, MaterializedRecipe):
        raise TypeError("recipe must be a MaterializedRecipe")
    if not isinstance(artifact_binding_set, ComponentArtifactBindingSet):
        raise TypeError("artifact_binding_set must be a ComponentArtifactBindingSet")
    if artifact_binding_set.recipe_id != recipe.recipe_id:
        raise ValueError("artifact binding set differs from MaterializedRecipe")
    resolved = recipe.resolved
    payload = {
        "schema_version": RECIPE_MANIFEST_SCHEMA_VERSION,
        "kind": "materialized_recipe",
        "identity": {
            "recipe_id": recipe.recipe_id,
            "resolved_fingerprint": resolved.resolved_fingerprint,
            "algorithm_declaration_id": resolved.algorithm.declaration_id,
            "algorithm_materialization_spec_id": resolved.algorithm_spec.spec_id,
            "execution_policy_id": resolved.execution_policy.policy_id,
            "reward_plan_id": recipe.reward_plan.plan_id,
            "source_plan_id": resolved.source_plan.plan_id,
            "source_content_binding_id": (
                recipe.source_content_binding.content_binding_id
            ),
            "component_artifact_binding_set_id": (artifact_binding_set.binding_set_id),
        },
        "resolved_recipe": resolved.canonical_semantic_payload(),
        "materialized_recipe": recipe.canonical_semantic_payload(),
        "component_artifact_bindings": artifact_binding_set.to_payload(),
        "compatibility_inspection": resolved.compatibility.inspection_payload(),
    }
    plain = to_plain_dict(payload)
    if not isinstance(plain, dict):  # pragma: no cover - root is statically a mapping
        raise TypeError("recipe manifest projection must be a plain mapping")
    return plain


def write_recipe_manifest(
    path: str | Path,
    recipe: MaterializedRecipe,
    artifact_binding_set: ComponentArtifactBindingSet,
) -> Path:
    """Atomically write canonical ``recipe.resolved.json`` bytes."""

    return _write_manifest(
        path,
        recipe_manifest_payload(recipe, artifact_binding_set),
        kind="recipe",
    )


def launch_manifest_payload(
    runtime: RuntimeBindResult,
    launch: LaunchSpec,
) -> dict[str, Any]:
    """Project launch-only facts while retaining the referenced recipe id."""

    if not isinstance(runtime, RuntimeBindResult):
        raise TypeError("runtime must be a RuntimeBindResult")
    if not isinstance(launch, LaunchSpec):
        raise TypeError("launch must be a LaunchSpec")
    return {
        **runtime.launch_manifest_payload(),
        "launch": {
            "output_dir": str(launch.output_dir),
            "resume_from": (
                None if launch.resume_from is None else str(launch.resume_from)
            ),
            "checkpoint_every_optimizer_steps": (
                launch.checkpoint_every_optimizer_steps
            ),
            "artifacts": launch.artifacts.to_payload(),
        },
    }


def write_launch_manifest(
    path: str | Path,
    runtime: RuntimeBindResult,
    launch: LaunchSpec,
) -> Path:
    """Atomically write canonical ``launch.resolved.json`` bytes."""

    return _write_manifest(
        path,
        launch_manifest_payload(runtime, launch),
        kind="launch",
    )


def assert_recipe_manifest_resume_compatible(
    path: str | Path,
    recipe: MaterializedRecipe,
    artifact_binding_set: ComponentArtifactBindingSet,
) -> Path:
    """Require exact typed identities while ignoring diagnostics-only wording."""

    expected = recipe_manifest_payload(recipe, artifact_binding_set)
    observed, destination = _read_canonical_manifest(path, kind="recipe")
    if set(observed) != set(expected):
        raise ArtifactError(
            "existing recipe manifest schema differs from the current recipe",
            path=str(destination),
        )
    if (
        observed.get("schema_version") != RECIPE_MANIFEST_SCHEMA_VERSION
        or observed.get("kind") != "materialized_recipe"
    ):
        raise ArtifactError(
            "existing recipe manifest uses an unsupported identity schema",
            path=str(destination),
        )
    ignored = frozenset({"compatibility_inspection"})
    if _without_top_level(observed, ignored) != _without_top_level(expected, ignored):
        raise ArtifactError(
            "existing recipe manifest semantic payload differs from the current recipe",
            path=str(destination),
        )
    return destination


def assert_launch_manifest_resume_compatible(
    path: str | Path,
    runtime: RuntimeBindResult,
    launch: LaunchSpec,
) -> Path:
    """Keep the historical resume locator while requiring one launch topology."""

    expected = launch_manifest_payload(runtime, launch)
    observed, destination = _read_canonical_manifest(path, kind="launch")
    if set(observed) != set(expected):
        raise ArtifactError(
            "existing launch manifest schema differs from the current launch",
            path=str(destination),
        )
    observed_launch = observed.get("launch")
    expected_launch = expected["launch"]
    if not isinstance(observed_launch, dict) or not isinstance(expected_launch, dict):
        raise ArtifactError(
            "existing launch manifest has an invalid launch payload",
            path=str(destination),
        )
    if set(observed_launch) != set(expected_launch):
        raise ArtifactError(
            "existing launch manifest launch schema differs",
            path=str(destination),
        )
    # Artifact locations are launch audit only.  Resume equivalence is already
    # locked by the path-free MaterializedRecipe and checkpoint contract.
    ignored = frozenset({"output_dir", "resume_from", "artifacts"})
    observed_projection = {
        **_without_top_level(
            observed,
            frozenset({"runtime_facts", "launch_audit", "launch_audit_id"}),
        ),
        "launch": _without_top_level(observed_launch, ignored),
    }
    expected_projection = {
        **_without_top_level(
            expected,
            frozenset({"runtime_facts", "launch_audit", "launch_audit_id"}),
        ),
        "launch": _without_top_level(expected_launch, ignored),
    }
    if observed_projection != expected_projection:
        raise ArtifactError(
            "existing launch manifest payload differs from the current launch",
            path=str(destination),
        )
    return destination


def _write_manifest(
    path: str | Path,
    payload: dict[str, Any],
    *,
    kind: str,
) -> Path:
    if not isinstance(path, (str, Path)) or isinstance(path, bool):
        raise TypeError("manifest path must be str or Path")
    data = (canonical_json_text(payload) + "\n").encode("utf-8")
    destination = _manifest_destination(path, kind=kind, create_parent=True)
    if _existing_manifest_is_exact(destination, data, kind=kind):
        return destination
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            text=True,
        )
    except OSError as exc:
        raise ArtifactError(
            f"cannot stage {kind} manifest",
            path=str(destination),
        ) from exc
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            if not _existing_manifest_is_exact(destination, data, kind=kind):
                raise AssertionError("unreachable manifest identity branch")
        temporary.unlink()
        _fsync_directory(destination.parent)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise ArtifactError(
            f"cannot commit {kind} manifest",
            path=str(destination),
        ) from exc
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return destination


def _manifest_destination(
    path: str | Path,
    *,
    kind: str,
    create_parent: bool,
) -> Path:
    if not isinstance(path, (str, Path)) or isinstance(path, bool):
        raise TypeError("manifest path must be str or Path")
    requested = Path(path).expanduser()
    destination = Path(os.path.abspath(requested))
    parent = destination.parent
    if parent.is_symlink():
        raise ArtifactError(
            f"{kind} manifest parent cannot be a symlink",
            path=str(parent),
        )
    if create_parent:
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ArtifactError(
                f"cannot create {kind} manifest directory",
                path=str(parent),
            ) from exc
    if not parent.is_dir() or parent.is_symlink():
        raise ArtifactError(
            f"{kind} manifest parent must be a real directory",
            path=str(parent),
        )
    return destination


def _existing_manifest_is_exact(
    destination: Path,
    expected: bytes,
    *,
    kind: str,
) -> bool:
    try:
        metadata = destination.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ArtifactError(
            f"cannot inspect existing {kind} manifest",
            path=str(destination),
        ) from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise ArtifactError(
            f"{kind} manifest destination cannot be a symlink",
            path=str(destination),
        )
    if not stat.S_ISREG(metadata.st_mode):
        raise ArtifactError(
            f"{kind} manifest destination must be a regular file",
            path=str(destination),
        )
    try:
        observed = destination.read_bytes()
        after = destination.lstat()
    except OSError as exc:
        raise ArtifactError(
            f"cannot read existing {kind} manifest",
            path=str(destination),
        ) from exc
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if not stat.S_ISREG(after.st_mode) or identity(metadata) != identity(after):
        raise ArtifactError(
            f"existing {kind} manifest changed while it was read",
            path=str(destination),
        )
    if observed != expected:
        raise ArtifactError(
            f"existing {kind} manifest differs from canonical payload",
            path=str(destination),
        )
    return True


def _read_canonical_manifest(
    path: str | Path,
    *,
    kind: str,
) -> tuple[dict[str, Any], Path]:
    destination = _manifest_destination(path, kind=kind, create_parent=False)
    try:
        metadata = destination.lstat()
    except OSError as exc:
        raise ArtifactError(
            f"cannot inspect existing {kind} manifest",
            path=str(destination),
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ArtifactError(
            f"existing {kind} manifest must be a regular non-symlink file",
            path=str(destination),
        )
    try:
        payload = strict_json_load(destination)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(
            f"existing {kind} manifest is not strict JSON",
            path=str(destination),
        ) from exc
    if not isinstance(payload, dict):
        raise ArtifactError(
            f"existing {kind} manifest must contain a JSON object",
            path=str(destination),
        )
    canonical = (canonical_json_text(payload) + "\n").encode("utf-8")
    if not _existing_manifest_is_exact(destination, canonical, kind=kind):
        raise AssertionError("unreachable canonical manifest branch")
    return payload, destination


def _without_top_level(
    payload: dict[str, Any],
    ignored: frozenset[str],
) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in ignored}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
