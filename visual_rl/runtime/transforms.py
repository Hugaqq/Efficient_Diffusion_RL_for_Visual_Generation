"""Execution-transform binding for one already prepared runtime root."""

from __future__ import annotations

from visual_rl.runtime.types import (
    ProductionRuntimeError,
    TransformExecution,
    TransformExecutor,
    TransformRequest,
)

__all__ = ("execute_runtime_transforms",)


def execute_runtime_transforms(
    request: TransformRequest,
    executor: TransformExecutor | None,
) -> TransformExecution:
    """Apply one immutable plan while preserving prepared owner identities."""

    if not isinstance(request, TransformRequest):
        raise TypeError("request must be TransformRequest")
    plan = request.plan
    if not plan.transforms:
        return TransformExecution(plan.plan_id, ())
    unsafe = tuple(
        transform.transform_id
        for transform in plan.transforms
        if not transform.preserves_parameter_identity
        or not transform.preserves_state_dict_keys
    )
    if unsafe:
        raise ProductionRuntimeError(
            "the current transform executor cannot replace the prepared "
            "root/optimizer; transforms must preserve parameter identity "
            f"and state-dict keys: {list(unsafe)}"
        )
    if executor is None:
        raise ProductionRuntimeError(
            "a non-empty execution transform plan requires an executor"
        )
    result = executor.execute(request)
    if not isinstance(result, TransformExecution):
        raise TypeError("transform executor must return TransformExecution")
    if result.plan_id != plan.plan_id:
        raise ProductionRuntimeError("transform executor returned the wrong plan id")
    if result.applied_transform_ids != plan.transform_ids:
        raise ProductionRuntimeError(
            "transform executor did not preserve declared transform order"
        )
    return result
