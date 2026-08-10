"""Session-owned container for one materialized reward resource pool."""

from __future__ import annotations

from collections.abc import Mapping

from visual_rl.core.contracts import RewardPlanSpec
from visual_rl.core.types import FrozenMapping
from visual_rl.runtime.reward_resources import (
    AcquiredRewardResource,
    RewardPool,
    RewardPoolView,
    RewardResourceAcquireRequest,
    RewardResourceBindingFacts,
    RewardResourceFactory,
    RewardResourceState,
    RuntimeResourceAcquisitionError,
    _acquisition_request_fingerprint,
    bound_reward_resource_id,
)

__all__ = ("DefaultRuntimeResourceContainer",)


class DefaultRuntimeResourceContainer:
    """One session owner for a materialized reward pool and its G3 evidence."""

    def __init__(self, factory: RewardResourceFactory) -> None:
        if not isinstance(factory, RewardResourceFactory):
            raise TypeError("factory must implement RewardResourceFactory")
        self._factory = factory
        self._state = RewardResourceState.DECLARED
        self._pool: RewardPool | None = None
        self._view: RewardPoolView | None = None
        self._plan: RewardPlanSpec | None = None
        self._bound_ids = FrozenMapping()
        self._acquisition_request_fingerprints = FrozenMapping()
        self._ever_acquired = False

    @property
    def state(self) -> RewardResourceState:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state is RewardResourceState.ACTIVE

    @property
    def bound_reward_resource_ids(self) -> FrozenMapping:
        if not self._ever_acquired:
            raise RuntimeError("reward resources have not been acquired")
        return self._bound_ids

    @property
    def acquisition_request_fingerprints(self) -> FrozenMapping:
        """Canonical hashes of the exact requests accepted by ``acquire``."""

        if not self._ever_acquired:
            raise RuntimeError("reward resources have not been acquired")
        return self._acquisition_request_fingerprints

    @property
    def plan(self) -> RewardPlanSpec:
        if self._plan is None:
            raise RuntimeError("reward resources have not been acquired")
        return self._plan

    def acquire(
        self,
        plan: RewardPlanSpec,
        requests: tuple[RewardResourceAcquireRequest, ...],
    ) -> RewardPoolView:
        """Acquire exact unique specs once and return a non-owning view."""

        if self._state is not RewardResourceState.DECLARED:
            raise RuntimeError("runtime reward resources are acquire-once")
        try:
            request_by_spec = self._validate_requests(plan, requests)
            request_fingerprints = self._request_fingerprints(plan, request_by_spec)
        except BaseException:
            self._state = RewardResourceState.CLOSED
            raise

        observed: dict[str, RewardResourceBindingFacts] = {}
        accepted_resource_objects: set[int] = set()

        def acquire_one(resource_spec_id: str) -> object:
            request = request_by_spec[resource_spec_id]
            candidate = self._factory(request)
            if not isinstance(candidate, AcquiredRewardResource):
                primary = TypeError(
                    "reward resource factory must return AcquiredRewardResource"
                )
                self._close_invalid_output(candidate, primary)
                raise primary
            resource = candidate.resource
            already_owned = id(resource) in accepted_resource_objects
            try:
                if already_owned:
                    raise RuntimeResourceAcquisitionError(
                        "factory reused one physical object for distinct resource specs"
                    )
                self._validate_binding_facts(request, candidate.binding_facts)
            except BaseException as primary:
                if not already_owned:
                    self._close_candidate(resource, primary)
                raise
            accepted_resource_objects.add(id(resource))
            observed[resource_spec_id] = candidate.binding_facts
            return resource

        pool: RewardPool | None = None
        try:
            pool = RewardPool(plan, acquire_one)
            bound_ids = FrozenMapping(
                {
                    resource_spec_id: bound_reward_resource_id(
                        resource_spec_id,
                        observed[resource_spec_id],
                    )
                    for resource_spec_id in plan.resource_identities
                }
            )
            view = pool.view()
        except BaseException as primary:
            self._state = RewardResourceState.CLOSED
            if pool is not None:
                try:
                    pool.close()
                except BaseException as cleanup_error:  # noqa: BLE001
                    if hasattr(primary, "add_note"):
                        primary.add_note(
                            "post-acquisition reward pool rollback failed: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
            raise

        self._pool = pool
        self._view = view
        self._plan = plan
        self._bound_ids = bound_ids
        self._acquisition_request_fingerprints = request_fingerprints
        self._ever_acquired = True
        self._state = RewardResourceState.ACQUIRED
        return view

    def assert_acquisition_requests_match(
        self,
        plan: RewardPlanSpec,
        requests: tuple[RewardResourceAcquireRequest, ...],
    ) -> None:
        """Fail closed unless a reuse request is exactly the acquired request set."""

        if self._state not in {
            RewardResourceState.ACQUIRED,
            RewardResourceState.ACTIVE,
        }:
            raise RuntimeError(
                "acquisition requests are unavailable in the current state"
            )
        if self._plan != plan:
            raise RuntimeResourceAcquisitionError(
                "pre-acquired reward container holds a different plan"
            )
        request_by_spec = self._validate_requests(plan, requests)
        fingerprints = self._request_fingerprints(plan, request_by_spec)
        if fingerprints != self._acquisition_request_fingerprints:
            raise RuntimeResourceAcquisitionError(
                "pre-acquired reward acquisition request fingerprints differ"
            )

    def view(self) -> RewardPoolView:
        """Return the same non-owning view while the session owner is live."""

        if self._state not in {
            RewardResourceState.ACQUIRED,
            RewardResourceState.ACTIVE,
        }:
            raise RuntimeError("reward pool view is unavailable in the current state")
        assert self._view is not None
        return self._view

    def activate(self) -> None:
        if self._state is not RewardResourceState.ACQUIRED:
            raise RuntimeError("reward resource activation requires ACQUIRED state")
        assert self._pool is not None
        try:
            self._pool.activate()
        except BaseException:
            self._state = RewardResourceState.CLOSED
            raise
        self._state = RewardResourceState.ACTIVE

    def close(self) -> None:
        if self._state is RewardResourceState.CLOSED:
            return
        pool = self._pool
        self._state = RewardResourceState.CLOSED
        if pool is not None:
            pool.close()

    @staticmethod
    def _validate_requests(
        plan: RewardPlanSpec,
        requests: tuple[RewardResourceAcquireRequest, ...],
    ) -> Mapping[str, RewardResourceAcquireRequest]:
        if not isinstance(plan, RewardPlanSpec):
            raise TypeError("plan must be a RewardPlanSpec")
        if plan.provisional:
            raise ValueError("runtime resources require a materialized reward plan")
        if type(requests) is not tuple or not requests:
            raise ValueError("requests must be a non-empty tuple")
        if any(not isinstance(item, RewardResourceAcquireRequest) for item in requests):
            raise TypeError("requests must contain RewardResourceAcquireRequest values")
        observed_ids = tuple(item.reward_resource_spec_id for item in requests)
        if observed_ids != plan.resource_identities:
            raise ValueError(
                "acquisition requests must exactly cover unique plan resource specs "
                "in canonical order"
            )
        return {item.reward_resource_spec_id: item for item in requests}

    @staticmethod
    def _request_fingerprints(
        plan: RewardPlanSpec,
        request_by_spec: Mapping[str, RewardResourceAcquireRequest],
    ) -> FrozenMapping:
        return FrozenMapping(
            {
                resource_spec_id: _acquisition_request_fingerprint(
                    request_by_spec[resource_spec_id]
                )
                for resource_spec_id in plan.resource_identities
            }
        )

    @staticmethod
    def _validate_binding_facts(
        request: RewardResourceAcquireRequest,
        facts: RewardResourceBindingFacts,
    ) -> None:
        if not isinstance(facts, RewardResourceBindingFacts):
            raise TypeError("factory binding_facts must be RewardResourceBindingFacts")
        descriptor = request.descriptor
        if facts.protocol != descriptor.protocol:
            raise RuntimeResourceAcquisitionError(
                "observed reward protocol differs from its descriptor"
            )
        if facts.protocol_version != descriptor.protocol_version:
            raise RuntimeResourceAcquisitionError(
                "observed reward protocol version differs from its descriptor"
            )
        device_domain = facts.device.split(":", 1)[0]
        policy = descriptor.allowed_runtime_policy
        if device_domain not in policy.allowed_devices:
            raise RuntimeResourceAcquisitionError(
                f"observed reward device {facts.device!r} violates allowed policy"
            )
        if facts.dtype not in policy.allowed_dtypes:
            raise RuntimeResourceAcquisitionError(
                f"observed reward dtype {facts.dtype!r} violates allowed policy"
            )
        if facts.worker_domain not in policy.allowed_worker_domains:
            raise RuntimeResourceAcquisitionError(
                "observed reward worker domain violates allowed policy"
            )

    @staticmethod
    def _close_invalid_output(
        candidate: object,
        primary: BaseException,
    ) -> None:
        close = getattr(candidate, "close", None)
        if callable(close):
            try:
                close()
            except BaseException as cleanup_error:  # noqa: BLE001
                if hasattr(primary, "add_note"):
                    primary.add_note(
                        "invalid factory output rollback failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )

    @staticmethod
    def _close_candidate(resource: object, primary: BaseException) -> None:
        try:
            resource.close()  # type: ignore[attr-defined]
        except BaseException as cleanup_error:  # noqa: BLE001
            if hasattr(primary, "add_note"):
                primary.add_note(
                    "candidate reward resource rollback failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
