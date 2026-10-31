"""Observed H3 current-only numerics and checkpoint binding; no run loop/PG owner.

Content hashes use the existing filesystem-artifact.v1 codec: all-files.v1
for the complete model directory (transformer AND both VAEs), python-code.v1
for code. Paths never replace content identity. UUID availability/authorization
is a prerequisite of the caller's runtime session, not inferred here.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

import torch

from visual_rl.algorithms.dynamics.h3_cps import H3CPSDynamics
from visual_rl.algorithms.dynamics.h3_schedule import H3ScheduleSnapshot
from visual_rl.algorithms.dynamics.selection import DynamicsSelectionPolicyState
from visual_rl.algorithms.optimization.h3_world import _world_rank
from visual_rl.algorithms.rewards.h3_hps import H3HPSReward
from visual_rl.artifacts.checkpoint.coordination import CheckpointSafePoint
from visual_rl.artifacts.checkpoint.h3_state import H3StateIdentity
from visual_rl.artifacts.checkpoint.protocol import (
    CheckpointContract, CheckpointProgress, ComponentContractRef, ParameterContract, OptimizerGroupContract,
    CHECKPOINT_PROGRESS_SCHEMA_VERSION,
)
from visual_rl.artifacts.checkpoint.state import RankRNGSnapshot
from visual_rl.composition.config.h3_profile import H3Profile, H3LocalFacts, preflight_h3_profile
from visual_rl.composition.preflight.artifacts import _hash_artifact
from visual_rl.core.serialization import to_plain_dict
from visual_rl.data.h3_source import H3PromptSource, PARSE_VERSION, SAMPLING_VERSION, STREAM_VERSION
from visual_rl.models.h3_interface import H3ModelInput, H3PromptInput, H3_GUIDANCE_ID
from visual_rl.models.implementations.h3 import H3Adapter
from visual_rl.models.lifecycle.components import ComponentBinding, ComponentRole, ModelComponents, ExecutionMode
from visual_rl.models.lifecycle.h3_coordination import H3Coordination
from visual_rl.models.lifecycle.h3_fsdp import H3PreparedModel, _canonical_parameter_names, _cuda_device
from visual_rl.models.lifecycle.prepared import validate_optimizer_parameter_subset
from visual_rl.models.numerics.execution import StageExecutionPolicy, ParameterView
from visual_rl.models.numerics.policy import (
    ParameterDTypePolicy, ForwardAutocastPolicy, ParameterViewEvidence,
    ParameterViewMode, ModelExecutionNumericsEvidence,
)
from visual_rl.models.state.parameters import ParameterStateManager


def _hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _id(namespace, payload):
    return namespace + ":" + _hash(payload)


def _unpack_id(value, namespace):
    if type(value) is not str or re.fullmatch(re.escape(namespace) + r":[0-9a-f]{64}", value) is None:
        raise ValueError(f"expected exact {namespace} identity")
    return value[len(namespace)+1:]


def verify_h3_content(profile: H3Profile, *, code_root: Path, expected_code_sha256: str):
    """Always stream actual bytes; no caller-provided verified=True escape hatch."""
    if not isinstance(profile, H3Profile):
        raise TypeError("H3Profile required")
    model = _hash_artifact(profile.launch.model, content_policy="all-files.v1")
    code = _hash_artifact(Path(code_root), content_policy="python-code.v1")
    if (model["content_sha256"] != profile.expected["model_sha256"]
            or profile.expected["cache_identity"]["model_sha256"] != model["content_sha256"]):
        raise ValueError("full model content differs from declared model/cache SHA256")
    if code["content_sha256"] != expected_code_sha256:
        raise ValueError("actual code content identity mismatch")
    return {"model": model, "code": code}


def _block_policy(block):
    """Observe actual FSDP2 compute/reduction settings; CPU tests replace this leaf."""
    from torch.distributed.fsdp import FSDPModule
    if not isinstance(block, FSDPModule):
        raise ValueError("main transformer block is not FSDP2 prepared")
    state = block._get_fsdp_state()
    # PyTorch 2.8 uses one group; newer releases expose a list. Observe
    # the real group in both layouts, never infer precision/division defaults.
    groups = getattr(state, '_fsdp_param_groups', None)
    if groups is None:
        group = getattr(state, '_fsdp_param_group', None)
        groups = () if group is None else (group,)
    if not groups:
        raise ValueError("H3 block has no prepared parameter groups")
    for group in groups:
        policy = group.mp_policy
        if (policy.param_dtype != torch.bfloat16 or policy.reduce_dtype != torch.float32
                or policy.cast_forward_inputs or group.gradient_divide_factor != 8):
            raise ValueError("H3 block mixed precision/reducer drift")
    return dict(param_dtype="bfloat16", reduce_dtype="float32", cast_forward_inputs=False, divisor=8)


class H3RuntimeBinding:
    """Borrow final objects; bind once after prepare, never cast or own their close.

    Typed checkpoint namespaces below identify this CLOSED H3 materialization
    payload, not successful traversal of the old Gaussian recipe materializer.
    The model component ref uses the actual registry declaration identity.
    """
    def __init__(self, *, profile: H3Profile, facts: H3LocalFacts, adapter: H3Adapter,
                 prepared: H3PreparedModel, source: H3PromptSource, scorer: H3HPSReward,
                 process_group, rank: int, device, gpu_uuids: tuple[str, ...],
                 code_root: Path, expected_code_sha256: str):
        if not all(isinstance(value, cls) for value, cls in (
            (profile, H3Profile), (facts, H3LocalFacts), (adapter, H3Adapter),
            (prepared, H3PreparedModel), (source, H3PromptSource), (scorer, H3HPSReward))):
            raise TypeError("actual H3 runtime objects required")
        if type(rank) is not int or rank != _world_rank(process_group):
            raise ValueError("rank differs from actual WORLD8")
        requested = torch.device(device)
        if requested != torch.device(f"cuda:{rank}") or gpu_uuids != profile.launch.gpu_uuids:
            raise ValueError("runtime rank/device/UUID topology differs from profile")
        self.device = _cuda_device(requested)
        self.profile, self.facts = profile, facts
        self.adapter, self.prepared, self.source, self.scorer = adapter, prepared, source, scorer
        self.process_group, self.rank = process_group, rank
        self.code_root, self.expected_code_sha256 = Path(code_root), expected_code_sha256
        self._closed = False
        self._active = ContextVar("h3-current-forward-policy", default=None)
        self.dynamics = H3CPSDynamics(adapter.geometry.joint_layout, H3ScheduleSnapshot.build(profile.training.num_steps))
        self.topology = dict(world_size=8, gpu_uuids=list(gpu_uuids), placement="rank-local-cuda-index",
                             sp=1, tp=1, reducer_divisor=8)
        self.content = verify_h3_content(profile, code_root=self.code_root, expected_code_sha256=expected_code_sha256)
        self._check_resources()
        self._source_binding = source.state_dict()["binding"]
        self._check_source()
        model = adapter.meta_model.model
        if (prepared.prepared_root.component("transformer") is not model
                or prepared.component_names != ("transformer",)):
            raise ValueError("adapter and prepared transformer ownership mismatch")
        try:
            bound = adapter.prepared_components
        except Exception as exc:
            from visual_rl.models.interface import ModelPortError
            if not isinstance(exc, ModelPortError):
                raise
            adapter._bind_prepared_components(prepared)
        else:
            if bound is not prepared:
                raise ValueError("adapter is bound to a different prepared handle")
        self._inventory = self._observe_model()
        components = ModelComponents((ComponentBinding("transformer", model,
            (ComponentRole.INFERENCE, ComponentRole.TRAINABLE), managed_residency=False),))
        self.parameter_state = ParameterStateManager(components)  # final objects, never meta
        projection = self.parameter_state.state_projection
        self.execution_policies = tuple(StageExecutionPolicy.canonical(stage) for stage in
                                         (ExecutionMode.ROLLOUT, ExecutionMode.TRAIN))
        view = ParameterViewEvidence(ParameterView.CURRENT, ParameterViewMode.CURRENT, ("transformer",),
                                     projection.standalone_parameter_names, projection.projection_id, False)
        # compute_dtype describes the root scope; BF16 subblocks are separately
        # observed above. Disabled root autocast must not turn FP32 islands BF16.
        self.numerics = ModelExecutionNumericsEvidence(ParameterDTypePolicy("float32"), tuple(
            ForwardAutocastPolicy(p.stage, ParameterView.CURRENT, self.device.type, "float32", False)
            for p in self.execution_policies), (view,))
        self.contract = self._contract()
        live = self._live_ids()
        self.state_identity = H3StateIdentity(live["topology"], live["model"], live["semantics"],
                                               _unpack_id(source.geometry_id, "h3-geometry"))
        adapter._bind_model_execution_numerics(self.numerics, execution_policy_provider=self._policy)

    def _check_resources(self):
        current = preflight_h3_profile(self.profile)
        if current != self.facts:
            raise ValueError("profile local facts/content drift")
        if self.adapter.config != self.profile.model.config:
            raise ValueError("adapter config differs from actual profile")
        meta = self.adapter.meta_model
        if meta.transformer_dir.resolve() != (self.profile.launch.model / "transformer").resolve():
            raise ValueError("adapter uses a different model artifact")
        if meta.config_sha256 != current.model_config_hashes["transformer"]:
            raise ValueError("loaded model config differs from artifact")
        self.scorer._check_manifest()  # actual re-read; identity alone is insufficient
        if self.scorer._closed or self.scorer._client is None:
            raise ValueError("HPS must already be acquired and open")
        identity = self.scorer.identity
        launch = self.profile.launch
        if (identity.profile != "h3_hpsv3_video_only_v1" or identity.weight != 1
                or identity.service_identity != self.profile.expected["hps_service_identity"]
                or identity.manifest_sha256 != current.hps_manifest_sha256
                or identity.video_contract.payload() != to_plain_dict(current.semantic_payload["video_contract"])
                or self.scorer._request.runtime_binding.endpoint != launch.hps_origin
                or self.scorer._request.runtime_binding.timeout_s != launch.hps_timeout_s
                or self.scorer._request.runtime_binding.max_response_bytes != launch.hps_max_response_bytes):
            raise ValueError("acquired HPS contract differs from profile")

    def _check_source(self):
        state = self.source.state_dict()
        binding, launch = state["binding"], self.profile.launch
        if (self.source._adapter is not self.adapter or self.source._dynamics != self.dynamics
                or state["rank"] != self.rank or state["device"] != str(self.device)
                or binding["world_size"] != 8 or binding["base_seed"] != launch.seed
                or binding["group_size"] != self.profile.training.group_size
                or binding.get('group_layout', 'rank_local') != self.profile.training.group_layout
                or binding["budgets"] != [self.profile.training.local_samples,
                    self.profile.training.local_samples * self.profile.training.num_steps, launch.budgets.trajectory_bytes]
                or binding["train_sha256"] != self.facts.train_sha256 or binding["eval_sha256"] != self.facts.eval_sha256
                or binding["geometry"]["cache_identity"] != self.facts.cache_identity
                or binding["parse"] != PARSE_VERSION or binding["sampling"] != SAMPLING_VERSION
                or binding["stream"] != STREAM_VERSION or binding != self._source_binding):
            raise ValueError("source/profile/model/dynamics identity mismatch")
        return state

    def _observe_model(self):
        meta, root = self.adapter.meta_model, self.prepared.prepared_root
        if root.component("transformer") is not meta.model or self.adapter.prepared_components is not self.prepared:
            raise ValueError("prepared model ownership changed")
        validate_optimizer_parameter_subset(root, self.prepared.optimizer)
        if type(self.prepared.optimizer) is not torch.optim.AdamW or len(self.prepared.optimizer.param_groups) != 1:
            raise ValueError("single-group final AdamW required")
        group = self.prepared.optimizer.param_groups[0]
        if any(group.get(key, False) for key in ("maximize", "amsgrad", "capturable", "differentiable", "fused")):
            raise ValueError("unsupported AdamW numerical mode")
        optimizer_spec = _hash({key: value for key, value in group.items() if key != "params"})
        if hasattr(self, "_optimizer_spec") and optimizer_spec != self._optimizer_spec:
            raise ValueError("optimizer hyperparameter contract drift")
        self._optimizer_spec = optimizer_spec
        for key, expected in dict(lr=1e-4, betas=(.9, .999), eps=1e-12, weight_decay=0.).items():
            if group[key] != expected:
                raise ValueError(f"optimizer {key} differs from fixed H3 profile")
        self._observe_scheduler(group)
        if self.prepared.reducer_divisor != 8:
            raise ValueError("explicit reducer required")
        params = _canonical_parameter_names(meta.model, meta.block_names)
        if {name for name, p in params.items() if p.requires_grad} != set(meta.lora_names):
            raise ValueError("final LoRA parameter scope changed")
        for name, p in params.items():
            spec = meta.state_specs[name]
            if p.is_meta or p.device != self.device or p.dtype != spec.dtype or tuple(p.shape) != spec.shape:
                raise ValueError("final parameter precision/shape/device mismatch: " + name)
        policies = [_block_policy(meta.model.get_submodule(name)) for name in meta.block_names]
        buffers = tuple(meta.model.named_buffers())
        for name, saved in meta.nonpersistent_buffers.items():
            actual = meta.model.get_buffer(name)
            if actual.dtype != saved.dtype or actual.shape != saved.shape or not torch.equal(actual.detach().cpu(), saved):
                raise ValueError("constructor/rope buffer changed")
        # Validate persistent buffer metadata at initial bind too, not merely
        # against the subsequently captured object inventory.
        persistent = set()
        for name, buffer in buffers:
            canonical = name
            for block in meta.block_names:
                prefix = block + '._checkpoint_wrapped_module.'
                if name.startswith(prefix):
                    canonical = block + '.' + name[len(prefix):]
                    break
            if buffer.is_meta or buffer.device != self.device:
                raise ValueError("buffer materialization/device mismatch")
            if canonical in meta.state_specs:
                persistent.add(canonical)
                spec = meta.state_specs[canonical]
                if buffer.dtype != spec.dtype or tuple(buffer.shape) != spec.shape:
                    raise ValueError("persistent buffer precision/shape mismatch")
            elif canonical not in meta.nonpersistent_buffers:
                raise ValueError("unexpected model buffer")
        if set(params) | persistent != set(meta.state_specs):
            raise ValueError("final model parameter/buffer coverage mismatch")
        inventory = tuple((name, id(p), tuple(p.shape), str(p.dtype), p.requires_grad)
                          for name, p in meta.model.named_parameters())
        inventory += tuple((name, id(b), tuple(b.shape), str(b.dtype), False) for name, b in buffers)
        if hasattr(self, "_inventory") and inventory != self._inventory:
            raise ValueError("prepared parameter/buffer object identity or precision drift")
        self.block_policies = tuple(policies)
        return inventory

    def _observe_scheduler(self, group):
        optimizer, scheduler = self.prepared.optimizer, self.prepared.scheduler
        if (type(scheduler) is not torch.optim.lr_scheduler.LambdaLR
                or scheduler.optimizer is not optimizer):
            raise ValueError("constant scheduler must own the prepared optimizer")
        if hasattr(self, "_scheduler") and (scheduler is not self._scheduler or optimizer is not self._optimizer):
            raise ValueError("bound scheduler/optimizer instance changed")
        # Observe the existing prepare/codec constant-LambdaLR representation;
        # never invoke its callable or step either object as a validation probe.
        state = scheduler.state_dict()
        epoch = state["last_epoch"]
        if (type(epoch) is not int or epoch < 0 or type(state["_step_count"]) is not int
                or state["_step_count"] != epoch + 1 or state["base_lrs"] != [1e-4]
                or state["_last_lr"] != [1e-4] or group.get("initial_lr") != 1e-4
                or group["lr"] != 1e-4 or state["lr_lambdas"] != [None]):
            raise ValueError("constant scheduler learning rate/counter state drift")
        if not optimizer.state:
            if epoch != 0:
                raise ValueError("scheduler position differs from cold Adam")
        else:
            if set(optimizer.state) != set(group["params"]):
                raise ValueError("scheduler requires complete Adam counters")
            for saved in optimizer.state.values():
                counter = saved.get("step")
                if (not isinstance(counter, torch.Tensor) or counter.shape != ()
                        or not bool(torch.isfinite(counter)) or counter.item() != epoch or epoch <= 0):
                    raise ValueError("scheduler position differs from Adam step")
        self._scheduler, self._optimizer = scheduler, optimizer

    def _policy(self):
        if self._closed:
            raise RuntimeError("H3 binding closed")
        policy = self._active.get()
        if policy is None:
            return None  # base port rejects missing scope
        if (policy.parameter_view is not ParameterView.CURRENT or policy not in self.execution_policies
                or policy.grad_enabled != torch.is_grad_enabled()):
            raise ValueError("wrong current-view stage/grad scope")
        if torch.is_autocast_enabled(self.device.type):
            raise ValueError("H3 root autocast must be disabled")
        self._observe_model()
        self.parameter_state.named_trainable_parameters()
        if any(module.training for module in self.adapter.meta_model.model.modules()):
            raise ValueError("H3 forward must preserve EVAL mode")
        if self.adapter.model_execution_numerics != self.numerics:
            raise ValueError("bound numerics evidence changed")
        return policy

    @contextmanager
    def scope(self, stage, *, parameter_view=ParameterView.CURRENT):
        if self._closed or self._active.get() is not None or parameter_view is not ParameterView.CURRENT:
            raise ValueError("only one open CURRENT scope is permitted")
        policy = StageExecutionPolicy.canonical(stage, parameter_view=parameter_view)
        if policy not in self.execution_policies:
            raise ValueError("unsupported H3 model stage")
        token = self._active.set(policy)
        try:
            with torch.set_grad_enabled(policy.grad_enabled):
                self._policy()
                yield
        finally:
            self._active.reset(token)

    def predict(self, prepared, *, sample_id, x, index, condition, dynamics):
        if prepared is not self.prepared or dynamics != self.dynamics:
            raise ValueError("prediction uses foreign prepared/dynamics binding")
        typed = self.source.conditioning_for(sample_id)
        if typed.cache_identity != self.facts.cache_identity or not torch.equal(condition, typed.embeddings):
            raise ValueError("prediction condition differs from source/cache identity")
        spec = self.adapter.latent_spec_for_batch(H3PromptInput(typed.prompt), device=x.device, dtype=x.dtype)
        vs, _, aus, _ = dynamics.schedule.transition(index)
        model_input = H3ModelInput(x, vs.to(x.device), typed, None, spec, typed.condition_identity,
                                  (H3_GUIDANCE_ID,), audio_sigma=aus.to(x.device))
        mode = ExecutionMode.TRAIN if torch.is_grad_enabled() else ExecutionMode.ROLLOUT
        with self.scope(mode):
            return self.adapter.predict(model_input).value

    def decode(self, terminal):
        if self._closed:
            raise RuntimeError("H3 binding closed")
        spec = self.adapter.latent_spec_for_batch(H3PromptInput(""), device="cpu", dtype=torch.float16)
        return self.adapter.decode(terminal, spec)

    def score_one(self, **kwargs):
        if self._closed:
            raise RuntimeError("H3 binding closed")
        return self.scorer.score_one(**kwargs)

    def coordination_factory(self, plan):
        if self._closed:
            raise RuntimeError("H3 binding closed")
        return H3Coordination(plan, process_group=self.process_group, device=f"cuda:{self.rank}")

    def _live_ids(self):
        semantics = dict(profile=to_plain_dict(self.facts.semantic_payload),
                         source=self._source_binding, numerics=self.numerics.to_payload(),
                         blocks=self.block_policies)
        return dict(model=self.content["model"]["content_sha256"], topology=_hash(self.topology),
                    semantics=_hash(semantics), data=_unpack_id(self.source.data_identity, "h3-data"),
                    cache=self.facts.cache_identity, score=_hash(asdict(self.scorer.identity)),
                    dynamics=_hash({k: self._source_binding[k] for k in ("num_steps", "video_sigmas", "audio_sigmas", "eta")}),
                    resources=_hash(dict(content=self.content, cache=self.facts.cache_identity,
                                         score=asdict(self.scorer.identity))), code=self.content["code"]["content_sha256"])

    def validate_live(self):
        if self._closed:
            raise RuntimeError("H3 binding closed")
        if (_world_rank(self.process_group) != self.rank
                or _cuda_device(torch.device(f"cuda:{self.rank}")) != self.device
                or self.topology["gpu_uuids"] != list(self.profile.launch.gpu_uuids)):
            raise ValueError("live rank/device/topology changed")
        if verify_h3_content(self.profile, code_root=self.code_root, expected_code_sha256=self.expected_code_sha256) != self.content:
            raise ValueError("artifact content identity changed")
        self._check_resources()
        self._check_source()
        self._observe_model()
        self.parameter_state.named_trainable_parameters()
        self.parameter_state.state_projection
        if self.adapter.model_execution_numerics != self.numerics:
            raise ValueError("numerics binding drift")
        return self._live_ids()

    def _contract(self):
        live = self._live_ids()
        projection = self.parameter_state.state_projection
        params = self.parameter_state.named_trainable_parameters()
        policy_payload = [p.to_payload() for p in self.execution_policies]
        transforms = dict(blocks=self.adapter.meta_model.block_names, activation_checkpoint="non_reentrant",
                          fsdp="world8_shard0", precision=self.block_policies)
        transform_id = _hash(transforms)
        runtime_id = _hash(dict(live=live, projection=projection.to_payload(), numerics=self.numerics.to_payload()))
        artifact_payload = dict(declaration=self.profile.model.declaration_id, model=live["model"], cache=live["cache"])
        component = ComponentContractRef("model", "model", self.profile.model.declaration_id,
                                         _id("component-artifact-binding.v1", artifact_payload), runtime_id)
        dual = self.profile.training.algorithm != "cps"
        implementation = ("visual_rl.algorithms.trainer.h3_dual_stages:H3DualIterationStages" if dual
                          else "visual_rl.algorithms.trainer.h3_stages:H3IterationStages")
        algorithm = dict(implementation=implementation,
                         math=self.profile.training.payload(), dynamics=live["dynamics"], code=live["code"])
        return CheckpointContract(
            recipe_id=_id("materialized-recipe.v2", dict(h3=live["semantics"], resources=live["resources"])),
            resolved_fingerprint=_id("resolved-recipe.v2", to_plain_dict(self.facts.semantic_payload)),
            algorithm_materialization_spec_id=_id("algorithm-materialization-spec.v1", algorithm),
            execution_policy_id=_id("execution-policy.v1", dict(stages=policy_payload, topology=self.topology)),
            reward_plan_id=_id("reward-plan-spec.v1", asdict(self.scorer.identity)),
            source_content_binding_id=_id("source-content-binding.v1", self._source_binding),
            component_artifact_binding_set_id=_id("component-artifact-binding-set.v1", artifact_payload),
            runtime_bound_contract_id=runtime_id, immutable_model_revision=live["model"], code_identity=live["code"],
            components=(component,), model_state_projection=projection, model_state_projection_id=projection.projection_id,
            model_execution_numerics=self.numerics, model_execution_numerics_id=self.numerics.execution_numerics_id,
            trainable_parameters=tuple(ParameterContract(p.name, tuple(p.parameter.shape), str(p.parameter.dtype)) for p in params),
            optimizer_groups=(OptimizerGroupContract("adamw", tuple(p.name for p in params), self._optimizer_spec),),
            scaler_schema="none", lr_scheduler_schema="h3.constant.v1", precision="bf16-blocks-fp32-islands-lora",
            preprocess_identity=_hash(dict(cache=live["cache"], geometry=self.source.geometry_id)),
            preprocess_requirement_set_id=_hash(dict(layer=50, width=5120, trimmed=True, guidance=1)),
            group_size=self.profile.training.group_size, global_batch_size=self.profile.training.global_samples,
            gradient_accumulation_steps=1, data_sharding_version=SAMPLING_VERSION, sampler_state_schema="h3.prompt-source.v1",
            rng_policy=STREAM_VERSION, dynamics_state_schema=("h3.dual-flow-sde.v1" if dual else "h3.cps-dual-all-n.v1"),
            progress_state_schema=f"checkpoint-progress.v{CHECKPOINT_PROGRESS_SCHEMA_VERSION}",
            ema_state_schema="none", reference_state_schema="none", execution_transform_plan_id=transform_id,
            execution_transform_chain=(("h3.world8-ac-fsdp2.v1", transform_id),), world_size=8, world_size_policy="strict",
            state_schema_versions=(("model", 2), ("optimizer", 1), ("data", 1), ("rng", 1)),
        )

    def checkpoint_boundary(self, stages, trainer):
        if stages.data is not self.source or stages.prepared is not self.prepared:
            raise ValueError("checkpoint requested for a foreign stage binding")
        state = stages.checkpoint_boundary(trainer)
        self.validate_live()
        step = state["next_optimizer_step"]
        for _, p in self.prepared.trainable_named_parameters():
            saved = self.prepared.optimizer.state.get(p, {})
            if not saved or saved["step"].item() != step:
                raise ValueError("Adam step does not match complete trainer tail")
        if self.prepared.scheduler.last_epoch != step:
            raise ValueError("scheduler position differs from completed tail")
        rng = RankRNGSnapshot.capture_current(self.rank)
        batch_id = _unpack_id(self.source.next_global_batch_id, "h3-batch")
        selection = DynamicsSelectionPolicyState(0, "h3.all-policy-steps-including-q0.v1", "h3.no-selection-rng", 1)
        progress = CheckpointProgress(step, step, step, self.source.source_id, batch_id, "train",
            ("h3_hpsv3_video_only_v1",), ((self.source.source_id, state["global_cursor"]),), selection,
            0, False, False, self.contract.execution_transform_plan_id, rng.state_identity)
        proof = CheckpointSafePoint(self.rank, 8, "committed", step, 0, 0, 0, True, 0, False,
                                    self.state_identity.group_geometry_id)
        proof.assert_ready(progress)
        data = dict(schema_version=1, rank=self.rank, world_size=8,
                    content_id=_unpack_id(self.source.data_identity, "h3-data"), source_id=self.source.source_id,
                    next_optimizer_step=step, next_global_batch_id=batch_id, cursor=state["global_cursor"],
                    behavior_id=state["behavior_id"], open_reservations=0, payload=state)
        return progress, proof, data, rng

    def close(self):
        self._closed = True  # borrowed adapter/prepared/scorer and WORLD remain caller-owned
