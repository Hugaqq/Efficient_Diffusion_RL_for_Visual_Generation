"""Concrete single-process H3 LoRA owner; no inference offload hooks in backward.

The same prepared root is called for collection and differentiable replay.
BF16 frozen base (native FP32 patch/time heads), FP32 LoRA, no GradScaler:
BF16 does not use FP16 scaling. Full DiT residency during forward/backward;
whole-model CPU parking is permitted ONLY with no live graph, before decode.
"""
from contextlib import contextmanager
import importlib.metadata
import logging
import json

import torch

from .h3_components import H3InferenceAdapter, chunk_feed_forwards, record_component, release_memory, stage_budget

TARGETS = ("to_q", "to_k", "to_v", "to_out.0")


class H3PreparedRoot(torch.nn.Module):
    """One actual nn.Module root, preserving every optimizer Parameter object."""
    def __init__(self, transformer):
        super().__init__()
        self.dit = transformer

    def forward(self, **kwargs):
        return self.dit(**kwargs)


class H3TrainingAdapter(H3InferenceAdapter):
    gpu_memory_policy = "oom"  # user-authorized: no GPU reserve on H3 training

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = None
        self.trainables = ()
        self.optimizer = None
        self.lr_scheduler = None
        self.graph_live = False

    def encode(self, request):
        # The encoder itself is correctly inference-only. Clone AFTER that scope
        # so its output can safely be saved by autograd in the DiT text projection.
        return super().encode(request).clone()

    def prepare_policy(self, *, rank, alpha, learning_rate):
        from diffusers import MiniMaxH3Transformer3DModel
        from peft import LoraConfig
        if self.root is not None or self.phase is not None or self.closed:
            raise RuntimeError("H3 policy can be prepared exactly once")
        if importlib.metadata.version("peft") != "0.18.1":
            raise RuntimeError("require peft==0.18.1 in the isolated training environment")
        files = list((self.converted_root / "transformer").glob("*.safetensors"))
        if self.resource_probe:
            stage_budget(sum(p.stat().st_size for p in files), max(p.stat().st_size for p in files),
                         self.resource_probe(), gpu_memory_policy=self.gpu_memory_policy)
        model = MiniMaxH3Transformer3DModel.from_pretrained(
            self.converted_root / "transformer", torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, local_files_only=True, use_safetensors=True)
        record_component(model, "trainable_denoiser")
        model.eval().requires_grad_(False)
        for name, parameter in model.named_parameters():
            expected = torch.float32 if any(s in name for s in ("proj_in", "proj_out", "time_embedder")) else torch.bfloat16
            if parameter.dtype != expected:
                raise ValueError(f"native H3 precision mismatch: {name}")
        # Preserve original attention/FF modules; only exact token-local FF chunking.
        chunk_feed_forwards(model)
        matched = [name for name, module in model.named_modules()
                   if isinstance(module, torch.nn.Linear) and any(name.endswith('.' + target) for target in TARGETS)]
        if not matched or any(not any(n.endswith('.' + t) for n in matched) for t in TARGETS):
            raise ValueError("H3 attention LoRA targets unavailable")
        model.add_adapter(LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0,
                                     target_modules=list(TARGETS), init_lora_weights=True, bias="none"))
        # Non-reentrant native checkpointing supports gradients when inputs/base
        # are frozen. No accelerate.cpu_offload/group-offload hooks on this DiT.
        model.enable_gradient_checkpointing()
        if not model.gradient_checkpointing or not model.token_refiner.gradient_checkpointing:
            raise RuntimeError("native H3 non-reentrant checkpointing not enabled")
        self.root = H3PreparedRoot(model).eval()
        pairs = tuple((n, p) for n, p in self.root.named_parameters() if p.requires_grad)
        if not pairs or any("lora_A" not in n and "lora_B" not in n for n, _ in pairs):
            raise ValueError("trainable set must consist ONLY of real LoRA A/B parameters")
        for _, p in pairs:
            p.data = p.data.float()
        self.trainables = pairs
        self.optimizer = torch.optim.AdamW([p for _, p in pairs], lr=learning_rate, weight_decay=0.0)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)
        self.assert_owner()
        logging.info(json.dumps({"event": "h3_policy_prepared", "rank": rank, "alpha": alpha,
            "matched_modules": matched, "trainable_names": [n for n, _ in pairs],
            "trainable_scalars": sum(p.numel() for _, p in pairs),
            "checkpointing": "native_nonreentrant", "group_offload": False, "grad_scaler": False}))

    def assert_owner(self):
        live = tuple((n, p) for n, p in self.root.named_parameters() if p.requires_grad)
        actual = [p for group in self.optimizer.param_groups for p in group["params"]]
        if (len(live) != len(self.trainables) or len(actual) != len(live)
                or len({id(p) for p in actual}) != len(actual)
                or any(n != old_n or p is not old_p for (n, p), (old_n, old_p) in zip(live, self.trainables))
                or {id(p) for p in actual} != {id(p) for _, p in live}
                or any(p.dtype != torch.float32 for _, p in live)):
            raise RuntimeError("H3 prepared root/optimizer Parameter ownership drift")

    def _move_policy(self, device):
        # PyTorch2.10's documented conversion mode preserves Parameter objects
        # even across device Tensor types. Scope the process-global flag to this
        # single-threaded model conversion; the heartbeat only reads telemetry.
        previous = torch.__future__.get_swap_module_params_on_conversion()
        torch.__future__.set_swap_module_params_on_conversion(True)
        try:
            self.root.to(device)
        finally:
            torch.__future__.set_swap_module_params_on_conversion(previous)
        self.assert_owner()

    def park_policy(self):
        if self.graph_live or self.transformer is not None:
            raise RuntimeError("cannot park H3 policy with active forward/backward")
        self._move_policy("cpu")
        release_memory()

    @contextmanager
    def policy(self, *, gradients=False):
        if self.root is None or self.phase is not None or self.transformer is not None or self.closed:
            raise RuntimeError("policy requires a prepared, idle H3 adapter")
        logging.info(json.dumps({"event": "h3_policy_onload", "gradients": gradients,
                                 "resources": self._resource_snapshot()}))
        self._move_policy(self.device)  # no dtype conversion: native mixed precision stays intact
        logging.info(json.dumps({"event": "h3_policy_resident", "gradients": gradients,
                                 "resources": self._resource_snapshot()}))
        self.root.eval()
        self.transformer = self.root
        self.graph_live = gradients
        try:
            with torch.set_grad_enabled(gradients):
                yield
        finally:
            # Caller must finish backward within this scope. Root is NOT freed.
            self.graph_live = False
            self.transformer = None

    def close(self):
        if self.graph_live or self.transformer is not None:
            raise RuntimeError("cannot close a live training graph")
        self.optimizer = self.lr_scheduler = self.root = None
        self.trainables = ()
        super().close()
