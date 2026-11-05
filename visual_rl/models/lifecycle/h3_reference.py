"""Read-only, adapter-disabled H3 reference on the existing prepared model."""
from contextlib import contextmanager

import torch

from visual_rl.artifacts.checkpoint.state import RankRNGSnapshot


class H3LoRAReference:
    def __init__(self, model, *, rank):
        from peft.tuners.lora import LoraLayer
        self.model, self.rank = model, rank
        self.layers = tuple(m for m in model.modules() if isinstance(m, LoraLayer))
        if not self.layers:
            raise ValueError('H3 reference requires injected PEFT LoRA layers')
        self.active = False
        self.validate()

    def validate(self):
        if any(m.training for m in self.model.modules()):
            raise ValueError('H3 reference requires EVAL mode')
        for layer in self.layers:
            if (layer.merged or layer.disable_adapters != self.active
                    or layer.active_adapters != ['default'] or any(layer.use_dora.values())
                    or getattr(layer, '_caches', {})):
                raise ValueError('H3 reference requires unmerged, cache-free default LoRA (no DoRA)')

    @contextmanager
    def scope(self):
        if self.active:
            raise RuntimeError('H3 reference scope is not reentrant')
        self.validate()
        rng = RankRNGSnapshot.capture_current(self.rank)
        flags = tuple((p, p.requires_grad) for p in self.model.parameters())
        buffers = tuple((b, b.detach().clone()) for b in self.model.buffers())
        try:
            for layer in self.layers:
                layer.enable_adapters(False)
            # PEFT 的公开接口也关闭 requires_grad；恢复标志，保持 FSDP 参数集合不变。
            # no_grad 才负责禁止参考图；绝不修改权重、优化器或 active adapter。
            for parameter, flag in flags:
                parameter.requires_grad_(flag)
            self.active = True
            self.validate()
            with torch.no_grad():
                yield
        finally:
            try:
                for layer in self.layers:
                    layer.enable_adapters(True)
                for parameter, flag in flags:
                    parameter.requires_grad_(flag)
                self.active = False
                with torch.no_grad():
                    changed = any(not torch.equal(buffer, saved) for buffer, saved in buffers)
                    if changed:
                        for buffer, saved in buffers:
                            buffer.copy_(saved)
                self.validate()
                if changed:
                    raise RuntimeError('reference forward mutated a model buffer; restored and refused')
            finally:
                rng.restore_current()
