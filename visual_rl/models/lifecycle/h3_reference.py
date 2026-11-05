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
        self._flags = None
        self.validate()

    def validate(self, *, full=True):
        if full and any(m.training for m in self.model.modules()):
            raise ValueError('H3 reference requires EVAL mode')
        for layer in self.layers:
            if (layer.merged or layer.disable_adapters != self.active
                    or layer.active_adapters != ['default'] or any(layer.use_dora.values())
                    or getattr(layer, '_caches', {})):
                raise ValueError('H3 reference requires unmerged, cache-free default LoRA (no DoRA)')

    @contextmanager
    def update_scope(self):
        """每个 optimize 一次快照；退出时先恢复并校验，再允许 optimizer.step。"""
        if self._flags is not None or self.active:
            raise RuntimeError('H3 reference update scope is not reentrant')
        self.validate()
        rng = RankRNGSnapshot.capture_current(self.rank)
        self._rng = rng
        buffers = tuple((b, b.detach().clone()) for b in self.model.buffers())
        self._flags = tuple((p, p.requires_grad) for p in self.model.parameters())
        try:
            yield
        finally:
            try:
                self._restore_adapters()
                with torch.no_grad():
                    changed = any(not torch.equal(buffer, saved) for buffer, saved in buffers)
                    if changed:
                        for buffer, saved in buffers:
                            buffer.copy_(saved)
                self.validate()
                if changed:
                    raise RuntimeError('reference forward mutated a model buffer; restored and refused')
            finally:
                self._flags = self._rng = None
                rng.restore_current()

    def _restore_adapters(self):
        for layer in self.layers:
            layer.enable_adapters(True)
        for parameter, flag in self._flags:
            parameter.requires_grad_(flag)
        self.active = False

    @contextmanager
    def scope(self):
        if self.active:
            raise RuntimeError('H3 reference scope is not reentrant')
        # 单独调用仍有完整保护；训练循环显式持有一次 update_scope。
        if self._flags is None:
            with self.update_scope(), self.scope():
                yield
            return
        self.validate(full=False)
        try:
            for layer in self.layers:
                layer.enable_adapters(False)
            # PEFT 同时修改 requires_grad；保持 FSDP 参数集合，只用 no_grad 禁止参考图。
            for parameter, flag in self._flags:
                parameter.requires_grad_(flag)
            self.active = True
            with torch.no_grad():
                yield
        finally:
            try:
                self._restore_adapters()
                self.validate(full=False)
            finally:
                # H3 的 EVAL optimize 无随机算子；复用更新快照，参考 RNG 不泄漏到策略前向。
                self._rng.restore_current()
