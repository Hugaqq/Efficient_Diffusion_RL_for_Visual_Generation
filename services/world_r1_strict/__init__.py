"""World-R1 strict_v2 + json_v1 fail-closed companion reward service.

This package is a deployment adapter for the patched native World-R1 reward
managers.  It is not part of the ``visual_rl`` public API, never enters the
training wheel, and imports no Torch/CUDA at package level so the fake-manager
contract tests run on plain CPU hosts.
"""

__all__ = ()
