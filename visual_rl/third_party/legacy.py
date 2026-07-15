"""Helpers for importing vendored legacy projects without permanently polluting sys.path."""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Iterator


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def reference_code_roots() -> list[Path]:
    """Candidate roots that contain upstream reference repositories."""

    roots: list[Path] = []
    if env_root := os.environ.get("VISUAL_RL_REFERENCE_CODE_ROOT"):
        roots.append(Path(env_root).expanduser())
    roots.extend(
        [
            project_root() / "reference_code",
            # Current source checkout: <workspace>/code_base/<upstream-repo>.
            project_root().parent / "code_base",
            # Keep the older nested code_base/reference_code layout compatible.
            project_root().parent / "code_base" / "reference_code",
        ]
    )
    return roots


def _reference_relative_path(raw: Path) -> Path:
    parts = raw.parts
    if parts and parts[0] == "reference_code":
        return Path(*parts[1:]) if len(parts) > 1 else Path()
    return raw


def resolve_legacy_repo(repo_root: str | Path) -> Path:
    """Resolve a legacy/reference repo path.

    Keep old `reference_code/<repo>` configs working while also supporting the
    current `code_base/<repo>` checkout and explicit env roots.
    """

    raw = Path(repo_root).expanduser()
    if raw.is_absolute():
        return raw.resolve()

    reference_rel = _reference_relative_path(raw)
    candidates = [
        Path.cwd() / raw,
        project_root() / raw,
    ]
    candidates.extend(root / reference_rel for root in reference_code_roots())
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[-1].resolve()


@contextlib.contextmanager
def legacy_repo_path(repo_root: str | Path, purge_flow_grpo: bool = True) -> Iterator[Path]:
    root = resolve_legacy_repo(repo_root)
    if not root.exists():
        raise FileNotFoundError(f"Legacy repo root does not exist: {root}")

    old_path = list(sys.path)
    purged = {}
    if purge_flow_grpo:
        for name in list(sys.modules):
            if name == "flow_grpo" or name.startswith("flow_grpo."):
                purged[name] = sys.modules.pop(name)

    sys.path.insert(0, str(root))
    try:
        yield root
    finally:
        sys.path[:] = old_path
        for name in list(sys.modules):
            module = sys.modules.get(name)
            module_file = getattr(module, "__file__", "") if module is not None else ""
            if module_file and str(root) in module_file and (name == "flow_grpo" or name.startswith("flow_grpo.")):
                sys.modules.pop(name, None)
        sys.modules.update(purged)
