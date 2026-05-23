"""Helpers for importing vendored legacy projects without permanently polluting sys.path."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Iterator


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_legacy_repo(repo_root: str | Path) -> Path:
    """Resolve a legacy/reference repo path.

    The preferred layout is `reference_code/<repo>`, but older configs may still
    pass `<repo>` from the workspace root. Keep both forms working.
    """

    raw = Path(repo_root).expanduser()
    if raw.is_absolute():
        return raw.resolve()

    candidates = [
        Path.cwd() / raw,
        project_root() / raw,
        project_root() / "reference_code" / raw,
    ]
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
