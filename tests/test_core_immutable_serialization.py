"""Focused contracts for the import-safe immutable core values."""

from __future__ import annotations

import pickle
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from visual_rl.core.immutable import FrozenMapping
from visual_rl.core.serialization import to_plain_dict


def test_frozen_mapping_recursively_copies_pickles_and_rejects_non_plain() -> None:
    source = {"nested": [1, {"path": Path("checkpoint")}]}
    frozen = FrozenMapping(source)
    source["nested"].append(3)

    assert frozen == FrozenMapping(
        {"nested": (1, FrozenMapping({"path": Path("checkpoint")}))}
    )
    assert pickle.loads(pickle.dumps(frozen)) == frozen
    with pytest.raises(TypeError, match="JSON-safe"):
        FrozenMapping({"runtime": object()})
    with pytest.raises(ValueError, match="non-finite"):
        FrozenMapping({"loss": float("nan")})


def test_plain_projection_uses_field_aliases_and_rejects_runtime_objects() -> None:
    @dataclass(frozen=True)
    class _Resume:
        from_: Path | None = field(metadata={"plain_name": "from"})
        rows: tuple[int, ...] = ()

    assert to_plain_dict(
        _Resume(from_=Path("checkpoint-1"), rows=(1, 2))
    ) == {
        "from": "checkpoint-1",
        "rows": [1, 2],
    }
    with pytest.raises(TypeError, match="does not accept object"):
        to_plain_dict(object())


def test_immutable_core_imports_without_training_libraries() -> None:
    script = """
import sys
from visual_rl.core.immutable import FrozenMapping
from visual_rl.core.serialization import to_plain_dict
assert 'torch' not in sys.modules
assert 'numpy' not in sys.modules
assert to_plain_dict(FrozenMapping({'ok': (1, 2)})) == {'ok': [1, 2]}
print('import-safe')
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "import-safe"
