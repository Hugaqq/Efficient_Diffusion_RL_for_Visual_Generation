"""04: 将 batch reward 对齐为逐样本记录，并原子写 JSON。"""

import json
import tempfile
from pathlib import Path

import torch


def build_records(step, prompts, rewards):
    # FILL_ME 1: 检查 prompts 与 rewards 长度。
    # FILL_ME 2: 每个 prompt 返回一条独立 dict，包含 sample_id/step/prompt/reward。
    raise NotImplementedError("FILL_ME")


def atomic_write_json(path: Path, payload) -> None:
    # FILL_ME 3: 写入 path.json.tmp，再用 replace 替换正式文件。
    raise NotImplementedError("FILL_ME")


def check() -> None:
    records = build_records(3, ["red", "blue"], torch.tensor([0.2, 0.8]))
    assert [item["sample_id"] for item in records] == ["step-000003-sample-000000", "step-000003-sample-000001"]
    assert all(
        abs(actual - expected) < 1e-6
        for actual, expected in zip(
            [item["reward"] for item in records],
            [0.2, 0.8],
            strict=True,
        )
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "manifest.json"
        atomic_write_json(path, {"records": records})
        assert json.loads(path.read_text(encoding="utf-8"))["records"] == records
        assert not path.with_suffix(".json.tmp").exists()
    print("04 PASS")


if __name__ == "__main__":
    check()
