"""06: 保存并恢复模型、optimizer 和 step。"""

import tempfile
from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, step):
    # FILL_ME 1: 保存 model.state_dict、optimizer.state_dict 和 step。
    raise NotImplementedError("FILL_ME")


def load_checkpoint(path, model, optimizer):
    # FILL_ME 2: 加载文件并恢复 model 和 optimizer。
    # FILL_ME 3: 返回 step。
    raise NotImplementedError("FILL_ME")


def check() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    loss = (model(torch.ones(1, 1)) - 2).square().mean()
    loss.backward()
    optimizer.step()
    expected_weight = model.weight.detach().clone()
    expected_exp_avg = next(iter(optimizer.state.values()))["exp_avg"].clone()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "state.pt"
        save_checkpoint(path, model, optimizer, step=1)
        with torch.no_grad():
            model.weight.zero_()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        step = load_checkpoint(path, model, optimizer)
    assert step == 1
    assert torch.equal(model.weight, expected_weight)
    assert torch.equal(next(iter(optimizer.state.values()))["exp_avg"], expected_exp_avg)
    print("06 PASS")


if __name__ == "__main__":
    check()
