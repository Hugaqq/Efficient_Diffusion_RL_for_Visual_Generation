"""05: 在指定 step 前共享状态，在该 step 后产生不同分支。"""

import torch


def branch_trajectories(seed=7, steps=4, branch_step_index=2, branch_count=3):
    generator = torch.Generator().manual_seed(seed)  # noqa: F841
    # FILL_ME 1: 生成一条长度到 branch_step_index 的共享 prefix。
    # FILL_ME 2: 为每个 branch 复制 prefix，再独立生成 suffix。
    # 返回 shape [branch_count, steps + 1] 的 states。
    raise NotImplementedError("FILL_ME")


def branch_metadata(branch_step_index, timestep_values, branch_count):
    # FILL_ME 3: 每条记录同时保存 branch_step_index 和对应 timestep value。
    raise NotImplementedError("FILL_ME")


def check() -> None:
    states = branch_trajectories()
    assert states.shape == (3, 5)
    assert torch.equal(states[0, :3], states[1, :3])
    assert not torch.equal(states[0, 3:], states[1, 3:])
    metadata = branch_metadata(2, [900, 600, 300, 100], 3)
    assert [item["branch_step_index"] for item in metadata] == [2, 2, 2]
    assert [item["branch_timestep_value"] for item in metadata] == [300, 300, 300]
    print("05 PASS")


if __name__ == "__main__":
    check()
