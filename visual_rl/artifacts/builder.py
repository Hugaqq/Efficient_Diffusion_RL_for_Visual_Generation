"""Build the sole persistent sample records from typed runtime contracts."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from visual_rl.artifacts.manifest import SampleRecord
from visual_rl.core.types import (
    FrozenMapping,
    RewardBatch,
    RolloutBatch,
    to_plain_dict,
)


class ManifestBuilder:
    """Pure per-rank projection with one stable run/model/rollout identity."""

    def __init__(
        self,
        *,
        run_id: str,
        media_type: str,
        rollout_type: str,
    ) -> None:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        if media_type not in {"image", "video"}:
            raise ValueError("media_type must be 'image' or 'video'")
        if rollout_type not in {
            "full_trajectory",
            "single_step",
            "branching",
        }:
            raise ValueError(
                "rollout_type must be full_trajectory, single_step, or branching"
            )
        self.run_id = run_id
        self.media_type = media_type
        self.rollout_type = rollout_type

    def build_records(
        self,
        batch: RolloutBatch,
        rewards: RewardBatch,
        *,
        media_paths: tuple[str | Path | None, ...],
    ) -> tuple[SampleRecord, ...]:
        """Project a validated batch without accepting duplicate identity inputs."""

        if not isinstance(batch, RolloutBatch):
            raise TypeError("batch must be a RolloutBatch")
        if not isinstance(rewards, RewardBatch):
            raise TypeError("rewards must be a RewardBatch")
        rewards.validate_against(batch)
        if type(media_paths) is not tuple or len(media_paths) != batch.batch_size:
            raise ValueError("media_paths must be a tuple with shape [B]")
        normalized_media_paths = tuple(
            self._relative_path(f"media_paths[{index}]", value)
            for index, value in enumerate(media_paths)
        )

        records = []
        for index in range(batch.batch_size):
            branch_id = (
                None if batch.branch_id is None else batch.branch_id[index]
            )
            records.append(
                SampleRecord(
                    run_id=self.run_id,
                    sample_id=batch.sample_id[index],
                    sample_index=index,
                    step=batch.context.step,
                    rank=batch.context.rank,
                    prompt=batch.prompts[index],
                    media_type=self.media_type,
                    prompt_metadata=FrozenMapping(batch.metadata[index]),
                    seed=batch.context.seed,
                    rollout_type=self.rollout_type,
                    timestep_summary=FrozenMapping(
                        self._timestep_summary(batch, index)
                    ),
                    reward_values=FrozenMapping(
                        self._reward_values(rewards, index)
                    ),
                    media_path=normalized_media_paths[index],
                    rollout_cache_path=None,
                    checkpoint_path=None,
                    model_metadata=FrozenMapping(batch.artifact_metadata),
                    prompt_id=batch.prompt_id[index],
                    group_id=batch.group_id[index],
                    branch_id=branch_id,
                )
            )
        return tuple(records)

    def _timestep_summary(
        self,
        batch: RolloutBatch,
        index: int,
    ) -> dict[str, Any]:
        values = self._finite_number_list(
            batch.timesteps[index],
            label="timesteps",
        )
        summary: dict[str, Any] = {
            "values": values,
            "count": batch.transition_count,
        }
        if len(values) != batch.transition_count:
            raise ValueError("timesteps row does not match transition_count")
        if self.rollout_type == "single_step":
            selected = batch.selected_timestep_index
            if selected is None:
                raise ValueError(
                    "single_step rollout requires selected_timestep_index"
                )
            summary["selected_timestep_index"] = int(selected[index].item())
        elif self.rollout_type == "branching":
            branch_step = batch.branch_step_index
            trajectory_step = batch.trajectory_step_index
            if branch_step is None or trajectory_step is None:
                raise ValueError(
                    "branching rollout requires branch_step_index and "
                    "trajectory_step_index"
                )
            summary["branch_step_index"] = int(branch_step[index].item())
            summary["trajectory_step_index"] = self._finite_number_list(
                trajectory_step,
                label="trajectory_step_index",
            )
        elif (
            batch.selected_timestep_index is not None
            or batch.branch_step_index is not None
            or batch.trajectory_step_index is not None
        ):
            raise ValueError(
                "full_trajectory rollout cannot carry selected/branch indices"
            )
        return summary

    @staticmethod
    def _reward_values(
        rewards: RewardBatch,
        index: int,
    ) -> dict[str, Any]:
        raw: dict[str, float] = {}
        weighted: dict[str, float] = {}
        shared: dict[str, Any] = {}
        sample: dict[str, Any] = {}
        for name in rewards.raw:
            raw[name] = ManifestBuilder._finite_scalar(
                rewards.raw[name][index],
                label=f"raw.{name}",
            )
            weighted[name] = ManifestBuilder._finite_scalar(
                rewards.weighted[name][index],
                label=f"weighted.{name}",
            )
            shared[name] = to_plain_dict(rewards.shared_metadata[name])
            sample[name] = to_plain_dict(rewards.sample_metadata[name][index])
        return {
            "raw": raw,
            "weighted": weighted,
            "weighted_total": ManifestBuilder._finite_scalar(
                rewards.weighted_total[index],
                label="weighted_total",
            ),
            "valid": True,
            "shared_metadata": shared,
            "sample_metadata": sample,
        }

    @staticmethod
    def _finite_scalar(value: Any, *, label: str) -> float:
        import math

        if getattr(value, "numel", lambda: 0)() != 1:
            raise ValueError(f"{label} must be a scalar tensor")
        result = float(value.detach().cpu().item())
        if not math.isfinite(result):
            raise ValueError(f"{label} must be finite")
        return result

    @staticmethod
    def _finite_number_list(value: Any, *, label: str) -> list[int | float]:
        import math

        plain = value.detach().cpu().tolist()
        if not isinstance(plain, list):
            plain = [plain]
        result: list[int | float] = []
        for item in plain:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise TypeError(f"{label} must contain JSON numbers")
            if not math.isfinite(float(item)):
                raise ValueError(f"{label} must contain finite values")
            result.append(item)
        return result

    @staticmethod
    def _relative_path(
        name: str,
        value: str | Path | None,
    ) -> str | None:
        if value is None:
            return None
        candidate = value.as_posix() if isinstance(value, Path) else value
        if not isinstance(candidate, str) or not candidate or "\\" in candidate:
            raise ValueError(f"{name} must be a relative POSIX path")
        path = PurePosixPath(candidate)
        if path.is_absolute() or any(
            part in {"", ".", ".."} for part in path.parts
        ):
            raise ValueError(f"{name} must be a normalized relative POSIX path")
        if path.as_posix() != candidate:
            raise ValueError(f"{name} must be a normalized relative POSIX path")
        return candidate
