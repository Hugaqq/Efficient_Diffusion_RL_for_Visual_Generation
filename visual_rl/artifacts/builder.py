"""Convert rollout and reward batches into per-sample manifest records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from visual_rl.artifacts.manifest import SampleRecord
from visual_rl.artifacts.serialization import to_jsonable
from visual_rl.core.types import RewardBatch, RolloutBatch


class ManifestBuilder:
    """Stateless per-batch conversion with a stable run identifier."""

    def __init__(self, run_id: str):
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        self.run_id = run_id

    def build_records(
        self,
        *,
        step: int,
        batch: RolloutBatch,
        rewards: RewardBatch,
        media_type: str,
        rollout_type: str | None = None,
        media_paths: Sequence[str | Path | None] | str | Path | None = None,
        rollout_cache_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> list[SampleRecord]:
        if step < 0:
            raise ValueError("step must be non-negative")
        if media_type not in {"image", "video"}:
            raise ValueError(f"Unsupported media type: {media_type!r}")

        batch.validate_lightweight()
        batch_size = len(batch.prompts)
        rewards.validate_against(batch)
        if batch.context is not None and batch.context.step != step:
            raise ValueError(
                "artifact step must match batch.context.step: "
                f"{step} != {batch.context.step}"
            )
        resolved_media_paths = self._resolve_media_paths(media_paths, batch_size)
        model_metadata = to_jsonable(dict(batch.model_metadata))
        resolved_rollout_type = rollout_type or batch.model_metadata.get("rollout")

        records: list[SampleRecord] = []
        for index in range(batch_size):
            prompt_metadata = to_jsonable(dict(batch.metadata[index]))
            branch_id = to_jsonable(self._batch_item(batch.branch_id, index))
            record = SampleRecord(
                run_id=self.run_id,
                sample_id=str(self._batch_item(batch.sample_id, index)),
                sample_index=index,
                step=step,
                prompt=batch.prompts[index],
                media_type=media_type,
                prompt_metadata=prompt_metadata,
                seed=(
                    int(batch.context.seed)
                    if batch.context is not None
                    else None
                ),
                rollout_type=str(resolved_rollout_type)
                if resolved_rollout_type is not None
                else None,
                timestep_summary=self._timestep_summary(
                    batch,
                    index,
                    prompt_metadata,
                    branch_id,
                ),
                reward_values=self._reward_values(rewards, index),
                media_path=resolved_media_paths[index],
                rollout_cache_path=self._path_string(rollout_cache_path),
                checkpoint_path=self._path_string(checkpoint_path),
                model_metadata=dict(model_metadata),
                prompt_id=str(self._batch_item(batch.prompt_id, index)),
                group_id=str(self._batch_item(batch.group_id, index)),
                branch_id=branch_id,
            )
            records.append(record)
        return records

    @staticmethod
    def _batch_item(value: Any, index: int) -> Any:
        return value[index]

    @classmethod
    def _reward_values(cls, rewards: RewardBatch, index: int) -> dict[str, Any]:
        values = {
            "raw": {
                name: to_jsonable(cls._batch_item(values, index))
                for name, values in rewards.raw.items()
            },
            "weighted": {
                name: to_jsonable(cls._batch_item(values, index))
                for name, values in rewards.weighted.items()
            },
            "weighted_total": to_jsonable(
                cls._batch_item(rewards.weighted_total, index)
            ),
            "valid": bool(to_jsonable(cls._batch_item(rewards.valid_mask, index))),
        }
        if "_schedule" in rewards.metadata:
            values["schedule"] = to_jsonable(rewards.metadata["_schedule"])
        provenance = {}
        shared_keys = (
            "protocol",
            "protocol_mode",
            "protocol_version",
            "scorer_identity",
            "identity",
            "formula",
            "server_revision",
            "server_revision_echo",
        )
        for name, metadata in rewards.metadata.items():
            if name.startswith("_") or not isinstance(metadata, Mapping):
                continue
            evidence = metadata.get("sample_evidence")
            if (
                isinstance(evidence, Sequence)
                and not isinstance(evidence, (str, bytes))
                and index < len(evidence)
            ):
                provenance[name] = {
                    **{key: metadata[key] for key in shared_keys if key in metadata},
                    "sample": evidence[index],
                }
        if provenance:
            values["provenance"] = to_jsonable(provenance)
        return values

    @classmethod
    def _timestep_summary(
        cls,
        batch: RolloutBatch,
        index: int,
        prompt_metadata: Mapping[str, Any],
        branch_id: Any,
    ) -> dict[str, Any]:
        values = to_jsonable(cls._batch_item(batch.timesteps, index))
        summary: dict[str, Any] = {
            "values": values,
            "count": len(values) if isinstance(values, list) else 1,
            "branch_id": branch_id,
        }
        for key in (
            "selected_timestep",
            "selected_timestep_index",
            "branch_step_index",
            "branch_timestep_value",
        ):
            if key in prompt_metadata:
                summary[key] = prompt_metadata[key]
        return summary

    @classmethod
    def _resolve_media_paths(
        cls,
        media_paths: Sequence[str | Path | None] | str | Path | None,
        batch_size: int,
    ) -> list[str | None]:
        if media_paths is None:
            return [None] * batch_size
        if isinstance(media_paths, (str, Path)):
            return [cls._path_string(media_paths)] * batch_size
        if len(media_paths) != batch_size:
            raise ValueError(
                f"media_paths length must be {batch_size}, got {len(media_paths)}"
            )
        return [cls._path_string(path) for path in media_paths]

    @staticmethod
    def _path_string(path: str | Path | None) -> str | None:
        return str(path) if path is not None else None
