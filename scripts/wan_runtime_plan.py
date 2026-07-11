"""World-R1 Wan runtime planning layer.

This module is intentionally import-light. It captures the runtime contract for
the upcoming real Wan trainer without importing diffusers, accelerate, or CUDA
components during local smoke tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from visual_rl.configs.schema import VisualRLConfig, section_to_dict
from visual_rl.datasets.prompt_dataset import PromptDataset
from visual_rl.feedback.world_r1_rewards import WORLD_R1_REWARD_CLIENT_NAMES, validate_reward_server_url
from visual_rl.third_party.legacy import resolve_legacy_repo


@dataclass
class WanRuntimePlan:
    backend: str
    model_name: str
    model_family: str
    model_path: str
    output_dir: str
    world_r1_root: str
    prompt_count: int
    sample: dict[str, Any]
    train: dict[str, Any]
    algorithm: dict[str, Any]
    reward_weights: dict[str, float]
    reward_servers: dict[str, Any]
    readiness: dict[str, bool]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WanRuntimePlanner:
    """Static readiness planner for the World-R1 Wan reference path.

    `build_runtime_plan()` is the current local validation path. Real training
    remains deferred until model checkpoints, reward services, and GPU runtime
    are available.
    """

    def __init__(self, config: VisualRLConfig):
        self.config = config
        self.dataset = PromptDataset.from_config(config.dataset)

    def _reference_root(self) -> Path:
        world_r1_root = self.config.model.extra.get("world_r1_root", "reference_code/World-R1-main")
        # Keep configs stable while allowing the actual snapshots to live outside
        # this repo, e.g. under ../code_base/reference_code.
        return resolve_legacy_repo(world_r1_root)

    def _reward_server_plan(self) -> tuple[dict[str, Any], list[str]]:
        reward_clients = self.config.rewards.clients or {}
        required_clients: list[str] = []
        urls: dict[str, str] = {}
        missing_urls: list[str] = []
        invalid_urls: dict[str, str] = {}

        for reward_name in sorted(self.config.rewards.weights):
            client_config = dict(reward_clients.get(reward_name, {}))
            client_name = str(client_config.get("name", reward_name))
            uses_remote_server = client_name == "remote_pickle" or reward_name in WORLD_R1_REWARD_CLIENT_NAMES
            if not uses_remote_server:
                continue

            required_clients.append(reward_name)
            params = dict(client_config.get("params", {}))
            raw_url = client_config.get("url", params.get("url", ""))
            if not raw_url:
                missing_urls.append(reward_name)
                continue
            try:
                urls[reward_name] = validate_reward_server_url(str(raw_url), reward_name=reward_name)
            except ValueError as exc:
                invalid_urls[reward_name] = str(exc)

        valid = not missing_urls and not invalid_urls
        reward_servers = {
            "required_clients": required_clients,
            "urls": urls,
            "missing_urls": missing_urls,
            "invalid_urls": invalid_urls,
            "valid": valid,
        }

        warnings: list[str] = []
        for reward_name in missing_urls:
            warnings.append(
                f"reward server URL is missing for {reward_name!r}; set rewards.clients.{reward_name}.url "
                "before real Wan training."
            )
        for message in invalid_urls.values():
            warnings.append(message)
        return reward_servers, warnings

    def build_runtime_plan(self) -> WanRuntimePlan:
        sample = section_to_dict(self.config.sample)
        train = section_to_dict(self.config.train)
        algorithm = section_to_dict(self.config.algorithm)
        model = section_to_dict(self.config.model)
        world_r1_root = self._reference_root()
        reward_servers, reward_server_warnings = self._reward_server_plan()

        readiness = {
            "world_r1_root_exists": world_r1_root.exists(),
            "model_path_set": bool(self.config.model.model_path),
            "mock_rewards_only": set(self.config.rewards.weights) == {"mock"},
            "reward_server_required": bool(reward_servers["required_clients"]),
            "reward_server_urls_valid": bool(reward_servers["valid"]),
        }
        warnings: list[str] = []
        if not readiness["model_path_set"]:
            warnings.append("model.model_path is empty; this plan is local-only and cannot launch real Wan training.")
        if readiness["mock_rewards_only"]:
            warnings.append("only mock reward is configured; replace rewards before real training.")
        elif not readiness["reward_server_required"]:
            warnings.append(
                "non-mock rewards are configured without World-R1/remote_pickle reward servers; "
                "verify they are local and video-compatible before real Wan training."
            )
        warnings.extend(reward_server_warnings)
        if not readiness["world_r1_root_exists"]:
            warnings.append(f"World-R1 reference root is missing: {world_r1_root}")

        return WanRuntimePlan(
            backend="wan",
            model_name=str(model.get("name", "")),
            model_family=str(model.get("model_family", "")),
            model_path=str(model.get("model_path", "")),
            output_dir=str(self.config.paths.output_dir),
            world_r1_root=str(world_r1_root),
            prompt_count=len(self.dataset),
            sample=sample,
            train=train,
            algorithm=algorithm,
            reward_weights=dict(self.config.rewards.weights),
            reward_servers=reward_servers,
            readiness=readiness,
            warnings=warnings,
        )
