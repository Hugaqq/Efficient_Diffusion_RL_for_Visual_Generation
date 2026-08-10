from __future__ import annotations

import hashlib
import json
from pathlib import Path

from visual_rl.composition.config import (
    bootstrap_recipe_v2,
    compile_recipe_v2,
    load_source_recipe,
)
from visual_rl.core.filesystem_identity import (
    filesystem_file_identity_from_snapshot,
)
from visual_rl.core.immutable import FrozenMapping
from visual_rl.data import (
    DatasetArtifactBinding,
    SourceLoadRequest,
    SourceLocationBinding,
    load_stable_source_sequences,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_world_r1_dynamic_demo_artifact_matches_provenance() -> None:
    prompt_path = REPO_ROOT / "data/prompts/world_r1_dynamic_v1.txt"
    provenance_path = REPO_ROOT / "data/prompts/world_r1_provenance_v1.json"

    raw = prompt_path.read_bytes()
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    selection = provenance["selection"]

    assert prompt_path.is_file()
    assert selection["path"] == "data/prompts/world_r1_dynamic_v1.txt"
    assert hashlib.sha256(raw).hexdigest() == selection["content_sha256"]
    prompts = raw.decode("utf-8", errors="strict").splitlines()
    assert len(prompts) == selection["count"] == 20
    assert all(prompt and prompt == prompt.strip() for prompt in prompts)
    assert len(prompts) == len(set(prompts))


def test_world_r1_dynamic_demo_provenance_does_not_claim_full_dataset() -> None:
    provenance_path = REPO_ROOT / "data/prompts/world_r1_provenance_v1.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    assert provenance["source_rows"] > provenance["selection"]["count"]
    assert "not the full upstream training split" in provenance["purpose"]


def test_release_surrogate_config_loads_both_checked_in_prompt_sources() -> None:
    config_path = REPO_ROOT / "configs/v2/world_r1_release_surrogate_wan.yaml"
    source = load_source_recipe(config_path)
    bootstrap = bootstrap_recipe_v2(source)
    resolved = compile_recipe_v2(source)
    artifacts = bootstrap.require_launch().artifacts
    locations = SourceLocationBinding(
        source_plan_id=resolved.source_plan.plan_id,
        artifacts=tuple(
            DatasetArtifactBinding(
                artifact_ref=artifact_ref,
                artifact_location=artifacts.dataset(artifact_ref),
                expected_content_identity=FrozenMapping(
                    filesystem_file_identity_from_snapshot(
                        artifacts.dataset(artifact_ref).read_bytes()
                    )
                ),
            )
            for artifact_ref in sorted(
                {item.artifact_ref for item in resolved.source_plan.sources}
            )
        ),
    )

    sequences = load_stable_source_sequences(
        SourceLoadRequest(plan=resolved.source_plan, locations=locations)
    )

    assert tuple(sequence.source_id for sequence in sequences) == ("dynamic", "main")
    assert tuple(len(sequence.items) for sequence in sequences) == (20, 9)
