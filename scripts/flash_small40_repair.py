"""Launcher-only allocator handoff and narrowly authorized run-002 owner upgrade.

No checkpoint/recipe rewriting. The production restore service still checks the
actual prepared and full contracts before any resumed optimizer work.
"""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess

from visual_rl.algorithms.rewards.clients.hpsv3 import Hpsv3VideoClient

REPAIR_BASE = "00cee786df783f36cb254a9a666a48a5952dc0a3"
REPAIR_OUTPUT = Path("/mnt/data/qifan/runs/flash-small40-20261022/run-002")
ALLOWED_CHANGES = frozenset({
    "scripts/launch_flash140.py", "scripts/flash_small40_repair.py",
    "tests/test_flash_collection_runtime.py", "docs/flash_small40_repair.md", "AGENTS.md",
})


def release_reward_cache(phase):
    """Only unused allocator blocks; never offload parameters or collect graphs.

    Called from the reward leaf AFTER the rollout execution context has exited.
    Rollout/evaluation inputs are already detached CPU data. Grad mode alone is
    not a live-graph detector; the synchronous stage boundary is the guarantee.
    """
    import torch
    from visual_rl.runtime.flash140_logging import progress
    if not torch.cuda.is_available():
        return  # CPU tests cannot establish actual CUDA reclamation.
    def sample():
        free, total = torch.cuda.mem_get_info(0)
        return dict(allocated_bytes=torch.cuda.memory_allocated(0),
                    reserved_bytes=torch.cuda.memory_reserved(0),
                    card_free_bytes=free, card_total_bytes=total)
    with torch.no_grad():
        torch.cuda.synchronize(0)
        before = sample()
        progress("reward_cuda_handoff_before", phase=phase, **before)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(0)
        after = sample()
        progress("reward_cuda_handoff_after", phase=phase, **after)
    # A busy external device can change card_free independently. No invented
    # minimum reclaimed bytes, service admission bypass, or automatic retry.


class CacheHandoffHpsClient(Hpsv3VideoClient):
    """Owned adapter around the unchanged real client, not a patched instance.

    Subtyping retains the evaluator's strict HPS client check. All scoring,
    validation, transport and close ownership delegate to the acquired client.
    """
    def __init__(self, client):
        self.client = client
        self.service_identity = client.service_identity
        self.contract = client.contract

    def healthcheck(self):
        return self.client.healthcheck()

    def close(self):
        return self.client.close()

    def activate(self):
        activate = getattr(self.client, "activate", None)
        if callable(activate):
            return activate()

    def score(self, *, batch):
        from visual_rl.algorithms.rewards.clients.input import resolve_pointwise_reward_input
        media = resolve_pointwise_reward_input(batch).media
        if media.device.type != "cpu" or media.requires_grad or media.grad_fn is not None:
            raise ValueError("cache handoff requires detached CPU reward media")
        release_reward_cache("collection_chunk")
        return self.client.score(batch=batch)  # exactly one handoff per chunk

    def score_video(self, **kwargs):
        release_reward_cache("evaluation_video")
        return self.client.score_video(**kwargs)


class CacheHandoffRewardFactory:
    """Use the default factory seam, preserving its live identity/binding facts."""
    def __init__(self, factory=None):
        from visual_rl.runtime.reward_resources import DefaultRewardResourceFactory
        self.factory = DefaultRewardResourceFactory() if factory is None else factory

    def __call__(self, request):
        acquired = self.factory(request)
        if not isinstance(acquired.resource, Hpsv3VideoClient):
            acquired.resource.close()
            raise TypeError("SMALL40 repair requires the actual HPSv3 client")
        return replace(acquired, resource=CacheHandoffHpsClient(acquired.resource))


def git_read(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, timeout=30).strip()


def prove_launcher_upgrade(root, new_sha, config, *, git=None):
    """Bound ancestry/path scope AND exact runtime/config/data Git objects.

    Main has independently required clean HEAD and actual imports at this root.
    Config bytes are checked too; hashes are evidence, never substituted recipe
    identities. Only this fixed historical base can authorize an owner change.
    """
    git = (lambda *args: git_read(root, *args)) if git is None else git
    if git("merge-base", REPAIR_BASE, new_sha) != REPAIR_BASE:
        raise RuntimeError("launcher upgrade requires the approved base ancestry")
    changed = git("diff", "--name-only", REPAIR_BASE, new_sha).splitlines()
    if not set(changed) <= ALLOWED_CHANGES:
        raise RuntimeError("launcher upgrade changed runtime/config/source outside the allowed paths")
    trees = {}
    for path in ("visual_rl", "experiments", "data"):
        old, new = (git("rev-parse", f"{sha}:{path}") for sha in (REPAIR_BASE, new_sha))
        if old != new:
            raise RuntimeError(f"launcher upgrade changed {path} content")
        trees[path] = new
    relative = Path(config).relative_to(root).as_posix()
    if git("rev-parse", f"{REPAIR_BASE}:{relative}") != git("hash-object", str(config)):
        raise RuntimeError("launcher upgrade config bytes differ")
    return dict(base_sha=REPAIR_BASE, new_sha=new_sha, changed_paths=changed,
                unchanged_git_objects=trees,
                source_config_sha256=hashlib.sha256(Path(config).read_bytes()).hexdigest(),
                runtime_contract="not overridden; production prepared/full restore gates remain mandatory")


def resume_owner(output, identity, resume, *, upgrade_from, root, config):
    """Read-only authorization; return exact existing owner and audited new owner.

    Caller validates latest/complete/schema first, then publishes the entire new
    owner+audit in ONE atomic write under the global launch lock. Preflight never
    publishes. Old checkpoints are not modified, nor are recipe/code identities.
    """
    output = Path(output).absolute()
    owner_path = output / "flash140-owner.json"
    if (output.is_symlink() or owner_path.is_symlink() or not owner_path.is_file()
            or owner_path.stat().st_uid != os.getuid() or output.stat().st_uid != os.getuid()):
        raise RuntimeError("resume requires a real launcher-owned output for this UID")
    owner = json.loads(owner_path.read_text())
    if not isinstance(owner, dict):
        raise RuntimeError("invalid launcher owner")
    if owner == identity:
        if upgrade_from is not None:
            raise RuntimeError("upgrade flag is only for the approved old run-002 owner")
        return owner, owner
    audit = owner.get("launcher_upgrade")
    if audit is not None and {k: v for k, v in owner.items() if k != "launcher_upgrade"} == identity:
        if upgrade_from not in (None, REPAIR_BASE):
            raise RuntimeError("unapproved launcher upgrade base")
        proof = prove_launcher_upgrade(root, identity["git_sha"], config)
        if audit != proof or output != REPAIR_OUTPUT:
            raise RuntimeError("launcher upgrade audit/source mismatch")
        return owner, owner
    if (upgrade_from != REPAIR_BASE or output != REPAIR_OUTPUT or resume is None
            or Path(resume).absolute() != output / "checkpoints/step-2"
            or owner != {**identity, "git_sha": REPAIR_BASE}):
        raise RuntimeError("different source/config owner; only approved run-002 step2 launcher upgrade is allowed")
    proof = prove_launcher_upgrade(root, identity["git_sha"], config)
    if identity["source_config_sha256"] != proof["source_config_sha256"]:
        raise RuntimeError("launcher upgrade source config hash mismatch")
    return owner, {**identity, "launcher_upgrade": proof}


def validate_small40_resume(output, resume):
    """No fallback to step0, incomplete checkpoints, or older history."""
    from scripts.launch_flash140 import inspect_checkpoint
    output, resume = Path(output).absolute(), Path(resume).absolute()
    if (resume.parent != output / "checkpoints" or resume.is_symlink()
            or any(p.is_symlink() for p in resume.parents)
            or not resume.name.removeprefix("step-").isdigit()):
        raise RuntimeError("SMALL40 resume must name an owned latest checkpoint")
    candidates = [p for p in resume.parent.glob("step-*") if p.name.removeprefix("step-").isdigit()]
    if not candidates or resume != max(candidates, key=lambda p: int(p.name[5:])):
        raise RuntimeError("SMALL40 resume must select latest checkpoint")
    if (output / "SUCCESS").exists():
        raise RuntimeError("SMALL40 output is already terminal")
    result = inspect_checkpoint(resume)  # verifies complete tree, schema2, finite CPU model/EMA
    marker = json.loads((resume / "complete.json").read_text())
    latest = json.loads((resume.parent / "latest.json").read_text())
    if latest != {"path": resume.name, **{key: marker[key] for key in
                  ("step", "checkpoint_contract_id", "progress_id")}}:
        raise RuntimeError("SMALL40 latest.json disagrees with explicit checkpoint")
    step = int(resume.name[5:])
    if (step < 2 or step >= 40 or step % 2 or result["committed_steps"] != step
            or result["commit_cursor"] != 0 or result["pending_rows"] != 0
            or result["evaluations_completed"] != ([0] if step < 20 else [0, 20])):
        raise RuntimeError("SMALL40 resume step/cursor/evaluation state mismatch")
    return result
