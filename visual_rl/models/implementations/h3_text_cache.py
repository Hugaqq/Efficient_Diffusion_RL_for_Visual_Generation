"""Strict read-only H3 text cache, schema ``visual_rl.h3-text-cache.v1``.

index.json has exactly {schema_version, identity, shards, entries}:
* identity: model_sha256, encoder_sha256, tokenizer_sha256, processor_sha256,
  template_sha256, truncation_sha256 (full lowercase SHA256); target_layer=50,
  normalized=false, add_special_tokens=false, feature_dim=5120, and dtype
  ("float32", "bfloat16", or "float16"). The caller supplies this entire
  expected identity from its artifact/recipe, never inferred from the cache.
* shards: {local_basename.safetensors: full_file_sha256}.
* entries: {SHA256(exact_UTF8_prompt): {shard, tensor, tokens}}. Each (shard,
  tensor) is unique. Each tensor is exactly [tokens,5120], unpadded.
Each shard has metadata {"h3_text_identity_sha256": SHA256(canonical identity
JSON: sorted keys, separators=(',', ':'), ensure_ascii=True)} and precisely
its indexed tensors. Index/header JSON is limited to 16 MiB each.

Raw UniRL 1.0 indexes (16-hex keys and incomplete provenance) are NOT accepted.
Semantic reference: UniRL 8cfbac17765004afb68fef08a4e011a4c8daf555,
minimax_h3/{offline_text_embed,text_embed}.py, Apache-2.0. Independently
implemented; no encoder, writer, reference import, online miss or normalization.
Declarations bind provenance, not proof that embeddings were actually generated
by layer 50 without final norm. max_embedding_bytes bounds ONE tensor payload,
not total RSS. Reads clone one tensor only, never materialize a full shard.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

_SCHEMA = "visual_rl.h3-text-cache.v1"
_JSON_LIMIT = 16 * 1024 * 1024
_HASH_FIELDS = ("model_sha256", "encoder_sha256", "tokenizer_sha256", "processor_sha256",
                "template_sha256", "truncation_sha256")
_DTYPES = {"float32": ("F32", 4), "bfloat16": ("BF16", 2), "float16": ("F16", 2)}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_hash(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _json(data: bytes) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(data, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def _canonical(value: Mapping) -> bytes:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _prompt_key(prompt: str) -> str:
    if not isinstance(prompt, str):
        raise TypeError("prompt must be an exact string")
    return _sha256(prompt.encode("utf-8"))


class H3TextCache:
    """CPU-only reader; no cached tensors or open shard handles.

    Construction validates every shard's metadata/hash. verify_coverage checks
    train AND eval before model phases and revalidates their shards. get repeats
    integrity checks, so edits cannot silently reuse a previously admitted hash.
    Cache files must remain immutable during use; detected replacements/writes
    fail closed. Content hashes, not mtime, are the identity authority.
    """

    def __init__(self, directory: str | Path, *, expected_identity: Mapping,
                 max_tokens: int, max_embedding_bytes: int) -> None:
        self._root = Path(directory).expanduser().resolve()
        if not self._root.is_dir():
            raise ValueError("text cache requires an existing local directory")
        for name, value in (("max_tokens", max_tokens), ("max_embedding_bytes", max_embedding_bytes)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be an explicit positive integer")
        self._max_tokens, self._max_bytes = max_tokens, max_embedding_bytes
        required = set(_HASH_FIELDS) | {"target_layer", "normalized", "add_special_tokens", "feature_dim", "dtype"}
        if not isinstance(expected_identity, Mapping) or set(expected_identity) != required:
            raise ValueError("expected_identity requires the complete explicit H3 identity")
        identity = dict(expected_identity)
        if any(not _is_hash(identity[name]) for name in _HASH_FIELDS):
            raise ValueError("expected artifact identities must be full SHA256")
        if (type(identity["target_layer"]) is not int or identity["target_layer"] != 50
                or identity["normalized"] is not False or identity["add_special_tokens"] is not False
                or type(identity["feature_dim"]) is not int or identity["feature_dim"] != 5120
                or not isinstance(identity["dtype"], str) or identity["dtype"] not in _DTYPES):
            raise ValueError("H3 identity requires raw layer50, text-only/no specials, 5120 features and supported dtype")
        self._dtype, self._element_bytes = _DTYPES[identity["dtype"]]
        self._embedding_identity_sha256 = _sha256(_canonical(identity))
        raw = self._index_bytes()
        data = _json(raw)
        if set(data) != {"schema_version", "identity", "shards", "entries"} or data["schema_version"] != _SCHEMA:
            raise ValueError("unsupported H3 text cache schema; raw UniRL cache is not compatible")
        if not isinstance(data["identity"], dict) or _canonical(data["identity"]) != _canonical(identity):
            raise ValueError("cache identity mismatch")
        self._identity_sha256 = _sha256(raw)
        self._shards, self._entries = data["shards"], data["entries"]
        if not isinstance(self._shards, dict) or not self._shards or not isinstance(self._entries, dict) or not self._entries:
            raise ValueError("cache requires nonempty shards and entries")
        self._by_shard = {name: {} for name in self._shards}
        for name, digest in self._shards.items():
            if not name.endswith(".safetensors") or not _is_hash(digest):
                raise ValueError("invalid shard filename or SHA256")
            self._path(name)
        for key, entry in self._entries.items():
            if not _is_hash(key) or not isinstance(entry, dict) or set(entry) != {"shard", "tensor", "tokens"}:
                raise ValueError("invalid prompt key or entry schema")
            shard, tensor, tokens = entry["shard"], entry["tensor"], entry["tokens"]
            if not isinstance(shard, str) or shard not in self._by_shard:
                raise ValueError("entry refers to undeclared shard")
            if not isinstance(tensor, str) or not tensor or tensor == "__metadata__":
                raise ValueError("invalid tensor key")
            if tensor in self._by_shard[shard]:
                raise ValueError("duplicate shard/tensor mapping")
            self._check_tokens(tokens)
            self._by_shard[shard][tensor] = tokens
        if any(not entries for entries in self._by_shard.values()):
            raise ValueError("unreferenced shard")
        for name in self._shards:
            with self._checked_shard(name):
                pass

    @property
    def identity_sha256(self) -> str:
        """Fixed full index digest, binding provenance AND all declared content."""
        return self._identity_sha256

    @property
    def embedding_identity_sha256(self) -> str:
        return self._embedding_identity_sha256

    def _path(self, name: str) -> Path:
        # Version 1 accepts basenames only; no symlinks, even within the cache.
        if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("cache path escape: expected local basename")
        path = self._root / name
        if path.is_symlink() or not path.is_file() or path.resolve().parent != self._root:
            raise ValueError(f"missing cache file or symlink/path escape: {name}")
        return path

    def _index_bytes(self) -> bytes:
        with self._path("index.json").open("rb") as stream:
            raw = stream.read(_JSON_LIMIT + 1)
        if len(raw) > _JSON_LIMIT:
            raise ValueError("index JSON exceeds 16 MiB limit")
        return raw

    def _check_index(self) -> None:
        if _sha256(self._index_bytes()) != self._identity_sha256:
            raise ValueError("cache index changed after admission")

    def _check_tokens(self, tokens: int) -> None:
        if type(tokens) is not int or not 1 <= tokens <= self._max_tokens:
            raise ValueError("embedding token limit or invalid token count")
        if tokens * 5120 * self._element_bytes > self._max_bytes:
            raise ValueError("embedding byte limit exceeded")

    @contextmanager
    def _checked_shard(self, name: str):
        from safetensors import safe_open

        path = self._path(name)
        before = path.stat()
        with path.open("rb") as stream:
            size = int.from_bytes(stream.read(8), "little")
            if not 2 <= size <= min(_JSON_LIMIT, before.st_size - 8):
                raise ValueError("invalid or oversized safetensors header")
            header = _json(stream.read(size))  # Reject duplicate keys before Rust parsing.
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if handle.metadata() != {"h3_text_identity_sha256": self._embedding_identity_sha256}:
                raise ValueError("shard metadata identity mismatch")
            expected = self._by_shard[name]
            if set(handle.keys()) != set(expected) or set(header) - {"__metadata__"} != set(expected):
                raise ValueError("shard tensor keys differ from index")
            for key, tokens in expected.items():
                view = handle.get_slice(key)
                if view.get_shape() != [tokens, 5120] or view.get_dtype() != self._dtype:
                    raise ValueError("embedding header shape/dtype mismatch")
                self._check_tokens(tokens)
            # Only after shape/dtype/limits admission, stream the content hash.
            self._check_file_hash(path, name, before)
            yield handle
            self._check_file_hash(path, name, before)
        self._check_index()

    def _check_file_hash(self, path: Path, name: str, before) -> None:
        if self._path(name) != path:
            raise ValueError("cache shard path changed")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                or digest.hexdigest() != self._shards[name]):
            raise ValueError("cache shard content/hash changed")

    def verify_coverage(self, train_prompts: Iterable[str], eval_prompts: Iterable[str]) -> None:
        self._check_index()
        shards = set()
        for label, prompts in (("train", train_prompts), ("eval", eval_prompts)):
            if isinstance(prompts, (str, bytes)):
                raise TypeError("coverage requires collections of exact prompt strings")
            for prompt in prompts:
                key = _prompt_key(prompt)
                if key not in self._entries:
                    raise ValueError(f"missing {label} prompt in text cache: {prompt!r}")
                shards.add(self._entries[key]["shard"])
        for name in sorted(shards):
            with self._checked_shard(name):
                pass

    def get(self, prompt: str):
        import torch

        self._check_index()
        key = _prompt_key(prompt)
        if key not in self._entries:
            raise ValueError(f"missing prompt in text cache: {prompt!r}")
        entry = self._entries[key]
        with self._checked_shard(entry["shard"]) as handle:
            tensor = handle.get_tensor(entry["tensor"]).detach().unsqueeze(0).clone()
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("embedding must be finite")
        return tensor
