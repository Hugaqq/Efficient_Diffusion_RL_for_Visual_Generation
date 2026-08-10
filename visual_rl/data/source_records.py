"""Pure selector, parsing, and item materialization semantics for sources.

This module never opens an artifact and never inspects a path.  Callers must
choose the explicit ``DatasetFormat`` before passing an immutable byte snapshot
to :func:`parse_source_records`.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from visual_rl.data.source_plan import DatasetFormat, SourceLoadError
from visual_rl.data.source_sampler import SourceSequence
from visual_rl.data.samples import SampleItem, SourceItemContext, T2IItem, T2VItem

__all__ = (
    "SOURCE_DESCRIPTOR_BY_SELECTOR",
    "DatasetSourceDescriptor",
    "SourceRecord",
    "build_source_sequence",
    "parse_source_records",
    "source_descriptor",
)

DatasetTask = Literal["t2i", "t2v"]

_CONTENT_IDENTITY_SCHEMA = "visual-rl.dataset-content.v1"
_SOURCE_ITEM_IDENTITY_SCHEMA = "visual-rl.source-item.v1"


@dataclass(frozen=True, slots=True)
class DatasetSourceDescriptor:
    """Typed task and item meaning of one built-in source selector."""

    selector: str
    task: DatasetTask
    item_type: type[SampleItem]

    def __post_init__(self) -> None:
        if not isinstance(self.selector, str) or not self.selector:
            raise ValueError("source selector must be non-empty")
        if self.task not in {"t2i", "t2v"}:
            raise ValueError("source task must be t2i or t2v")
        expected_type = T2IItem if self.task == "t2i" else T2VItem
        if self.item_type is not expected_type:
            raise ValueError("source item type must match its typed task")


SOURCE_DESCRIPTOR_BY_SELECTOR: Mapping[str, DatasetSourceDescriptor] = MappingProxyType(
    {
        "prompt-image": DatasetSourceDescriptor(
            selector="prompt-image",
            task="t2i",
            item_type=T2IItem,
        ),
        "prompt-video": DatasetSourceDescriptor(
            selector="prompt-video",
            task="t2v",
            item_type=T2VItem,
        ),
        "world-r1-dynamic-prompts": DatasetSourceDescriptor(
            selector="world-r1-dynamic-prompts",
            task="t2v",
            item_type=T2VItem,
        ),
        "world-r1-prompts": DatasetSourceDescriptor(
            selector="world-r1-prompts",
            task="t2v",
            item_type=T2VItem,
        ),
    }
)


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """One validated record in stable source order."""

    prompt: str
    metadata: dict[str, object]

    def identity_payload(self) -> dict[str, object]:
        return {"prompt": self.prompt, "metadata": self.metadata}


class _DuplicateJSONKey(ValueError):
    pass


def source_descriptor(selector: str, *, source_id: str) -> DatasetSourceDescriptor:
    """Resolve one built-in selector without consulting recipes or file paths."""

    descriptor = SOURCE_DESCRIPTOR_BY_SELECTOR.get(selector)
    if descriptor is None:
        raise SourceLoadError(
            f"data.sources.{source_id}.id has unknown built-in selector {selector!r}"
        )
    return descriptor


def parse_source_records(
    snapshot: bytes,
    *,
    source_id: str,
    file_format: DatasetFormat,
) -> tuple[SourceRecord, ...]:
    """Parse one already-stable snapshot with explicit fail-closed semantics."""

    if not isinstance(snapshot, bytes):
        raise TypeError("source snapshot must be bytes")
    if file_format not in {"text", "jsonl"}:
        raise SourceLoadError("dataset format must be text or jsonl")
    try:
        text = snapshot.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise SourceLoadError(
            f"dataset artifact for source {source_id!r} is not strict UTF-8"
        ) from exc
    if text.startswith("\ufeff"):
        raise SourceLoadError(
            f"dataset artifact for source {source_id!r} must not contain a UTF-8 BOM"
        )

    records = (
        _parse_text(text, source_id=source_id)
        if file_format == "text"
        else _parse_jsonl(text, source_id=source_id)
    )
    if not records:
        raise SourceLoadError(f"dataset source {source_id!r} is empty")
    seen_prompts: dict[str, int] = {}
    for record_number, record in enumerate(records, start=1):
        previous = seen_prompts.get(record.prompt)
        if previous is not None:
            raise SourceLoadError(
                f"dataset source {source_id!r} has duplicate prompt at record "
                f"{record_number}; first seen at record {previous}"
            )
        seen_prompts[record.prompt] = record_number
    return records


def build_source_sequence(
    *,
    source_id: str,
    descriptor: DatasetSourceDescriptor,
    records: tuple[SourceRecord, ...],
) -> SourceSequence:
    """Materialize typed items while preserving validated record order."""

    if not isinstance(descriptor, DatasetSourceDescriptor):
        raise TypeError("descriptor must be a DatasetSourceDescriptor")
    if type(records) is not tuple or not records:
        raise SourceLoadError("source records must be a non-empty tuple")
    if any(not isinstance(record, SourceRecord) for record in records):
        raise TypeError("records must contain SourceRecord values")

    revision = _content_revision(descriptor, records)
    items = tuple(
        descriptor.item_type(
            prompt=record.prompt,
            metadata=record.metadata,
            source=SourceItemContext(
                source_item_id=_source_item_id(
                    source_id=source_id,
                    dataset_index=index,
                    revision=revision,
                ),
                dataset_source_id=source_id,
                dataset_index=index,
                dataset_revision=revision,
            ),
        )
        for index, record in enumerate(records)
    )
    return SourceSequence(
        source_id=source_id,
        revision=revision,
        items=items,
    )


def _parse_text(text: str, *, source_id: str) -> tuple[SourceRecord, ...]:
    result: list[SourceRecord] = []
    for line_number, prompt in enumerate(text.splitlines(), start=1):
        _validate_prompt(prompt, source_id=source_id, line_number=line_number)
        result.append(SourceRecord(prompt=prompt, metadata={}))
    return tuple(result)


def _parse_jsonl(text: str, *, source_id: str) -> tuple[SourceRecord, ...]:
    result: list[SourceRecord] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise SourceLoadError(
                f"dataset source {source_id!r} has an empty line at {line_number}"
            )
        try:
            value = json.loads(
                line,
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (_DuplicateJSONKey, json.JSONDecodeError, ValueError) as exc:
            raise SourceLoadError(
                f"dataset source {source_id!r} has invalid JSON at line "
                f"{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise SourceLoadError(
                f"dataset source {source_id!r} JSONL line {line_number} must "
                "be an object"
            )
        keys = set(value)
        if keys not in ({"prompt"}, {"prompt", "metadata"}):
            raise SourceLoadError(
                f"dataset source {source_id!r} JSONL line {line_number} must "
                "contain exactly prompt and optional metadata"
            )
        prompt = value["prompt"]
        _validate_prompt(prompt, source_id=source_id, line_number=line_number)
        metadata = value.get("metadata", {})
        if not isinstance(metadata, dict):
            raise SourceLoadError(
                f"dataset source {source_id!r} JSONL line {line_number} "
                "metadata must be an object"
            )
        _validate_json_value(
            metadata,
            location=(
                f"dataset source {source_id!r} JSONL line {line_number} metadata"
            ),
        )
        result.append(SourceRecord(prompt=prompt, metadata=metadata))
    return tuple(result)


def _validate_prompt(prompt: object, *, source_id: str, line_number: int) -> None:
    if not isinstance(prompt, str) or not prompt:
        raise SourceLoadError(
            f"dataset source {source_id!r} has an empty/non-string prompt at "
            f"line {line_number}"
        )
    if prompt != prompt.strip():
        raise SourceLoadError(
            f"dataset source {source_id!r} prompt at line {line_number} has "
            "leading or trailing whitespace"
        )


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _validate_json_value(value: object, *, location: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SourceLoadError(f"{location} contains a non-finite float")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, location=f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_json_value(item, location=f"{location}.{key}")
        return
    raise SourceLoadError(
        f"{location} contains unsupported value type {type(value).__name__}"
    )


def _content_revision(
    descriptor: DatasetSourceDescriptor,
    records: tuple[SourceRecord, ...],
) -> str:
    payload = {
        "schema": _CONTENT_IDENTITY_SCHEMA,
        "selector": descriptor.selector,
        "task": descriptor.task,
        "records": [record.identity_payload() for record in records],
    }
    return f"sha256:{hashlib.sha256(_canonical_bytes(payload)).hexdigest()}"


def _source_item_id(
    *,
    source_id: str,
    dataset_index: int,
    revision: str,
) -> str:
    payload = {
        "schema": _SOURCE_ITEM_IDENTITY_SCHEMA,
        "source_id": source_id,
        "dataset_index": dataset_index,
        "revision": revision,
    }
    return f"source-item-{hashlib.sha256(_canonical_bytes(payload)).hexdigest()}"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
