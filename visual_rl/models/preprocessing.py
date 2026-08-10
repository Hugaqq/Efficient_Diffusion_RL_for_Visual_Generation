"""Model-owned preprocessing consumer contract."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from visual_rl.core.serialization import canonical_json_text
from visual_rl.data.preprocess import PreprocessProducerSpec

__all__ = ("ModelPreprocessConsumerSpec", "ModelPreprocessSpec")

# One-slice source compatibility only. The producer schema has exactly one
# runtime identity and is owned by visual_rl.data.preprocess.
ModelPreprocessSpec = PreprocessProducerSpec


def _canonical_text(value: object, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\r" in value
        or "\n" in value
    ):
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _canonical_string_tuple(
    value: object,
    *,
    field_name: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field_name} must be a tuple")
    if not allow_empty and not value:
        raise ValueError(f"{field_name} must not be empty")
    for item in value:
        _canonical_text(item, field_name=f"{field_name} entry")
    canonical = tuple(sorted(value))
    if len(canonical) != len(set(canonical)):
        raise ValueError(f"{field_name} must contain unique values")
    return canonical


@dataclass(frozen=True, slots=True)
class ModelPreprocessConsumerSpec:
    """Model-owned declaration of fields consumed by its forward path.

    This is deliberately separate from :class:`PreprocessProducerSpec`: the
    producer and consumer declarations can therefore be compared at G3 instead
    of silently mirroring one another.
    """

    implementation_revision: str
    payload_type: str
    required_modalities: tuple[str, ...]
    positive_output_fields: tuple[str, ...]
    negative_output_fields: tuple[str, ...] = ()
    uses_negative_condition: bool = False

    def __post_init__(self) -> None:
        _canonical_text(
            self.implementation_revision,
            field_name="implementation_revision",
        )
        _canonical_text(self.payload_type, field_name="payload_type")
        object.__setattr__(
            self,
            "required_modalities",
            _canonical_string_tuple(
                self.required_modalities,
                field_name="required_modalities",
                allow_empty=False,
            ),
        )
        positive = _canonical_string_tuple(
            self.positive_output_fields,
            field_name="positive_output_fields",
            allow_empty=False,
        )
        negative = _canonical_string_tuple(
            self.negative_output_fields,
            field_name="negative_output_fields",
            allow_empty=True,
        )
        if set(positive).intersection(negative):
            raise ValueError(
                "positive_output_fields and negative_output_fields must be disjoint"
            )
        if type(self.uses_negative_condition) is not bool:
            raise TypeError("uses_negative_condition must be bool")
        if self.uses_negative_condition and not negative:
            raise ValueError("uses_negative_condition requires negative_output_fields")
        object.__setattr__(self, "positive_output_fields", positive)
        object.__setattr__(self, "negative_output_fields", negative)

    @property
    def required_output_fields(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                self.positive_output_fields
                + (self.negative_output_fields if self.uses_negative_condition else ())
            )
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "implementation_revision": self.implementation_revision,
            "payload_type": self.payload_type,
            "required_modalities": list(self.required_modalities),
            "positive_output_fields": list(self.positive_output_fields),
            "negative_output_fields": list(self.negative_output_fields),
            "uses_negative_condition": self.uses_negative_condition,
        }

    @property
    def consumer_spec_id(self) -> str:
        return hashlib.sha256(
            canonical_json_text(self.to_payload()).encode("utf-8")
        ).hexdigest()
