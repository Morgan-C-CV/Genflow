import json
from typing import Any, Dict

from app.agent.runtime_models import NormalizedSchema


REQUIRED_METADATA_FIELDS = (
    "prompt",
    "negative_prompt",
    "cfgscale",
    "steps",
    "sampler",
    "seed",
    "model",
    "clipskip",
    "style",
    "lora",
    "full_metadata_string",
)


class MetadataParseError(ValueError):
    pass


class MetadataValidationError(ValueError):
    pass


def parse_metadata_json(raw_json: str) -> Dict[str, Any]:
    if not isinstance(raw_json, str):
        raise MetadataParseError("Metadata payload must be a JSON string.")
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise MetadataParseError(f"Failed to parse metadata JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise MetadataValidationError("Metadata JSON root must be an object.")
    return parsed


def validate_metadata_fields(data: Dict[str, Any]) -> Dict[str, str]:
    missing = [field for field in REQUIRED_METADATA_FIELDS if field not in data]
    if missing:
        raise MetadataValidationError(
            f"Metadata is missing required fields: {', '.join(missing)}"
        )

    normalized: Dict[str, str] = {}
    for field in REQUIRED_METADATA_FIELDS:
        value = data[field]
        if value is None:
            raise MetadataValidationError(f"Metadata field '{field}' cannot be null.")
        if not isinstance(value, str):
            raise MetadataValidationError(f"Metadata field '{field}' must be a string.")
        cleaned = value.strip()
        if field != "lora" and field != "style" and not cleaned:
            raise MetadataValidationError(f"Metadata field '{field}' cannot be empty.")
        normalized[field] = cleaned
    return normalized


def normalize_metadata_schema(data: Dict[str, Any]) -> NormalizedSchema:
    validated = validate_metadata_fields(data)
    return NormalizedSchema(
        prompt=validated["prompt"],
        negative_prompt=validated["negative_prompt"],
        cfgscale=validated["cfgscale"],
        steps=validated["steps"],
        sampler=validated["sampler"],
        seed=validated["seed"],
        model=validated["model"],
        clipskip=validated["clipskip"],
        style=_split_csv_field(validated["style"]),
        lora=_split_csv_field(validated["lora"]),
        full_metadata_string=validated["full_metadata_string"],
        raw_fields=dict(validated),
    )


def parse_and_normalize_metadata(raw_json: str) -> NormalizedSchema:
    return normalize_metadata_schema(parse_metadata_json(raw_json))


def _split_csv_field(value: str) -> list[str]:
    stripped = value.strip()
    if not stripped or stripped.lower() == "none":
        return []
    return [item.strip() for item in stripped.split(",") if item.strip()]
