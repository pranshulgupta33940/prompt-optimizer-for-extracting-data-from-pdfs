"""Schema parsing utilities for ExtractBench JSON schemas."""

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FieldEvalConfig:
    """Evaluation configuration for a single schema field.

    Attributes:
        metric_id: The metric identifier (e.g. ``'string_exact'``).
        params: Optional parameters dict (e.g. ``{'tolerance': 0.001}``).
    """

    metric_id: str
    params: dict[str, Any] | None = None


@dataclass
class SchemaInfo:
    """Parsed schema with resolved references and field-level eval configs.

    Attributes:
        raw_schema: The original, unmodified schema dict.
        resolved_schema: Schema with all ``$ref`` entries resolved.
        field_configs: Mapping from dotted field paths to eval configs.
        schema_text: Pretty-printed JSON string of the raw schema.
    """

    raw_schema: dict
    resolved_schema: dict = field(repr=False)
    field_configs: dict[str, FieldEvalConfig] = field(repr=False)
    schema_text: str = field(repr=False)


# ---------------------------------------------------------------------------
#  $ref resolution
# ---------------------------------------------------------------------------

def resolve_refs(schema: dict) -> dict:
    """Recursively resolve ``$ref`` references using ``$defs``.

    Only ``#/$defs/<name>`` references are supported (the format used by
    ExtractBench schemas).  Additional keys alongside ``$ref`` are merged
    into the resolved result.

    Args:
        schema: Root JSON-Schema dict that may contain ``$defs``.

    Returns:
        A deep copy of the schema with all ``$ref`` entries inlined.
    """
    defs = schema.get("$defs", {})

    def _resolve(node: Any, depth: int = 0) -> Any:
        if depth > 50:
            return node
        if isinstance(node, dict):
            if "$ref" in node:
                return _resolve_ref(node, defs, depth)
            return {k: _resolve(v, depth) for k, v in node.items()}
        if isinstance(node, list):
            return [_resolve(item, depth) for item in node]
        return node

    resolved = _resolve(copy.deepcopy(schema))
    resolved.pop("$defs", None)
    return resolved


def _resolve_ref(
    node: dict,
    defs: dict,
    depth: int,
) -> Any:
    """Resolve a single ``$ref`` node against ``$defs``."""
    ref_path: str = node["$ref"]
    if not ref_path.startswith("#/$defs/"):
        return node

    def_name = ref_path.split("/")[-1]
    if def_name not in defs:
        return node

    resolved = copy.deepcopy(defs[def_name])
    for key, value in node.items():
        if key != "$ref":
            resolved[key] = value

    return _resolve_recursive(resolved, defs, depth + 1)


def _resolve_recursive(node: Any, defs: dict, depth: int) -> Any:
    """Continue resolving inside an already-resolved subtree."""
    if depth > 50:
        return node
    if isinstance(node, dict):
        if "$ref" in node:
            return _resolve_ref(node, defs, depth)
        return {k: _resolve_recursive(v, defs, depth) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_recursive(item, defs, depth) for item in node]
    return node


# ---------------------------------------------------------------------------
#  evaluation_config extraction
# ---------------------------------------------------------------------------

def extract_eval_configs(
    schema: dict,
    prefix: str = "",
) -> dict[str, FieldEvalConfig]:
    """Walk a resolved schema tree and collect ``evaluation_config`` entries.

    Field paths use dot notation for nested objects and ``[]`` for array
    items.  For example: ``authors[].name``, ``terms.loan_commitment.amount``.

    Args:
        schema: A resolved JSON-Schema dict (no ``$ref`` entries).
        prefix: Current path prefix (used during recursion).

    Returns:
        Dict mapping field paths to ``FieldEvalConfig`` instances.
    """
    configs: dict[str, FieldEvalConfig] = {}

    if "evaluation_config" in schema:
        parsed = _parse_eval_config(schema["evaluation_config"])
        if parsed is not None and prefix:
            configs[prefix] = parsed

    if "properties" in schema:
        for name, prop_schema in schema["properties"].items():
            child = f"{prefix}.{name}" if prefix else name
            configs.update(extract_eval_configs(prop_schema, child))

    if "items" in schema and isinstance(schema["items"], dict):
        items_prefix = f"{prefix}[]" if prefix else "[]"
        configs.update(extract_eval_configs(schema["items"], items_prefix))

    for union_key in ("anyOf", "oneOf"):
        if union_key in schema:
            for variant in schema[union_key]:
                if isinstance(variant, dict) and variant.get("type") != "null":
                    configs.update(extract_eval_configs(variant, prefix))

    if "schema_definition" in schema:
        configs.update(extract_eval_configs(schema["schema_definition"], prefix))

    return configs


def _parse_eval_config(raw: Any) -> FieldEvalConfig | None:
    """Parse a raw ``evaluation_config`` value.

    Supports two formats:
      * **String shorthand**: ``"string_exact"``
      * **Object with metrics list**:
        ``{"metrics": [{"metric_id": "number_tolerance", "params": {...}}]}``

    Returns:
        A ``FieldEvalConfig`` or ``None`` if parsing fails.
    """
    if isinstance(raw, str):
        return FieldEvalConfig(metric_id=raw)

    if isinstance(raw, dict):
        if "metrics" in raw and isinstance(raw["metrics"], list):
            first = raw["metrics"][0] if raw["metrics"] else None
            if first and isinstance(first, dict):
                return FieldEvalConfig(
                    metric_id=first.get("metric_id", ""),
                    params=first.get("params"),
                )
        if "metric_id" in raw:
            return FieldEvalConfig(
                metric_id=raw["metric_id"],
                params=raw.get("params"),
            )

    return None


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

class SchemaParser:
    """Loads and parses ExtractBench JSON schemas."""

    @staticmethod
    def parse(schema_path: Path) -> SchemaInfo:
        """Load a schema file, resolve refs, and extract eval configs.

        Args:
            schema_path: Filesystem path to a ``*-schema.json`` file.

        Returns:
            A ``SchemaInfo`` instance.
        """
        with open(schema_path, "r", encoding="utf-8") as fh:
            raw_schema = json.load(fh)

        resolved = resolve_refs(raw_schema)
        field_configs = extract_eval_configs(resolved)

        return SchemaInfo(
            raw_schema=raw_schema,
            resolved_schema=resolved,
            field_configs=field_configs,
            schema_text=json.dumps(raw_schema, indent=2),
        )

    @staticmethod
    def get_extraction_schema(schema_info: SchemaInfo) -> dict:
        """Return a clean schema suitable for inclusion in LLM prompts.

        Strips ``evaluation_config`` keys since they are internal metadata
        and would only confuse the extraction LLM.

        Args:
            schema_info: A parsed ``SchemaInfo`` instance.

        Returns:
            Schema dict without ``evaluation_config`` entries.
        """
        return _strip_eval_configs(copy.deepcopy(schema_info.raw_schema))


def _strip_eval_configs(node: Any) -> Any:
    """Recursively remove ``evaluation_config`` from a schema tree."""
    if isinstance(node, dict):
        node.pop("evaluation_config", None)
        return {k: _strip_eval_configs(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_strip_eval_configs(item) for item in node]
    return node
