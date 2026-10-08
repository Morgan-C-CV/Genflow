"""Convert a Genflow ``NormalizedSchema`` into ComfyUI workflow JSON.

Two output formats are produced:

* **API format** (``graph``) — the ``{"<id>": {"class_type": ..., "inputs": ...}}``
  payload accepted by ComfyUI's ``POST /prompt`` endpoint.
* **UI format** (``build_ui_workflow``) — the ``nodes``/``links`` document the
  ComfyUI canvas loads, derived generically from ``GET /object_info`` so widget
  ordering and slot types always match the installed ComfyUI version.

This module is pure: no network access, no global state.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ComfyUI input types that arrive over a link rather than as a widget value.
_LINK_TYPES = {
    "MODEL",
    "CLIP",
    "VAE",
    "CONDITIONING",
    "LATENT",
    "IMAGE",
    "MASK",
    "CONTROL_NET",
    "STYLE_MODEL",
    "CLIP_VISION",
    "CLIP_VISION_OUTPUT",
    "GLIGEN",
    "UPSCALE_MODEL",
    "SAMPLER",
    "SIGMAS",
    "GUIDER",
    "NOISE",
    "AUDIO",
    "PHOTOMAKER",
}

# Scalar widget types whose spec entry is a plain type name.
_WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}

_LORA_TAG_RE = re.compile(r"<lora:([^:>]+)(?::([^:>]+))?>", re.IGNORECASE)

# A1111 sampler name -> (ComfyUI sampler_name, default scheduler).
_SAMPLER_MAP: Dict[str, Tuple[str, str]] = {
    "euler": ("euler", "normal"),
    "euler a": ("euler_ancestral", "normal"),
    "euler ancestral": ("euler_ancestral", "normal"),
    "lms": ("lms", "normal"),
    "heun": ("heun", "normal"),
    "dpm2": ("dpm_2", "normal"),
    "dpm2 a": ("dpm_2_ancestral", "normal"),
    "dpm2 ancestral": ("dpm_2_ancestral", "normal"),
    "dpm++ 2s a": ("dpmpp_2s_ancestral", "normal"),
    "dpm++ 2s ancestral": ("dpmpp_2s_ancestral", "normal"),
    "dpm++ 2m": ("dpmpp_2m", "normal"),
    "dpm++ 2m sde": ("dpmpp_2m_sde", "normal"),
    "dpm++ 3m sde": ("dpmpp_3m_sde", "exponential"),
    "dpm++ sde": ("dpmpp_sde", "normal"),
    "dpm fast": ("dpm_fast", "normal"),
    "dpm adaptive": ("dpm_adaptive", "normal"),
    "ddim": ("ddim", "ddim_uniform"),
    "unipc": ("uni_pc", "normal"),
    "uni_pc": ("uni_pc", "normal"),
    "lcm": ("lcm", "normal"),
    "plms": ("lms", "normal"),
}

_SCHEDULER_ALIASES: Dict[str, str] = {
    "karras": "karras",
    "exponential": "exponential",
    "sgm uniform": "sgm_uniform",
    "sgm_uniform": "sgm_uniform",
    "beta": "beta",
    "normal": "normal",
    "simple": "simple",
    "ddim uniform": "ddim_uniform",
    "ddim_uniform": "ddim_uniform",
    "linear quadratic": "linear_quadratic",
    "kl optimal": "kl_optimal",
}


@dataclass
class GraphBuildResult:
    """Outcome of converting a schema into a ComfyUI API graph."""

    graph: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    checkpoint: str = ""
    checkpoint_resolved: bool = False
    applied_loras: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_loras: List[str] = field(default_factory=list)
    controls: Dict[str, Any] = field(default_factory=dict)

    @property
    def checkpoint_available(self) -> bool:
        return self.checkpoint_resolved


def _normalize_name(value: str) -> str:
    """Lowercase and drop every non-alphanumeric character for fuzzy matching."""
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def resolve_against(candidate: str, available: Sequence[str]) -> Optional[str]:
    """Pick the available entry that best matches ``candidate``.

    Matching is deliberately forgiving because Genflow emits Civitai display
    names while ComfyUI stores on-disk filenames.
    """
    if not candidate or not available:
        return None

    target = _normalize_name(candidate)
    if not target:
        return None

    normalized = [(_normalize_name(name), name) for name in available]

    for norm, original in normalized:
        if norm == target:
            return original

    # Prefer the longest available name contained in the candidate (or vice versa).
    best: Optional[str] = None
    best_len = 0
    for norm, original in normalized:
        if not norm:
            continue
        if norm in target or target in norm:
            if len(norm) > best_len:
                best, best_len = original, len(norm)
    return best


def rank_similar(candidate: str, available: Sequence[str], limit: int = 5) -> List[str]:
    """Rank installed names by similarity, for "did you mean" suggestions."""
    names = [str(name) for name in available if str(name).strip()]
    if not names:
        return []

    target = _normalize_name(candidate)
    if not target:
        return names[:limit]

    scored: List[Tuple[float, str]] = []
    for name in names:
        norm = _normalize_name(name)
        if not norm:
            continue
        ratio = difflib.SequenceMatcher(None, target, norm).ratio()
        if target in norm or norm in target:
            ratio = max(ratio, 0.75)
        scored.append((ratio, name))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [name for _, name in scored[:limit]]


def _coerce_int(value: Any, default: int, *, minimum: Optional[int] = None) -> Tuple[int, Optional[str]]:
    raw = str(value or "").strip()
    if not raw:
        return default, None
    match = re.search(r"-?\d+", raw)
    if not match:
        return default, f"Could not read an integer from {raw!r}; using {default}."
    number = int(match.group(0))
    if minimum is not None and number < minimum:
        return default, f"Value {number} is below the minimum {minimum}; using {default}."
    return number, None


def _coerce_float(value: Any, default: float, *, minimum: Optional[float] = None) -> Tuple[float, Optional[str]]:
    raw = str(value or "").strip()
    if not raw:
        return default, None
    match = re.search(r"-?\d+(?:\.\d+)?", raw)
    if not match:
        return default, f"Could not read a number from {raw!r}; using {default}."
    number = float(match.group(0))
    if minimum is not None and number < minimum:
        return default, f"Value {number} is below the minimum {minimum}; using {default}."
    return number, None


def map_sampler(sampler: str, available: Sequence[str]) -> Tuple[str, str, List[str]]:
    """Map an A1111-style sampler string to ``(sampler_name, scheduler, warnings)``."""
    warnings: List[str] = []
    raw = str(sampler or "").strip()
    available = list(available) or ["euler"]
    lowered = raw.lower().replace("＋", "+")

    scheduler = ""
    # Split an explicit scheduler suffix ("DPM++ 2M Karras").
    for alias, resolved in _SCHEDULER_ALIASES.items():
        for suffix in (f" {alias}", f"_{alias}", f"-{alias}"):
            if lowered.endswith(suffix):
                scheduler = resolved
                lowered = lowered[: -len(suffix)].strip(" _-")
                break
        if scheduler:
            break

    canonical = lowered.replace("_", " ").strip()
    sampler_name, default_scheduler = _SAMPLER_MAP.get(canonical, ("", ""))

    if not sampler_name:
        # Try a normalized lookup before giving up.
        for key, value in _SAMPLER_MAP.items():
            if _normalize_name(key) == _normalize_name(canonical):
                sampler_name, default_scheduler = value
                break

    if not sampler_name:
        fallback = resolve_against(canonical, available)
        if fallback:
            sampler_name = fallback
            default_scheduler = "normal"
            warnings.append(
                f"Sampler {raw!r} is not a known A1111 name; matched ComfyUI sampler {fallback!r}."
            )
        else:
            sampler_name, default_scheduler = "euler", "normal"
            warnings.append(f"Unrecognised sampler {raw!r}; falling back to 'euler'.")

    if sampler_name not in available:
        # The mapped name is absent from this ComfyUI build; try a prefix match.
        matched = resolve_against(sampler_name, available)
        if matched and matched != sampler_name:
            warnings.append(f"Sampler {sampler_name!r} unavailable; using {matched!r}.")
            sampler_name = matched
        elif sampler_name not in available:
            fallback = "euler" if "euler" in available else available[0]
            warnings.append(f"Sampler {sampler_name!r} unavailable; using {fallback!r}.")
            sampler_name = fallback

    if not scheduler:
        scheduler = default_scheduler
    if scheduler not in _SCHEDULER_ALIASES.values():
        scheduler = "normal"

    available_schedulers = {"simple", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta", "normal", "linear_quadratic", "kl_optimal"}
    if scheduler not in available_schedulers:
        scheduler = "normal"

    return sampler_name, scheduler, warnings


def split_lora_tags(prompt: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Strip ``<lora:name:weight>`` tags out of ``prompt``.

    Core ComfyUI has no syntactic support for these tags, so they cannot stay in
    the encoded text. Returns the cleaned prompt plus the parsed tags.
    """
    tags: List[Dict[str, Any]] = []

    def _collect(match: re.Match) -> str:
        name = (match.group(1) or "").strip()
        weight_raw = (match.group(2) or "").strip()
        weight = 1.0
        if weight_raw:
            try:
                weight = float(weight_raw)
            except ValueError:
                weight = 1.0
        if name:
            tags.append({"name": name, "weight": weight})
        return ""

    cleaned = _LORA_TAG_RE.sub(_collect, str(prompt or ""))
    # Removing a tag often leaves an orphaned separator behind ("a, , b").
    cleaned = re.sub(r"\s*,\s*(?=,)", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = cleaned.strip().strip(",").strip()
    cleaned = re.sub(r",\s*,+", ",", cleaned).strip(", ")
    return cleaned, tags


def resolve_loras(
    tags: Iterable[Dict[str, Any]],
    available: Sequence[str],
    declared: Sequence[str] = (),
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Resolve parsed LoRA tags (plus the schema's ``lora`` field) to on-disk files.

    Returns ``(applied, unresolved)``. The schema's ``lora`` field lists the same
    LoRAs as the inline tags but without weights, so entries already satisfied by
    a tag are not reported as unresolved.
    """
    applied: List[Dict[str, Any]] = []
    unresolved: List[str] = []
    seen_files: set[str] = set()
    seen_names: set[str] = set()
    seen_unresolved: set[str] = set()

    def _record(requested: str, weight: float) -> None:
        key = _normalize_name(requested)
        resolved = resolve_against(requested, available)
        if resolved:
            file_key = resolved.lower()
            if file_key in seen_files:
                return
            seen_files.add(file_key)
            seen_names.update({key, _normalize_name(resolved)})
            applied.append({"requested": requested, "resolved": resolved, "weight": weight})
            return
        # The schema's `lora` field repeats the inline tags, so an unresolved name
        # would otherwise be reported once per source.
        if key in seen_names or key in seen_unresolved:
            return
        seen_unresolved.add(key)
        unresolved.append(requested)

    for tag in tags:
        name = str(tag.get("name", "")).strip()
        if name:
            _record(name, float(tag.get("weight", 1.0)))

    for name in declared:
        cleaned = str(name).strip()
        if not cleaned or cleaned.lower() == "none":
            continue
        _record(cleaned, 1.0)

    return applied, unresolved


def build_api_graph(
    schema: Any,
    *,
    checkpoints: Sequence[str] = (),
    loras: Sequence[str] = (),
    samplers: Sequence[str] = (),
    width: int = 1024,
    height: int = 1024,
    batch_size: int = 1,
    filename_prefix: str = "Genflow",
    seed_override: Optional[int] = None,
    checkpoint_override: Optional[str] = None,
) -> GraphBuildResult:
    """Build a ComfyUI API-format txt2img graph from a Genflow schema."""
    result = GraphBuildResult()
    warnings = result.warnings

    checkpoints = list(checkpoints)
    loras = list(loras)
    samplers = list(samplers)

    # --- checkpoint ------------------------------------------------------
    # Missing checkpoints/LoRAs are reported as structured remediation items by
    # ComfyUIService rather than as free-text warnings, so they are not repeated here.
    requested_model = str(getattr(schema, "model", "") or "").strip()
    override = str(checkpoint_override or "").strip()
    if override:
        # The caller explicitly chose this checkpoint (a remediation); honour it
        # verbatim and only report whether ComfyUI actually has it.
        result.checkpoint = override
        result.checkpoint_resolved = override in checkpoints
    else:
        resolved_checkpoint = resolve_against(requested_model, checkpoints)
        if resolved_checkpoint:
            result.checkpoint = resolved_checkpoint
            result.checkpoint_resolved = True
        else:
            result.checkpoint = requested_model
            result.checkpoint_resolved = False

    # --- scalars ---------------------------------------------------------
    steps, note = _coerce_int(getattr(schema, "steps", ""), 30, minimum=1)
    if note:
        warnings.append(note)
    cfg, note = _coerce_float(getattr(schema, "cfgscale", ""), 7.0, minimum=0.0)
    if note:
        warnings.append(note)
    clipskip, note = _coerce_int(getattr(schema, "clipskip", ""), 1, minimum=1)
    if note:
        warnings.append(note)

    if seed_override is not None:
        seed = int(seed_override)
    else:
        seed, note = _coerce_int(getattr(schema, "seed", ""), 0, minimum=0)
        if note:
            warnings.append(note)
        if not str(getattr(schema, "seed", "") or "").strip():
            warnings.append("Schema carried no seed; using 0.")

    sampler_name, scheduler, sampler_warnings = map_sampler(getattr(schema, "sampler", ""), samplers)
    warnings.extend(sampler_warnings)

    # --- prompt text -----------------------------------------------------
    raw_prompt = str(getattr(schema, "prompt", "") or "")
    cleaned_prompt, lora_tags = split_lora_tags(raw_prompt)
    if lora_tags:
        warnings.append(
            f"Removed {len(lora_tags)} inline <lora:...> tag(s) from the prompt: core ComfyUI "
            "does not parse that syntax."
        )

    declared_loras = list(getattr(schema, "lora", []) or [])
    applied_loras, unresolved_loras = resolve_loras(lora_tags, loras, declared_loras)
    result.applied_loras = applied_loras
    result.unresolved_loras = unresolved_loras

    negative_prompt = str(getattr(schema, "negative_prompt", "") or "")
    _, negative_tags = split_lora_tags(negative_prompt)
    if negative_tags:
        negative_prompt, _ = split_lora_tags(negative_prompt)

    # --- graph assembly --------------------------------------------------
    graph: Dict[str, Dict[str, Any]] = {}
    next_id = 1

    def _add(class_type: str, inputs: Dict[str, Any]) -> str:
        nonlocal next_id
        node_id = str(next_id)
        next_id += 1
        graph[node_id] = {"class_type": class_type, "inputs": inputs}
        return node_id

    checkpoint_id = _add("CheckpointLoaderSimple", {"ckpt_name": result.checkpoint})
    model_ref: List[Any] = [checkpoint_id, 0]
    clip_ref: List[Any] = [checkpoint_id, 1]
    vae_ref: List[Any] = [checkpoint_id, 2]

    for entry in applied_loras:
        lora_id = _add(
            "LoraLoader",
            {
                "model": model_ref,
                "clip": clip_ref,
                "lora_name": entry["resolved"],
                "strength_model": float(entry["weight"]),
                "strength_clip": float(entry["weight"]),
            },
        )
        model_ref = [lora_id, 0]
        clip_ref = [lora_id, 1]

    if clipskip > 1:
        clip_id = _add("CLIPSetLastLayer", {"clip": clip_ref, "stop_at_clip_layer": -abs(clipskip)})
        clip_ref = [clip_id, 0]

    positive_id = _add("CLIPTextEncode", {"text": cleaned_prompt, "clip": clip_ref})
    negative_id = _add("CLIPTextEncode", {"text": negative_prompt, "clip": clip_ref})
    latent_id = _add(
        "EmptyLatentImage",
        {"width": int(width), "height": int(height), "batch_size": int(batch_size)},
    )
    sampler_id = _add(
        "KSampler",
        {
            "seed": int(seed),
            "steps": int(steps),
            "cfg": float(cfg),
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": 1.0,
            "model": model_ref,
            "positive": [positive_id, 0],
            "negative": [negative_id, 0],
            "latent_image": [latent_id, 0],
        },
    )
    decode_id = _add("VAEDecode", {"samples": [sampler_id, 0], "vae": vae_ref})
    _add("SaveImage", {"images": [decode_id, 0], "filename_prefix": filename_prefix})

    result.graph = graph
    result.controls = {
        "steps": steps,
        "cfg": float(cfg),
        "seed": int(seed),
        "clipskip": clipskip,
        "sampler_name": sampler_name,
        "scheduler": scheduler,
        "width": int(width),
        "height": int(height),
        "batch_size": int(batch_size),
    }
    return result


# ---------------------------------------------------------------------------
# UI ("canvas") workflow format
# ---------------------------------------------------------------------------


def _input_kind(spec: Any) -> str:
    """Classify an ``object_info`` input spec as ``link`` or ``widget``."""
    if not isinstance(spec, (list, tuple)) or not spec:
        return "widget"
    type_token = spec[0]
    if isinstance(type_token, list):
        return "widget"
    if not isinstance(type_token, str):
        return "widget"
    if type_token in _WIDGET_TYPES:
        return "widget"
    if type_token in _LINK_TYPES:
        return "link"
    # Unknown string tokens in ComfyUI are node-connection types (e.g. custom types).
    return "link"


def _iter_input_specs(node_def: Dict[str, Any]) -> List[Tuple[str, Any, bool]]:
    """Yield ``(name, spec, required)`` in declaration order."""
    inputs = node_def.get("input", {}) or {}
    ordered: List[Tuple[str, Any, bool]] = []
    for name, spec in (inputs.get("required", {}) or {}).items():
        ordered.append((name, spec, True))
    for name, spec in (inputs.get("optional", {}) or {}).items():
        ordered.append((name, spec, False))
    return ordered


def _widget_value(spec: Any, raw: Any) -> Any:
    """Coerce an API-graph value into the representation the canvas expects."""
    options = spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict) else {}
    type_token = spec[0] if isinstance(spec, (list, tuple)) and spec else None
    if isinstance(type_token, list):
        return raw
    if type_token == "INT":
        parsed, _ = _coerce_int(raw, int(options.get("default", 0) or 0))
        return parsed
    if type_token == "FLOAT":
        parsed, _ = _coerce_float(raw, float(options.get("default", 0.0) or 0.0))
        return parsed
    if type_token == "BOOLEAN":
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}
    return raw


def _topological_order(graph: Dict[str, Dict[str, Any]]) -> List[str]:
    """Order node ids so every dependency precedes its consumer."""
    dependencies: Dict[str, set[str]] = {}
    for node_id, node in graph.items():
        deps: set[str] = set()
        for value in (node.get("inputs") or {}).values():
            if isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[0], (str, int)):
                source = str(value[0])
                if source in graph and source != node_id:
                    deps.add(source)
        dependencies[node_id] = deps

    ordered: List[str] = []
    remaining = dict(dependencies)
    while remaining:
        ready = [nid for nid, deps in remaining.items() if not (deps - set(ordered))]
        if not ready:
            # Cycle or dangling reference: fall back to insertion order.
            ordered.extend(sorted(remaining, key=lambda n: int(n) if str(n).isdigit() else 0))
            break
        ready.sort(key=lambda n: int(n) if str(n).isdigit() else 0)
        for nid in ready:
            ordered.append(nid)
            remaining.pop(nid, None)
    return ordered


def build_ui_workflow(
    graph: Dict[str, Dict[str, Any]],
    object_info: Dict[str, Any],
    *,
    title: str = "Genflow Workflow",
) -> Dict[str, Any]:
    """Convert an API-format graph into ComfyUI's UI (canvas) workflow format.

    Slot names, types and widget ordering are derived from ``object_info`` so the
    document stays correct across ComfyUI versions.
    """
    order = _topological_order(graph)
    depth_cache: Dict[str, int] = {}
    node_records: Dict[str, Dict[str, Any]] = {}
    links: List[List[Any]] = []
    link_counter = 0

    def _depth(node_id: str) -> int:
        if node_id in depth_cache:
            return depth_cache[node_id]
        depth_cache[node_id] = 0  # guard against cycles
        depth = 0
        for value in (graph[node_id].get("inputs") or {}).values():
            if isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[0], (str, int)):
                source = str(value[0])
                if source in graph and source != node_id:
                    depth = max(depth, _depth(source) + 1)
        depth_cache[node_id] = depth
        return depth

    # Pass 1: create node shells and resolve widget values.
    for node_id in order:
        node = graph[node_id]
        class_type = node.get("class_type", "")
        values = node.get("inputs") or {}
        node_def = object_info.get(class_type) or {}
        specs = _iter_input_specs(node_def)

        widgets_values: List[Any] = []
        input_slots: List[Dict[str, Any]] = []
        seen_widget_names: List[str] = []

        for name, spec, _required in specs:
            kind = _input_kind(spec)
            if kind == "widget":
                widgets_values.append(_widget_value(spec, values.get(name)))
                seen_widget_names.append(name)
                options = spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict) else {}
                if options.get("control_after_generate"):
                    # The canvas injects an extra combo widget right after this input.
                    widgets_values.append("randomize")
            else:
                input_slots.append({"name": name, "type": spec[0] if spec else "*", "link": None})

        output_names = node_def.get("output_name") or node_def.get("output") or []
        output_types = node_def.get("output") or []
        output_slots = [
            {
                "name": output_names[idx] if idx < len(output_names) else f"out{idx}",
                "type": output_types[idx] if idx < len(output_types) else "*",
                "links": [],
                "slot_index": idx,
            }
            for idx in range(len(output_types))
        ]

        depth = _depth(node_id)
        node_records[node_id] = {
            "id": int(node_id) if str(node_id).isdigit() else node_id,
            "type": class_type,
            "pos": [40 + depth * 340, 40 + len(node_records) % 6 * 200],
            "size": [320, max(90, 26 * max(1, len(widgets_values)) + 60)],
            "flags": {},
            "order": order.index(node_id),
            "mode": 0,
            "inputs": input_slots,
            "outputs": output_slots,
            "properties": {"Node name for S&R": class_type},
            "widgets_values": widgets_values,
        }

    # Pass 2: materialise links now that every node shell exists.
    for node_id in order:
        node = graph[node_id]
        record = node_records[node_id]
        for name, value in (node.get("inputs") or {}).items():
            if not (isinstance(value, (list, tuple)) and len(value) == 2):
                continue
            source_id, source_slot = str(value[0]), value[1]
            if source_id not in node_records:
                continue
            slot = next((s for s in record["inputs"] if s["name"] == name), None)
            if slot is None:
                continue
            source_record = node_records[source_id]
            slot_index = int(source_slot) if isinstance(source_slot, int) else 0
            if slot_index >= len(source_record["outputs"]):
                continue
            link_counter += 1
            slot["link"] = link_counter
            source_record["outputs"][slot_index]["links"].append(link_counter)
            links.append(
                [
                    link_counter,
                    source_record["id"],
                    slot_index,
                    record["id"],
                    record["inputs"].index(slot),
                    slot["type"],
                ]
            )

    max_id = max((int(nid) for nid in graph if str(nid).isdigit()), default=0)
    return {
        "id": title,
        "revision": 0,
        "last_node_id": max_id,
        "last_link_id": link_counter,
        "nodes": [node_records[nid] for nid in order],
        "links": links,
        "groups": [],
        "config": {},
        "extra": {"ds": {"title": title}},
        "version": 0.4,
    }
