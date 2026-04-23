from __future__ import annotations

from typing import Callable, Dict, Optional
from uuid import uuid4

from app.agent.runtime_models import (
    CommittedPatch,
    NormalizedSchema,
    PreviewProbe,
    PreviewResult,
    ResultPayload,
    ResultSummary,
)


class ResultExecutor:
    def __init__(self, id_factory: Optional[Callable[[], str]] = None):
        self._id_factory = id_factory or (lambda: str(uuid4()))

    def produce_initial_result(
        self,
        schema: NormalizedSchema,
        reference_bundle: Optional[Dict[str, object]] = None,
    ) -> tuple[ResultPayload, ResultSummary]:
        reference_bundle = reference_bundle or {}
        reference_count = len(reference_bundle.get("references", []))
        result_id = self._id_factory()
        payload = ResultPayload(
            result_id=result_id,
            result_type="mock_initial_result",
            content={
                "prompt": schema.prompt,
                "negative_prompt": schema.negative_prompt,
                "model": schema.model,
                "sampler": schema.sampler,
                "style": list(schema.style),
                "lora": list(schema.lora),
                "reference_count": reference_count,
            },
            artifacts={
                "render_mode": "mock",
                "schema_snapshot": {
                    "model": schema.model,
                    "sampler": schema.sampler,
                    "seed": schema.seed,
                },
            },
        )
        summary = ResultSummary(
            summary_text=(
                f"Mock initial result generated with model={schema.model or 'unknown'}, "
                f"sampler={schema.sampler or 'unknown'}, references={reference_count}."
            ),
            changed_axes=["initial_generation"],
            preserved_axes=[],
            notes=[
                f"prompt_length={len(schema.prompt)}",
                f"style_count={len(schema.style)}",
                f"lora_count={len(schema.lora)}",
            ],
        )
        return payload, summary

    def execute_preview_probe(
        self,
        schema: NormalizedSchema,
        probe: PreviewProbe,
    ) -> PreviewResult:
        payload = ResultPayload(
            result_id=self._id_factory(),
            result_type="mock_preview_result",
            content={
                "prompt": schema.prompt,
                "probe_id": probe.probe_id,
                "target_axes": list(probe.target_axes),
                "source_kind": probe.source_kind,
                "preview_execution_spec": dict(probe.preview_execution_spec),
            },
            artifacts={
                "render_mode": "mock_preview",
                "preserve_axes": list(probe.preserve_axes),
            },
        )
        summary = ResultSummary(
            summary_text=(
                f"Mock preview for probe={probe.probe_id}, "
                f"target_axes={','.join(probe.target_axes) or 'none'}, "
                f"source_kind={probe.source_kind or 'unknown'}."
            ),
            changed_axes=list(probe.target_axes),
            preserved_axes=list(probe.preserve_axes),
            notes=[
                f"patch_family={probe.preview_execution_spec.get('patch_family', '')}",
                f"reference_anchor={probe.preview_execution_spec.get('reference_anchor', '')}",
            ],
        )
        return PreviewResult(
            probe_id=probe.probe_id,
            summary=summary,
            payload=payload,
            comparison_notes=[
                f"preview_source={probe.source_kind}",
                f"target_axes={','.join(probe.target_axes)}",
            ],
        )

    def execute_committed_patch(
        self,
        schema: NormalizedSchema,
        patch: CommittedPatch,
    ) -> tuple[ResultPayload, ResultSummary]:
        changed_axes = list(patch.target_axes or patch.target_fields)
        payload = ResultPayload(
            result_id=self._id_factory(),
            result_type="mock_committed_result",
            content={
                "prompt": schema.prompt,
                "patch_id": patch.patch_id,
                "target_fields": list(patch.target_fields),
                "target_axes": changed_axes,
                "rationale": patch.rationale,
            },
            artifacts={
                "render_mode": "mock_commit",
                "changes": dict(patch.changes),
                "preserve_axes": list(patch.preserve_axes),
            },
        )
        summary = ResultSummary(
            summary_text=(
                f"Mock committed patch result for patch={patch.patch_id}, "
                f"target_fields={','.join(patch.target_fields) or 'none'}, "
                f"target_axes={','.join(changed_axes) or 'none'}."
            ),
            changed_axes=changed_axes,
            preserved_axes=list(patch.preserve_axes),
            notes=[
                patch.rationale,
                f"change_keys={','.join(patch.changes.keys())}",
            ] if patch.rationale else [f"change_keys={','.join(patch.changes.keys())}"],
        )
        return payload, summary
