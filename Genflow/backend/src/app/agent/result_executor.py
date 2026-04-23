from __future__ import annotations

from typing import Callable, Dict, Optional
from uuid import uuid4

from app.agent.runtime_models import (
    CommittedPatch,
    NormalizedSchema,
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
        probe: Dict[str, object],
    ) -> PreviewResult:
        payload = ResultPayload(
            result_id=self._id_factory(),
            result_type="mock_preview_result",
            content={
                "prompt": schema.prompt,
                "probe": dict(probe),
            },
            artifacts={"render_mode": "mock_preview"},
        )
        summary = ResultSummary(
            summary_text="Mock preview result.",
            changed_axes=[],
            preserved_axes=[],
            notes=[],
        )
        return PreviewResult(
            probe_id=str(probe.get("probe_id", "")),
            summary=summary,
            payload=payload,
            comparison_notes=[],
        )

    def execute_committed_patch(
        self,
        schema: NormalizedSchema,
        patch: CommittedPatch,
    ) -> tuple[ResultPayload, ResultSummary]:
        payload = ResultPayload(
            result_id=self._id_factory(),
            result_type="mock_committed_result",
            content={
                "prompt": schema.prompt,
                "patch_id": patch.patch_id,
            },
            artifacts={"render_mode": "mock_commit"},
        )
        summary = ResultSummary(
            summary_text="Mock committed patch result.",
            changed_axes=list(patch.target_fields),
            preserved_axes=[],
            notes=[patch.rationale] if patch.rationale else [],
        )
        return payload, summary
