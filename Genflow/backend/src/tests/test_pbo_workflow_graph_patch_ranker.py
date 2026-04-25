import unittest

from app.agent.benchmark_comparison_summary import BenchmarkComparisonSummary
from app.agent.memory import AgentMemoryService
from app.agent.pbo_workflow_graph_patch_ranker import rank_workflow_graph_patch_candidates
from app.agent.workflow_graph_patch_models import (
    WorkflowEdgePatch,
    WorkflowGraphPatchCandidate,
    WorkflowNodePatch,
    WorkflowRegionPatch,
)


class PboWorkflowGraphPatchRankerTest(unittest.TestCase):
    def _make_session(self):
        session = AgentMemoryService().create_session("make a portrait")
        session.dissatisfaction_axes = ["style"]
        session.preserve_constraints = ["composition"]
        session.benchmark_comparison_summary = BenchmarkComparisonSummary(
            focus_axes=["style"],
            preserve_axes=["composition"],
            metadata={"benchmark_source": "refinement_search_bundle"},
        )
        return session

    def test_ranking_changes_with_context(self):
        session = self._make_session()
        candidates = [
            WorkflowGraphPatchCandidate(
                workflow_id="workflow-1",
                candidate_id="gpc-1",
                candidate_kind="conservative",
                node_patches=[WorkflowNodePatch(node_id="render.model", target_axes=["style"])],
                edge_patches=[WorkflowEdgePatch(edge_id="e1")],
                region_patches=[WorkflowRegionPatch(region_id="r1")],
                target_axes=["style"],
                preserve_axes=["composition"],
            ),
            WorkflowGraphPatchCandidate(
                workflow_id="workflow-1",
                candidate_id="gpc-2",
                candidate_kind="aggressive",
                node_patches=[WorkflowNodePatch(node_id="render.model", target_axes=["lighting"])],
                edge_patches=[],
                region_patches=[],
                target_axes=["lighting"],
                preserve_axes=[],
            ),
        ]

        ranked = rank_workflow_graph_patch_candidates(candidates, session)
        self.assertEqual(ranked[0].candidate_id, "gpc-1")

        session.dissatisfaction_axes = ["lighting"]
        session.benchmark_comparison_summary.focus_axes = ["lighting"]
        session.benchmark_comparison_summary.preserve_axes = []
        session.preserve_constraints = []
        ranked = rank_workflow_graph_patch_candidates(candidates, session)
        self.assertEqual(ranked[0].candidate_id, "gpc-2")

    def test_ranking_is_stable_for_same_context(self):
        session = self._make_session()
        candidate = WorkflowGraphPatchCandidate(
            workflow_id="workflow-1",
            candidate_id="gpc-1",
            candidate_kind="balanced",
            node_patches=[WorkflowNodePatch(node_id="render.model")],
            edge_patches=[WorkflowEdgePatch(edge_id="e1")],
            region_patches=[WorkflowRegionPatch(region_id="r1")],
            target_axes=["style"],
            preserve_axes=["composition"],
        )

        first = rank_workflow_graph_patch_candidates([candidate], session)
        second = rank_workflow_graph_patch_candidates([candidate], session)

        self.assertEqual(first[0].metadata["pbo_score"], second[0].metadata["pbo_score"])
        self.assertEqual(first[0].metadata["pbo_rationale"], second[0].metadata["pbo_rationale"])


if __name__ == "__main__":
    unittest.main()
