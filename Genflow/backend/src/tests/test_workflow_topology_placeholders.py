import unittest

from app.agent.memory import AgentMemoryService
from app.agent.result_executor import ResultExecutor
from app.agent.runtime_service import AgentRuntimeService
from tests.test_runtime_service import (
    FakeFeedbackParser,
    FakeHypothesisBuilder,
    FakeOrchestrationService,
    FakePatchPlanner,
    FakeProbeGenerator,
    FakeSearchService,
)


class WorkflowTopologyPlaceholdersTest(unittest.TestCase):
    def _build_service(self, ids):
        memory = AgentMemoryService()
        service = AgentRuntimeService(
            memory_service=memory,
            orchestration_service=FakeOrchestrationService(memory),
            search_service=FakeSearchService(),
            execution_adapter=ResultExecutor(id_factory=lambda: next(ids)),
            feedback_parser=FakeFeedbackParser(),
            hypothesis_builder=FakeHypothesisBuilder(),
            probe_generator=FakeProbeGenerator(),
            patch_planner=FakePatchPlanner(),
        )
        return service

    def test_topology_placeholder_defaults_are_safe(self):
        session = AgentMemoryService().create_session("intent")

        self.assertEqual(session.workflow_graph_placeholder.graph_id, "")
        self.assertEqual(session.workflow_graph_placeholder.node_refs, [])
        self.assertEqual(session.workflow_topology_hints, {})

    def test_sync_populates_topology_placeholder_after_initial_result(self):
        service = self._build_service(iter(["result-1"]))
        session = service.start_episode("make a portrait")
        session = service.generate_initial_candidates(session.session_id)
        session = service.select_initial_reference(session.session_id, 7)
        session = service.generate_initial_schema(session.session_id)
        session = service.produce_initial_result(session.session_id)

        self.assertTrue(session.workflow_graph_placeholder.graph_id)
        self.assertEqual(session.workflow_graph_placeholder.graph_kind, "surrogate_topology")
        self.assertTrue(session.workflow_graph_placeholder.node_refs)
        self.assertTrue(session.workflow_graph_placeholder.topology_slices)
        self.assertEqual(session.workflow_state.workflow_graph_placeholder.graph_id, session.workflow_graph_placeholder.graph_id)
        self.assertEqual(session.workflow_topology_hints["region_label"], "initial_region")

    def test_sync_updates_topology_hints_after_probe_and_patch(self):
        service = self._build_service(iter(["initial-1", "preview-1", "commit-1"]))
        session = service.start_episode("make a portrait")
        session = service.generate_initial_candidates(session.session_id)
        session = service.select_initial_reference(session.session_id, 7)
        session = service.generate_initial_schema(session.session_id)
        session = service.produce_initial_result(session.session_id)
        session = service.submit_feedback(session.session_id, "Keep the composition, but improve style.")
        session = service.build_repair_hypotheses(session.session_id)
        session = service.generate_local_probes(session.session_id)
        session = service.select_probe(session.session_id, "p_001")
        session = service.commit_patch(session.session_id)

        self.assertEqual(session.workflow_topology_hints["selected_probe_id"], "p_001")
        self.assertEqual(session.workflow_topology_hints["accepted_patch_id"], "cp_p_001")
        self.assertEqual(session.workflow_graph_placeholder.metadata["region_label"], "repair_region")
        self.assertTrue(
            any(node.node_kind == "surrogate_probe" for node in session.workflow_graph_placeholder.node_refs)
        )
        self.assertTrue(
            any(node.node_kind == "surrogate_patch" for node in session.workflow_graph_placeholder.node_refs)
        )


if __name__ == "__main__":
    unittest.main()
