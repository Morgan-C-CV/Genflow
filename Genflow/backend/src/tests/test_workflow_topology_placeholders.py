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
        self.assertEqual(session.workflow_graph_placeholder.entry_node_ids, ["reference.bundle", "intent.prompt"])
        self.assertEqual(session.workflow_graph_placeholder.exit_node_ids, ["result.output"])
        self.assertEqual(session.workflow_graph_placeholder.metadata["graph_regions"], ["initial_region"])
        self.assertTrue(any(node.role == "input" for node in session.workflow_graph_placeholder.node_refs))
        self.assertTrue(any(node.role == "output" for node in session.workflow_graph_placeholder.node_refs))

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
        session = service.commit_patch(session.session_id)

        self.assertEqual(session.workflow_topology_hints["selected_probe_id"], "p_002")
        self.assertEqual(session.workflow_topology_hints["accepted_patch_id"], "cp_p_002")
        self.assertEqual(session.workflow_graph_placeholder.metadata["region_label"], "repair_region")
        self.assertTrue(
            any(node.node_kind == "surrogate_probe" for node in session.workflow_graph_placeholder.node_refs)
        )
        self.assertTrue(
            any(node.node_kind == "surrogate_patch" for node in session.workflow_graph_placeholder.node_refs)
        )
        self.assertTrue(
            any(node.role == "repair_probe" for node in session.workflow_graph_placeholder.node_refs)
        )
        self.assertTrue(
            any(node.role == "repair_patch" for node in session.workflow_graph_placeholder.node_refs)
        )
        self.assertEqual(session.workflow_graph_placeholder.metadata["graph_regions"], ["repair_region"])
        self.assertEqual(session.workflow_graph_placeholder.topology_slices[0].slice_kind, "repair_region")
        self.assertEqual(session.workflow_graph_placeholder.topology_slices[0].entry_node_ids, ["reference.bundle", "intent.prompt"])
        self.assertEqual(session.workflow_graph_placeholder.topology_slices[0].exit_node_ids, ["patch.cp_p_002", "result.output"])

    def test_topology_sync_is_stable_for_same_input(self):
        service = self._build_service(iter(["result-1", "result-2"]))
        session = service.start_episode("make a portrait")
        session = service.generate_initial_candidates(session.session_id)
        session = service.select_initial_reference(session.session_id, 7)
        session = service.generate_initial_schema(session.session_id)
        session = service.produce_initial_result(session.session_id)

        first_roles = [(node.node_id, node.role, list(node.upstream_ids), list(node.downstream_ids)) for node in session.workflow_graph_placeholder.node_refs]
        first_entry = list(session.workflow_graph_placeholder.entry_node_ids)
        first_exit = list(session.workflow_graph_placeholder.exit_node_ids)

        session = service.produce_initial_result(session.session_id)

        second_roles = [(node.node_id, node.role, list(node.upstream_ids), list(node.downstream_ids)) for node in session.workflow_graph_placeholder.node_refs]
        self.assertEqual(first_roles, second_roles)
        self.assertEqual(first_entry, session.workflow_graph_placeholder.entry_node_ids)
        self.assertEqual(first_exit, session.workflow_graph_placeholder.exit_node_ids)


if __name__ == "__main__":
    unittest.main()
