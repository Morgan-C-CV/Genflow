import unittest

from app.agent.memory import AgentMemoryService
from app.agent.runtime_models import CommittedPatch, PreviewProbe
from app.agent.workflow_scope_materializer import (
    materialize_editable_scopes,
    materialize_protected_scopes,
    materialize_workflow_scopes,
)


class WorkflowScopeMaterializerTest(unittest.TestCase):
    def _make_session(self):
        return AgentMemoryService().create_session("make a portrait")

    def test_patch_priority_is_higher_than_probe_priority(self):
        session = self._make_session()
        session.accepted_patch = CommittedPatch(
            patch_id="patch-1",
            target_fields=["style", "model"],
        )
        session.selected_probe = PreviewProbe(
            probe_id="probe-1",
            target_axes=["prompt"],
            preserve_axes=["composition"],
        )
        session.current_schema_raw = "schema"

        editable = materialize_editable_scopes(session)

        self.assertEqual(editable[0].node_ids, ["style", "model"])

    def test_probe_priority_is_higher_than_schema_fallback(self):
        session = self._make_session()
        session.selected_probe = PreviewProbe(
            probe_id="probe-1",
            target_axes=["style"],
            preserve_axes=["composition"],
        )
        session.current_schema_raw = "schema"

        editable = materialize_editable_scopes(session)

        self.assertEqual(editable[0].node_ids, ["style"])

    def test_schema_fallback_is_used_when_no_patch_or_probe_exists(self):
        session = self._make_session()
        session.current_schema_raw = "schema"

        editable = materialize_editable_scopes(session)

        self.assertEqual(editable[0].node_ids, ["prompt", "model", "style", "lora"])

    def test_protected_scope_prefers_preserve_constraints(self):
        session = self._make_session()
        session.preserve_constraints = ["Keep the composition"]
        session.selected_probe = PreviewProbe(
            probe_id="probe-1",
            target_axes=["style"],
            preserve_axes=["composition"],
        )

        protected = materialize_protected_scopes(session)

        self.assertEqual(protected[0].node_ids, ["Keep the composition"])

    def test_materialize_workflow_scopes_is_stable_for_same_input(self):
        session = self._make_session()
        session.current_schema_raw = "schema"
        session.preserve_constraints = ["Keep the composition"]

        first = materialize_workflow_scopes(session)
        second = materialize_workflow_scopes(session)

        self.assertEqual(first[0][0].node_ids, second[0][0].node_ids)
        self.assertEqual(first[1][0].node_ids, second[1][0].node_ids)


if __name__ == "__main__":
    unittest.main()
