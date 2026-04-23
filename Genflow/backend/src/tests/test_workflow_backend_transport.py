import unittest

from app.agent.live_backend_config import LiveBackendConfig
from app.agent.live_backend_errors import (
    LiveBackendNotConfiguredError,
    LiveBackendNotImplementedError,
    LiveBackendUnavailableError,
)
from app.agent.live_execution_models import ExecutionRequest
from app.agent.local_workflow_facade import LocalWorkflowFacade
from app.agent.workflow_backend_transport import WorkflowBackendTransport


class WorkflowBackendTransportTest(unittest.TestCase):
    def test_transport_raises_not_configured_when_disabled(self):
        transport = WorkflowBackendTransport(LiveBackendConfig(enabled=False))

        with self.assertRaisesRegex(LiveBackendNotConfiguredError, "Live backend substrate is not configured."):
            transport.dispatch("initial", ExecutionRequest(execution_kind="initial"))

    def test_transport_raises_not_configured_when_kind_missing(self):
        transport = WorkflowBackendTransport(LiveBackendConfig(enabled=True, backend_kind=""))

        with self.assertRaisesRegex(LiveBackendNotConfiguredError, "Live backend kind is missing from configuration."):
            transport.dispatch("initial", ExecutionRequest(execution_kind="initial"))

    def test_transport_rejects_unsupported_kind(self):
        transport = WorkflowBackendTransport(LiveBackendConfig(enabled=True, backend_kind="unsupported"))

        with self.assertRaisesRegex(LiveBackendUnavailableError, "Unsupported live backend kind: unsupported."):
            transport.dispatch("preview", ExecutionRequest(execution_kind="preview"))

    def test_transport_raises_not_implemented_for_supported_shell_kind(self):
        transport = WorkflowBackendTransport(
            LiveBackendConfig(enabled=True, backend_kind="workflow_shell", endpoint="memory://shell")
        )

        with self.assertRaisesRegex(
            LiveBackendNotImplementedError,
            "Live backend substrate 'workflow_shell' is configured but no workflow facade is available.",
        ):
            transport.dispatch("commit", ExecutionRequest(execution_kind="commit"))

    def test_transport_dispatches_through_local_workflow_facade(self):
        transport = WorkflowBackendTransport(
            LiveBackendConfig(enabled=True, backend_kind="workflow_shell", endpoint="memory://shell"),
            workflow_facade=LocalWorkflowFacade(),
        )

        response = transport.dispatch(
            "initial",
            ExecutionRequest(
                execution_kind="initial",
                schema_snapshot={"prompt": "portrait", "model": "sdxl-base"},
                reference_info={"reference_ids": [101]},
            ),
        )

        self.assertEqual(response.execution_kind, "initial")
        self.assertEqual(response.response_id, "local-initial")
        self.assertEqual(response.output_payload["reference_count"], 1)


if __name__ == "__main__":
    unittest.main()
