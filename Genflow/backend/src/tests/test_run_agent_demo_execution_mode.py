import unittest

from app.agent.default_live_backend_client import DefaultLiveBackendClient
from app.agent.live_backend_errors import (
    LiveBackendNotConfiguredError,
    LiveBackendNotImplementedError,
)
from app.agent.live_execution_adapter import LiveExecutionAdapter
from app.agent.result_executor import ResultExecutor
from run_agent_demo import (
    build_execution_adapter,
    build_workflow_facade,
    build_workflow_backend_transport,
    build_live_backend_client,
    describe_cli_failure,
    resolve_execution_mode,
    resolve_live_backend_config,
)


class RunAgentDemoExecutionModeTest(unittest.TestCase):
    def test_resolve_execution_mode_defaults_to_mock(self):
        self.assertEqual(resolve_execution_mode({}), "mock")

    def test_resolve_execution_mode_accepts_live(self):
        self.assertEqual(resolve_execution_mode({"GENFLOW_EXECUTION_MODE": "live"}), "live")

    def test_resolve_live_backend_config_defaults_to_disabled(self):
        config = resolve_live_backend_config({})
        self.assertFalse(config.enabled)

    def test_build_execution_adapter_returns_mock_adapter(self):
        adapter = build_execution_adapter("mock")
        self.assertIsInstance(adapter, ResultExecutor)

    def test_build_execution_adapter_returns_live_adapter(self):
        adapter = build_execution_adapter("live")
        self.assertIsInstance(adapter, LiveExecutionAdapter)

    def test_invalid_execution_mode_raises_clear_error(self):
        with self.assertRaisesRegex(
            ValueError,
            "Invalid execution mode: invalid-mode. Allowed modes: live, mock.",
        ):
            build_execution_adapter("invalid-mode")

    def test_build_live_backend_client_requires_configured_substrate(self):
        config = resolve_live_backend_config({})
        with self.assertRaisesRegex(LiveBackendNotConfiguredError, "Live backend substrate is not configured."):
            build_live_backend_client(config)

    def test_build_live_backend_client_returns_default_client_for_live_shell(self):
        config = resolve_live_backend_config(
            {
                "GENFLOW_LIVE_BACKEND_ENABLED": "true",
                "GENFLOW_LIVE_BACKEND_KIND": "workflow_shell",
                "GENFLOW_LIVE_BACKEND_ENDPOINT": "memory://workflow-shell",
            }
        )
        client = build_live_backend_client(config)
        self.assertIsInstance(client, DefaultLiveBackendClient)

    def test_build_workflow_facade_returns_local_facade_for_local_profile(self):
        config = resolve_live_backend_config(
            {
                "GENFLOW_LIVE_BACKEND_ENABLED": "true",
                "GENFLOW_LIVE_BACKEND_KIND": "workflow_shell",
                "GENFLOW_LIVE_BACKEND_PROFILE": "local",
            }
        )

        facade = build_workflow_facade(config)

        self.assertIsNotNone(facade)
        self.assertEqual(type(facade).__name__, "LocalWorkflowFacade")

    def test_build_live_backend_client_uses_configured_local_profile_without_manual_facade(self):
        config = resolve_live_backend_config(
            {
                "GENFLOW_LIVE_BACKEND_ENABLED": "true",
                "GENFLOW_LIVE_BACKEND_KIND": "workflow_shell",
                "GENFLOW_LIVE_BACKEND_PROFILE": "local",
                "GENFLOW_LIVE_BACKEND_ENDPOINT": "memory://workflow-shell",
            }
        )

        client = build_live_backend_client(config)
        response = client.run_initial(
            type(
                "Req",
                (),
                {
                    "execution_kind": "initial",
                    "schema_snapshot": {"prompt": "portrait", "model": "sdxl-base"},
                    "workflow_payload": {},
                    "reference_info": {"reference_ids": [1, 2]},
                },
            )()
        )

        self.assertEqual(response.response_id, "local-initial")
        self.assertEqual(response.execution_kind, "initial")

    def test_build_workflow_backend_transport_without_local_profile_stays_unimplemented(self):
        config = resolve_live_backend_config(
            {
                "GENFLOW_LIVE_BACKEND_ENABLED": "true",
                "GENFLOW_LIVE_BACKEND_KIND": "workflow_shell",
                "GENFLOW_LIVE_BACKEND_ENDPOINT": "memory://workflow-shell",
            }
        )

        transport = build_workflow_backend_transport(config)

        with self.assertRaisesRegex(
            LiveBackendNotImplementedError,
            "Live backend substrate 'workflow_shell' is configured but no workflow facade is available.",
        ):
            transport.dispatch(
                "initial",
                type(
                    "Req",
                    (),
                    {
                        "execution_kind": "initial",
                        "schema_snapshot": {"prompt": "portrait"},
                        "workflow_payload": {},
                        "reference_info": {},
                    },
                )(),
            )

    def test_cli_failure_message_for_live_mode_not_implemented_is_short_and_actionable(self):
        message = describe_cli_failure(
            LiveBackendNotImplementedError(
                "Live backend substrate 'workflow_shell' is configured but dispatch is not implemented yet."
            )
        )
        self.assertIn("Live substrate is configured, but dispatch is not implemented yet.", message)
        self.assertIn("GENFLOW_EXECUTION_MODE=mock", message)
        self.assertNotIn("Traceback", message)

    def test_cli_failure_message_for_live_mode_not_configured_is_short_and_actionable(self):
        message = describe_cli_failure(LiveBackendNotConfiguredError("Live backend substrate is not configured."))
        self.assertIn("Live execution mode selected, but no live substrate is configured.", message)
        self.assertIn("GENFLOW_LIVE_BACKEND_KIND", message)
        self.assertNotIn("Traceback", message)


if __name__ == "__main__":
    unittest.main()
