import unittest

from app.agent.live_execution_adapter import LiveExecutionAdapter
from app.agent.result_executor import ResultExecutor
from run_agent_demo import build_execution_adapter, describe_cli_failure, resolve_execution_mode


class RunAgentDemoExecutionModeTest(unittest.TestCase):
    def test_resolve_execution_mode_defaults_to_mock(self):
        self.assertEqual(resolve_execution_mode({}), "mock")

    def test_resolve_execution_mode_accepts_live(self):
        self.assertEqual(resolve_execution_mode({"GENFLOW_EXECUTION_MODE": "live"}), "live")

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

    def test_cli_failure_message_for_live_mode_not_implemented_is_short_and_actionable(self):
        message = describe_cli_failure(NotImplementedError("Live execution adapter is not wired yet."))
        self.assertIn("Live execution mode is reserved but not wired yet.", message)
        self.assertIn("GENFLOW_EXECUTION_MODE=mock", message)
        self.assertNotIn("Traceback", message)


if __name__ == "__main__":
    unittest.main()
