import unittest
from io import StringIO
from unittest.mock import patch

from app.agent.runtime_models import NormalizedSchema, ResultPayload, ResultSummary
from run_agent_demo import (
    build_live_runtime_components_from_env,
    format_local_live_smoke_summary,
    main,
    run_demo_mode,
    run_local_live_smoke,
)


class RunAgentDemoLocalLiveSmokeTest(unittest.TestCase):
    def setUp(self):
        self.env = {
            "GENFLOW_EXECUTION_MODE": "live",
            "GENFLOW_LIVE_BACKEND_ENABLED": "true",
            "GENFLOW_LIVE_BACKEND_KIND": "workflow_shell",
            "GENFLOW_LIVE_BACKEND_PROFILE": "local",
            "GENFLOW_LIVE_BACKEND_ENDPOINT": "memory://workflow-shell",
        }

    def test_build_live_runtime_components_from_env_returns_live_stack(self):
        components = build_live_runtime_components_from_env(env=self.env)

        self.assertEqual(components["execution_mode"], "live")
        self.assertEqual(components["live_backend_config"].backend_kind, "workflow_shell")
        self.assertEqual(components["live_backend_config"].workflow_profile, "local")
        self.assertEqual(type(components["backend_client"]).__name__, "DefaultLiveBackendClient")
        self.assertEqual(type(components["execution_adapter"]).__name__, "LiveExecutionAdapter")

    def test_run_local_live_smoke_produces_structured_initial_result(self):
        smoke = run_local_live_smoke(env=self.env)

        self.assertIsInstance(smoke["schema"], NormalizedSchema)
        self.assertIsInstance(smoke["result_payload"], ResultPayload)
        self.assertIsInstance(smoke["result_summary"], ResultSummary)
        self.assertEqual(smoke["result_payload"].result_type, "live_initial_result")
        self.assertEqual(smoke["result_payload"].content["reference_count"], 2)
        self.assertIn("Local workflow facade produced initial result", smoke["result_summary"].summary_text)

    def test_run_demo_mode_local_live_smoke_returns_stable_summary(self):
        result = run_demo_mode(env={**self.env, "GENFLOW_DEMO_MODE": "local_live_smoke"})

        self.assertEqual(result["demo_mode"], "local_live_smoke")
        self.assertIn("execution_mode: live", result["summary_text"])
        self.assertIn("backend_kind: workflow_shell", result["summary_text"])
        self.assertIn("workflow_profile: local", result["summary_text"])
        self.assertIn("result_type: live_initial_result", result["summary_text"])

    def test_format_local_live_smoke_summary_is_stable(self):
        smoke = run_local_live_smoke(env=self.env)

        summary_text = format_local_live_smoke_summary(smoke)

        self.assertIn("GenFlow Local Live Smoke", summary_text)
        self.assertIn("execution_mode: live", summary_text)
        self.assertIn("result_id: local-initial", summary_text)

    def test_main_uses_local_live_smoke_mode_without_interactive_input(self):
        stdout = StringIO()
        env = {**self.env, "GENFLOW_DEMO_MODE": "local_live_smoke"}

        with patch("run_agent_demo.os.environ", env), patch("sys.stdout", stdout), patch(
            "builtins.input",
            side_effect=AssertionError("interactive input should not be called in local_live_smoke mode"),
        ):
            main()

        output = stdout.getvalue()
        self.assertIn("GenFlow Local Live Smoke", output)
        self.assertIn("result_type: live_initial_result", output)

    def test_main_local_live_smoke_uses_stable_error_output_when_config_missing(self):
        stdout = StringIO()
        env = {
            "GENFLOW_DEMO_MODE": "local_live_smoke",
            "GENFLOW_EXECUTION_MODE": "live",
        }

        with patch("run_agent_demo.os.environ", env), patch("sys.stdout", stdout):
            with self.assertRaises(SystemExit) as exc:
                main()

        self.assertEqual(exc.exception.code, 1)
        output = stdout.getvalue()
        self.assertIn("Live execution mode selected, but no live substrate is configured.", output)


if __name__ == "__main__":
    unittest.main()
