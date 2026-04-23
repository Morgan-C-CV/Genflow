import unittest

from app.agent.runtime_models import NormalizedSchema, ResultPayload, ResultSummary
from run_agent_demo import build_live_runtime_components_from_env, run_local_live_smoke


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


if __name__ == "__main__":
    unittest.main()
