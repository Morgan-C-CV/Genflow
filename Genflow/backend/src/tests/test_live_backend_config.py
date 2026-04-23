import unittest

from app.agent.live_backend_config import LiveBackendConfig, resolve_live_backend_config


class LiveBackendConfigTest(unittest.TestCase):
    def test_resolve_live_backend_config_defaults_to_disabled(self):
        config = resolve_live_backend_config({})

        self.assertIsInstance(config, LiveBackendConfig)
        self.assertFalse(config.enabled)
        self.assertEqual(config.backend_kind, "")
        self.assertEqual(config.timeout_seconds, 30.0)

    def test_resolve_live_backend_config_from_env(self):
        config = resolve_live_backend_config(
            {
                "GENFLOW_LIVE_BACKEND_ENABLED": "true",
                "GENFLOW_LIVE_BACKEND_KIND": "workflow_shell",
                "GENFLOW_LIVE_BACKEND_ENDPOINT": "memory://workflow-shell",
                "GENFLOW_LIVE_BACKEND_TIMEOUT_SECONDS": "12.5",
                "GENFLOW_LIVE_BACKEND_NAMESPACE": "demo",
                "GENFLOW_LIVE_BACKEND_PROFILE": "local",
            }
        )

        self.assertTrue(config.enabled)
        self.assertEqual(config.backend_kind, "workflow_shell")
        self.assertEqual(config.workflow_profile, "local")
        self.assertEqual(config.endpoint, "memory://workflow-shell")
        self.assertEqual(config.timeout_seconds, 12.5)
        self.assertEqual(config.metadata["namespace"], "demo")
        self.assertEqual(config.metadata["workflow_profile"], "local")


if __name__ == "__main__":
    unittest.main()
