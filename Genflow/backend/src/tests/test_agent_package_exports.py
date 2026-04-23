import os
import unittest


class AgentPackageExportsTest(unittest.TestCase):
    def test_package_exports_memory_symbols_without_api_key(self):
        old_value = os.environ.pop("GOOGLE_API_KEY", None)
        try:
            from app.agent import AgentMemoryService, AgentSessionState

            self.assertTrue(callable(AgentMemoryService))
            self.assertTrue(hasattr(AgentSessionState, "__dataclass_fields__"))
        finally:
            if old_value is not None:
                os.environ["GOOGLE_API_KEY"] = old_value

    def test_import_does_not_require_runtime_repositories(self):
        from app.agent import AgentMemoryService, AgentSessionState

        memory = AgentMemoryService()
        session = memory.create_session("package export test")

        self.assertEqual(session.original_intent, "package export test")
        self.assertIs(AgentSessionState, type(session))


if __name__ == "__main__":
    unittest.main()
