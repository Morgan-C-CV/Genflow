import unittest

from app.agent.memory import AgentMemoryService


class MemoryCompatibilityTest(unittest.TestCase):
    def test_create_session_keeps_existing_constructor_flow(self):
        memory = AgentMemoryService()
        session = memory.create_session("test intent")

        self.assertEqual(session.original_intent, "test intent")
        self.assertEqual(session.clarified_intent, "test intent")
        self.assertEqual(session.current_schema.prompt, "")
        self.assertEqual(session.current_schema_raw, "")
        self.assertEqual(session.selected_reference_bundle, {})
        self.assertEqual(session.feedback_history, [])
        self.assertEqual(session.patch_history, [])
        self.assertFalse(session.continue_recommended)

    def test_save_and_get_session_preserve_new_default_fields(self):
        memory = AgentMemoryService()
        session = memory.create_session("another intent")
        session.feedback_history.append("too dark")
        session.selected_reference_ids.extend([1, 2, 3])
        memory.save_session(session)

        loaded = memory.get_session(session.session_id)
        self.assertEqual(loaded.feedback_history, ["too dark"])
        self.assertEqual(loaded.selected_reference_ids, [1, 2, 3])
        self.assertEqual(loaded.current_result_summary.summary_text, "")


if __name__ == "__main__":
    unittest.main()
