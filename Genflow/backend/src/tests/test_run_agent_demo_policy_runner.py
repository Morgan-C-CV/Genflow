import unittest
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from run_agent_demo import (
    format_policy_runner_demo_summary,
    main,
    run_demo_mode,
)


class RunAgentDemoPolicyRunnerTest(unittest.TestCase):
    def _fake_policy_demo_payload(self):
        final_session = SimpleNamespace(
            session_id="session-policy-demo",
            selected_probe=SimpleNamespace(probe_id="p_002"),
            accepted_patch=SimpleNamespace(patch_id="cp_p_002"),
            latest_verifier_result=SimpleNamespace(summary="verifier accepts current direction"),
        )
        run_result = SimpleNamespace(
            steps=[
                SimpleNamespace(action="build_hypotheses", rationale=["repair_hypotheses_missing"], session_id="session-policy-demo"),
                SimpleNamespace(action="generate_probes", rationale=["preview_probe_candidates_missing"], session_id="session-policy-demo"),
                SimpleNamespace(action="preview_selected_probe", rationale=["selected_probe_needs_preview=p_002"], session_id="session-policy-demo"),
                SimpleNamespace(action="commit_selected_patch", rationale=["selected_probe_ready_for_commit=p_002"], session_id="session-policy-demo"),
                SimpleNamespace(action="execute_patch", rationale=["accepted_patch_needs_execution=cp_p_002"], session_id="session-policy-demo"),
                SimpleNamespace(action="verify_latest_result", rationale=["accepted_patch_needs_verification=cp_p_002"], session_id="session-policy-demo"),
            ],
            final_session=final_session,
            stopped=False,
            stop_reason="verifier_accepts_current_direction",
        )
        return {
            "demo_mode": "policy_runner_demo",
            "execution_mode": "mock",
            "session_id": "session-policy-demo",
            "max_steps": 6,
            "run_result": run_result,
        }

    def test_run_demo_mode_policy_runner_returns_stable_summary(self):
        payload = self._fake_policy_demo_payload()

        with patch("run_agent_demo.run_policy_runner_demo", return_value=payload):
            result = run_demo_mode(env={"GENFLOW_DEMO_MODE": "policy_runner_demo"})

        self.assertEqual(result["demo_mode"], "policy_runner_demo")
        self.assertIn("actions_taken: build_hypotheses, generate_probes, preview_selected_probe", result["summary_text"])
        self.assertIn("accepted_patch: cp_p_002", result["summary_text"])

    def test_format_policy_runner_demo_summary_is_stable(self):
        summary_text = format_policy_runner_demo_summary(self._fake_policy_demo_payload())

        self.assertIn("GenFlow Policy Runner Demo", summary_text)
        self.assertIn("execution_mode: mock", summary_text)
        self.assertIn("session_id: session-policy-demo", summary_text)
        self.assertIn("stop_reason: verifier_accepts_current_direction", summary_text)

    def test_main_uses_policy_runner_demo_without_interactive_input(self):
        stdout = StringIO()
        payload = self._fake_policy_demo_payload()
        env = {"GENFLOW_DEMO_MODE": "policy_runner_demo"}

        with patch("run_agent_demo.os.environ", env), patch("run_agent_demo.run_policy_runner_demo", return_value=payload), patch(
            "sys.stdout",
            stdout,
        ), patch(
            "builtins.input",
            side_effect=AssertionError("interactive input should not be called in policy_runner_demo mode"),
        ):
            main()

        output = stdout.getvalue()
        self.assertIn("GenFlow Policy Runner Demo", output)
        self.assertIn("actions_taken: build_hypotheses, generate_probes, preview_selected_probe", output)


if __name__ == "__main__":
    unittest.main()
