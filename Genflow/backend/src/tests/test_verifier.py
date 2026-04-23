import unittest

from app.agent.runtime_models import CommittedPatch, PreviewProbe, ResultSummary
from app.agent.verifier import Verifier


class VerifierTest(unittest.TestCase):
    def test_verifier_returns_structured_decision(self):
        verifier = Verifier()
        previous = ResultSummary(summary_text="before", changed_axes=["initial_generation"])
        updated = ResultSummary(
            summary_text="after",
            changed_axes=["style"],
            preserved_axes=["composition"],
        )
        probe = PreviewProbe(
            probe_id="p_001",
            target_axes=["style"],
            preserve_axes=["composition"],
        )
        patch = CommittedPatch(
            patch_id="cp_p_001",
            target_fields=["style", "model"],
            target_axes=["style"],
            preserve_axes=["composition"],
            rationale="adjust style",
        )

        result = verifier.verify(previous, updated, probe, patch, ["Keep the composition"])

        self.assertIsInstance(result.improved, bool)
        self.assertIsInstance(result.continue_recommended, bool)
        self.assertIsInstance(result.confidence, float)
        self.assertIsInstance(result.regression_notes, list)
        self.assertTrue(result.summary)


if __name__ == "__main__":
    unittest.main()
