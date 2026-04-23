import unittest

from app.agent.default_live_backend_client import DefaultLiveBackendClient
from app.agent.live_backend_errors import (
    LiveBackendDispatchError,
    LiveBackendResponseError,
    LiveBackendUnavailableError,
)
from app.agent.live_execution_models import ExecutionRequest


class FakeTransport:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def dispatch(self, execution_kind, request):
        self.calls.append((execution_kind, request))
        return self.response


class DefaultLiveBackendClientTest(unittest.TestCase):
    def test_client_normalizes_successful_response(self):
        transport = FakeTransport(
            {
                "response_id": "resp-1",
                "execution_kind": "initial",
                "output_payload": {"image_id": "img-1"},
                "summary_text": "ok",
                "changed_axes": ("style",),
                "preserved_axes": "composition",
                "backend_artifacts": {"artifact_uri": "memory://1"},
                "backend_metadata": {"backend": "fake"},
                "comparison_notes": ("note-1",),
            }
        )
        client = DefaultLiveBackendClient(transport=transport)

        response = client.run_initial(ExecutionRequest(execution_kind="initial"))

        self.assertEqual(transport.calls[0][0], "initial")
        self.assertEqual(response.response_id, "resp-1")
        self.assertEqual(response.changed_axes, ["style"])
        self.assertEqual(response.preserved_axes, ["composition"])
        self.assertEqual(response.comparison_notes, ["note-1"])

    def test_client_raises_unavailable_for_missing_transport(self):
        client = DefaultLiveBackendClient()

        with self.assertRaisesRegex(LiveBackendUnavailableError, "Live backend transport is not configured."):
            client.run_initial(ExecutionRequest(execution_kind="initial"))

    def test_client_translates_transport_connection_errors(self):
        def failing_transport(execution_kind, request):
            raise ConnectionError("offline")

        client = DefaultLiveBackendClient(transport=failing_transport)

        with self.assertRaisesRegex(
            LiveBackendUnavailableError,
            "Live backend transport is unavailable for preview execution.",
        ):
            client.run_preview(ExecutionRequest(execution_kind="preview"))

    def test_client_translates_generic_dispatch_errors(self):
        def failing_transport(execution_kind, request):
            raise RuntimeError("boom")

        client = DefaultLiveBackendClient(transport=failing_transport)

        with self.assertRaisesRegex(
            LiveBackendDispatchError,
            "Live backend dispatch failed for commit execution.",
        ):
            client.run_commit(ExecutionRequest(execution_kind="commit"))

    def test_client_rejects_bad_response_shape(self):
        transport = FakeTransport(
            {
                "response_id": "",
                "execution_kind": "initial",
                "output_payload": [],
            }
        )
        client = DefaultLiveBackendClient(transport=transport)

        with self.assertRaisesRegex(LiveBackendResponseError, "Backend response is missing response_id."):
            client.run_initial(ExecutionRequest(execution_kind="initial"))


if __name__ == "__main__":
    unittest.main()
