import unittest

from app.agent.live_backend_client import LiveBackendClient
from app.agent.live_execution_models import (
    CommitExecutionRequest,
    ExecutionRequest,
    ExecutionResponse,
    PreviewExecutionRequest,
)


class FakeLiveBackendClient:
    def run_initial(self, request: ExecutionRequest) -> ExecutionResponse:
        return ExecutionResponse(response_id="initial")

    def run_preview(self, request: PreviewExecutionRequest) -> ExecutionResponse:
        return ExecutionResponse(response_id="preview")

    def run_commit(self, request: CommitExecutionRequest) -> ExecutionResponse:
        return ExecutionResponse(response_id="commit")


class LiveBackendClientContractTest(unittest.TestCase):
    def test_fake_client_satisfies_protocol(self):
        client = FakeLiveBackendClient()
        self.assertIsInstance(client, LiveBackendClient)


if __name__ == "__main__":
    unittest.main()
