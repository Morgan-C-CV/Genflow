class LiveExecutionError(Exception):
    """Stable base error for live execution adapter/client boundaries."""


class LiveBackendUnavailableError(LiveExecutionError):
    """Raised when the configured live backend transport is missing or unavailable."""


class LiveBackendResponseError(LiveExecutionError):
    """Raised when backend response data is malformed or inconsistent."""


class LiveBackendDispatchError(LiveExecutionError):
    """Raised when request dispatch fails for non-availability reasons."""
