class LiveExecutionError(Exception):
    """Stable base error for live execution adapter/client boundaries."""


class LiveBackendUnavailableError(LiveExecutionError):
    """Raised when the configured live backend transport is missing or unavailable."""


class LiveBackendResponseError(LiveExecutionError):
    """Raised when backend response data is malformed or inconsistent."""


class LiveBackendDispatchError(LiveExecutionError):
    """Raised when request dispatch fails for non-availability reasons."""


class LiveBackendNotConfiguredError(LiveBackendUnavailableError):
    """Raised when live execution is selected but no substrate config is provided."""


class LiveBackendNotImplementedError(LiveExecutionError):
    """Raised when a substrate shell is selected but dispatch is not implemented yet."""
