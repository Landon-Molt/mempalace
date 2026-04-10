"""Exceptions raised by the providers subsystem."""

from __future__ import annotations


class ProviderError(Exception):
    """Base class for provider-related errors."""


class ProviderUnavailableError(ProviderError):
    """Raised when a provider endpoint is unreachable or unhealthy.

    Callers that want a fallback can catch this and try the next provider.
    """


class DimensionMismatchError(ProviderError):
    """Raised when a palace's stored embedding dimension doesn't match the
    currently configured embedder.

    This prevents silent vector-space corruption: if the sidecar meta says
    the palace was indexed with a 384-dim model and the current config
    resolves to a 1024-dim model, every query would be computing a vector in
    the wrong space and retrieval would be garbage. Better to fail loudly
    and direct the user to ``mempalace reembed``.
    """

    def __init__(
        self,
        palace_path: str,
        stored_dim: int,
        stored_model: str,
        current_dim: int,
        current_model: str,
    ):
        self.palace_path = palace_path
        self.stored_dim = stored_dim
        self.stored_model = stored_model
        self.current_dim = current_dim
        self.current_model = current_model
        super().__init__(
            f"\n"
            f"  Palace at {palace_path} was indexed with:\n"
            f"    model = {stored_model!r}\n"
            f"    dim   = {stored_dim}\n"
            f"  Current config resolves to:\n"
            f"    model = {current_model!r}\n"
            f"    dim   = {current_dim}\n"
            f"\n"
            f"  Vector spaces are incompatible. Fix with either:\n"
            f"    1. mempalace reembed --palace {palace_path}\n"
            f"       (re-embeds all drawers with the new model)\n"
            f"    2. Revert config to the previous embedding model\n"
            f"    3. Create a fresh palace at a different path\n"
        )
