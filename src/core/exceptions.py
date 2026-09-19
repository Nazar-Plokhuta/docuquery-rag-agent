"""Domain exception hierarchy for the DocuQuery RAG Agent.

All application-specific errors derive from ``AppException`` so that the
centralised FastAPI exception handler can intercept every domain fault and
translate it to a deterministic HTTP response without leaking internal state.
"""

from __future__ import annotations


class AppException(Exception):
    """Base class for all domain-level exceptions raised within this service.

    Args:
        message: Human-readable description of the failure, safe to surface in
            API error responses.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message: str = message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r})"


class ConfigurationError(AppException):
    """Raised when required configuration values are missing or invalid.

    Typical triggers: missing API key, unresolvable file paths supplied via
    environment variables, or an unsupported enum value in settings.
    """


class ResourceNotFoundError(AppException):
    """Raised when a requested domain entity or document cannot be located.

    Maps to HTTP 404 at the presentation layer.
    """


class StorageError(AppException):
    """Raised when a persistence operation fails (ChromaDB or SQLite).

    Wraps lower-level driver exceptions to prevent storage internals from
    leaking into API responses or logs at unexpected severity levels.
    """


class DocumentParsingError(AppException):
    """Raised when uploaded bytes cannot be parsed into extractable text.

    Typical triggers: corrupted PDFs, encrypted PDFs, or empty binary payloads.
    Maps to HTTP 422 at the presentation layer.
    """


class UnsupportedFileTypeError(AppException):
    """Raised when no loader is registered for the uploaded file extension.

    Maps to HTTP 415 at the presentation layer when raised from the domain layer.
    """
