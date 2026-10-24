"""Unit tests for the domain-exception → HTTP status table in ``src.main``."""

from __future__ import annotations

import pytest

from src.core.exceptions import (
    AppException,
    ConfigurationError,
    DocumentParsingError,
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
    ResourceNotFoundError,
    StorageError,
    UnsupportedFileTypeError,
)
from src.main import _exception_to_status


@pytest.mark.parametrize(
    ("exc", "status"),
    [
        (ResourceNotFoundError("x"), 404),
        (DocumentParsingError("x"), 422),
        (UnsupportedFileTypeError("x"), 415),
        (StorageError("x"), 503),
        (LLMProviderError("x"), 502),
        (LLMAuthenticationError("x"), 502),
        (LLMRateLimitError("x"), 429),
        (LLMUnavailableError("x"), 503),
        (LLMTimeoutError("x"), 504),
        (ConfigurationError("x"), 500),
        (AppException("x"), 500),
    ],
)
def test_domain_exception_maps_to_status(exc: AppException, status: int) -> None:
    assert _exception_to_status(exc) == status


def test_unlisted_subclass_inherits_closest_ancestor_status() -> None:
    """A new subclass must not silently degrade to 500 because the table lacks it."""

    class QuotaExhaustedError(LLMRateLimitError):
        pass

    class ScannedPdfError(DocumentParsingError):
        pass

    assert _exception_to_status(QuotaExhaustedError("x")) == 429
    assert _exception_to_status(ScannedPdfError("x")) == 422
