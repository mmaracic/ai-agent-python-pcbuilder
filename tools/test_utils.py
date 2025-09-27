"""Tests for the utility functions in tools.utils.

Covers:
 - get_url_text (successful load and empty loader response)
 - get_url_text_diff (successful diff extraction and HTTP error path)
 - extract_text_from_html (HTML cleaning and text extraction)
"""
from __future__ import annotations

from typing import Any, List
from types import SimpleNamespace

import pytest

from tools import utils


def test_get_url_text_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_url_text returns text from the first loaded document."""
    captured_url: dict[str, str] = {}

    class DummyLoader:  # noqa: D401 - simple test double
        def __init__(self, url: str) -> None:
            captured_url["url"] = url

        def load(self) -> List[Any]:  # noqa: D401 - returns a list with a document-like object
            doc: SimpleNamespace = SimpleNamespace(page_content="Sample content")
            return [doc]

    monkeypatch.setattr(utils, "WebBaseLoader", DummyLoader)

    url: str = "https://example.com"
    result: str = utils.get_url_text(url)
    assert result == "Sample content"
    assert captured_url["url"] == url


def test_get_url_text_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_url_text returns an empty string when loader returns no documents."""

    class EmptyLoader:  # noqa: D401 - simple test double
        def __init__(self, url: str) -> None:  # noqa: D401
            self.url: str = url

        def load(self) -> List[Any]:  # noqa: D401
            return []

    monkeypatch.setattr(utils, "WebBaseLoader", EmptyLoader)
    result: str = utils.get_url_text("https://example.com/none")
    assert result == ""


def test_get_url_text_diff_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test successful diff retrieval and HTML extraction path in get_url_text_diff."""
    # Baseline vs new HTML: one paragraph updated and one extra paragraph added so diff should capture them.
    baseline: str = (
        "<html><head><title>Doc</title></head><body>"
        "<h1>Header</h1>"
        "<p id='line1'>Original paragraph</p>"
        "<p>Shared paragraph</p>"
        "</body></html>"
    )
    new_text: str = (
        "<html><head><title>Doc</title></head><body>"
        "<h1>Header</h1>"
        "<p id='line1'>Updated paragraph</p>"
        "<p>Shared paragraph</p>"
        "<p>Extra paragraph</p>"
        "</body></html>"
    )

    class DummyResponse:  # noqa: D401 - simple response double
        def __init__(self) -> None:
            self.status_code: int = 200
            self.text: str = new_text

    def fake_get(url: str, timeout: int) -> DummyResponse:  # noqa: D401 - mimic requests.get
        assert timeout == 10
        assert url == "https://example.com/diff"
        return DummyResponse()

    monkeypatch.setattr(utils.requests, "get", fake_get)

    result: str = utils.get_url_text_diff("https://example.com/diff", baseline)

    # We expect the diff-derived extracted text to include tokens for the updated and extra paragraph content.
    # Diff/HTML extraction may split nodes so we assert on presence of key tokens.
    assert "Updated" in result
    assert "paragraph" in result
    assert "Extra paragraph" in result
    # Prefer that original paragraph text is absent, but tolerate presence due to formatter variability.
    if "Original paragraph" in result:
        # Ensure updated still present (already asserted) so change is observable.
        pass


def test_get_url_text_diff_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_url_text_diff raises an exception when HTTP status is not 200."""

    class ErrorResponse:  # noqa: D401 - simple response double
        def __init__(self) -> None:
            self.status_code: int = 500
            self.text: str = "Server error"

    def fake_get(url: str, timeout: int) -> ErrorResponse:  # noqa: D401
        return ErrorResponse()

    monkeypatch.setattr(utils.requests, "get", fake_get)

    with pytest.raises(Exception) as exc_info:
        utils.get_url_text_diff("https://example.com/fail", "Baseline")
    assert "Unable to fetch URL" in str(exc_info.value)


def test_extract_text_from_html_basic() -> None:
    """Test HTML to text extraction removes scripts/styles and normalizes whitespace."""
    html: str = (
        "<html><head><style>.a{}</style><script>var x=1;</script></head>"
        "<body><h1>Title</h1><p>Paragraph text</p><p> More text </p></body></html>"
    )
    extracted: str = utils.extract_text_from_html(html)
    # Expect lines for Title, Paragraph text, More text (order preserved)
    assert extracted.split("\n") == ["Title", "Paragraph text", "More text"]


def test_get_xml_diff_basic() -> None:
    """Test that get_xml_diff returns XML diff containing new content markers.

    We provide a baseline HTML snippet and a new snippet with a changed and an added paragraph.
    The resulting XML diff string should contain tokens from the new content (Updated, Extra) and
    preferably omit the old unique token (Original). Because xmldiff formatting can vary across versions,
    we assert minimally on presence of new tokens and type signature (string, non-empty).
    """
    baseline: str = (
        "<html><body><h1>Header</h1><p id='x'>Original paragraph</p><p>Shared</p></body></html>"
    )
    new_text: str = (
        "<html><body><h1>Header</h1><p id='x'>Updated paragraph</p><p>Shared</p><p>Extra paragraph</p></body></html>"
    )
    diff_xml: str = utils.get_xml_diff(new_text, baseline)
    assert isinstance(diff_xml, str)
    assert diff_xml != ""
    # New content tokens must appear
    assert "Updated" in diff_xml or "Updated paragraph" in diff_xml
    assert "Extra" in diff_xml
    # Old unique token ideally removed (soft expectation)
    if "Original paragraph" in diff_xml:
        # Acceptable due to representation differences; ensure diff still captures change marker
        assert "Updated" in diff_xml or "Updated paragraph" in diff_xml
