from unittest.mock import MagicMock, patch

import pytest
from tools.links_tool import LinksTool


@patch("tools.links_tool.requests.get")
def test_run_mock_success(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "component data"
    mock_get.return_value = mock_response

    tool = LinksTool()
    result = tool._run("intel procesor", 200, 300)
    assert result == "component data"
    mock_get.assert_called_once_with(
        "https://www.links.hr/hr/search?orderby=10&pagesize=100&viewmode=grid&q=intel%20procesor&price=200-300",
        timeout=10
    )

def test_run_success():
    tool = LinksTool()
    result = tool._run("intel procesor", 200, 300)
    assert len(result) > 0

@patch("tools.links_tool.requests.get")
def test_run_mock_failure(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = "Not found"
    mock_get.return_value = mock_response

    tool = LinksTool()
    try:
        tool._run("intel procesor", 200, 300)
    except Exception:
        pass  # Test passes if an exception is raised
    else:
        pytest.fail("Exception was not raised when expected.")
