from unittest.mock import MagicMock, patch

import pytest
from tools.protis_tool import ProtisTool


@patch("tools.protis_tool.requests.get")
def test_run_mock_success(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "component data"
    mock_get.return_value = mock_response

    tool = ProtisTool()
    result = tool._run("cpu intel 1400")
    assert result == "component data"
    mock_get.assert_called_once_with(
        "https://www.protis.hr/products/search?exp=cpu+intel+1400",
        timeout=10
    )

def test_run_success():
    tool = ProtisTool()
    result = tool._run("cpu intel 1400")
    assert len(result) > 0

@patch("tools.protis_tool.requests.get")
def test_run_mock_failure(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = "Not found"
    mock_get.return_value = mock_response

    tool = ProtisTool()
    try:
        tool._run("cpu intel 1400")
    except Exception:
        pass  # Test passes if an exception is raised
    else:
        pytest.fail("Exception was not raised when expected.")
