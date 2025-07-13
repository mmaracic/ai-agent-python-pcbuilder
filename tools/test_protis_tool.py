from unittest.mock import MagicMock

import pytest
from tools.item_extractor_agent import ItemExtractorAgent
from tools.protis_tool import ProtisTool


def test_run_mock_success():
    tool = ProtisTool(extractor_agent=ItemExtractorAgent(model=MagicMock()))
    # Patch process_link to return the mock response text
    tool.extractor_agent.process_link = MagicMock(return_value="Intel cpu 1400, 200 euros")
    result = tool._run("cpu intel 1400")
    assert result == "Intel cpu 1400, 200 euros"

def test_run_mock_failure():
    tool = ProtisTool(extractor_agent=ItemExtractorAgent(model=MagicMock()))
    # Patch process_link to raise an exception
    tool.extractor_agent.process_link = MagicMock(side_effect=Exception("Not found"))
    with pytest.raises(Exception):
        tool._run("cpu intel 1400")
