from unittest.mock import MagicMock

import pytest
from tools.item_extractor_agent import ItemExtractorAgent
from tools.links_tool import LinksTool


def test_run_mock_success():
    tool = LinksTool(extractor_agent=ItemExtractorAgent(model=MagicMock()))
    # Patch process_link to return the mock response text
    tool.extractor_agent.process_link = MagicMock(return_value="component data")
    result = tool._run("intel procesor", 200, 300)
    assert result == "component data"

def test_run_mock_failure():
    tool = LinksTool(extractor_agent=ItemExtractorAgent(model=MagicMock()))
    # Patch process_link to raise an exception
    tool.extractor_agent.process_link = MagicMock(side_effect=Exception("Not found"))
    with pytest.raises(Exception):
        tool._run("intel procesor", 200, 300)
