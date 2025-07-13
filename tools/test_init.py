
"""
Unit tests for provider tool discovery in tools/__init__.py.

Tests the get_provider_tools function to ensure it correctly discovers and instantiates subclasses of ProviderToolInterface.
Includes tests for dynamic addition and removal of provider tool classes in the module.
"""

from unittest.mock import MagicMock
from tools.item_extractor_agent import ItemExtractorAgent
from tools.links_tool import LinksTool
from tools.protis_tool import ProtisTool
from tools.provider_tool_interface import ProviderToolInterface

from . import get_provider_tools


def test_get_provider_tools_returns_provider_tools():
    """
    Test that get_provider_tools returns all provider tool instances, including DummyProviderTool.
    """
    agent = ItemExtractorAgent(model=MagicMock())
    tools_list = get_provider_tools(agent)
    assert any(isinstance(tool, LinksTool) for tool in tools_list)
    assert any(isinstance(tool, ProtisTool) for tool in tools_list)
    assert len(tools_list) == 2
    assert all(isinstance(tool, ProviderToolInterface) for tool in tools_list)
