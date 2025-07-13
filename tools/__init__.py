
"""
Tools package initialization module.

Provides factory functions to retrieve available LangChain tool instances for use in the application. Requires an
ItemExtractorAgent instance and returns a list of LangChain-compatible tool objects with type annotations.
"""
import inspect
import sys

from langchain_core.tools import BaseTool

from tools.provider_tool_interface import ProviderToolInterface
from tools.item_extractor_agent import ItemExtractorAgent
from tools.search_tool import SearchTool
from tools.time_tool import TimeTool
from tools.links_tool import LinksTool
from tools.protis_tool import ProtisTool


def get_tools(extractor_agent: ItemExtractorAgent) -> list[BaseTool]:
    """
    Return a list of all available LangChain tool instances.

    Args:
        extractor_agent (ItemExtractorAgent): Agent for extracting items from web pages.

    Returns:
        list[BaseTool]: List of all available tool instances.
    """
    return [TimeTool(), SearchTool(), LinksTool(extractor_agent=extractor_agent), ProtisTool(extractor_agent=extractor_agent)]


def get_provider_tools(extractor_agent: ItemExtractorAgent) -> list[ProviderToolInterface]:
    """
    Return a list of provider-specific LangChain tool instances by checking subclasses of ProviderToolInterface.

    Args:
        extractor_agent (ItemExtractorAgent): Agent for extracting items from web pages.

    Returns:
        list[BaseTool]: List of provider-specific tool instances.
    """
    # Get all classes in the current module that inherit ProviderToolInterface
    provider_tools = []
    current_module = sys.modules[__name__]
    for _, obj in inspect.getmembers(current_module):
        if inspect.isclass(obj) and issubclass(obj, ProviderToolInterface) and obj is not ProviderToolInterface:
            provider_tools.append(obj(extractor_agent=extractor_agent)) # type: ignore
    return provider_tools
