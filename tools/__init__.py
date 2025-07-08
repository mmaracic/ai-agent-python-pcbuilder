
"""
Tools package initialization module.

This module provides a factory function to retrieve all available tool instances for use in the application.
"""
from langchain_core.tools import BaseTool

from tools.search_tool import SearchTool
from tools.time_tool import TimeTool
from tools.links_tool import LinksTool


def get_tools() -> list[BaseTool]:
    """
    Factory function to get a list of available tools.

    Returns:
        list[BaseTool]: A list containing all available tools.
    """
    return [TimeTool(), SearchTool(), LinksTool()]
