"""
Module providing a web search tool using DuckDuckGo and a Pydantic schema for input validation.
"""
import logging
from typing import Optional

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class SearchSchema(BaseModel):
    """
    Schema for the search tool input.
    Attributes:
        query (str): The search query string to look up.
    """
    query: str = Field(description="search query to look up")

class SearchTool(BaseTool):
    """
    Tool for performing web searches using DuckDuckGo.
    """

    name: str = "search_tool"
    description: str = "Performs a web search using DuckDuckGo."
    args_schema: Optional[ArgsSchema] = SearchSchema

    duckduck: DuckDuckGoSearchRun = DuckDuckGoSearchRun()

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Perform a synchronous web search using DuckDuckGo.

        Returns:
            str: The search results as a string.
        """
        if self.duckduck is not None:
            result = self.duckduck.run(tool_input=query, run_manager=run_manager)
            logger.info("Search tool called with query: %s\n Response: %s", query, result)
            return result
        raise ValueError("DuckDuckGoSearchRun is not initialized.")

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """
        Perform an asynchronous web search using DuckDuckGo.
        If the calculation is cheap, you can just delegate to the sync implementation
        as shown below.
        If the sync calculation is expensive, you should delete the entire _arun method.
        LangChain will automatically provide a better implementation that will
        kick off the task in a thread to make sure it doesn't block other async code.

        Returns:
            str: The search results as a string.
        """
        return self._run(query=query, run_manager=run_manager.get_sync()) if run_manager else self._run(query=query)
