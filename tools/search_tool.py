"""
Module providing a web search tool using DuckDuckGo and a Pydantic schema for input validation.
"""
import logging
from typing import Optional

from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
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

    duckduck: DuckDuckGoSearchAPIWrapper
    google: GoogleSearchAPIWrapper

    def __init__(self, duckduck: Optional[DuckDuckGoSearchAPIWrapper] = None, google: Optional[GoogleSearchAPIWrapper] = None):
        """
        Initialize the SearchTool with optional DuckDuckGo and Google search wrappers.

        Args:
            duckduck (Optional[DuckDuckGoSearchAPIWrapper]): DuckDuckGo search wrapper.
            google (Optional[GoogleSearchAPIWrapper]): Google search wrapper.
        """
        super().__init__(
            duckduck=duckduck if duckduck is not None else DuckDuckGoSearchAPIWrapper(),
            google=google if google is not None else GoogleSearchAPIWrapper()
        )

    def _run(self, query: str) -> str:
        """
        Perform a synchronous web search using DuckDuckGo.

        Returns:
            str: The search results as a string.
        """
        if self.duckduck is not None:
            return self.run_search_tool(query=query)
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
        return self._run(query=query)

    def run_search_tool(self, query: str) -> str:
        """
        Run the search tool with the given query.

        Args:
            query (str): The search query to look up.

        Returns:
            str: The search results as a string.
        """
        logger.info("Search tool called with query: %s", query)
        try:
            result = self.duckduck.run(query=query)
            logger.info("DuckDuckGo generated search response: %s", result)
            return result
        except Exception as e:
            logger.error("Error occurred while running DuckDuckGo search tool: %s", e)
            result = self.google.run(query=query)
            logger.info("Google search generated search response: %s", result)
            return result
