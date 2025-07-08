"""
Module providing a tool for retrieving unstructured computer components from provider Links.

This module defines a Pydantic schema for search parameters and a LangChain-compatible tool class for fetching and parsing
product listings from the Links web store. The tool supports both synchronous and asynchronous execution and returns
unstructured text extracted from the HTML response.
"""
import logging
from typing import Optional
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from langchain_core.callbacks import (AsyncCallbackManagerForToolRun,
                                      CallbackManagerForToolRun)
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class SearchSchema(BaseModel):
    """
    Pydantic schema for search tool input parameters.

    Attributes:
        query (str): The search query string to look up.
        min_price (int): Minimum price filter.
        max_price (int): Maximum price filter.
    """
    query: str = Field(description="search query to look up")
    min_price: int = Field(default=0, description="minimum price filter")
    max_price: int = Field(default=10000, description="maximum price filter")



class LinksTool(BaseTool):
    """
    LangChain-compatible tool for retrieving unstructured computer components from provider Links.

    This tool constructs a search URL using the provided query and price range, fetches the HTML page, and parses the
    content to extract unstructured text. It supports both synchronous and asynchronous execution.

    Example request format:
        https://www.links.hr/hr/search?orderby=10&pagesize=100&viewmode=grid&q=intel%20procesor&price=0-23400
    """

    name: str = "links_tool"
    description: str = "A tool that returns the unstructured computer components from provider Links."
    args_schema: Optional[ArgsSchema] = SearchSchema

    def _run(self, query: str, min_price: int, max_price: int, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Retrieve unstructured computer components from provider Links.

        Args:
            query (str): The search query string.
            min_price (int): The minimum price filter.
            max_price (int): The maximum price filter.
            run_manager (Optional[CallbackManagerForToolRun]): Optional callback manager for tool run.

        Returns:
            str: The unstructured computer components from provider Links as plain text.
        """
        logger.info("Links tool called with query: %s, min_price: %d, max_price: %d", query, min_price, max_price)
        response = requests.get(
            f"https://www.links.hr/hr/search?orderby=10&pagesize=100&viewmode=grid&q={quote(query)}&price={min_price}-{max_price}",
            timeout=10
        )
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, features="html.parser")

            # kill all script and style elements
            for script in soup(["script", "style"]):
                script.extract()    # rip it out

            # get text
            text = soup.get_text()

            # break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)

            logger.info("Links tool called and responded with: %d characters", len(text))
            return text
        else:
            logger.error("Failed to fetch data from provider Links, status code: %s", response.status_code)
            raise ValueError("Error while getting response from inks tool")

    async def _arun(self, query: str, min_price: int, max_price: int, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """
        Asynchronously retrieve unstructured computer components from provider Links.
        If the calculation is cheap, you can just delegate to the sync implementation as shown below.
        If the sync calculation is expensive, you should delete the entire _arun method.
        LangChain will automatically provide a better implementation that will kick off the task in a thread to make sure it doesn't block other async code.

        Args:
            query (str): The search query string.
            min_price (int): The minimum price filter.
            max_price (int): The maximum price filter.
            run_manager (Optional[AsyncCallbackManagerForToolRun]): Optional async callback manager for tool run.

        Returns:
            str: The unstructured computer components from provider Links as plain text.
        """
        return self._run(query, min_price, max_price, run_manager=run_manager.get_sync()) if run_manager else self._run(query, min_price, max_price)
