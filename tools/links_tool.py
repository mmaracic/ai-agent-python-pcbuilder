"""
Module for retrieving computer components from provider Links using LangChain and Pydantic models.

Defines a Pydantic model for search parameters and a LangChain-compatible tool class for fetching and parsing product
listings from the Links web store. The tool supports synchronous and asynchronous execution and returns structured
data using Pydantic models.
"""
import logging
from typing import Optional
from urllib.parse import quote

from langchain_core.callbacks import (CallbackManagerForToolRun)
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field
from tools.provider_tool_interface import ProviderToolInterface
from tools.item_extractor_agent import ItemExtractorAgent, ExtractedData

logger = logging.getLogger(__name__)

class SearchSchema(BaseModel):
    """
    Pydantic model for input parameters for Links search.

    Attributes:
        query (str): Search query string.
        min_price (int): Minimum price filter.
        max_price (int): Maximum price filter.
    """
    query: str = Field(description="search query to look up")
    min_price: int = Field(default=0, description="minimum price filter")
    max_price: int = Field(default=10000, description="maximum price filter")




class LinksTool(BaseTool, ProviderToolInterface):
    """
    LangChain-compatible tool for retrieving computer components from provider Links.

    Constructs a search URL using the provided query and price range, fetches the HTML page, and parses the
    content to extract structured data using Pydantic models. Supports synchronous and asynchronous execution.

    Example request format:
        https://www.links.hr/hr/search?orderby=10&pagesize=100&viewmode=grid&q=intel%20procesor&price=0-23400
    """

    name: str = "links_tool"
    description: str = "A tool that returns the unstructured computer components from provider Links."
    args_schema: Optional[ArgsSchema] = SearchSchema

    def __init__(self, extractor_agent: ItemExtractorAgent):
        """
        Initialize LinksTool with the specified extractor agent.

        Args:
            extractor_agent (ItemExtractorAgent): Agent used to process and extract data from the web page.
        """
        super().__init__(extractor_agent=extractor_agent)


    def _run(self, query: str, min_price: int = 0, max_price: int = 10000, run_manager: Optional[CallbackManagerForToolRun] = None) -> ExtractedData:
        """
        Retrieve computer components from provider Links and return structured data.

        Args:
            query (str): Search query string.
            min_price (int): Minimum price filter.
            max_price (int): Maximum price filter.
            run_manager (Optional[CallbackManagerForToolRun]): Optional callback manager for tool run.

        Returns:
            ExtractedData: Structured data containing extracted items and metadata.
        """
        logger.info("Links tool called with query: %s, min_price: %d, max_price: %d", query, min_price, max_price)
        url = f"https://www.links.hr/hr/search?orderby=10&pagesize=100&viewmode=grid&q={quote(query)}&price={min_price}-{max_price}"
        return self.extractor_agent.process_link(url)

    def get_data(self, params: dict) -> ExtractedData:
        """
        Extract computer component data from the given parameters.

        Args:
            params (dict): The parameters containing the search query and price filters.

        Returns:
            ExtractedData: Structured data extracted from the web page.
        """
        return self._run(
            query=params.get("query", ""),
            min_price=params.get("min_price", 0),
            max_price=params.get("max_price", 10000)
        )