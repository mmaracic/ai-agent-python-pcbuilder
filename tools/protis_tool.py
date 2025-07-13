"""
Module for retrieving unstructured computer components from provider Protis using LangChain and Pydantic models.

Defines a Pydantic model for search parameters and a LangChain-compatible tool class for fetching and parsing product
listings from the Protis web store. The tool supports synchronous and asynchronous execution and returns unstructured
text extracted from the HTML response.
"""
import logging
from typing import Optional

from langchain_core.callbacks import (CallbackManagerForToolRun)
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field

from tools.item_extractor_agent import ExtractedData, ItemExtractorAgent
from tools.provider_tool_interface import ProviderToolInterface

logger = logging.getLogger(__name__)

class SearchSchema(BaseModel):
    """
    Pydantic model for input parameters for Protis search.

    Attributes:
        query (str): Search query string.
    """
    query: str = Field(description="search query to look up")



class ProtisTool(BaseTool, ProviderToolInterface):
    """
    LangChain-compatible tool for retrieving unstructured computer components from provider Protis.

    Constructs a search URL using the provided query, fetches the HTML page, and parses the
    content to extract unstructured text. Supports synchronous and asynchronous execution.

    Example request format:
        https://www.protis.hr/products/search?exp=cpu+intel+1400
    """

    name: str = "protis_tool"
    description: str = "A tool that returns the unstructured computer components from provider Protis."
    args_schema: Optional[ArgsSchema] = SearchSchema

    def __init__(self, extractor_agent: ItemExtractorAgent):
        """
        Initialize ProtisTool with the specified extractor agent.

        Args:
            extractor_agent (ItemExtractorAgent): Agent used to process and extract data from the web page.
        """
        super().__init__(extractor_agent=extractor_agent)

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> ExtractedData:
        """
        Retrieve unstructured computer components from provider Protis and return extracted data.

        Args:
            query (str): Search query string.
            run_manager (Optional[CallbackManagerForToolRun]): Optional callback manager for tool run.

        Returns:
            ExtractedData: Unstructured computer components from provider Protis as plain text and metadata.
        """
        logger.info("Protis tool called with query: %s", query)
        url = f"https://www.protis.hr/products/search?exp={query.replace(' ', '+')}"
        return self.extractor_agent.process_link(url)

    def get_data(self, params: dict) -> ExtractedData:
        """
        Extract computer component data from the given parameters.

        Args:
            params (dict): The parameters containing the search query.

        Returns:
            ExtractedData: Structured data extracted from the web page.
        """
        return self._run(
            query=params.get("query", "")
        )