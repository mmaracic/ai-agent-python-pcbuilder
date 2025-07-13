
"""
Interface for provider tools that extract computer component data from a web page.

Defines an abstract base class for provider tools, requiring implementation of a get_data method that takes parameters and returns ExtractedData.
"""

from abc import ABC, abstractmethod
from tools.item_extractor_agent import ExtractedData, ItemExtractorAgent



class ProviderToolInterface(ABC):
    """
    Abstract base class for provider tools.

    Requires implementation of a get_data method that takes parameters and returns ExtractedData.
    """

    extractor_agent: ItemExtractorAgent

    @abstractmethod
    def get_data(self, params: dict) -> ExtractedData:
        """
        Extract computer component data from the given parameters.

        Args:
            params (dict): Parameters containing the URL and extraction options.

        Returns:
            ExtractedData: Structured data extracted from the web page.
        """

