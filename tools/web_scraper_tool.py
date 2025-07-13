"""
Module providing a tool for scraping web page content and returning it as plain text.
"""
import logging
from typing import Optional

from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from tools.utils import get_url_text

logger = logging.getLogger(__name__)

class WebScraperTool(BaseTool):
    """
    Tool for scraping web page content and returning it as plain text.
    """

    name: str = "web_scraper_tool" # Can not have spaces otherwise models other deepseek crash on query
    description: str = "A tool that returns the content of a web page as a plain text."

    def _run(self, url: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Returns the content of a web page as a string without html.

        Returns:
            str: The content of the web page as plain text.
        """
        logger.info("WebScraperTool called with URL: %s", url)
        text = get_url_text(url)
        logger.info("WebScraperTool returning text of length: %d characters", len(text))
        return text

    async def _arun(self, url: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """
        Asynchronously returns the content of a web page as a plain text.
        If the calculation is cheap, you can just delegate to the sync implementation
        as shown below.
        If the sync calculation is expensive, you should delete the entire _arun method.
        LangChain will automatically provide a better implementation that will
        kick off the task in a thread to make sure it doesn't block other async code.

        Returns:
            str: The content of the web page as plain text.
        """
        return self._run(url, run_manager=run_manager.get_sync()) if run_manager else self._run(url)
