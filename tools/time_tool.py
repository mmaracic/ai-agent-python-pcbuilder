"""
Module providing a tool for returning the current date and time in ISO format.
"""
from datetime import datetime
import logging
from typing import Optional

from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

logger = logging.getLogger(__name__)

class TimeTool(BaseTool):
    """
    A tool that returns the current time.
    """

    name: str = "date_and_time_tool" # Can not have spaces otherwise models other deepseek crash on query
    description: str = "A tool that returns the current date and time in ISO format."

    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Returns the current date and time in ISO format.

        Returns:
            str: The current time in ISO format.
        """
        current = datetime.now().isoformat()
        logger.info("Time tool called and responded with: %s", current)
        return current

    async def _arun(self, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """
        Asynchronously returns the current time in ISO format.
        If the calculation is cheap, you can just delegate to the sync implementation
        as shown below.
        If the sync calculation is expensive, you should delete the entire _arun method.
        LangChain will automatically provide a better implementation that will
        kick off the task in a thread to make sure it doesn't block other async code.

        Returns:
            str: The current time in ISO format.
        """
        return self._run(run_manager=run_manager.get_sync()) if run_manager else self._run()
