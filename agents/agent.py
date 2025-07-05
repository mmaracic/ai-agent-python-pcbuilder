"""
Agent module for message processing using LangGraph and LangChain.

This module provides abstract and concrete agent classes for handling conversational messages.
It includes:
    - AbstractAgent: An abstract base class defining the agent interface.
    - GraphAgent: A state-graph-based agent for message processing and prompt management.
    - ReActAgent: An agent with ReAct (Reasoning and Acting) capabilities using LangGraph's prebuilt tools.
"""
import logging
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


class AbstractAgent(ABC):
    """
    Abstract base class for agents.

    Defines the interface for an agent that can process messages and return responses.
    Subclasses must implement the process_message method.
    """

    @abstractmethod
    def process_message(self, messages: list[HumanMessage], user_id: str) -> dict[str, Any]:
        """
        Process a message and return the updated state.

        Args:
            messages (list[HumanMessage]): The list of messages to process.
            user_id (str): The ID of the user sending the messages.

        Returns:
            dict[str, Any]: The updated state or response from the agent.
        """
