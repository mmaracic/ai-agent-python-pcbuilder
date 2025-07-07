"""
ReActAgent implementation module.

Defines the ReActAgent class for tool-augmented reasoning and acting using LangGraph's prebuilt utilities.
"""
import logging
from typing import Any

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import trim_messages, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from agents.agent import AbstractAgent

logger = logging.getLogger(__name__)

class ReActAgent(AbstractAgent):
    """
    Agent with ReAct (Reasoning and Acting) capabilities using LangGraph's prebuilt tools.

    The ReActAgent compiles a state graph with a model node that processes incoming messages using a prompt template
    and chat model, and supports message trimming and memory for each user thread. It leverages LangGraph's
    create_react_agent utility for tool-augmented reasoning.

    Args:
        model (BaseChatModel): The chat model for generating responses.
        tools (list[BaseTool]): List of tools available to the agent.
        prompt_template (ChatPromptTemplate): Template for formatting prompts.
        prompt_size (int, optional): Maximum number of messages to include in the prompt. Defaults to 50.
    """

    def __init__(self, model: BaseChatModel,
                 tools: list[BaseTool],
                 prompt_template: ChatPromptTemplate,
                 prompt_size: int = 50):
        """
        Initialize a ReActAgent instance.

        Args:
            model (BaseChatModel): The chat model for generating responses.
            tools (list[BaseTool]): List of tools available to the agent.
            prompt_template (ChatPromptTemplate): Template for formatting prompts.
            prompt_size (int, optional): Maximum number of messages to include in the prompt. Defaults to 50.
        """
        self.model = model
        self.tools = tools
        self.prompt_template = prompt_template
        self.prompt_size = prompt_size
        self.compiled_graph: CompiledStateGraph = create_react_agent(
            model=self.model, tools=self.tools, prompt=self.prompt_template, checkpointer=MemorySaver())

    def process_message(self, messages: list[HumanMessage], user_id: str) -> dict[str, Any]:
        """
        Process a message using the compiled state graph and return the model's response.

        Args:
            messages (list[HumanMessage]): The list of messages to process.
            user_id (str): The ID of the user sending the messages.

        Returns:
            dict[str, Any]: The response from the model.
        """
        config = RunnableConfig(configurable={"thread_id": user_id})
        return self.compiled_graph.invoke({"messages": messages}, config)
