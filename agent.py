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

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import trim_messages, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

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


class GraphAgent(AbstractAgent):
    """
    Agent that uses a LangGraph state graph to process conversational messages.

    The GraphAgent builds a state graph with a model node that processes incoming messages using a prompt template and chat model. It supports message trimming and maintains a memory history for each user thread.

    Args:
        model (BaseChatModel): The chat model for generating responses.
        tools (list[BaseTool]): List of tools available to the agent.
        prompt_template (ChatPromptTemplate): Template for formatting prompts.
        prompt_size (int, optional): Maximum number of messages to include in the prompt. Defaults to 50.
    """

    def __init__(self, model: BaseChatModel, tools: list[BaseTool], prompt_template: ChatPromptTemplate, prompt_size: int = 50):
        """
        Initialize a GraphAgent instance.

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

        def call_model(state: MessagesState):
            """
            Node action for the state graph: trims messages, formats the prompt, and invokes the model.

            Args:
                state (MessagesState): The current state containing messages.

            Returns:
                dict: The model's response wrapped in a dictionary.
            """
            trimmed_messages = trim_messages(  # we are trimming only the prompt to 4 last messages, full in memory history is kept
                messages=state["messages"],
                max_tokens=self.prompt_size,
                strategy="last",
                token_counter=len,  # Here we can put a model to count tokens but mistral cant count tokens so len is counting messages
                include_system=True,
                allow_partial=False,
                start_on="human",
            )
            prompt = self.prompt_template.invoke(
                {"messages": trimmed_messages})
            response = self.model.invoke(prompt)
            return {"messages": response}

        graph = StateGraph(state_schema=MessagesState)
        graph.add_edge(START, "model")
        graph.add_node(node="model", action=call_model)
        self.compiled_graph: CompiledStateGraph = graph.compile(
            checkpointer=MemorySaver())

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


class ReActAgent(AbstractAgent):
    """
    Agent with ReAct (Reasoning and Acting) capabilities using LangGraph's prebuilt tools.

    The ReActAgent compiles a state graph with a model node that processes incoming messages using a prompt template and chat model, and supports message trimming and memory for each user thread. It leverages LangGraph's create_react_agent utility for tool-augmented reasoning.

    Args:
        model (BaseChatModel): The chat model for generating responses.
        tools (list[BaseTool]): List of tools available to the agent.
        prompt_template (ChatPromptTemplate): Template for formatting prompts.
        prompt_size (int, optional): Maximum number of messages to include in the prompt. Defaults to 50.
    """

    def __init__(self, model: BaseChatModel, tools: list[BaseTool], prompt_template: ChatPromptTemplate, prompt_size: int = 50):
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
            model, tools, checkpointer=MemorySaver())

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


def get_agent(agent_type: str, model: BaseChatModel, tools: list[BaseTool], prompt_template: ChatPromptTemplate, prompt_size: int = 50) -> AbstractAgent:
    """
    Factory function to get an instance of the specified agent type.

    Args:
        agent_type (str): The type of agent to create ("graph" or "react").
        model (BaseChatModel): The chat model for generating responses.
        tools (list[BaseTool]): List of tools available to the agent.
        prompt_template (ChatPromptTemplate): Template for formatting prompts.
        prompt_size (int, optional): Maximum number of messages to include in the prompt. Defaults to 50.

    Returns:
        AbstractAgent: An instance of the specified agent type.
    """
    if agent_type == "graph":
        return GraphAgent(model, tools, prompt_template, prompt_size)
    elif agent_type == "react":
        return ReActAgent(model, tools, prompt_template, prompt_size)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
