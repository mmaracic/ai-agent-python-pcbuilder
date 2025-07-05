from langchain.chat_models.base import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

from agents.agent import AbstractAgent
from agents.graph_agent import GraphAgent
from agents.react_agent import ReActAgent

def get_agent(agent_type: str,
              model: BaseChatModel,
              tools: list[BaseTool],
              prompt_template: ChatPromptTemplate,
              prompt_size: int = 50) -> AbstractAgent:
    """
    Factory function to get an instance of the specified agent type.

    Args:
        agent_type (str): The type of agent to create ("graph" or "react").
        model (BaseChatModel): The chat model for generating responses.
        tools (list[BaseTool]): List of tools available to the agent.
        prompt_template (ChatPromptTemplate): Template for formatting prompts.
        prompt_size (int, optional): Maximum number of messages to include in the prompt.
        Defaults to 50.

    Returns:M
        AbstractAgent: An instance of the specified agent type.
    """
    if agent_type == "graph":
        return GraphAgent(model, tools, prompt_template, prompt_size)
    elif agent_type == "react":
        return ReActAgent(model, tools, prompt_template, prompt_size)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
