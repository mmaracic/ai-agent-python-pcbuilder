"""
Main FastAPI application module for the AI agent service.

This module sets up the FastAPI app, application state, and endpoints for model setup and querying.
It integrates LangChain, OpenRouter, and custom agent/tool logic for conversational AI.
"""
import logging
import os
from typing import Annotated, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.params import Body, Depends
from langchain.chat_models.base import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, SecretStr

from agents.agent import AbstractAgent
from agents import get_agent
from database.azure_repository import AzureRepository
from database.retrieved_item_model import RetrievedDatabaseExtractedItem
from embedding.azure_llm_embedder import AzureLlmEmbedder
from embedding.embedder import Embedder
from tools.item_extractor_agent import ExtractedData, ItemExtractorAgent
from tools import get_provider_tools, get_tools
from utils import filter_messages_until_condition

# Constants
OPEN_ROUTER_API_KEY = "OPEN_ROUTER_API_KEY"
OPEN_ROUTER_API_KEY_ERROR = "OPEN_ROUTER_API_KEY environment variable is not set"
MODEL_NOT_INITIALIZED_ERROR = "Model is not initialized. Please call /setup first."


class AppState:
    """
    Holds application-wide state, such as the chat model instance.
    """

    def __init__(self):
        self.model: Optional[BaseChatModel] = None
        self.agent: Optional[AbstractAgent] = None
        self.prompt_template: Optional[ChatPromptTemplate] = None
        self.long_term_memory: Optional[AzureRepository] = None
        self.embedder: Optional[Embedder] = None

load_dotenv()
app = FastAPI()
app.state.app_state = AppState()
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_state(request: Request) -> AppState:
    """
    Retrieves the application state from the request.

    Returns:
        AppState: The current application state.
    """
    return request.app.state.app_state


@app.post("/setup")
def setup(
    application_state: Annotated[AppState, Depends(get_state)],
    prompt: Annotated[str, Body(..., media_type="text/plain")],
    prompt_size: int = 50,
    agent_type: str = "react"
) -> None:
    """
    Initializes the FastAPI application by checking for the required
    environment variable and setting up the chat model.
    Raises:
        ValueError: If the OPENAI_API_KEY environment variable is not set.
    """
    logger.info("Initializing FastAPI application")
    if not os.environ.get(OPEN_ROUTER_API_KEY):
        logger.error(OPEN_ROUTER_API_KEY_ERROR)
        raise ValueError(OPEN_ROUTER_API_KEY_ERROR)

    application_state.model = ChatOpenAI(
        #Used models deepseek/deepseek-chat-v3-0324:free,google/gemini-2.0-flash-001
        model="deepseek/deepseek-chat-v3-0324:free",
        api_key=SecretStr(os.environ[OPEN_ROUTER_API_KEY]),
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://mysite", "X-Title": "My App"},
    )
    logger.info("Prompt is set to: %s", prompt)
    application_state.prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=prompt
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    setup_embedder_and_lt_memory(application_state)
    application_state.agent = get_agent(
        agent_type=agent_type,
        model=application_state.model,
        tools=get_tools(
            ItemExtractorAgent(
                model=application_state.model,
                long_term_memory=application_state.long_term_memory,
                embedder=application_state.embedder
            )
        ),
        prompt_template=application_state.prompt_template,
        prompt_size=prompt_size
    )
    logger.info("FastAPI application initialized successfully")

def setup_embedder_and_lt_memory(
    application_state: AppState
) -> None:
    """
    Sets up the embedder and long-term memory for the application state.

    Args:
        application_state (AppState): The application state to update.
    """
    application_state.long_term_memory = AzureRepository(
        connection_string=os.environ.get("AZURE_COSMOS_CONNECTION_STRING", ""),
        database_name=os.environ.get("AZURE_COSMOS_DATABASE_NAME", "pcbuilder"),
        container_name=os.environ.get("AZURE_COSMOS_CONTAINER_NAME", "extracted_items")
    )
    application_state.embedder = AzureLlmEmbedder(
        endpoint=os.environ.get("AZURE_EMBEDDER_ENDPOINT", ""),
        api_key=os.environ.get("AZURE_EMBEDDER_API_KEY", ""),
        api_version=os.environ.get("AZURE_EMBEDDER_API_VERSION", ""),
        deployment=os.environ.get("AZURE_EMBEDDER_DEPLOYMENT", ""),
        model=os.environ.get("AZURE_EMBEDDER_MODEL", "text-embedding-3-large")
    )

@app.post("/query")
def query(state: Annotated[AppState, Depends(get_state)],
          text: Annotated[str, Body(media_type="text/plain")],
          user_id: str = "default_user"):
    """
    Handles POST requests to the '/query' endpoint.

    Args:
        request (Request): The incoming HTTP request object.
        text (str): The text provided in the request body.

    Returns:
        dict: A dictionary containing the response from the model.

    Logs:
        - The received query.
        - The response generated by the model.
    """
    if not state.agent or not state.model:
        logger.error(MODEL_NOT_INITIALIZED_ERROR)
        return {"response": MODEL_NOT_INITIALIZED_ERROR}
    logger.info("Received query: %s from user %s", text, user_id)

    query_embedding = state.embedder.embed(text)
    memory_items = state.long_term_memory.query_by_embedding(query_embedding)
    structured_memory_items = [
        RetrievedDatabaseExtractedItem.from_dict(item) for item in memory_items
    ]
    logger.info("Long term memory items found: %d", len(memory_items))
    if len(memory_items)>0:
        logger.info("Memory item 0: %s", memory_items[0])

    input_messages = [HumanMessage(text)]
    response = state.agent.process_message(input_messages, user_id)
    logger.info("Message count in history: %d", len(response["messages"]))
    reversed_list = response["messages"][::-1]
    new_messages = filter_messages_until_condition(
        reversed_list,
        lambda m: m.type == "human"
    )[::-1]
    logger.info("Response generated: %s", [m.content for m in new_messages])
    return {"response": new_messages}

class DbQuery(BaseModel):
    """
    Model for querying the database with a text input.
    
    Attributes:
        text (str): The text to query the database with.
        max_results (int): The maximum number of results to return.
        user_id (str): The ID of the user making the query. Default is "default_user".
    """
    text: str
    max_results: int = 10
    user_id: str = "default_user"

@app.post("/query_db")
def query_db(state: Annotated[AppState, Depends(get_state)],
              query: DbQuery) -> list[RetrievedDatabaseExtractedItem]:
    """
    Handles POST requests to the '/query_db' endpoint for semantic similarity search.

    This endpoint takes a query object containing text and max_results parameters, creates an 
    embedding using the application's embedder, and uses the embedding to search for 
    similar items in the database using vector similarity.

    Args:
        state (AppState): The application state containing the embedder and long-term memory.
        query (DbQuery): A query object with the following attributes:
            - text (str): The text to search for in the database
            - max_results (int): Maximum number of results to return
            - user_id (str, optional): The ID of the user making the query. Defaults to "default_user".

    Returns:
        list[RetrievedDatabaseExtractedItem]: A list of database items that match the query,
        sorted by similarity score (most similar first).

    Logs:
        - The number of items found in long-term memory.
    
    Raises:
        Exception: If the embedder is not available or the database query fails.
        
    Note:
        If the embedder and long-term memory are not initialized in the application state,
        this method will initialize them automatically using the setup_embedder_and_lt_memory function.
    """
    if not state.embedder or not state.long_term_memory:
        setup_embedder_and_lt_memory(state)
        logger.info("Embedder and long-term memory initialized.")
        
    query_embedding = state.embedder.embed(query.text)
    memory_items = state.long_term_memory.query_by_embedding(query_embedding, query.max_results)
    structured_memory_items = [
        RetrievedDatabaseExtractedItem.from_dict(item) for item in memory_items
    ]
    logger.info("Long term memory items found: %d", len(memory_items))
    return structured_memory_items


@app.post("/provider")
def test_providers(state: Annotated[AppState, Depends(get_state)],
          params: Annotated[dict, Body(media_type="text/json")],
          user_id: str = "default_user"):
    """
    Handles POST requests to the '/provider' endpoint.

    Args:
        request (Request): The incoming HTTP request object.
        text (str): The text provided in the request body.

    Returns:
        dict: A dictionary containing the response from the model.
    """
    if not state.agent or not state.model:
        logger.error(MODEL_NOT_INITIALIZED_ERROR)
        return {"response": MODEL_NOT_INITIALIZED_ERROR}
    logger.info("Received paraameters: %s from user %s", params, user_id)

    long_term_memory = AzureRepository(
        connection_string=os.environ.get("AZURE_COSMOS_CONNECTION_STRING", ""),
        database_name=os.environ.get("AZURE_COSMOS_DATABASE_NAME", "pcbuilder"),
        container_name=os.environ.get("AZURE_COSMOS_CONTAINER_NAME", "extracted_items")
    )
    provider_agent = ItemExtractorAgent(
        model=state.model,
        long_term_memory=long_term_memory,
        embedder=AzureLlmEmbedder(
            endpoint=os.environ.get("AZURE_EMBEDDER_ENDPOINT", ""),
            api_key=os.environ.get("AZURE_EMBEDDER_API_KEY", ""),
            api_version=os.environ.get("AZURE_EMBEDDER_API_VERSION", ""),
            deployment=os.environ.get("AZURE_EMBEDDER_DEPLOYMENT", ""),
            model=os.environ.get("AZURE_EMBEDDER_MODEL", "text-embedding-3-large")
        )
    )
    provider_tools = get_provider_tools(provider_agent)
    response: list[ExtractedData] = []
    for tool in provider_tools:
        try:
            tool_response = tool.get_data(params)
            logger.info("Tool %s returns : %d", tool.__class__.__name__, len(tool_response.items))
            response.append(tool_response)
        except Exception as e:
            logger.error("Error in tool %s: %s", tool.__class__.__name__, str(e))
            continue
    if not response:
        return {"response": "No provider tools returned data."}
    else:
        return {"response": response}
