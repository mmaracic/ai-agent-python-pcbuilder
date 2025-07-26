"""
Module implementing an item extraction agent with ReAct (Reasoning and Acting) capabilities.

Defines Pydantic models for extracted item and store data, and the ItemExtractorAgent class for extracting search result
items from computer component store web pages using LangGraph's tool-augmented reasoning and prompt templates.
"""
import logging
from typing import Any

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from agents.react_agent import ReActAgent
from database.azure_repository import AzureRepository
from database.extracted_item_model import DatabaseExtractedItem
from tools.time_tool import TimeTool
from tools.web_scraper_tool import WebScraperTool

logger = logging.getLogger(__name__)


class ExtractedItem(BaseModel):
    """
    Represents a single item extracted from a store page.

    Fields:
        price (str): Price of the item.
        description (str): Description of the item.
        item_code (str): Unique identifier for the item in the store.
    """
    price: str = Field(description="Price of the item")
    description: str = Field(description="Description of the item")
    item_code: str = Field(
        description="Unique identifier for the item in the store")


class ExtractedData(BaseModel):
    """
    Represents extracted data from a store page, including metadata and items.

    Fields:
        date_time (str): Date and time of extraction.
        store_name (str): Name of the store.
        items (list[ExtractedItem]): List of extracted items.
    """
    date_time: str = Field(description="Date and time of extraction")
    store_name: str = Field(description="Name of the store")
    items: list[ExtractedItem] = Field(
        description="List of extracted items", default_factory=list)


class ItemExtractorAgent(ReActAgent):
    """
    Agent for extracting items from computer component store web pages using ReAct (Reasoning and Acting).

    Uses a prompt template and LangGraph's tool-augmented reasoning to process incoming messages, extract search result
    items, and maintain memory for each user thread.

    Args:
        link (str): The URL of the store page to extract items from.
        model (BaseChatModel): The chat model for generating responses.
        prompt_size (int, optional): Maximum number of messages to include in the prompt. Defaults to 50.
    """
    long_term_memory: AzureRepository

    def __init__(self,
                 model: BaseChatModel,
                 long_term_memory: AzureRepository,
                 prompt_size: int = 50):
        self.long_term_memory = long_term_memory
        prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""
                    I will give you a link of a web page of computer components store.
                    Access the link to obtain the web page data.
                    The data consists of store menu, service information and search results.
                    Extract the store name and search result items in a structured way, no other information is needed.
                    """
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        super().__init__(model, [WebScraperTool(), TimeTool(
        )], prompt_template, prompt_size, response_format=ExtractedData)

    def process_link(self, link: str,) -> ExtractedData:
        """
        Process a message to extract items from the provided store page link.

        Args:
            link (str): The URL of the store page to extract items from.

        Returns:
            ExtractedData: The extracted data including store name and items.
        """
        messages = [HumanMessage(content=f"Extract items from the following store page: {link}")]
        result_dict: dict[str, Any] = self.process_message(messages, user_id="default_user")
        extracted_data: ExtractedData = result_dict['structured_response']
        if extracted_data is not None:
            logger.info("Extracted item count: %d", len(extracted_data.items))
            # Store each extracted item in the Azure Cosmos DB
            for item in extracted_data.items:
                # Map ExtractedItem to the database item model
                db_item: DatabaseExtractedItem = DatabaseExtractedItem(
                    price=item.price,
                    description=item.description,
                    item_code=item.item_code,
                    store_name=extracted_data.store_name,
                    date_time=extracted_data.date_time
                )
                self.long_term_memory.create_item(db_item.to_dict())
        else:
            logger.warning("No items were extracted from the provided link: %s", link)
        logger.info("Extraction completed for link: %s", link)
        return extracted_data
