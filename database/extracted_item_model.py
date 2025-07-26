"""
Database model for storing extracted items with store information and timestamp.

Defines a Pydantic model for encapsulating extracted item data, store name, and extraction timestamp.
"""
import uuid
from pydantic import BaseModel, Field

class DatabaseExtractedItem(BaseModel):
    """
    Represents an item extracted from a store page for database storage, including store info and timestamp.

    Attributes:
        id (str): Unique identifier for the item in the database.
        price (str): Price of the item.
        description (str): Description of the item.
        item_code (str): Unique identifier for the item in the store.
        store_name (str): Name of the store.
        date_time (str): Date and time of extraction.
    """
    id: str = Field(
        description="Unique identifier for the item in the database",
        default_factory=lambda: "item_" + str(uuid.uuid4())
    )
    price: str = Field(description="Price of the item")
    description: str = Field(description="Description of the item")
    item_code: str = Field(description="Unique identifier for the item in the store")
    store_name: str = Field(description="Name of the store")
    date_time: str = Field(description="Date and time of extraction")

    def to_dict(self) -> dict[str, str]:
        """
        Convert the DatabaseExtractedItem instance to a dictionary.

        Returns:
            dict[str, str]: Dictionary representation of the item.
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "DatabaseExtractedItem":
        """
        Create a DatabaseExtractedItem instance from a dictionary.

        Args:
            data (dict[str, str]): Dictionary containing item data.

        Returns:
            DatabaseExtractedItem: Instance created from the dictionary.
        """
        return cls(**data)
