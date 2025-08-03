"""
Pydantic model for retrieving items from the database, including store information, extraction timestamp, and similarity score.

This module defines the `RetrievedDatabaseExtractedItem` class, which is intended for use as a retrieval model when loading item records from the database. It encapsulates all relevant data for an item as stored in the database, such as price, description, store name, extraction time, and a similarity score for search or ranking purposes.

Note:
    This model does not include the vector embedding itself; only metadata and similarity scores are retrieved.
"""
import uuid
from pydantic import BaseModel, Field

class RetrievedDatabaseExtractedItem(BaseModel):
    """
    Retrieval model for an item loaded from the database, including all fields as stored.

    This model is used for representing items as they are retrieved from persistent storage (database),
    and may include additional fields such as similarity scores for ranking or search results.
    The actual vector embedding is not included in this model—only metadata and similarity scores are retrieved.

    Fields:
        id (str): Unique identifier for the item in the database (auto-generated if not provided).
        price (str): Price of the item as stored in the database.
        description (str): Textual description of the item.
        item_code (str): Store-specific unique identifier for the item.
        store_name (str): Name of the store from which the item was extracted.
        date_time (str): ISO-formatted date and time when the item was extracted and stored.
        similarity_score (float): Similarity score for ranking or search, defaults to 0.0.
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
    similarity_score: float = Field(
        description="Similarity score of the item based on embedding query",
        default=0.0
    )

    def to_dict(self) -> dict[str, str]:
        """
        Convert this retrieved item to a dictionary representation, suitable for serialization or further processing.

        Returns:
            dict[str, str]: Dictionary containing all item fields and values as loaded from the database.
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "RetrievedDatabaseExtractedItem":
        """
        Create a `RetrievedDatabaseExtractedItem` instance from a dictionary of field values, as typically loaded from the database.

        Args:
            data (dict[str, str]): Dictionary containing item data (should match model fields as stored in the database).

        Returns:
            RetrievedDatabaseExtractedItem: Instance created from the provided dictionary.
        """
        return cls(**data)
