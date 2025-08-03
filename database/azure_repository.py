"""
AzureRepository provides CRUD operations for Azure Cosmos DB containers.

This class manages connection and operations for a specified Cosmos DB database and container.
It supports creating, reading, updating, and deleting items using the Azure Cosmos Python SDK.
"""



import logging
from azure.cosmos import CosmosClient

logger = logging.getLogger(__name__)

class AzureRepository:
    """
    Repository for Azure Cosmos DB container operations.

    Args:
        connection_string (str): Connection string for Cosmos DB.
        database_name (str): Name of the Cosmos DB database.
        container_name (str): Name of the Cosmos DB container.
    """

    def __init__(self, connection_string: str, database_name: str , container_name: str):
        self.connection_string = connection_string
        self.client = CosmosClient.from_connection_string(connection_string)
        self.database = self.client.get_database_client(database_name)
        self.container = self.database.get_container_client(container_name)

    def create_item(self, item: dict) -> dict:
        """
        Create a new item in the Cosmos DB container.

        Args:
            item (dict): The item to be created.

        Returns:
            dict: The created item.
        """
        created = self.container.create_item(item)
        return created

    def read_item(self, item_id: str) -> dict:
        """
        Read an item from the Cosmos DB container by its ID.

        Args:
            item_id (str): The ID of the item to read.

        Returns:
            dict: The retrieved item.
        """
        item = self.container.read_item(item=item_id, partition_key=item_id)
        return item

    def update_item(self, updated_item: dict) -> dict:
        """
        Update an existing item in the Cosmos DB container.

        Args:
            updated_item (dict): The updated item data.
        Returns:
            dict: The upserted item.
        """
        upserted = self.container.upsert_item(updated_item)
        return upserted

    def delete_item(self, item_id: str) -> dict | None:
        """
        Delete an item from the Cosmos DB container by its ID.

        Args:
            item_id (str): The ID of the item to delete.

        Returns:
            dict: The result of the delete operation.
        """
        result = self.container.delete_item(item=item_id, partition_key=item_id)
        return result

    def query_by_embedding(self, embedding: list[float], max_results: int = 10) -> list[dict]:
        """
        Query items in the Cosmos DB container by vector similarity using the VectorDistance function.
        
        This method performs a semantic similarity search against the vector embeddings stored in Cosmos DB.
        It calculates the distance between the provided embedding vector and the embeddings in the database,
        returning the closest matches sorted by similarity.

        Args:
            embedding (list[float]): The embedding vector to query by. Should be a list of floating point 
                                     numbers representing the vector embedding of the query text.
            max_results (int, optional): Maximum number of results to return. Defaults to 10.

        Returns:
            list[dict]: A list of items matching the embedding query, with only specific fields:
                        - id: The item's unique identifier
                        - price: The price of the item
                        - description: The item description
                        - item_code: The product code from the retailer
                        - store_name: The name of the retailer
                        - date_time: When the item was retrieved
                        - similarity_score: The vector distance value (lower is better)
                        
        Note:
            This function leverages Cosmos DB's vector capabilities to perform similarity search.
            The results are ordered by increasing vector distance (meaning the most similar 
            items appear first).
        """
        # Only select the fields defined in RetrievedDatabaseExtractedItem
        select_fields = [
            "c.id",
            "c.price",
            "c.description",
            "c.item_code",
            "c.store_name",
            "c.date_time",
            "VectorDistance(c.embedding, @embedding) AS similarity_score"
        ]
        select_clause = ", ".join(select_fields)
        query = f"""
        SELECT TOP {max_results} {select_clause}
        FROM c
        ORDER BY VectorDistance(c.embedding, @embedding)
        """
        parameters = [{"name": "@embedding", "value": embedding}]
        items = self.container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True)
        return list(items)