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
        logger.info(f"Creating item: {item}")
        created = self.container.create_item(item)
        logger.info(f"Item created: {created}")
        return created

    def read_item(self, item_id: str) -> dict:
        """
        Read an item from the Cosmos DB container by its ID.

        Args:
            item_id (str): The ID of the item to read.

        Returns:
            dict: The retrieved item.
        """
        logger.info(f"Reading item with id: {item_id}")
        item = self.container.read_item(item=item_id, partition_key=item_id)
        logger.info(f"Item read: {item}")
        return item

    def update_item(self, updated_item: dict) -> dict:
        """
        Update an existing item in the Cosmos DB container.

        Args:
            updated_item (dict): The updated item data.
        Returns:
            dict: The upserted item.
        """
        logger.info(f"Updating item: {updated_item}")
        upserted = self.container.upsert_item(updated_item)
        logger.info(f"Item upserted: {upserted}")
        return upserted

    def delete_item(self, item_id: str) -> dict | None:
        """
        Delete an item from the Cosmos DB container by its ID.

        Args:
            item_id (str): The ID of the item to delete.

        Returns:
            dict: The result of the delete operation.
        """
        logger.info(f"Deleting item with id: {item_id}")
        result = self.container.delete_item(item=item_id, partition_key=item_id)
        logger.info(f"Delete result: {result}")
        return result