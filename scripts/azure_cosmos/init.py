#Run from root with: python ./scripts/azure_cosmos/init.py
import os

from azure.cosmos import CosmosClient, PartitionKey
from dotenv import load_dotenv

load_dotenv()

connection_string = os.environ.get("AZURE_COSMOS_CONNECTION_STRING", "")
database_name = os.environ.get("AZURE_COSMOS_DATABASE_NAME", "pcbuilder")
container_name = os.environ.get("AZURE_COSMOS_CONTAINER_NAME", "computer_parts")

if not connection_string or not database_name or not container_name:
    raise ValueError("Azure Cosmos DB connection string, database name, and container name must be set in environment variables.")
print(f"Connecting to Azure Cosmos DB with connection string:\n {connection_string}\n database: {database_name}\n container: {container_name}")

client = CosmosClient.from_connection_string(connection_string)
database = client.get_database_client(database_name)
container = database.create_container(
    id=container_name,
    partition_key=PartitionKey(path="/id"),
    unique_key_policy={
        "uniqueKeys": [{"paths": ["/description"]}]
    },
    indexing_policy={
        "indexingMode": "consistent",
        "automatic": True,
        "includedPaths": [
            {
                "path": "/*"
            }
        ],
        "excludedPaths": [
            {
                "path": "/\"_etag\"/?"
            },
            {
                "path": "/embedding/*"
            }
        ],
        "fullTextIndexes": [],
        "vectorIndexes": [
            {
                "path": "/embedding",
                "type": "diskANN",
                "quantizationByteSize": 128,
                "indexingSearchListSize": 100
            }
        ]
    },
    vector_embedding_policy={
        "vectorEmbeddings": [
            {
                "path": "/embedding",
                "dataType": "float32",
                "distanceFunction": "cosine",
                "dimensions": 3072
            }
        ]
    }
)
print(f"Container '{container_name}' created successfully in database '{database_name}'.")
