import logging
from openai import AzureOpenAI
from embedding.embedder import Embedder
from azure.core.credentials import AzureKeyCredential

logger = logging.getLogger(__name__)

class AzureLlmEmbedder(Embedder):
    """
    Class for embedding text using a Large Language Model (LLM).
    """
    client: AzureOpenAI
    model: str

    def __init__(self, endpoint: str, api_version: str, deployment: str, model: str, api_key: str):
        """
        Initialize the LLM embedder with a chat model.

        Args:
            model (BaseChatModel): The chat model to use for embedding.
        """

        self.model = model
        self.client = AzureOpenAI(
            azure_deployment=deployment,
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=api_key
        )

    def embed(self, text: str) -> list[float]:
        """
        Embed the given text using the LLM.

        Args:
            text (str): The text to embed.

        Returns:
            list[float]: The embedding vector.
        """
        response = self.client.embeddings.create(
            input=[text],
            model=self.model
        )
        logger.debug("Embedding response: %s", response)
        return response.data[0].embedding if response.data else []