from abc import ABC, abstractmethod


class Embedder(ABC):
    """
    Class for embedding text using a specified embedding model.
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """
        Embed the given text using the specified model.

        Args:
            text (str): The text to embed.

        Returns:
            list[float]: The embedding vector.
        """
        pass