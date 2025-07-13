
"""
Utility functions for document extraction and web content loading.

This module provides helpers for fetching and extracting text from URLs using LangChain document loaders.
"""
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader

def get_url_text(url: str) -> str:
    """
    Fetches the text content from a given URL.

    Args:
        url (str): The URL to fetch the text from.

    Returns:
        str: The text content of the page.
    """
    loader =  WebBaseLoader(url)
    docs: list[Document] = loader.load()
    text = docs[0].page_content if docs else ""
    return text
