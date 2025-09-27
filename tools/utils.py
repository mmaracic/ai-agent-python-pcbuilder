
"""
Utility functions for document extraction and web content loading.

This module provides helpers for fetching and extracting text from URLs using LangChain document loaders.
"""
import logging
import requests

from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
from xmldiff import main, formatting


logger = logging.getLogger(__name__)

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

def get_url_text_diff(url: str, baseline: str) -> str:
    """
    Fetches the text content from a given URL.

    Args:
        url (str): The URL to fetch the text from.
        baseline (str): The baseline text to compare against.

    Returns:
        str: The text content of the page.
    """
    response = requests.get(url, timeout=10)
    if response.status_code == 200:

        logger.info("Url responded with: %d characters", len(response.text))
        diff_xml = get_xml_diff(response.text, baseline)
        logger.info("Url responded with: %d characters after cleaning", len(diff_xml))
        text = extract_text_from_html(str(diff_xml))
        return text
    else:
        raise Exception(f"Unable to fetch URL: {url}, status code: {response.status_code}")

def get_xml_diff(new_text: str, baseline: str) -> str:
    """
    Computes the XML diff between the new text and the baseline text.
    Args:
        new_text (str): The new text to compare.
        baseline (str): The baseline text to compare against.
    Returns:
        str: The XML diff as a string.
    """
    diff_xml = main.diff_texts(baseline, new_text, formatter=formatting.XMLFormatter())
    return str(diff_xml)

def extract_text_from_html(html_content: str) -> str:
    """
    Extracts and returns the text content from an HTML string.

    Args:
        html_content (str): The HTML content as a string.

    Returns:
        str: The extracted text content.
    """
    soup = BeautifulSoup(html_content, features="html.parser")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text(separator="\n", strip=True)

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text