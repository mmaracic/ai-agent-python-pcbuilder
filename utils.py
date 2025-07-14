"""
Utility functions for text extraction and message filtering in the AI agent application.

This module provides helpers for extracting substrings between markers and for filtering message lists
up to a condition, using type-safe signatures and supporting flexible filtering via callables.
"""

from typing import Callable, List

from langchain_core.messages import BaseMessage

def extract_between(text: str, start_marker: str, end_marker: str) -> str:
    """
    Extracts and returns the substring between two markers in the given text.

    Args:
        text (str): The input text to search within.
        start_marker (str): The substring marking the start of the extraction.
        end_marker (str): The substring marking the end of the extraction.

    Returns:
        str: The substring found between start_marker and end_marker, or an empty string if not found or invalid.
    """
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start == -1 or end == -1 or end <= start:
        return ""
    start += len(start_marker)
    return text[start:end]


def filter_messages_until_condition(
    messages: List[BaseMessage],
    condition: Callable[[BaseMessage], bool]
) -> List[BaseMessage]:
    """
    Returns all messages from the list up to (but not including) the first message for which condition(message) is True.

    Args:
        messages (List[BaseMessage]): List of messages to filter.
        condition (Callable[[BaseMessage], bool]): Function that returns True to stop including messages.

    Returns:
        List[BaseMessage]: Filtered list of messages up to the first match of the condition.
    """
    result: list = []
    for message in messages:
        if condition(message):
            break
        result.append(message)
    return result
