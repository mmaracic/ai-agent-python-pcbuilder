
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
