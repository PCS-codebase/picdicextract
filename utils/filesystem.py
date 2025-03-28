import re

def sanitize_filename(text):
    """
    Sanitize the given text for use in a filename by replacing invalid characters
    with underscores and limiting the result to 32 characters.
    """
    return re.sub(r'[^A-Za-z0-9_-]', '_', text)[:32]