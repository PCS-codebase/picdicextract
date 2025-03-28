
import re
import enchant
# Initialize an English dictionary (US)
english_dict = enchant.Dict("en_US")

def validate_text(text, exceptions=None):
    """
    Validate text by checking that each word is found in the English dictionary
    or is in the curated exceptions list. Reject tokens that are purely numeric or
    contain disallowed special characters.
    
    :param text: The OCR recognized text.
    :param exceptions: A set of allowed special characters/words.
    :return: A string containing only valid tokens.
    """
    if exceptions is None:
        # Define your curated exceptions here (for example, allowing ampersands, hyphens, apostrophes)
        exceptions = {"&", "-", "'"}
    
    valid_tokens = []
    # Tokenize the text; allow words with hyphens or apostrophes
    tokens = re.findall(r"\b[\w'-]+\b", text)
    for token in tokens:
        # Skip tokens that are purely numeric
        if token.isdigit():
            continue
        # Strip common punctuation from the boundaries for checking
        cleaned_token = token.strip("'-")
        # Accept the token if it is in the exceptions or if it's a valid English word
        if cleaned_token in exceptions or english_dict.check(cleaned_token):
            valid_tokens.append(token)
    return " ".join(valid_tokens)