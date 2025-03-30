import re
import enchant

# Initialize an English dictionary (US)
english_dict = enchant.Dict("en_US")

def validate_text(text, exceptions=None):
    """
    Return True if all parts of the input (split by whitespace or hyphens) are valid English words
    or are in the exceptions list. Rejects numbers and words with invalid characters.
    
    Apostrophes are allowed within words.
    """
    if exceptions is None:
        exceptions = set()

    # Split on whitespace or hyphens
    tokens = re.split(r'[\s-]+', text)

    for token in tokens:
        if not token:
            continue  # Skip empty strings from consecutive spaces or hyphens
        if token.isdigit():
            return False
        if not re.fullmatch(r"[A-Za-z']+", token):
            return False
        if token not in exceptions and not english_dict.check(token):
            return False

    return True
