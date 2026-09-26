import re
import unicodedata

def unicode_to_ascii(s: str) -> str:
    """Normalize unicode characters to plain ASCII."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_text(text: str) -> str:
    """
    Clean dialogue text:
    - Lowercase
    - Expand common contractions
    - Remove emojis and non-alphanumeric symbols except standard punctuation
    - Clean whitespace
    """
    text = unicode_to_ascii(text.lower().strip())
    
    # Contraction expansions
    contractions = {
        r"can\'t": "cannot",
        r"won\'t": "will not",
        r"n\'t": " not",
        r"\'re": " are",
        r"\'s": " is",
        r"\'d": " would",
        r"\'ll": " will",
        r"\'t": " not",
        r"\'ve": " have",
        r"\'m": " am",
        r"whats": "what is",
        r"what\'s": "what is",
        r"hows": "how is",
        r"how\'s": "how is",
        r"it\'s": "it is",
        r"let\'s": "let us",
        r"you\'re": "you are",
        r"i\'m": "i am",
        r"dont": "do not",
        r"don\'t": "do not",
        r"doesnt": "does not",
        r"doesn\'t": "does not",
        r"didnt": "did not",
        r"didn\'t": "did not",
        r"havent": "have not",
        r"haven\'t": "have not",
        r"hasnt": "has not",
        r"hasn\'t": "has not"
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text)
        
    # Split punctuation with spaces to treat as tokens or remove non-alphanumeric except basic punctuation
    text = re.sub(r"([.!?])", r" \1", text)
    # Remove weird symbols, keep only alphanumeric and basic punctuation
    text = re.sub(r"[^a-zA-Z0-9.!?]+", r" ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", r" ", text).strip()
    return text

def filter_pair(q: str, a: str, min_len: int = 1, max_len: int = 15) -> bool:
    """Filter dialogue pairs based on sentence length constraints."""
    q_words = q.split()
    a_words = a.split()
    return min_len <= len(q_words) <= max_len and min_len <= len(a_words) <= max_len
