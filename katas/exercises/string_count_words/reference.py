"""String word count kata - reference solution."""


def count_words(s: str) -> int:
    """Count the number of words in a string."""
    if not s or s.isspace():
        return 0
    return len(s.split())
