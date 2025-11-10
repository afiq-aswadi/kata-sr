"""String palindrome checker kata - reference solution."""


def is_palindrome(s: str) -> bool:
    """Check if a string is a palindrome."""
    # Remove spaces and convert to lowercase
    cleaned = s.replace(" ", "").lower()
    return cleaned == cleaned[::-1]
