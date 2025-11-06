"""Trie (prefix tree) kata."""

from typing import List


class TrieNode:
    """Node in a Trie."""

    def __init__(self):
        # TODO: initialize children dict and is_end_of_word flag
        # BLANK_START
        pass
        # BLANK_END


class Trie:
    """Trie (prefix tree) data structure."""

    def __init__(self):
        """Initialize empty trie."""
        # TODO: create root node
        # BLANK_START
        pass
        # BLANK_END

    def insert(self, word: str) -> None:
        """Insert a word into the trie.

        Args:
            word: word to insert
        """
        # TODO: traverse/create nodes for each character
        # BLANK_START
        pass
        # BLANK_END

    def search(self, word: str) -> bool:
        """Check if word exists in trie.

        Args:
            word: word to search for

        Returns:
            True if word exists (as complete word)
        """
        # TODO: traverse nodes, check is_end_of_word
        # BLANK_START
        pass
        # BLANK_END

    def starts_with(self, prefix: str) -> bool:
        """Check if any word starts with given prefix.

        Args:
            prefix: prefix to search for

        Returns:
            True if any word has this prefix
        """
        # TODO: traverse nodes for prefix
        # BLANK_START
        pass
        # BLANK_END

    def get_words_with_prefix(self, prefix: str) -> List[str]:
        """Get all words with given prefix.

        Args:
            prefix: prefix to search for

        Returns:
            list of words with this prefix
        """
        # TODO: find prefix node, then DFS to collect all words
        # BLANK_START
        pass
        # BLANK_END
