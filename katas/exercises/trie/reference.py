"""Trie (prefix tree) kata - reference solution."""

from typing import List


class TrieNode:
    """Node in a Trie."""

    def __init__(self):
        self.children: dict[str, TrieNode] = {}
        self.is_end_of_word: bool = False


class Trie:
    """Trie (prefix tree) data structure."""

    def __init__(self):
        """Initialize empty trie."""
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """Insert a word into the trie."""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        """Check if word exists in trie."""
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def starts_with(self, prefix: str) -> bool:
        """Check if any word starts with given prefix."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

    def get_words_with_prefix(self, prefix: str) -> List[str]:
        """Get all words with given prefix."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        # DFS to collect all words from this node
        words = []
        self._dfs(node, prefix, words)
        return words

    def _dfs(self, node: TrieNode, current: str, words: List[str]) -> None:
        """Helper to collect all words via DFS."""
        if node.is_end_of_word:
            words.append(current)

        for char, child in node.children.items():
            self._dfs(child, current + char, words)
