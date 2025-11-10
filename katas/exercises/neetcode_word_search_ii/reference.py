"""Word Search II - LeetCode 212 - Reference Solution"""

class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = None

def find_words(board: list[list[str]], words: list[str]) -> list[str]:
    # Build trie
    root = TrieNode()
    for word in words:
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.word = word

    result = []
    rows, cols = len(board), len(board[0])

    def dfs(r: int, c: int, node: TrieNode):
        if node.word:
            result.append(node.word)
            node.word = None  # Avoid duplicates

        if r < 0 or r >= rows or c < 0 or c >= cols:
            return

        char = board[r][c]
        if char not in node.children:
            return

        board[r][c] = '#'  # Mark as visited
        next_node = node.children[char]

        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            dfs(r + dr, c + dc, next_node)

        board[r][c] = char  # Restore

    for r in range(rows):
        for c in range(cols):
            dfs(r, c, root)

    return result
