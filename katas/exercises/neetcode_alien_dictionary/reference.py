"""Alien Dictionary - LeetCode 269 - Reference Solution"""

from collections import defaultdict, deque

def alien_order(words: list[str]) -> str:
    # Build adjacency list and in-degree map
    adj = defaultdict(set)
    in_degree = {c: 0 for word in words for c in word}

    # Compare adjacent words to find character ordering
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        min_len = min(len(w1), len(w2))

        # Check for invalid case: w1 is prefix and longer
        if len(w1) > len(w2) and w1[:min_len] == w2[:min_len]:
            return ""

        for j in range(min_len):
            if w1[j] != w2[j]:
                if w2[j] not in adj[w1[j]]:
                    adj[w1[j]].add(w2[j])
                    in_degree[w2[j]] += 1
                break

    # Topological sort using Kahn's algorithm
    queue = deque([c for c in in_degree if in_degree[c] == 0])
    result = []

    while queue:
        c = queue.popleft()
        result.append(c)

        for neighbor in adj[c]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Check if all characters were processed (no cycle)
    if len(result) != len(in_degree):
        return ""

    return "".join(result)
