"""Word Ladder - LeetCode 127 - Reference Solution"""

from collections import defaultdict, deque

def ladder_length(begin_word: str, end_word: str, word_list: list[str]) -> int:
    if end_word not in word_list:
        return 0

    # Build pattern dictionary
    neighbors = defaultdict(list)
    word_list.append(begin_word)

    for word in word_list:
        for i in range(len(word)):
            pattern = word[:i] + "*" + word[i+1:]
            neighbors[pattern].append(word)

    # BFS
    visited = set([begin_word])
    queue = deque([begin_word])
    length = 1

    while queue:
        for _ in range(len(queue)):
            word = queue.popleft()

            if word == end_word:
                return length

            for i in range(len(word)):
                pattern = word[:i] + "*" + word[i+1:]
                for neighbor in neighbors[pattern]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

        length += 1

    return 0
