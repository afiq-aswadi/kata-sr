"""Encode and Decode Strings - LeetCode 271 - Reference Solution"""

def encode(strs: list[str]) -> str:
    result = []
    for s in strs:
        result.append(f"{len(s)}#{s}")
    return "".join(result)

def decode(s: str) -> list[str]:
    result = []
    i = 0
    while i < len(s):
        # Find the delimiter
        j = i
        while s[j] != '#':
            j += 1
        length = int(s[i:j])
        i = j + 1
        result.append(s[i:i + length])
        i += length
    return result
