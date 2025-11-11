"""Tests for Word Search II kata."""

try:
    from user_kata import find_words
except ImportError:
    from .reference import find_words


def test_word_search_ii_example():

    board = [
        ["o","a","a","n"],
        ["e","t","a","e"],
        ["i","h","k","r"],
        ["i","f","l","v"]
    ]
    words = ["oath","pea","eat","rain"]
    result = find_words(board, words)
    assert set(result) == {"eat", "oath"}

def test_word_search_ii_single_char():

    board = [["a","b"],["c","d"]]
    words = ["a","b","c","d","abcd"]
    result = find_words(board, words)
    assert set(result) == {"a", "b", "c", "d"}

def test_word_search_ii_no_words():

    board = [["a","a"]]
    words = ["aaa"]
    result = find_words(board, words)
    assert result == []
