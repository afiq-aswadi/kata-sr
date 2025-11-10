"""Tests for Word Search II kata."""

def test_word_search_ii_example():
    from template import find_words

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
    from template import find_words

    board = [["a","b"],["c","d"]]
    words = ["a","b","c","d","abcd"]
    result = find_words(board, words)
    assert set(result) == {"a", "b", "c", "d"}

def test_word_search_ii_no_words():
    from template import find_words

    board = [["a","a"]]
    words = ["aaa"]
    result = find_words(board, words)
    assert result == []
