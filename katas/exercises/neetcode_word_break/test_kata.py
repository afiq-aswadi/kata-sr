"""Tests for Word Break kata."""

try:
    from user_kata import word_break
except ImportError:
    from .reference import word_break


def test_word_break_example1():
    assert word_break("leetcode", ["leet","code"]) == True

def test_word_break_example2():
    assert word_break("applepenapple", ["apple","pen"]) == True

def test_word_break_example3():
    assert word_break("catsandog", ["cats","dog","sand","and","cat"]) == False

def test_word_break_single():
    assert word_break("a", ["a"]) == True
    assert word_break("a", ["b"]) == False
