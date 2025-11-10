"""Tests for Word Break kata."""

def test_word_break_example1():
    from template import word_break
    assert word_break("leetcode", ["leet","code"]) == True

def test_word_break_example2():
    from template import word_break
    assert word_break("applepenapple", ["apple","pen"]) == True

def test_word_break_example3():
    from template import word_break
    assert word_break("catsandog", ["cats","dog","sand","and","cat"]) == False

def test_word_break_single():
    from template import word_break
    assert word_break("a", ["a"]) == True
    assert word_break("a", ["b"]) == False
