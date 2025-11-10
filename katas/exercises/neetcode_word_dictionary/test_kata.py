"""Tests for Design Add and Search Words Data Structure kata."""

def test_word_dictionary_basic():
    from template import WordDictionary

    wd = WordDictionary()
    wd.add_word("bad")
    wd.add_word("dad")
    wd.add_word("mad")
    assert wd.search("pad") == False
    assert wd.search("bad") == True
    assert wd.search(".ad") == True
    assert wd.search("b..") == True

def test_word_dictionary_complex_patterns():
    from template import WordDictionary

    wd = WordDictionary()
    wd.add_word("at")
    wd.add_word("and")
    wd.add_word("an")
    wd.add_word("add")
    assert wd.search("a") == False
    assert wd.search(".at") == False
    assert wd.search("an") == True
    assert wd.search(".") == False
    assert wd.search("..") == True
    assert wd.search("a.") == True
    assert wd.search("a..") == True

def test_word_dictionary_no_match():
    from template import WordDictionary

    wd = WordDictionary()
    wd.add_word("hello")
    assert wd.search("hell") == False
    assert wd.search("helloa") == False
    assert wd.search(".....") == True
