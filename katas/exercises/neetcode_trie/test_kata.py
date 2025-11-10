"""Tests for Implement Trie kata."""

def test_trie_basic():
    from template import Trie

    trie = Trie()
    trie.insert("apple")
    assert trie.search("apple") == True
    assert trie.search("app") == False
    assert trie.starts_with("app") == True
    trie.insert("app")
    assert trie.search("app") == True

def test_trie_multiple_words():
    from template import Trie

    trie = Trie()
    trie.insert("hello")
    trie.insert("hell")
    trie.insert("heaven")
    assert trie.search("hello") == True
    assert trie.search("hell") == True
    assert trie.search("heaven") == True
    assert trie.search("heave") == False
    assert trie.starts_with("hea") == True
    assert trie.starts_with("hel") == True

def test_trie_prefix_not_word():
    from template import Trie

    trie = Trie()
    trie.insert("application")
    assert trie.starts_with("app") == True
    assert trie.search("app") == False
