"""Tests for Trie kata."""



def test_trie_insert_and_search():
    from template import Trie

    trie = Trie()
    trie.insert("apple")

    assert trie.search("apple") is True
    assert trie.search("app") is False


def test_trie_starts_with():
    from template import Trie

    trie = Trie()
    trie.insert("apple")

    assert trie.starts_with("app") is True
    assert trie.starts_with("apple") is True
    assert trie.starts_with("appl") is True
    assert trie.starts_with("b") is False


def test_trie_multiple_words():
    from template import Trie

    trie = Trie()
    words = ["apple", "app", "application", "apply", "banana"]
    for word in words:
        trie.insert(word)

    assert all(trie.search(word) for word in words)
    assert trie.search("ban") is False


def test_trie_get_words_with_prefix():
    from template import Trie

    trie = Trie()
    words = ["apple", "app", "application", "apply", "banana", "band"]
    for word in words:
        trie.insert(word)

    app_words = trie.get_words_with_prefix("app")
    assert set(app_words) == {"apple", "app", "application", "apply"}

    ban_words = trie.get_words_with_prefix("ban")
    assert set(ban_words) == {"banana", "band"}


def test_trie_empty():
    from template import Trie

    trie = Trie()
    assert trie.search("test") is False
    assert trie.starts_with("test") is False
    assert trie.get_words_with_prefix("test") == []


def test_trie_prefix_not_word():
    from template import Trie

    trie = Trie()
    trie.insert("apple")

    assert trie.starts_with("app") is True
    assert trie.search("app") is False
