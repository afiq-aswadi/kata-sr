"""Tests for Encode and Decode Strings kata."""

def test_encode_decode_example1():
    from template import encode, decode
    strs = ["hello", "world"]
    assert decode(encode(strs)) == strs

def test_encode_decode_empty():
    from template import encode, decode
    strs = [""]
    assert decode(encode(strs)) == strs

def test_encode_decode_special():
    from template import encode, decode
    strs = ["#","##","a#b"]
    assert decode(encode(strs)) == strs
