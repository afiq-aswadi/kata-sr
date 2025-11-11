"""Tests for Encode and Decode Strings kata."""

try:
    from user_kata import encode, decode
except ImportError:
    from .reference import encode, decode


def test_encode_decode_example1():
    strs = ["hello", "world"]
    assert decode(encode(strs)) == strs

def test_encode_decode_empty():
    strs = [""]
    assert decode(encode(strs)) == strs

def test_encode_decode_special():
    strs = ["#","##","a#b"]
    assert decode(encode(strs)) == strs
