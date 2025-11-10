"""Tests for Merge Triplets to Form Target Triplet kata."""

def test_merge_triplets_example1():
    from template import merge_triplets
    assert merge_triplets([[2,5,3],[1,8,4],[1,7,5]], [2,7,5]) == True

def test_merge_triplets_example2():
    from template import merge_triplets
    assert merge_triplets([[3,4,5],[4,5,6]], [3,2,5]) == False

def test_merge_triplets_exact_match():
    from template import merge_triplets
    assert merge_triplets([[1,2,3]], [1,2,3]) == True

def test_merge_triplets_impossible():
    from template import merge_triplets
    assert merge_triplets([[1,1,1],[2,2,2]], [3,3,3]) == False

def test_merge_triplets_multiple():
    from template import merge_triplets
    assert merge_triplets([[1,0,0],[0,1,0],[0,0,1]], [1,1,1]) == True
