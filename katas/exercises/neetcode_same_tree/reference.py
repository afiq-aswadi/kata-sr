"""Same Tree - LeetCode 100 - Reference Solution"""

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def is_same_tree(p: TreeNode | None, q: TreeNode | None) -> bool:
    # Both empty
    if not p and not q:
        return True

    # One empty, one not
    if not p or not q:
        return False

    # Values differ
    if p.val != q.val:
        return False

    # Check subtrees
    return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)
