"""Subtree of Another Tree - LeetCode 572 - Reference Solution"""

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def is_same_tree(p, q):
    if not p and not q:
        return True
    if not p or not q:
        return False
    if p.val != q.val:
        return False
    return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)

def is_subtree(root: TreeNode | None, sub_root: TreeNode | None) -> bool:
    if not sub_root:
        return True
    if not root:
        return False

    # Check if trees match at current node
    if is_same_tree(root, sub_root):
        return True

    # Check left and right subtrees
    return is_subtree(root.left, sub_root) or is_subtree(root.right, sub_root)
