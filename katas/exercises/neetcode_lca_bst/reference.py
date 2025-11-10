"""Lowest Common Ancestor of a Binary Search Tree - LeetCode 235 - Reference Solution"""

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def lowest_common_ancestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    current = root

    while current:
        # Both nodes are in left subtree
        if p.val < current.val and q.val < current.val:
            current = current.left
        # Both nodes are in right subtree
        elif p.val > current.val and q.val > current.val:
            current = current.right
        # Split point found (or one equals current)
        else:
            return current

    return None
