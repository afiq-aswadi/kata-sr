"""Diameter of Binary Tree - LeetCode 543 - Reference Solution"""

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def diameter_of_binary_tree(root: TreeNode | None) -> int:
    diameter = 0

    def height(node):
        nonlocal diameter
        if not node:
            return 0

        left_height = height(node.left)
        right_height = height(node.right)

        # Update diameter if path through current node is longer
        diameter = max(diameter, left_height + right_height)

        return 1 + max(left_height, right_height)

    height(root)
    return diameter
