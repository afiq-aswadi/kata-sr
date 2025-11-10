"""Construct Binary Tree from Preorder and Inorder Traversal - LeetCode 105 - Reference Solution"""

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def build_tree(preorder: list[int], inorder: list[int]) -> TreeNode | None:
    if not preorder or not inorder:
        return None

    # First element in preorder is root
    root = TreeNode(preorder[0])

    # Find root position in inorder
    mid = inorder.index(preorder[0])

    # Recursively build left and right subtrees
    root.left = build_tree(preorder[1:mid+1], inorder[:mid])
    root.right = build_tree(preorder[mid+1:], inorder[mid+1:])

    return root
