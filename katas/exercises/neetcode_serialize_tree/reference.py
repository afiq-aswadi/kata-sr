"""Serialize and Deserialize Binary Tree - LeetCode 297 - Reference Solution"""

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def serialize(root: TreeNode | None) -> str:
    """Encodes a tree to a single string using preorder traversal."""
    def dfs(node):
        if not node:
            return ["null"]
        return [str(node.val)] + dfs(node.left) + dfs(node.right)

    return ",".join(dfs(root))

def deserialize(data: str) -> TreeNode | None:
    """Decodes your encoded data to tree."""
    def dfs(vals):
        val = next(vals)
        if val == "null":
            return None
        node = TreeNode(int(val))
        node.left = dfs(vals)
        node.right = dfs(vals)
        return node

    return dfs(iter(data.split(",")))
