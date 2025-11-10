"""Binary Tree Maximum Path Sum - LeetCode 124 - Reference Solution"""

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def max_path_sum(root: TreeNode | None) -> int:
    max_sum = float('-inf')

    def max_gain(node):
        nonlocal max_sum
        if not node:
            return 0

        # Max gain from left and right subtrees (ignore negative gains)
        left_gain = max(max_gain(node.left), 0)
        right_gain = max(max_gain(node.right), 0)

        # Path sum through current node
        current_path_sum = node.val + left_gain + right_gain

        # Update global max
        max_sum = max(max_sum, current_path_sum)

        # Return max gain if we continue from this node
        return node.val + max(left_gain, right_gain)

    max_gain(root)
    return max_sum
