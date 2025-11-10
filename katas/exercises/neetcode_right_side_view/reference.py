"""Binary Tree Right Side View - LeetCode 199 - Reference Solution"""

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def right_side_view(root: TreeNode | None) -> list[int]:
    if not root:
        return []

    result = []
    queue = [root]

    while queue:
        level_size = len(queue)

        for i in range(level_size):
            node = queue.pop(0)

            # Add rightmost node of this level
            if i == level_size - 1:
                result.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

    return result
